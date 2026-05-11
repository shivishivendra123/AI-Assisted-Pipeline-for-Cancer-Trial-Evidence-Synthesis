"""
Agent-based MeSH term generator that replicates the mesh API functionality.
Uses an LLM agent to generate MeSH terms from PICO and augmented PICO data.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from agents.factory import build_vertex_agent
from configs.env_config import config
from meshOnDemand.mesh_utils.mesh_response_parser import build_mesh_query
from meshOnDemand.advanced_query_builder import build_pico_query_advanced, build_simple_mesh_query
from pubmedSearch.pubmedApiSearch import query_pubmed_api


def _agent():
    """Create a Vertex AI agent for MeSH term generation."""
    return build_vertex_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
        endpoint_id=config.GCP_ENDPOINT_ID,
        dedicated_dns_or_predict_url=config.GCP_DEDICATED_DNS,
        temperature=0.2,
        max_tokens=4000  # Increased to prevent truncation
    )


def _load_prompt_template() -> str:
    """Load the MeSH generation prompt template."""
    base = Path(__file__).resolve().parent
    # Try to use v2 prompt, fallback to v1
    prompt_path_v2 = base.parent / "prompts" / "mesh_generation_prompt_v2.txt"
    if prompt_path_v2.exists():
        return prompt_path_v2.read_text(encoding="utf-8")
    else:
        prompt_path = base.parent / "prompts" / "mesh_generation_prompt.txt"
        return prompt_path.read_text(encoding="utf-8")


def _prepare_prompt(pico_json: Dict[str, Any]) -> str:
    """
    Prepare the prompt by substituting PICO and augmented PICO values.

    Args:
        pico_json: Dictionary containing 'pico_valid' and optionally 'augmented' fields

    Returns:
        str: The prepared prompt with all substitutions
    """
    template = _load_prompt_template()

    pico = pico_json.get('pico_valid', {})
    augmented = pico_json.get('augmented', {})

    # Extract values
    population = pico.get('Population', 'Not specified')
    intervention = pico.get('Intervention', 'Not specified')
    comparator = pico.get('Comparator', 'Not specified') if pico.get('Comparator') else 'Not specified'
    outcomes = pico.get('Outcomes', [])

    # Extract synonyms from augmented PICO
    population_synonyms = []
    intervention_synonyms = []
    comparator_synonyms = []
    outcomes_synonyms = []

    if augmented:
        if 'Population' in augmented:
            population_synonyms = augmented['Population'].get('synonyms', [])
        if 'Intervention' in augmented:
            intervention_synonyms = augmented['Intervention'].get('synonyms', [])
        if 'Comparator' in augmented and augmented['Comparator']:
            comparator_synonyms = augmented['Comparator'].get('synonyms', [])
        if 'Outcomes' in augmented:
            for outcome in augmented['Outcomes']:
                outcomes_synonyms.extend(outcome.get('synonyms', []))

    # Format lists as JSON arrays for the prompt
    outcomes_str = json.dumps(outcomes, ensure_ascii=False)
    population_synonyms_str = json.dumps(population_synonyms, ensure_ascii=False)
    intervention_synonyms_str = json.dumps(intervention_synonyms, ensure_ascii=False)
    comparator_synonyms_str = json.dumps(comparator_synonyms, ensure_ascii=False)
    outcomes_synonyms_str = json.dumps(outcomes_synonyms, ensure_ascii=False)

    # Substitute in template
    prompt = template.replace("{{POPULATION}}", population)
    prompt = prompt.replace("{{POPULATION_SYNONYMS}}", population_synonyms_str)
    prompt = prompt.replace("{{INTERVENTION}}", intervention)
    prompt = prompt.replace("{{INTERVENTION_SYNONYMS}}", intervention_synonyms_str)
    prompt = prompt.replace("{{COMPARATOR}}", comparator)
    prompt = prompt.replace("{{COMPARATOR_SYNONYMS}}", comparator_synonyms_str)
    prompt = prompt.replace("{{OUTCOMES}}", outcomes_str)
    prompt = prompt.replace("{{OUTCOMES_SYNONYMS}}", outcomes_synonyms_str)

    return prompt


def _parse_mesh_response_v2(response_text: str) -> Dict[str, List[str]]:
    """
    Parse the agent's response to extract categorized MeSH terms (v2 format).

    Args:
        response_text: The agent's JSON response with PICO-categorized terms

    Returns:
        dict: Dictionary with PICO-categorized MeSH terms
    """
    try:
        # Clean the response text
        cleaned_text = response_text.strip()

        # Remove markdown code fences if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        cleaned_text = cleaned_text.strip()

        # Try to parse as JSON
        data = json.loads(cleaned_text)

        # Validate structure (v2 format)
        if 'population_terms' in data or 'intervention_terms' in data:
            return {
                'population_terms': data.get('population_terms', []),
                'intervention_terms': data.get('intervention_terms', []),
                'comparator_terms': data.get('comparator_terms', []),
                'outcome_terms': data.get('outcome_terms', []),
                'general_terms': data.get('general_terms', [])
            }
        else:
            # Old format, return empty v2 structure
            return {
                'population_terms': [],
                'intervention_terms': [],
                'comparator_terms': [],
                'outcome_terms': [],
                'general_terms': []
            }

    except json.JSONDecodeError:
        # Try to salvage using regex
        population_terms = []
        intervention_terms = []
        comparator_terms = []
        outcome_terms = []
        general_terms = []

        # Try to extract each category
        for category, terms_list in [
            ('population_terms', population_terms),
            ('intervention_terms', intervention_terms),
            ('comparator_terms', comparator_terms),
            ('outcome_terms', outcome_terms),
            ('general_terms', general_terms)
        ]:
            match = re.search(rf'"{category}"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
            if match:
                terms_str = match.group(1)
                extracted_terms = re.findall(r'"([^"]+)"', terms_str)
                terms_list.extend(extracted_terms)

        return {
            'population_terms': population_terms,
            'intervention_terms': intervention_terms,
            'comparator_terms': comparator_terms,
            'outcome_terms': outcome_terms,
            'general_terms': general_terms
        }
    except ValueError:
        return {
            'population_terms': [],
            'intervention_terms': [],
            'comparator_terms': [],
            'outcome_terms': [],
            'general_terms': []
        }


def _parse_mesh_response(response_text: str) -> Dict[str, List[str]]:
    """
    Parse the agent's response to extract MeSH terms.

    Args:
        response_text: The agent's JSON response

    Returns:
        dict: Dictionary with 'mesh_terms' and 'relevant_mesh_terms' lists
    """
    try:
        # Clean the response text
        cleaned_text = response_text.strip()

        # Remove markdown code fences if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]  # Remove ```json
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]  # Remove ```

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]  # Remove trailing ```

        cleaned_text = cleaned_text.strip()

        # Try to parse as JSON
        data = json.loads(cleaned_text)

        # Validate structure
        if 'mesh_terms' not in data or 'relevant_mesh_terms' not in data:
            raise ValueError("Response missing required fields")

        return {
            'mesh_terms': data.get('mesh_terms', []),
            'relevant_mesh_terms': data.get('relevant_mesh_terms', [])
        }
    except json.JSONDecodeError:
        # Try to salvage what we can using regex
        mesh_terms = []
        relevant_mesh_terms = []

        # Try to extract mesh_terms array
        mesh_match = re.search(r'"mesh_terms"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
        if mesh_match:
            terms_str = mesh_match.group(1)
            mesh_terms = re.findall(r'"([^"]+)"', terms_str)

        # Try to extract relevant_mesh_terms array
        relevant_match = re.search(r'"relevant_mesh_terms"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
        if relevant_match:
            terms_str = relevant_match.group(1)
            relevant_mesh_terms = re.findall(r'"([^"]+)"', terms_str)

        if mesh_terms or relevant_mesh_terms:
            return {
                'mesh_terms': mesh_terms,
                'relevant_mesh_terms': relevant_mesh_terms
            }

        # Complete fallback: return empty lists
        return {
            'mesh_terms': [],
            'relevant_mesh_terms': []
        }
    except ValueError:
        # Fallback: return empty lists
        return {
            'mesh_terms': [],
            'relevant_mesh_terms': []
        }


def generate_mesh_terms_with_agent(pico_json: Dict[str, Any]) -> str:
    """
    Generate MeSH terms using an LLM agent instead of the mesh API.

    This function replicates the functionality of query_mesh_api() but uses
    an agent to generate MeSH terms from PICO and augmented PICO data.

    Args:
        pico_json: Dictionary containing:
            - qid: Query ID
            - pico_valid: PICO dictionary with Population, Intervention, Comparator, Outcomes
            - augmented: AugmentedPICO dictionary with synonyms (optional)

    Returns:
        str: The DNF mesh query string that can be passed to PubMed search
    """
    # Prepare the prompt
    prompt = _prepare_prompt(pico_json)

    # Create agent and generate MeSH terms
    agent = _agent()
    agent.set_system("Return only valid JSON. No explanations. No code fences. No markdown. Just pure JSON.")

    try:
        response = agent.say(prompt)

        # Parse the response (try v2 format first)
        mesh_response = _parse_mesh_response_v2(response)
        mesh_response['qid'] = pico_json['qid']

        # Save artifacts
        base = Path(__file__).resolve().parent
        artifacts_dir = base.parent / "artifacts_day3"
        artifacts_dir.mkdir(exist_ok=True)

        # Save the raw agent response
        (artifacts_dir / f"mesh_agent_{pico_json['qid']}.txt").write_text(
            response, encoding="utf-8"
        )

        # Save the parsed response
        (artifacts_dir / f"mesh_agent_parsed{pico_json['qid']}.json").write_text(
            json.dumps(mesh_response, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Check if we have any terms
        all_terms = (
            mesh_response.get('population_terms', []) +
            mesh_response.get('intervention_terms', []) +
            mesh_response.get('comparator_terms', []) +
            mesh_response.get('outcome_terms', []) +
            mesh_response.get('general_terms', [])
        )

        if not all_terms:
            return _fallback_query(pico_json)

        # Build advanced PICO query
        mesh_query = build_pico_query_advanced(pico_json, mesh_response)

        # Save the mesh query
        temp_dict = {"qid": pico_json['qid'], "mesh_query": mesh_query}
        with open(artifacts_dir / "mesh_query.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")

        # Query PubMed with the generated mesh terms
        pubmed_response = query_pubmed_api(mesh_query)
        pubmed_response['qid'] = pico_json['qid']
        (artifacts_dir / f"pubmed_parsed{pico_json['qid']}.json").write_text(
            json.dumps(pubmed_response, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

        print(mesh_query)
        return mesh_query
    

    except Exception:
        return _fallback_query(pico_json)


def _fallback_query(pico_json: Dict[str, Any]) -> str:
    """
    Generate a fallback query from PICO terms when agent generation fails.

    Args:
        pico_json: Dictionary containing pico_valid

    Returns:
        str: Basic PICO-based query
    """
    pico_valid = pico_json.get('pico_valid', {})
    terms = []

    if pico_valid.get('Population'):
        terms.append(f"({pico_valid['Population']}[MeSH Terms])")
    if pico_valid.get('Intervention'):
        terms.append(f"({pico_valid['Intervention']}[MeSH Terms])")
    if pico_valid.get('Comparator'):
        terms.append(f"({pico_valid['Comparator']}[MeSH Terms])")
    if pico_valid.get('Outcomes'):
        for outcome in pico_valid['Outcomes']:
            terms.append(f"({outcome}[MeSH Terms])")

    fallback_query = " AND ".join(terms) if terms else "clinical trial[Publication Type]"
    return fallback_query


if __name__ == "__main__":
    # Example usage
    example_pico = {
        "qid": "test_123",
        "pico_valid": {
            "Population": "adults with septic shock",
            "Intervention": "early norepinephrine",
            "Comparator": "dopamine",
            "Outcomes": ["28-day mortality"]
        },
        "augmented": {
            "Population": {
                "value": "adults with septic shock",
                "synonyms": ["patients with septic shock", "individuals with septic shock"]
            },
            "Intervention": {
                "value": "early norepinephrine",
                "synonyms": ["norepinephrine", "early-norepinephrine"]
            },
            "Comparator": {
                "value": "dopamine",
                "synonyms": ["dopamine treatment", "dopamines"]
            },
            "Outcomes": [
                {
                    "value": "28-day mortality",
                    "synonyms": ["28-day mortality rate", "28 day mortality rate"]
                }
            ]
        }
    }

    result = generate_mesh_terms_with_agent(example_pico)
    print(f"\nFinal mesh query: {result}")
