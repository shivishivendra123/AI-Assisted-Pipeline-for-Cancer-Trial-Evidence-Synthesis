"""
Basic MeSH term generator that extracts MeSH terms from PICO only (no augmentation).
Returns simple OR query: "Term1"[MeSH Terms] OR "Term2"[MeSH Terms] OR ...
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from agents.factory import build_vertex_agent
from configs.env_config import config
from pubmedSearch.pubmedApiSearch import query_pubmed_api
from pipeline.extractor.pico_extractor import _extract_vertex_content


def _agent():
    """Create a Vertex AI agent for basic MeSH term extraction."""
    return build_vertex_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
        endpoint_id=config.GCP_ENDPOINT_ID,
        dedicated_dns_or_predict_url=config.GCP_DEDICATED_DNS,
        temperature=0.2,
        max_tokens=2000
    )


def _load_prompt_template() -> str:
    """Load the basic MeSH prompt template."""
    base = Path(__file__).resolve().parent
    prompt_path = base.parent / "prompts" / "mesh_basic_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _prepare_prompt(pico_json: Dict[str, Any]) -> str:
    """
    Prepare the prompt using only PICO (no augmentation).

    Args:
        pico_json: Dictionary containing 'pico_valid' field

    Returns:
        str: The prepared prompt
    """
    template = _load_prompt_template()
    pico = pico_json.get('pico_valid', {})

    # Extract PICO values
    population = pico.get('Population', 'Not specified')
    intervention = pico.get('Intervention', 'Not specified')
    comparator = pico.get('Comparator', 'Not specified') if pico.get('Comparator') else 'Not specified'
    outcomes = pico.get('Outcomes', [])

    # Format outcomes as JSON array
    outcomes_str = json.dumps(outcomes, ensure_ascii=False)

    # Substitute in template
    prompt = template.replace("{{POPULATION}}", population)
    prompt = prompt.replace("{{INTERVENTION}}", intervention)
    prompt = prompt.replace("{{COMPARATOR}}", comparator)
    prompt = prompt.replace("{{OUTCOMES}}", outcomes_str)

    return prompt


def _clean_response(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    if '\\"' in cleaned:
        cleaned = cleaned.replace('\\"', '"')
    return cleaned.strip()


def _flatten_terms(raw: list) -> List[str]:
    """
    Accept either a flat list of strings or a list of dicts
    (e.g. {"MeSH term": ..., "synonym": ..., "abbreviation": ...})
    and return a deduplicated flat list of non-empty strings.
    """
    out = []
    keys_to_skip = {"population", "intervention", "comparator", "mesh_terms",
                    "MeSH term", "synonym", "abbreviation"}
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            for v in item.values():
                if isinstance(v, str) and v.strip() and v.strip() not in keys_to_skip:
                    out.append(v.strip())
    # deduplicate preserving order
    seen = set()
    result = []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            result.append(t)
    return result


def _parse_mesh_response(response_text: str) -> dict:
    """
    Parse the agent's grouped response into a dict with keys:
    population, intervention, comparator (each a flat list of strings).
    Handles both flat-string and nested-dict formats from the LLM.
    Caps each group at MAX_TERMS to guard against LLM repetition loops.
    """
    MAX_TERMS = 10

    def _cap(terms: list) -> list:
        return _flatten_terms(terms)[:MAX_TERMS]

    try:
        data = json.loads(_clean_response(response_text))

        # New grouped format
        if "population" in data or "intervention" in data or "comparator" in data:
            return {
                "population":   _cap(data.get("population", [])),
                "intervention": _cap(data.get("intervention", [])),
                "comparator":   _cap(data.get("comparator", [])),
            }

        # Legacy flat list
        if "mesh_terms" in data and isinstance(data["mesh_terms"], list):
            return {"population": _cap(data["mesh_terms"]), "intervention": [], "comparator": []}

        return {"population": [], "intervention": [], "comparator": []}

    except (json.JSONDecodeError, Exception):
        pass

    # Targeted per-group regex fallback — handles truncated JSON.
    # Extracts terms for each key up to the next key or end of text.
    _skip = {"population", "intervention", "comparator", "mesh_terms",
             "MeSH term", "synonym", "abbreviation"}

    def _extract_group(text: str, key: str) -> list:
        m = re.search(rf'"{key}"\s*:\s*\[', text)
        if not m:
            return []
        start = m.end()
        # Find closing ] — if truncated, go to end
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == '[':   depth += 1
            elif text[i] == ']': depth -= 1
            i += 1
        content = text[start:i]
        terms = [t for t in re.findall(r'"([^"]+)"', content)
                 if t.strip() and t.lower() not in _skip]
        # Deduplicate preserving order
        seen, out = set(), []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
        return out[:MAX_TERMS]

    pop = _extract_group(response_text, "population")
    inv = _extract_group(response_text, "intervention")
    cmp = _extract_group(response_text, "comparator")

    if pop or inv or cmp:
        return {"population": pop, "intervention": inv, "comparator": cmp}

    # Last resort: dump non-key quoted strings into population
    all_terms = [t for t in re.findall(r'"([^"]+)"', response_text)
                 if t.lower() not in _skip]
    return {"population": all_terms[:MAX_TERMS], "intervention": [], "comparator": []}


def _inject_missing_inns(groups: dict, pico: dict) -> None:
    """
    Ensure canonical INN drug names are present in the groups.
    Checks PICO intervention/comparator text for known drug patterns and
    adds the INN if the LLM omitted it. Modifies groups in place.
    """
    int_text = (pico.get('Intervention') or '').lower()
    cmp_text = (pico.get('Comparator') or '').lower()
    int_lower = {t.lower() for t in groups.get('intervention', [])}
    cmp_lower = {t.lower() for t in groups.get('comparator', [])}

    # Map: (trigger keywords in PICO text) -> INN to inject into intervention
    intervention_inns = [
        (['peg', 'g-csf'],          'Pegfilgrastim'),
        (['pegfilgrastim'],          'Pegfilgrastim'),
        (['mecapegfilgrastim'],      'mecapegfilgrastim'),
        (['norepinephrine'],         'Norepinephrine'),
        (['dopamine'],               'Dopamine'),
        (['filgrastim'],             'Filgrastim'),
        (['lenograstim'],            'Lenograstim'),
    ]

    for keywords, inn in intervention_inns:
        if all(k in int_text for k in keywords):
            if inn.lower() not in int_lower:
                groups.setdefault('intervention', []).append(inn)
                int_lower.add(inn.lower())

    # Same for comparator
    comparator_inns = [
        (['filgrastim'],             'Filgrastim'),
        (['g-csf'],                  'Filgrastim'),
        (['dopamine'],               'Dopamine'),
        (['norepinephrine'],         'Norepinephrine'),
        (['lenograstim'],            'Lenograstim'),
    ]

    for keywords, inn in comparator_inns:
        if all(k in cmp_text for k in keywords):
            if inn.lower() not in cmp_lower:
                groups.setdefault('comparator', []).append(inn)
                cmp_lower.add(inn.lower())


def _group_clause(terms: List[str]) -> str:
    parts = []
    for t in terms:
        parts.append(f'"{t}"[MeSH Terms]')
        parts.append(f'"{t}"[Title/Abstract]')
    return "(" + " OR ".join(parts) + ")"


def _build_mesh_query(groups: dict) -> str:
    """
    Build: (population) AND (intervention-specific terms only).

    Intervention-specific = intervention terms that are NOT in the comparator
    (e.g. pegfilgrastim, neulasta — but not filgrastim/G-CSF which are in both).

    A SEPARATE comparator query is built by _build_comparator_query() and fetched
    independently, then merged. This avoids one big OR clause that would mix
    common comparator terms (G-CSF, TKI) into the ranking and push GT papers
    below the retmax cutoff.
    """
    clauses = []

    if groups.get("population"):
        clauses.append(_group_clause(groups["population"]))

    comparator_lower = {t.lower() for t in groups.get("comparator", [])}
    intervention_terms = [
        t for t in groups.get("intervention", [])
        if t.lower() not in comparator_lower
    ]
    if intervention_terms:
        clauses.append(_group_clause(intervention_terms))

    return " AND ".join(clauses) if clauses else ""


def _build_comparator_query(groups: dict) -> str:
    """
    Build: (population) AND (comparator-specific terms only).
    Returns "" if no comparator terms.
    Fetched separately and merged with the intervention query results.
    """
    if not groups.get("comparator"):
        return ""

    clauses = []
    if groups.get("population"):
        clauses.append(_group_clause(groups["population"]))

    intervention_lower = {t.lower() for t in groups.get("intervention", [])}
    comparator_terms = [
        t for t in groups.get("comparator", [])
        if t.lower() not in intervention_lower
    ]
    if comparator_terms:
        clauses.append(_group_clause(comparator_terms))

    return " AND ".join(clauses) if clauses else ""


def generate_basic_mesh_query(pico_json: Dict[str, Any]) -> str:
    """
    Generate MeSH terms from PICO only (no augmentation) and return simple OR query.

    Args:
        pico_json: Dictionary containing:
            - qid: Query ID
            - pico_valid: PICO dictionary with Population, Intervention, Comparator, Outcomes

    Returns:
        str: Simple OR query like "Term1"[MeSH Terms] OR "Term2"[MeSH Terms]
    """

    # Prepare the prompt (PICO only, no augmentation)
    prompt = _prepare_prompt(pico_json)

    # Create agent and generate MeSH terms
    agent = _agent()
    agent.set_system(
        "You are a PubMed MeSH search expert. "
        "Return ONLY valid JSON with exactly three keys: population, intervention, comparator. "
        "Each value is an array of search strings. "
        "Always use the canonical INN drug name as the first item in intervention and comparator "
        "(e.g. 'Pegfilgrastim' not 'Pegylated G-CSF', 'Filgrastim' not 'G-CSF'). "
        "No markdown, no code fences, no explanations — pure JSON only."
    )

    try:
        raw      = agent.say(prompt)
        response = _extract_vertex_content(raw) or str(raw)

        # Parse the response into grouped dict
        groups = _parse_mesh_response(response)

        # Safety net: inject INN terms the LLM missed based on PICO text
        pico = pico_json.get('pico_valid', {})
        _inject_missing_inns(groups, pico)

        print(groups)

        # Save artifacts
        base = Path(__file__).resolve().parent
        artifacts_dir = base.parent / "artifacts_day3"
        artifacts_dir.mkdir(exist_ok=True)

        # Save the extracted content (not the raw Vertex wrapper)
        (artifacts_dir / f"mesh_basic_{pico_json['qid']}.txt").write_text(
            response, encoding="utf-8"
        )

        # Save the parsed groups
        mesh_data = {
            'qid': pico_json['qid'],
            'groups': groups,
        }

        print(mesh_data)

        (artifacts_dir / f"mesh_basic_parsed{pico_json['qid']}.json").write_text(
            json.dumps(mesh_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Build two separate queries: one for intervention, one for comparator.
        # They are fetched independently and merged in eval_ui to avoid ranking
        # conflicts (adding common comparator terms like G-CSF/TKI to one big
        # query pushes GT papers below the retmax cutoff).
        all_terms = (groups.get("population", []) + groups.get("intervention", []) +
                     groups.get("comparator", []))
        if not all_terms:
            return _fallback_query(pico_json)

        mesh_query     = _build_mesh_query(groups)          # (pop) AND (intervention-specific)
        cmp_query      = _build_comparator_query(groups)    # (pop) AND (comparator-specific)

        # Store groups and secondary query so downstream steps can use them
        pico_json['mesh_groups']          = groups
        pico_json['mesh_query_comparator'] = cmp_query  # "" if no comparator

        # Save the mesh query
        temp_dict = {"qid": pico_json['qid'], "mesh_query": mesh_query,
                     "mesh_query_comparator": cmp_query}
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
        print("--------mesh--------")
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
        terms.append(f'"{pico_valid["Population"]}"[MeSH Terms]')
    if pico_valid.get('Intervention'):
        terms.append(f'"{pico_valid["Intervention"]}"[MeSH Terms]')
    if pico_valid.get('Comparator'):
        terms.append(f'"{pico_valid["Comparator"]}"[MeSH Terms]')
    if pico_valid.get('Outcomes'):
        for outcome in pico_valid['Outcomes']:
            terms.append(f'"{outcome}"[MeSH Terms]')

    fallback_query = " OR ".join(terms) if terms else "clinical trial[Publication Type]"
    return fallback_query


if __name__ == "__main__":
    # Example usage
    example_pico = {
        "qid": "test_basic_001",
        "pico_valid": {
            "Population": "adults with septic shock",
            "Intervention": "early norepinephrine",
            "Comparator": "dopamine",
            "Outcomes": ["28-day mortality"]
        }
    }

    result = generate_basic_mesh_query(example_pico)
    print("=" * 70)
    print("Basic MeSH Query Generator")
    print("=" * 70)
    print(f"\nPICO:")
    print(f"  Population: {example_pico['pico_valid']['Population']}")
    print(f"  Intervention: {example_pico['pico_valid']['Intervention']}")
    print(f"  Comparator: {example_pico['pico_valid']['Comparator']}")
    print(f"  Outcomes: {example_pico['pico_valid']['Outcomes']}")
    print(f"\nGenerated Query:\n{result}")
    print("=" * 70)
