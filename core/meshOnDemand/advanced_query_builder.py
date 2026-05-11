"""
Advanced PubMed query builder that creates sophisticated search queries
similar to PubMed's auto-expansion with field tags and synonyms.
"""

from typing import List, Dict, Any, Optional


def build_term_group(term: str, synonyms: List[str] = None, use_mesh: bool = True) -> str:
    """
    Build a group of OR'd terms with field tags for a single concept.

    Args:
        term: The main term
        synonyms: List of synonym terms
        use_mesh: Whether to include MeSH Terms field

    Returns:
        str: OR'd group like ("term"[MeSH Terms] OR "term"[All Fields] OR "synonym1"[All Fields])
    """
    if synonyms is None:
        synonyms = []

    parts = []

    # Add MeSH term if requested
    if use_mesh:
        parts.append(f'"{term}"[MeSH Terms]')

    # Add main term in All Fields
    parts.append(f'"{term}"[All Fields]')

    # Add synonyms in All Fields
    for syn in synonyms[:3]:  # Limit to top 3 synonyms to avoid query explosion
        if syn.lower() != term.lower():
            parts.append(f'"{syn}"[All Fields]')

    return "(" + " OR ".join(parts) + ")"


def build_mesh_terms_group(mesh_terms: List[str]) -> str:
    """
    Build a group of OR'd MeSH terms.

    Args:
        mesh_terms: List of MeSH terms

    Returns:
        str: OR'd MeSH terms like ("Term1"[MeSH Terms] OR "Term2"[MeSH Terms])
    """
    if not mesh_terms:
        return ""

    parts = [f'"{term}"[MeSH Terms]' for term in mesh_terms]
    return "(" + " OR ".join(parts) + ")"


def build_pico_query_advanced(pico_json: Dict[str, Any], mesh_response: Dict[str, Any]) -> str:
    """
    Build an advanced PubMed query using PICO components and MeSH terms.

    Query structure:
    (Population MeSH OR synonyms) AND
    (Intervention MeSH OR synonyms) AND
    (Comparator MeSH OR synonyms) AND
    (Outcome MeSH OR synonyms)

    Args:
        pico_json: Dictionary with pico_valid and augmented fields
        mesh_response: Dictionary with categorized MeSH terms by PICO component

    Returns:
        str: Complex PubMed query
    """
    pico = pico_json.get('pico_valid', {})
    augmented = pico_json.get('augmented', {})

    query_parts = []

    # Population
    if pico.get('Population') and mesh_response.get('population_terms'):
        pop_mesh = build_mesh_terms_group(mesh_response['population_terms'])
        pop_synonyms = augmented.get('Population', {}).get('synonyms', [])
        pop_text = build_term_group(pico['Population'], pop_synonyms, use_mesh=False)

        if pop_mesh and pop_text:
            query_parts.append(f"({pop_mesh} OR {pop_text})")
        elif pop_mesh:
            query_parts.append(pop_mesh)
        elif pop_text:
            query_parts.append(pop_text)

    # Intervention
    if pico.get('Intervention') and mesh_response.get('intervention_terms'):
        int_mesh = build_mesh_terms_group(mesh_response['intervention_terms'])
        int_synonyms = augmented.get('Intervention', {}).get('synonyms', [])
        int_text = build_term_group(pico['Intervention'], int_synonyms, use_mesh=False)

        if int_mesh and int_text:
            query_parts.append(f"({int_mesh} OR {int_text})")
        elif int_mesh:
            query_parts.append(int_mesh)
        elif int_text:
            query_parts.append(int_text)

    # Comparator (optional)
    if pico.get('Comparator') and mesh_response.get('comparator_terms'):
        comp_mesh = build_mesh_terms_group(mesh_response['comparator_terms'])
        comp_synonyms = augmented.get('Comparator', {}).get('synonyms', [])
        comp_text = build_term_group(pico['Comparator'], comp_synonyms, use_mesh=False)

        if comp_mesh and comp_text:
            query_parts.append(f"({comp_mesh} OR {comp_text})")
        elif comp_mesh:
            query_parts.append(comp_mesh)
        elif comp_text:
            query_parts.append(comp_text)

    # Outcomes
    if pico.get('Outcomes') and mesh_response.get('outcome_terms'):
        outcome_mesh = build_mesh_terms_group(mesh_response['outcome_terms'])

        # Build outcome text groups
        outcome_text_parts = []
        outcomes_augmented = augmented.get('Outcomes', [])
        for i, outcome in enumerate(pico['Outcomes']):
            outcome_synonyms = []
            if i < len(outcomes_augmented):
                outcome_synonyms = outcomes_augmented[i].get('synonyms', [])
            outcome_text_parts.append(build_term_group(outcome, outcome_synonyms, use_mesh=False))

        outcome_text = "(" + " OR ".join(outcome_text_parts) + ")" if outcome_text_parts else ""

        if outcome_mesh and outcome_text:
            query_parts.append(f"({outcome_mesh} OR {outcome_text})")
        elif outcome_mesh:
            query_parts.append(outcome_mesh)
        elif outcome_text:
            query_parts.append(outcome_text)

    # Join with AND
    if not query_parts:
        return "clinical trial[Publication Type]"

    return " AND ".join(query_parts)


def build_simple_mesh_query(mesh_terms: List[str], operator: str = "OR") -> str:
    """
    Build a simple query from MeSH terms (backward compatible).

    Args:
        mesh_terms: List of MeSH terms
        operator: Boolean operator (OR or AND)

    Returns:
        str: Simple query like "Term1"[MeSH Terms] OR "Term2"[MeSH Terms]
    """
    if not mesh_terms:
        return ""

    return f" {operator} ".join([f'"{term}"[MeSH Terms]' for term in mesh_terms])


if __name__ == "__main__":
    # Test example
    pico_json = {
        "pico_valid": {
            "Population": "adults with septic shock",
            "Intervention": "early norepinephrine",
            "Comparator": "dopamine",
            "Outcomes": ["28-day mortality"]
        },
        "augmented": {
            "Population": {
                "value": "adults with septic shock",
                "synonyms": ["patients with septic shock", "septic shock patients"]
            },
            "Intervention": {
                "value": "early norepinephrine",
                "synonyms": ["norepinephrine", "noradrenaline"]
            },
            "Comparator": {
                "value": "dopamine",
                "synonyms": ["dopamine therapy"]
            },
            "Outcomes": [
                {
                    "value": "28-day mortality",
                    "synonyms": ["mortality", "death"]
                }
            ]
        }
    }

    mesh_response = {
        "population_terms": ["Septic Shock", "Shock"],
        "intervention_terms": ["Norepinephrine", "Vasoconstrictor Agents"],
        "comparator_terms": ["Dopamine"],
        "outcome_terms": ["Mortality", "Hospital Mortality"],
        "general_terms": ["Humans", "Adult"]
    }

    query = build_pico_query_advanced(pico_json, mesh_response)
    print("Generated Query:")
    print(query)
    print()
    print("Query length:", len(query), "characters")
