import re

def extract_from_pre_block(text):
    # Extract MeSH Terms
    mesh_terms_block = re.search(r"-- MeSH Terms --(.*?)-- Relevant MeSH Terms --", text, re.DOTALL)
    mesh_terms = [line.strip() for line in mesh_terms_block.group(1).strip().splitlines() if line.strip()] if mesh_terms_block else []

    # Extract Relevant MeSH Terms
    relevant_mesh_block = re.search(r"-- Relevant MeSH Terms --(.*?)-- PubMed/MEDLINE Similar Citations --", text, re.DOTALL)
    relevant_mesh_terms = [line.strip() for line in relevant_mesh_block.group(1).strip().splitlines() if line.strip()] if relevant_mesh_block else []

    # Extract PubMed citations
    pmid_block = re.search(r"-- PubMed/MEDLINE Similar Citations --(.*)", text, re.DOTALL)
    pmid_lines = [line.strip() for line in pmid_block.group(1).strip().splitlines() if line.strip()] if pmid_block else []

    pmid_entries = []
    for line in pmid_lines:
        match = re.match(r"(\d+):\s+(.*)", line)
        if match:
            pmid_entries.append({
                "PMID": match.group(1),
                "Title": match.group(2)
            })
    

    return {"mesh_terms": mesh_terms, "relevant_mesh_terms": relevant_mesh_terms, "pmid_entries": pmid_entries}

def build_mesh_query(mesh_terms, operator="OR"):
    return f" {operator} ".join([f'"{term}"[MeSH Terms]' for term in mesh_terms])