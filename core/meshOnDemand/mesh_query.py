import requests
from pathlib import Path
import json
from meshOnDemand.mesh_utils.mesh_response_parser import extract_from_pre_block, build_mesh_query
from pubmedSearch.pubmedApiSearch import query_pubmed_api

# json_query = {
#   "qid": "6d5487fd37ad",
#   "created_at": "2025-10-07T00:32:09Z",
#   "pico_valid": {
#     "Population": "adults with septic shock",
#     "Intervention": "early norepinephrine",
#     "Comparator": "dopamine",
#     "Outcomes": [
#       "28-day mortality"
#     ]
#   },
#   "augmented": {
#     "Population": {
#       "value": "adults with septic shock",
#       "synonyms": [
#         "patients with septic shock",
#         "individuals with septic shock",
#         "patients-with-septic-shock",
#         "individuals-with-septic-shock"
#       ]
#     },
#     "Intervention": {
#       "value": "early norepinephrine",
#       "synonyms": [
#         "norepinephrine",
#         "norepinephrines",
#         "early-norepinephrine"
#       ]
#     },
#     "Comparator": {
#       "value": "dopamine",
#       "synonyms": [
#         "dopamine treatment",
#         "dopamines",
#         "dopamine-treatment"
#       ]
#     },
#     "Outcomes": [
#       {
#         "value": "28-day mortality",
#         "synonyms": [
#           "28-day mortality rate",
#           "28 day mortality rate",
#           "28-day-mortality-rate"
#         ]
#       }
#     ]
#   },
#   "model": "gemma-3 via Vertex",
#   "temperature": 0.2
# }

def query_mesh_api(pico_json: json):
    """
    Queries the Mesh API with the given Pico query and returns the response.

    Args:
        pico_json (dict): The Pico query JSON.

    Returns:
        dict: The txt response from the Mesh API.
    """
    headers = {"Content-Type": "application/json"}

    # Prepare payload
    payload = {"input": json.dumps(pico_json['pico_valid'])}
    print("-----------")
    print(payload)
    print("-----------")
    try:
        response = requests.post(
            
            "https://meshb.nlm.nih.gov/api/MOD",
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] MeSH API request failed: {e}")
        raise

    if response.status_code == 200:
        base = Path(__file__).resolve().parent
        (base.parent / "artifacts_day3" / f"mesh_{pico_json['qid']}.txt").write_text(response.text, encoding="utf-8")
        mesh_response = extract_from_pre_block(response.text)
        mesh_response['qid'] = pico_json['qid']
        (base.parent / "artifacts_day3" / f"mesh_parsed{pico_json['qid']}.json").write_text(json.dumps(mesh_response, ensure_ascii=False, indent=2), encoding="utf-8")

        mesh_query = build_mesh_query(mesh_response['mesh_terms']+mesh_response['relevant_mesh_terms'])
        print(mesh_query)
        temp_dict = {"qid": pico_json['qid'], "mesh_query": mesh_query}
        with open(base.parent / "artifacts_day3" / "mesh_query.jsonl", "a", encoding="utf-8") as f:
          f.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")  

        pubmed_response = query_pubmed_api(mesh_query)
        pubmed_response['qid'] = pico_json['qid']
        (base.parent / "artifacts_day3" / f"pubmed_parsed{pico_json['qid']}.json").write_text(
            json.dumps(pubmed_response, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        return mesh_query
    else:
        print(f"[ERROR] MeSH API returned status {response.status_code}")
        print(f"[ERROR] Request payload: {json.dumps(payload, indent=2)}")
        print(f"[ERROR] Response: {response.text[:500]}")

        # Fallback: Generate basic query from PICO terms
        print("[WARN] Falling back to basic PICO-based query...")
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
        print(f"[INFO] Fallback query: {fallback_query}")
        return fallback_query

if __name__ == "__main__":
    result = query_mesh_api()
    # print(result)