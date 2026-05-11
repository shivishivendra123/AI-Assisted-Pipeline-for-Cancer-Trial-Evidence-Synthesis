
import requests

def query_pubmed_api(mesh_query):   
    """
    Queries the PubMed API with the given MeSH query and returns the response.

    Args:
        mesh_query (str): The MeSH query string.

    Returns:
        dict: The JSON response from the PubMed API.
    """

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": mesh_query,
        "retmode": "json",
        "retmax": 200
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()