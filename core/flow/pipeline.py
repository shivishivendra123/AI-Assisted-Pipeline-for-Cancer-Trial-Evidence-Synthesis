
from pipeline.pico import pico_pipe
from pipeline.augment.term_expander import expand_terms_with_llm
from schemas.pico import PICO
from meshOnDemand.mesh_query import query_mesh_api
from meshOnDemand.mesh_generator_agent import generate_mesh_terms_with_agent
from efetch_utility.efetch import fetch_pubmed_articles
from eligibility_builder.built_eligibility import build_eligibility
from screening.screening import screen_studies
from extraction.study_char_outcome_ext import extract_study_char_outcomes
from extraction.outcome_extract import extract_study_outcomes
from basic_sythesis.synthesis import run_evidence_synthesis
import csv
import json
from pathlib import Path


def get_valid_studies(csv_path):
    pmids_high_score = []

    # Handle both string and Path objects
    if isinstance(csv_path, str):
        from pathlib import Path
        csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_str = row.get("cumulative_score", "")
            try:
                score = float(score_str)
            except ValueError:
                continue  # skip rows with bad or empty score

            if score >=3:
                pmid = row.get("study_id")
                if pmid:
                    pmids_high_score.append(pmid)

    print("PMIDs with cumulative_score >= 8:")
    for pmid in pmids_high_score:
        print(pmid)
    return pmids_high_score



def run_flow(use_agent_mesh=True, question=None):
    """
    Run the complete evidence synthesis pipeline.

    Args:
        use_agent_mesh (bool): If True, use agent-based MeSH generation. If False, use mesh API.
        question (str): The research question. If None, uses the default question.
    """
    max_studies = 5

    if question is None:
        question = "In adults with early-stage HER2-positive breast cancer, does adjuvant trastuzumab plus chemotherapy, compared with chemotherapy alone, reduce 5-year disease-free survival events (recurrence or death)?"

    print("[Info]======== Started PICO ==========")
    pico_json = pico_pipe(question)
    print("[Info]========= Extracted PICO =============")

    # Stage 2: PICO Augmentation (Term Expansion)
    print("[Info]========= Augmenting PICO terms =============")
    syn_prompt_path = Path("prompts/synonym_prompt.txt")
    syn_prompt = syn_prompt_path.read_text(encoding="utf-8")

    pico_obj = PICO.model_validate(pico_json['pico_valid'])
    augmented_pico = expand_terms_with_llm(pico_obj, syn_prompt)

    # Add augmented PICO to pico_json
    pico_json['augmented'] = json.loads(augmented_pico.model_dump_json())
    print("[Info]========= Finished PICO Augmentation =============")

    # Stage 3: MeSH Query Generation
    print("[Info]========= Extracting Mesh terms =============")
    if use_agent_mesh:
        print("[Info] Using agent-based MeSH generation...")
        mesh_query = generate_mesh_terms_with_agent(pico_json)
    else:
        print("[Info] Using MeSH API...")
        mesh_query = query_mesh_api(pico_json)
    print("[Info]========== Finished Extracting Mesh Terms===========")

    # Stage 4: PubMed Search
    print("[Info]========== Extracting article text ===========")
    data = fetch_pubmed_articles(pico_json['qid'], mesh_query)
    articles , full_text = data['articles'],data['full_text']
    print("[Info]========== Finished Extracting article text ===========")

    # Stage 5: Eligibility Criteria Construction
    print('[Info]========== Contructing Eligibility Criteria=============')
    eligibility_criteria = build_eligibility(pico_json['question'] , pico_json['pico_valid'],pico_json['qid'])
    print('[Info]========== Created Eligibility Criteria')

    # Stage 6: Study Screening
    print('[Info]========== Study Screening Started===========')
    csv_path_screened_studies = screen_studies(articles, full_text, eligibility_criteria, pico_json, max_studies)
    pmids_relevant = get_valid_studies(csv_path_screened_studies)
    print('[Info]========== Study Screening Ended ===========')

    # Stage 7: Data Extraction - Study Characteristics
    print('[Info]========== Extract study chars Started===========')
    csv_path_study_chars = extract_study_char_outcomes(articles,full_text,pico_json,pmids_relevant)
    print('[Info]========== Extract study chars Ended ===========')

    # Stage 8: Data Extraction - Outcomes
    print('[Info]========== Extract study outcome Started===========')
    csv_path_study_outcomes = extract_study_outcomes(articles,full_text,pico_json,pmids_relevant)
    print('[Info]========== Extract study outcome Ended ===========')

    # Stage 9: Evidence Synthesis
    print('[Info]========== ES Started===========')

    es = run_evidence_synthesis(pico_json,Path(csv_path_study_chars),
    Path(csv_path_study_outcomes))
    print(es)
    print('[Info]========== ES Ended ===========')

if __name__ == "__main__":
    # Default: use agent-based MeSH generation
    # To use the mesh API instead, call: run_flow(use_agent_mesh=False)
    run_flow(use_agent_mesh=True)