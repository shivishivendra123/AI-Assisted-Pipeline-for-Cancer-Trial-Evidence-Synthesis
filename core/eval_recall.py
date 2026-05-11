"""
Recall evaluation for the pipeline up to the screened studies stage.

Usage (from evsy/core/):
    python eval_recall.py [--testcase PATH] [--max-retrieve N] [--max-screen N] [--no-agent-mesh]

Metrics reported:
  - Retrieval recall  : fraction of ground-truth PMIDs retrieved by PubMed search
  - Screening recall  : fraction of ground-truth PMIDs that pass screening (score >= 3)
"""

import argparse
import json
import sys
import csv
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth(testcase_path: str):
    """Return (query, set_of_ground_truth_pmids) from a testcase JSON file."""
    with open(testcase_path, "r", encoding="utf-8") as f:
        tc = json.load(f)
    query = tc["Title"]
    pmids = {c["pmid"] for c in tc.get("Involved_Citations", []) if c.get("pmid")}
    return query, pmids


def read_screened_pmids(csv_path: str, score_threshold: float = 3.0):
    """Return set of PMIDs from the screening CSV whose cumulative_score >= threshold."""
    pmids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row.get("cumulative_score", 0))
            except ValueError:
                continue
            if score >= score_threshold:
                pmid = row.get("study_id", "").strip()
                if pmid:
                    pmids.add(pmid)
    return pmids


def recall(pipeline_pmids: set, ground_truth: set) -> float:
    if not ground_truth:
        return 0.0
    return len(pipeline_pmids & ground_truth) / len(ground_truth)


def print_recall_report(stage: str, pipeline_pmids: set, gt_pmids: set):
    found  = pipeline_pmids & gt_pmids
    missed = gt_pmids - pipeline_pmids
    r = recall(pipeline_pmids, gt_pmids)

    print(f"\n{'='*60}")
    print(f"  {stage}")
    print(f"{'='*60}")
    print(f"  Pipeline PMIDs :  {len(pipeline_pmids)}")
    print(f"  Ground truth   :  {len(gt_pmids)}")
    print(f"  Hits           :  {len(found)}")
    print(f"  Missed         :  {len(missed)}")
    print(f"  Recall         :  {r:.1%}  ({len(found)}/{len(gt_pmids)})")
    if found:
        print(f"\n  Found:")
        for p in sorted(found):
            print(f"     + {p}")
    if missed:
        print(f"\n  Missed:")
        for p in sorted(missed):
            print(f"     - {p}")


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_evaluation(testcase_path: str, max_retrieve: int, max_screen: int, use_agent_mesh: bool):
    # 1. Ground truth
    query, gt_pmids = load_ground_truth(testcase_path)

    print(f"\nQuery : {query}")
    print(f"\nGround-truth PMIDs ({len(gt_pmids)}):")
    for p in sorted(gt_pmids):
        print(f"  - {p}")

    # 2. Imports (done here so CLI arg errors surface before heavy imports)
    from pipeline.pico import pico_pipe
    from pipeline.augment.term_expander import expand_terms_with_llm
    from schemas.pico import PICO
    from meshOnDemand.mesh_query import query_mesh_api
    from meshOnDemand.mesh_generator_agent import generate_mesh_terms_with_agent
    from efetch_utility.efetch import fetch_pubmed_articles
    from eligibility_builder.built_eligibility import build_eligibility
    from screening.screening import screen_studies

    # 3. PICO extraction
    print("\n[1/6] Extracting PICO...")
    pico_json = pico_pipe(query)

    # 4. PICO augmentation
    print("[2/6] Augmenting PICO terms...")
    syn_prompt = Path("prompts/synonym_prompt.txt").read_text(encoding="utf-8")
    pico_obj = PICO.model_validate(pico_json["pico_valid"])
    augmented_pico = expand_terms_with_llm(pico_obj, syn_prompt)
    pico_json["augmented"] = json.loads(augmented_pico.model_dump_json())

    # 5. MeSH query
    print("[3/6] Generating MeSH query...")
    mesh_query = generate_mesh_terms_with_agent(pico_json) if use_agent_mesh else query_mesh_api(pico_json)
    print(f"      MeSH query: {mesh_query}")

    # 6. PubMed search
    print(f"[4/6] Searching PubMed (max_retrieve={max_retrieve})...")
    data = fetch_pubmed_articles(pico_json["qid"], mesh_query, max_results=max_retrieve)
    articles, full_text = data["articles"], data["full_text"]
    retrieved_pmids = set(articles.keys())

    print_recall_report("RETRIEVAL RECALL (after PubMed search)", retrieved_pmids, gt_pmids)

    # 7. Eligibility criteria
    print("\n[5/6] Building eligibility criteria...")
    eligibility_criteria = build_eligibility(
        pico_json["question"], pico_json["pico_valid"], pico_json["qid"]
    )

    # 8. Screening
    print(f"[6/6] Screening studies (max_screen={max_screen})...")
    csv_path = screen_studies(
        articles, full_text, eligibility_criteria, pico_json,
        max_studies=max_screen
    )

    if csv_path is None:
        print("[ERROR] Screening produced no output CSV.")
        sys.exit(1)

    screened_pmids = read_screened_pmids(csv_path, score_threshold=3.0)

    print_recall_report("SCREENING RECALL (score >= 3)", screened_pmids, gt_pmids)

    # Summary
    r_ret = recall(retrieved_pmids, gt_pmids)
    r_scr = recall(screened_pmids, gt_pmids)
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Retrieval recall  : {r_ret:.1%}  ({len(retrieved_pmids & gt_pmids)}/{len(gt_pmids)})")
    print(f"  Screening recall  : {r_scr:.1%}  ({len(screened_pmids & gt_pmids)}/{len(gt_pmids)})")
    print(f"  Screening CSV     : {csv_path}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline recall vs ground-truth PMIDs.")
    parser.add_argument(
        "--testcase",
        default="../../testcase.json",
        help="Path to testcase JSON (default: ../../testcase.json)",
    )
    parser.add_argument(
        "--max-retrieve",
        type=int,
        default=500,
        help="Max PubMed results to retrieve (default: 500)",
    )
    parser.add_argument(
        "--max-screen",
        type=int,
        default=500,
        help="Max studies to screen — set equal to max-retrieve to screen all (default: 500)",
    )
    parser.add_argument(
        "--no-agent-mesh",
        action="store_true",
        help="Use MeSH API instead of agent-based MeSH generation",
    )
    args = parser.parse_args()

    run_evaluation(
        testcase_path=args.testcase,
        max_retrieve=args.max_retrieve,
        max_screen=args.max_screen,
        use_agent_mesh=not args.no_agent_mesh,
    )
