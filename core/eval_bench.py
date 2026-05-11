"""
Batch recall evaluator for TrialReviewBench.

Iterates every test case in the benchmark JSONL, runs PICO → MeSH → dual-fetch
(intervention + comparator queries), and records per-case retrieval recall.

Usage (from evsy/core/):
    python eval_bench.py [--bench PATH] [--max-retrieve N] [--out CSV] [--limit N]
                         [--icite] [--top-k N] [--alpha F]

Flags:
  --icite          Re-rank retrieved pool by iCite RCR + BM25 before recall
                   measurement. Promotes landmark papers the relevance sort buried.
  --top-k N        After iCite reranking keep only top-N papers (default 200).
  --alpha F        iCite weight in combined score 0–1 (default 0.55).

Outputs:
  - A CSV with per-case recall (stdout summary + saved file)
  - Console summary: mean, median, min/max, and distribution by GT-size bucket
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CORE_DIR))
# Always run with evsy/core/ as cwd so relative prompt paths resolve correctly
import os
os.chdir(_CORE_DIR)

BENCH_DEFAULT = _CORE_DIR.parent.parent / "TrialReviewBench" / "TrialReviewBench-study-search-screening.jsonl"


# ── LLM backend override: route mesh_basic / mesh_expand through Gemini 2.5 Flash
# Done here (in the evaluator only) so the rest of the pipeline keeps its
# configured backend untouched. Patches the `_agent` factory in each module
# after import so generate_*_mesh_query picks up the override transparently.

_GEMINI_PRO_MODEL = "gemini-2.5-pro"


def _make_gemini_pro_agent(max_tokens: int = 8000, temperature: float = 0.0):
    from agents.factory import build_vertex_gemini_agent
    from configs.env_config import config
    return build_vertex_gemini_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GEMINI_LOCATION,
        model=_GEMINI_PRO_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _patch_mesh_agents_to_gemini_pro():
    """Replace `_agent` in mesh_basic / mesh_expand so the eval runs on Gemini 2.5 Flash."""
    try:
        from meshOnDemand import mesh_basic
        mesh_basic._agent = lambda: _make_gemini_pro_agent(max_tokens=2000, temperature=0.2)
    except Exception as e:
        print(f"[warn] could not patch mesh_basic._agent: {e}")
    try:
        from meshOnDemand import mesh_expand
        mesh_expand._agent = lambda: _make_gemini_pro_agent(max_tokens=8000)
    except Exception as e:
        print(f"[warn] could not patch mesh_expand._agent: {e}")


def _suppress_artifact_writes():
    """
    Suppress per-case temp writes from mesh_basic / mesh_expand.

    Those modules dump mesh_basic_<qid>.txt, mesh_basic_parsed<qid>.json,
    pubmed_parsed<qid>.json, mesh_query.jsonl, mesh_expand_<qid>.txt, etc.
    into core/artifacts_day3/ as a side effect of generate_*_mesh_query.
    For batch eval only the aggregated bench CSV matters, so silently
    redirect any write to an `artifacts_day*` path to a no-op.

    Patches pathlib.Path.write_text, Path.mkdir, and builtins.open.
    The bench CSV (in evals/) is unaffected — only paths whose string
    representation contains '/artifacts_day' are swallowed.
    """
    import builtins
    import os as _os
    from pathlib import Path as _Path

    _orig_write_text = _Path.write_text
    _orig_mkdir      = _Path.mkdir
    _orig_open       = builtins.open

    def _is_artifact(p) -> bool:
        try:
            return "artifacts_day" in str(p)
        except Exception:
            return False

    def _patched_write_text(self, *a, **kw):
        if _is_artifact(self):
            return 0
        return _orig_write_text(self, *a, **kw)

    def _patched_mkdir(self, *a, **kw):
        if _is_artifact(self):
            return None
        return _orig_mkdir(self, *a, **kw)

    def _patched_open(file, *a, **kw):
        if _is_artifact(file):
            return _orig_open(_os.devnull, *a, **kw)
        return _orig_open(file, *a, **kw)

    _Path.write_text = _patched_write_text
    _Path.mkdir      = _patched_mkdir
    builtins.open    = _patched_open


def _patch_expand_terms_per_axis(per_axis_cap: int):
    """
    Raise the per-axis term cap in mesh_expand from the default 8 → `per_axis_cap`,
    and rewrite the prompt count hints so the LLM emits enough terms to fill it.

    Patches:
      - mesh_expand._parse_axes  (replaces the local AXIS_CAP=8 with the new cap)
      - mesh_expand._load_prompt (replaces '3–5 terms' / '2–3 terms' hints with
                                  half-of-cap each across CORE + EXPAND)

    No-op for --strategy basic (mesh_basic has its own MAX_TERMS path).
    """
    if per_axis_cap is None or per_axis_cap <= 0:
        return

    from meshOnDemand import mesh_expand
    import json as _json

    # ── 1. Replace _parse_axes with a version using the configurable cap.
    def _parse_axes(response: str):
        cleaned = mesh_expand._clean_json(response)
        s2, s3 = {}, {}
        try:
            data = _json.loads(cleaned)
            s2 = data.get("step 2", {}) or {}
            s3 = data.get("step 3", {}) or {}
        except Exception:
            for key in (
                "CORE_CONDITIONS", "CORE_TREATMENTS", "CORE_OUTCOMES",
                "EXPAND_CONDITIONS", "EXPAND_TREATMENTS", "EXPAND_OUTCOMES",
            ):
                terms = mesh_expand._extract_axis_regex(cleaned, key)
                (s2 if key.startswith("CORE_") else s3)[key] = terms

        def merged(core_key, expand_key):
            combined = (s2.get(core_key) or []) + (s3.get(expand_key) or [])
            return mesh_expand._dedupe_with_near(combined)[:per_axis_cap]

        return {
            "conditions": merged("CORE_CONDITIONS", "EXPAND_CONDITIONS"),
            "treatments": merged("CORE_TREATMENTS", "EXPAND_TREATMENTS"),
            "outcomes":   merged("CORE_OUTCOMES",   "EXPAND_OUTCOMES"),
        }

    mesh_expand._parse_axes = _parse_axes

    # ── 2. Rewrite prompt count hints so the LLM produces enough terms to fill
    # the new cap after dedup. Ask for ~cap/2 CORE + ~cap/2 EXPAND per axis,
    # padded by +1 to absorb dedup/near-dup losses.
    half = max(1, per_axis_cap // 2)
    lo, hi = half, half + 1
    _original_load_prompt = mesh_expand._load_prompt
    def _load_prompt_patched() -> str:
        txt = _original_load_prompt()
        replacements = [
            ("CORE_CONDITIONS: 3–5 terms",   f"CORE_CONDITIONS: {lo}–{hi} terms"),
            ("CORE_TREATMENTS: 3–5 terms",   f"CORE_TREATMENTS: {lo}–{hi} terms"),
            ("CORE_OUTCOMES:   2–3 terms",   f"CORE_OUTCOMES:   {lo}–{hi} terms"),
            ("EXPAND_CONDITIONS: 3–5 terms", f"EXPAND_CONDITIONS: {lo}–{hi} terms"),
            ("EXPAND_TREATMENTS: 3–5 terms", f"EXPAND_TREATMENTS: {lo}–{hi} terms"),
            ("EXPAND_OUTCOMES:   3–5 terms", f"EXPAND_OUTCOMES:   {lo}–{hi} terms"),
        ]
        for old, new in replacements:
            txt = txt.replace(old, new)
        return txt

    mesh_expand._load_prompt = _load_prompt_patched


# ── recall helpers ────────────────────────────────────────────────────────────

def recall(retrieved: set, gt: set) -> float:
    if not gt:
        return 0.0
    return len(retrieved & gt) / len(gt)


def load_bench(path: Path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def gt_pmids(tc: dict) -> set:
    return {c["pmid"] for c in tc.get("Involved_Citations", []) if c.get("pmid")}


# ── ranked esearch helper ─────────────────────────────────────────────────────

# Cutoffs we report rank-recall at. Top-100 mirrors what the screening stage
# typically caps at; @500 matches the existing retmax default; @2000 shows
# the deeper picture so we can distinguish "query problem" from "depth problem".
_RANK_CUTOFFS = (100, 500, 2000)
_RANK_DEEP    = max(_RANK_CUTOFFS)


def esearch_ranked_pmids(term: str, retmax: int = _RANK_DEEP,
                         use_relevance_sort: bool = True) -> list:
    """
    Cheap esearch-only call returning PMIDs in PubMed's chosen order.

    Used to compute recall@K for K in (100, 500, 2000).

    use_relevance_sort:
        True  — passes sort=relevance (PubMed's "Best Match" learning-to-rank).
        False — omits the sort parameter; PubMed uses its default sort
                (typically Most Recent / chronological, depending on query).
                Mirrors a "let the query do the filtering" approach.
    """
    if not term:
        return []
    from Bio import Entrez
    from lxml import etree
    from efetch_utility.efetch import retry_fetch
    try:
        sort_kwargs = {"sort": "relevance"} if use_relevance_sort else {}
        xml  = retry_fetch(
            Entrez.esearch, db="pubmed", term=term,
            retmax=retmax, retmode="xml", **sort_kwargs,
        )
        root = etree.fromstring(xml)
        return list(root.xpath("//Id/text()"))
    except Exception as e:
        print(f"[esearch_ranked] failed: {e}")
        return []


def rank_recalls(ranked: list, gt: set) -> dict:
    """recall@K for K in _RANK_CUTOFFS. Returns {'rank_recall_K': float, ...}."""
    if not gt:
        return {f"rank_recall_{k}": 0.0 for k in _RANK_CUTOFFS}
    out = {}
    for k in _RANK_CUTOFFS:
        top_k = set(ranked[:k])
        out[f"rank_recall_{k}"] = round(len(top_k & gt) / len(gt), 4)
    return out


def pico_from_testcase(tc: dict) -> dict:
    """
    Build the pico_json shape that generate_basic_mesh_query expects directly
    from the benchmark test case's PICO field — no LLM extraction.

    Benchmark PICO keys are P/I/C/O (strings). Comparator may be "N/A" when
    the review is single-arm. Outcomes is a free-text string; wrap it as a
    single-element list to match the downstream contract.
    """
    pico = tc.get("PICO") or {}
    comparator = (pico.get("C") or "").strip()
    if comparator.upper() in {"N/A", "NA", "NONE", ""}:
        comparator = ""

    outcomes_raw = (pico.get("O") or "").strip()
    outcomes = [outcomes_raw] if outcomes_raw else []

    return {
        "qid": tc.get("PMID", "unknown"),
        "pico_valid": {
            "Population":   (pico.get("P") or "").strip(),
            "Intervention": (pico.get("I") or "").strip(),
            "Comparator":   comparator,
            "Outcomes":     outcomes,
        },
    }


# ── single-case runner ────────────────────────────────────────────────────────

def run_case(
    tc: dict,
    max_retrieve: int,
    query_builder,
    fetch_pubmed_articles,
    use_icite: bool = False,
    icite_top_k: int = 200,
    icite_alpha: float = 0.55,
    use_relevance_sort: bool = True,
) -> dict:
    pmid  = tc.get("PMID", "unknown")
    title = tc["Title"]
    gt    = gt_pmids(tc)

    result = {
        "pmid":              pmid,
        "title":             title[:80],
        "gt_count":          len(gt),
        "retrieved":         0,
        "hits":              0,
        "recall":            0.0,
        "retrieved_pre_rerank": 0,
        "hits_pre_rerank":   0,
        "recall_pre_rerank": 0.0,
        "rank_recall_100":   0.0,
        "rank_recall_500":   0.0,
        "rank_recall_2000":  0.0,
        "missed_pmids":      "",
        "error":             "",
    }

    if not gt:
        result["error"] = "no GT PMIDs"
        return result

    try:
        pico_json  = pico_from_testcase(tc)
        mesh_query = query_builder(pico_json)

        # Rank-distribution recall on the primary query alone (no dual-fetch,
        # no iCite). Tells us whether GT papers are ranking high in PubMed's
        # chosen order — the upper bound for any downstream screening cap.
        # When --no-sort is on, this measures default-sort (chronological)
        # ranking rather than relevance.
        ranked = esearch_ranked_pmids(mesh_query, retmax=_RANK_DEEP,
                                      use_relevance_sort=use_relevance_sort)
        result.update(rank_recalls(ranked, gt))

        data = fetch_pubmed_articles(
            pico_json["qid"], mesh_query,
            max_results=max_retrieve, fetch_fulltext=False,
        )
        articles = data["articles"] if data else {}

        # Dual-fetch: comparator query merged separately
        cmp_query = pico_json.get("mesh_query_comparator", "")
        if cmp_query:
            data2 = fetch_pubmed_articles(
                pico_json["qid"], cmp_query,
                max_results=max_retrieve, fetch_fulltext=False,
            )
            if data2 and data2.get("articles"):
                articles = {**data2["articles"], **articles}

        # Record pre-rerank stats
        retrieved_pre = set(articles.keys())
        hits_pre      = retrieved_pre & gt
        result["retrieved_pre_rerank"] = len(retrieved_pre)
        result["hits_pre_rerank"]      = len(hits_pre)
        result["recall_pre_rerank"]    = round(recall(retrieved_pre, gt), 4)

        # Optional iCite reranking
        if use_icite and articles:
            from pipeline.icite_reranker import icite_rerank, icite_rerank_stats
            groups   = pico_json.get("mesh_groups", {})
            bm25_q   = " ".join(
                groups.get("population", []) +
                groups.get("intervention", []) +
                groups.get("comparator", [])
            ) or title
            articles = icite_rerank(bm25_q, articles, top_k=icite_top_k, alpha=icite_alpha)
            stats    = icite_rerank_stats(
                {p: {} for p in retrieved_pre}, articles, gt
            )
            print(f"          {stats}")

        retrieved = set(articles.keys())
        hits      = retrieved & gt
        missed    = gt - retrieved
        r         = recall(retrieved, gt)

        result["retrieved"]    = len(retrieved)
        result["hits"]         = len(hits)
        result["recall"]       = round(r, 4)
        result["missed_pmids"] = "|".join(sorted(missed))

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


# ── summary stats ──────────────────────────────────────────────────────────────

def print_summary(results: list):
    valid = [r for r in results if not r["error"] and r["gt_count"] > 0]
    if not valid:
        print("No valid results to summarise.")
        return

    recalls = [r["recall"] for r in valid]
    n = len(recalls)

    recalls_sorted = sorted(recalls)
    mean   = sum(recalls) / n
    median = recalls_sorted[n // 2] if n % 2 else (recalls_sorted[n//2-1] + recalls_sorted[n//2]) / 2
    mn, mx = recalls_sorted[0], recalls_sorted[-1]

    # Bucket by GT size
    buckets = {
        "easy   (1–5 GT)":    [],
        "medium (6–15 GT)":   [],
        "hard   (16–30 GT)":  [],
        "v.hard (31+ GT)":    [],
    }
    for r in valid:
        gt_c = r["gt_count"]
        if gt_c <= 5:
            buckets["easy   (1–5 GT)"].append(r["recall"])
        elif gt_c <= 15:
            buckets["medium (6–15 GT)"].append(r["recall"])
        elif gt_c <= 30:
            buckets["hard   (16–30 GT)"].append(r["recall"])
        else:
            buckets["v.hard (31+ GT)"].append(r["recall"])

    errors = [r for r in results if r["error"]]

    print(f"\n{'='*62}")
    print(f"  BENCHMARK RETRIEVAL RECALL  —  {n} cases evaluated")
    print(f"{'='*62}")
    print(f"  Mean recall   :  {mean:.1%}")
    print(f"  Median recall :  {median:.1%}")
    print(f"  Min recall    :  {mn:.1%}")
    print(f"  Max recall    :  {mx:.1%}")
    print(f"  Errors        :  {len(errors)}")
    print()
    print(f"  {'Bucket':<22}  {'N':>4}  {'Mean':>7}  {'Median':>8}")
    print(f"  {'-'*44}")
    for label, vals in buckets.items():
        if not vals:
            continue
        bm = sum(vals) / len(vals)
        bmed = sorted(vals)[len(vals) // 2]
        print(f"  {label:<22}  {len(vals):>4}  {bm:>6.1%}  {bmed:>7.1%}")
    print()

    # Cases with 0% recall
    zero = [r for r in valid if r["recall"] == 0.0]
    if zero:
        print(f"  Zero-recall cases ({len(zero)}):")
        for r in zero[:10]:
            print(f"    PMID {r['pmid']}  GT={r['gt_count']}  {r['title']}")
        if len(zero) > 10:
            print(f"    … and {len(zero)-10} more")
    print()

    # ── rank-distribution summary (recall@K from primary query alone) ────────
    # Tells us where the GT papers actually sit in PubMed's relevance order.
    # If rank@2000 is high but rank@500 is low → depth problem; raise retmax.
    # If rank@2000 itself is low → query problem; retmax can't help.
    print(f"  {'='*44}")
    print(f"  RANK-RECALL on primary query (no dual-fetch, no iCite)")
    print(f"  {'-'*44}")

    def _stats(vals):
        s = sorted(vals)
        m = sum(vals) / len(vals)
        med = s[len(vals) // 2] if len(vals) % 2 else (s[len(vals)//2-1] + s[len(vals)//2]) / 2
        return m, med

    for k in _RANK_CUTOFFS:
        col = f"rank_recall_{k}"
        vals = [r[col] for r in valid]
        m, med = _stats(vals)
        n_full = sum(1 for v in vals if v == 1.0)
        n_zero = sum(1 for v in vals if v == 0.0)
        print(f"  recall@{k:<5}        mean={m:>6.1%}  median={med:>6.1%}  "
              f"perfect={n_full:>3}  zero={n_zero:>3}")

    # Diagnostic split: of the cases that have GT in top-2000, how many
    # are stuck past rank 500? That's the population a higher retmax fixes.
    in_2k_not_500 = sum(
        1 for r in valid
        if r["rank_recall_2000"] > 0 and r["rank_recall_500"] < r["rank_recall_2000"]
    )
    none_in_2k = sum(1 for r in valid if r["rank_recall_2000"] == 0.0)
    print(f"  {'-'*44}")
    print(f"  Cases with GT in [501..2000] (raise retmax to recover): {in_2k_not_500}")
    print(f"  Cases with NO GT in top-2000 (query problem):           {none_in_2k}")
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch recall evaluator for TrialReviewBench.")
    parser.add_argument(
        "--bench",
        default=str(BENCH_DEFAULT),
        help="Path to TrialReviewBench JSONL (default: auto-detected)",
    )
    parser.add_argument(
        "--max-retrieve",
        type=int,
        default=500,
        help="Max PubMed results per query (default: 500)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: bench_recall_<timestamp>.csv in evals/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N cases (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--pmids",
        default=None,
        help="Comma-separated PMIDs to evaluate (e.g. 30854085,12137670). Overrides --limit.",
    )
    parser.add_argument(
        "--worst",
        default=None,
        help="Path to a previous bench CSV — re-evaluate the N lowest-recall cases. "
             "Format: PATH:N  e.g.  artifacts_day5/bench_recall_xyz.csv:20",
    )
    parser.add_argument(
        "--icite",
        action="store_true",
        help="Re-rank retrieved pool by iCite RCR + BM25 before recall measurement",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="After iCite reranking keep only top-K papers (default: 200)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="iCite score weight in combined score, 0–1 (default: 0.55)",
    )
    parser.add_argument(
        "--strategy",
        choices=["basic", "expand"],
        default="basic",
        help="Query-construction strategy. "
             "'basic' = (P) AND (I) MeSH-locked (current). "
             "'expand' = 2-stage retrieval-augmented expansion: "
             "fetches 7 reference papers from a primary probe, then asks the LLM "
             "to emit CORE+EXPAND terms across (conditions, treatments, outcomes) "
             "axes; final query is fat-OR per axis, AND across axes, no MeSH lock-in.",
    )
    parser.add_argument(
        "--multi-sort",
        action="store_true",
        help="Use multi-sort PubMed retrieval (relevance + newest + oldest, "
             "1000/500/500 union deduped, ~1500-1900 unique). Recovers older "
             "landmark trials that sort=relevance buries due to PubMed's "
             "recency bias. Note: --max-retrieve is ignored when this is on; "
             "the multi-sort function uses its own per-sort budgets.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Drop sort=relevance from esearch calls. Uses PubMed's default "
             "sort instead (typically Most Recent / chronological). Mirrors "
             "the approach of letting the query do the filtering and "
             "leaving order to PubMed's default. Tests whether 'Best Match' "
             "relevance ranking is helping or hurting recall@K for SR retrieval. "
             "Affects BOTH rank-recall measurement and main fetch. Ignored if "
             "--multi-sort is also on.",
    )
    parser.add_argument(
        "--by-decade",
        action="store_true",
        help="Date-stratified retrieval: 4 buckets × 500 PMIDs (2020+, 2010s, "
             "2000s, 1980s-1990s), each with sort=relevance within its date "
             "range. Guarantees per-era coverage by giving each decade its own "
             "retmax slot — old landmark trials don't compete with newer paper "
             "volume. Mutually exclusive with --multi-sort.",
    )
    parser.add_argument(
        "--exclude-reviews",
        action="store_true",
        help="Append `NOT (review[pt] OR meta-analysis[pt] OR systematic review[pt])` "
             "to every query. Drops reviews/meta-analyses from the candidate pool "
             "BEFORE ranking, so original trials (TrialReviewBench GT) aren't "
             "outranked by review-class papers in Best Match / iCite. Standard "
             "Cochrane SR retrieval methodology. Only affects the expand strategy.",
    )
    parser.add_argument(
        "--terms-per-axis",
        type=int,
        default=None,
        help="Override the per-axis term cap in --strategy expand (default 8). "
             "Raises mesh_expand AXIS_CAP and rewrites the prompt count hints "
             "(CORE + EXPAND) so the LLM emits enough terms to fill the cap "
             "after dedup. Use 15 for fat-OR per axis. No-op for --strategy basic.",
    )
    args = parser.parse_args()

    # Late imports so CLI arg errors surface before heavy imports
    if args.strategy == "expand":
        from meshOnDemand.mesh_expand import generate_expand_mesh_query as _gen_expand
        # Closure carries the --exclude-reviews flag through to _build_query.
        if args.exclude_reviews:
            def query_builder(pj):
                return _gen_expand(pj, exclude_reviews=True)
        else:
            query_builder = _gen_expand
    else:
        if args.exclude_reviews:
            print("[warn] --exclude-reviews only applies to --strategy expand; ignored for basic")
        from meshOnDemand.mesh_basic import generate_basic_mesh_query as query_builder

    # Route mesh term extraction through Gemini 2.5 Flash instead of the
    # default medgemma (:predict) endpoint. Eval-scoped override only.
    _patch_mesh_agents_to_gemini_pro()
    print(f"LLM backend  : Gemini 2.5 Flash (eval override)")

    # Drop per-case temp writes (mesh_basic_*.txt, pubmed_parsed*.json, ...)
    # that mesh_basic / mesh_expand normally dump into core/artifacts_day3/.
    # Only the bench CSV (in evals/) is kept.
    _suppress_artifact_writes()
    print(f"artifacts    : suppressed (only bench CSV will be written)")

    # Optional: bump per-axis term cap (and prompt count hints) for --strategy expand.
    if args.terms_per_axis and args.strategy == "expand":
        _patch_expand_terms_per_axis(args.terms_per_axis)
        print(f"axis cap     : {args.terms_per_axis} terms / axis (prompt + parser patched)")
    elif args.terms_per_axis and args.strategy != "expand":
        print(f"[warn] --terms-per-axis only applies to --strategy expand; ignored for {args.strategy}")

    if args.multi_sort and args.by_decade:
        sys.exit("ERROR: --multi-sort and --by-decade are mutually exclusive.")

    if args.multi_sort:
        # Adapter — multi-sort ignores max_results, uses DEFAULT_SORT_STRATEGIES.
        # --no-sort doesn't apply here (multi-sort defines its own sort orders).
        from efetch_utility.efetch import fetch_pubmed_articles_multi_sort
        def fetch_pubmed_articles(qid, q, max_results=None, fetch_fulltext=True):
            return fetch_pubmed_articles_multi_sort(qid, q, fetch_fulltext=fetch_fulltext)
    elif args.by_decade:
        # Date-stratified per-decade retrieval. Ignores max_results; uses
        # DEFAULT_DECADE_BUCKETS (4 × 500 PMIDs across 1980-now).
        from efetch_utility.efetch import fetch_pubmed_articles_by_decade
        def fetch_pubmed_articles(qid, q, max_results=None, fetch_fulltext=True):
            return fetch_pubmed_articles_by_decade(qid, q, fetch_fulltext=fetch_fulltext)
    else:
        from efetch_utility.efetch import fetch_pubmed_articles as _fetch_pubmed_articles
        # Adapter applies sort policy globally — single source of truth for the
        # whole run. None means "let PubMed use its default".
        _sort_value = None if args.no_sort else "relevance"
        def fetch_pubmed_articles(qid, q, max_results=None, fetch_fulltext=True):
            return _fetch_pubmed_articles(
                qid, q,
                max_results=max_results,
                fetch_fulltext=fetch_fulltext,
                sort=_sort_value,
            )

    bench_path = Path(args.bench)
    if not bench_path.exists():
        print(f"[ERROR] Benchmark file not found: {bench_path}", file=sys.stderr)
        sys.exit(1)

    cases = load_bench(bench_path)

    # -- filter by explicit PMID list
    if args.pmids:
        wanted = {p.strip() for p in args.pmids.split(",")}
        cases  = [c for c in cases if c.get("PMID", "") in wanted]
        print(f"[filter] --pmids: kept {len(cases)} cases")

    # -- filter to worst N from a previous run
    elif args.worst:
        try:
            prev_csv, n_str = args.worst.rsplit(":", 1)
            n_worst = int(n_str)
        except ValueError:
            prev_csv, n_worst = args.worst, 37   # default: all zero-recall

        prev_rows = []
        with open(prev_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prev_rows.append(row)
        prev_rows.sort(key=lambda r: (float(r.get("recall", 0)), -int(r.get("retrieved", 0))))
        worst_pmids = {r["pmid"] for r in prev_rows[:n_worst]}
        cases = [c for c in cases if c.get("PMID", "") in worst_pmids]
        # Re-sort cases in same worst-first order
        order = {p: i for i, p in enumerate(r["pmid"] for r in prev_rows[:n_worst])}
        cases.sort(key=lambda c: order.get(c.get("PMID", ""), 999))
        print(f"[filter] --worst {n_worst} from {Path(prev_csv).name}: kept {len(cases)} cases")

    elif args.limit:
        cases = cases[: args.limit]

    if args.multi_sort:
        retrieval_mode = "multi-sort (relevance 1000 + newest 500 + oldest 500, deduped)"
    elif args.by_decade:
        retrieval_mode = "by-decade (4 × 500: 2020+, 2010s, 2000s, 1980s-1990s, deduped)"
    else:
        retrieval_mode = f"single-sort (max_retrieve={args.max_retrieve})"

    print(f"Benchmark    : {bench_path.name}")
    print(f"Cases        : {len(cases)}")
    print(f"strategy     : {args.strategy}")
    print(f"retrieval    : {retrieval_mode}")
    if not args.multi_sort and not args.by_decade:
        print(f"sort policy  : {'PubMed default (chronological)' if args.no_sort else 'sort=relevance (Best Match)'}")
    print(f"iCite rerank : {'ON  (top_k=%d, alpha=%.2f)' % (args.top_k, args.alpha) if args.icite else 'OFF'}")
    print(f"exclude rev. : {'ON  (drops review/meta-analysis/systematic review at query)' if args.exclude_reviews else 'OFF'}")
    print()

    # Output CSV path
    if args.out:
        out_path = Path(args.out)
    else:
        artifacts_dir = Path(__file__).resolve().parent / "evals"
        artifacts_dir.mkdir(exist_ok=True)
        parts = [args.strategy]
        if args.multi_sort:
            parts.append("multisort")
        elif args.by_decade:
            parts.append("bydecade")
        elif args.no_sort:
            parts.append("nosort")
        if args.icite:
            parts.append("icite")
        if args.exclude_reviews:
            parts.append("noreviews")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = artifacts_dir / f"bench_recall_{'_'.join(parts)}_{ts}.csv"

    fieldnames = [
        "pmid", "title", "gt_count",
        "retrieved", "hits", "recall",
        "retrieved_pre_rerank", "hits_pre_rerank", "recall_pre_rerank",
        "rank_recall_100", "rank_recall_500", "rank_recall_2000",
        "missed_pmids", "error",
    ]

    results = []
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for i, tc in enumerate(cases, 1):
            pmid = tc.get("PMID", "?")
            print(f"[{i:3d}/{len(cases)}]  PMID {pmid}  GT={len(gt_pmids(tc))}  {tc['Title'][:60]}…")
            r = run_case(
                tc, args.max_retrieve,
                query_builder, fetch_pubmed_articles,
                use_icite=args.icite,
                icite_top_k=args.top_k,
                icite_alpha=args.alpha,
                use_relevance_sort=not args.no_sort,
            )
            results.append(r)
            writer.writerow(r)
            csvf.flush()  # incremental write — partial runs are recoverable

            if r["error"]:
                print(f"          → ERROR: {r['error']}")
            elif args.icite and r["recall_pre_rerank"] != r["recall"]:
                print(f"          → recall={r['recall']:.1%} ({r['hits']}/{r['gt_count']})  "
                      f"[pre-rerank: {r['recall_pre_rerank']:.1%}]")
                print(f"          → rank@100={r['rank_recall_100']:.1%}  "
                      f"@500={r['rank_recall_500']:.1%}  @2000={r['rank_recall_2000']:.1%}")
            else:
                print(f"          → recall={r['recall']:.1%} ({r['hits']}/{r['gt_count']})  "
                      f"|  rank@100={r['rank_recall_100']:.1%}  "
                      f"@500={r['rank_recall_500']:.1%}  @2000={r['rank_recall_2000']:.1%}")

    print_summary(results)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
