"""
iCite-based reranker for PubMed article pools.

Uses NIH's iCite API to fetch citation metrics (Relative Citation Ratio, citation
count) for a pool of PMIDs, then re-scores each article combining:

    score = α × RCR_norm + (1 - α) × BM25_norm

Landmark papers that ranked low in PubMed's relevance sort (common failure mode)
are promoted to the top of the screening queue.

Usage:
    from pipeline.icite_reranker import icite_rerank, icite_rerank_stats

    articles = icite_rerank(query_terms, articles, top_k=150)
"""

import re
import time
import math
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi


# ── iCite API ─────────────────────────────────────────────────────────────────

ICITE_BASE = "https://icite.od.nih.gov/api/pubs"
ICITE_FIELDS = "pmid,citation_count,relative_citation_ratio,is_clinical_trial,year"
BATCH_SIZE = 100   # iCite max per request
RETRY = 3
RETRY_WAIT = 2.0


def _fetch_icite_batch(pmids: List[str]) -> Dict[str, dict]:
    """Fetch iCite metrics for up to BATCH_SIZE PMIDs. Returns {pmid_str: metrics}."""
    if not pmids:
        return {}

    params = urllib.parse.urlencode({
        "pmids": ",".join(pmids),
        "fields": ICITE_FIELDS,
    })
    url = f"{ICITE_BASE}?{params}"

    for attempt in range(RETRY):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            return {str(d["pmid"]): d for d in data.get("data", []) if d.get("pmid")}
        except Exception as e:
            if attempt < RETRY - 1:
                time.sleep(RETRY_WAIT)
            else:
                print(f"  [iCite] Failed after {RETRY} attempts: {e}")
                return {}


def fetch_icite_metrics(pmids: List[str]) -> Dict[str, dict]:
    """
    Fetch iCite metrics for an arbitrary number of PMIDs (auto-batched).

    Returns:
        {pmid_str: {"citation_count": int, "relative_citation_ratio": float,
                    "is_clinical_trial": bool, "year": int}}
    """
    results = {}
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i : i + BATCH_SIZE]
        results.update(_fetch_icite_batch(batch))
        if i + BATCH_SIZE < len(pmids):
            time.sleep(0.1)   # stay well under rate limit
    return results


# ── BM25 helper ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _bm25_scores(query: str, articles: Dict) -> Dict[str, float]:
    """Return {pmid: bm25_score} for all articles."""
    if not query.strip():
        return {pmid: 0.0 for pmid in articles}

    pmids  = list(articles.keys())
    corpus = [
        _tokenize((a.get("title") or "") + " " + (a.get("abstract") or ""))
        for a in articles.values()
    ]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    return dict(zip(pmids, scores))


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize a score dict to [0, 1] via min-max."""
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Combined scorer ───────────────────────────────────────────────────────────

def icite_rerank(
    query: str,
    articles: Dict,
    top_k: int = 150,
    alpha: float = 0.55,
    min_year: Optional[int] = None,
) -> Dict:
    """
    Re-rank `articles` by a combined iCite + BM25 score and return top_k.

    Score formula:
        combined = α × RCR_norm + (1 - α) × BM25_norm

    Where RCR_norm is log-scaled Relative Citation Ratio (0→1 range).
    Papers with no iCite record get RCR_norm = 0 (recent/uncited papers
    are still surfaced via BM25 component).

    Args:
        query:      Free-text query (PICO terms joined) for BM25 component.
        articles:   {pmid: {"title": ..., "abstract": ..., ...}}
        top_k:      Number of top articles to return (default 150).
        alpha:      Weight of iCite score vs BM25 (default 0.55 — slight
                    favour to citation signal while keeping text relevance).
        min_year:   If set, articles published before this year get a 20%
                    RCR penalty (avoids promoting very old off-topic papers).

    Returns:
        Reranked {pmid: article} dict, at most top_k entries.
    """
    if not articles:
        return articles
    if top_k >= len(articles):
        top_k = len(articles)

    pmids = list(articles.keys())

    # ── 1. Fetch iCite metrics ─────────────────────────────────────────────────
    metrics = fetch_icite_metrics(pmids)

    # ── 2. Build RCR scores ────────────────────────────────────────────────────
    # Use log(1 + RCR) so very high-RCR papers don't completely dominate.
    rcr_raw: Dict[str, float] = {}
    for pmid in pmids:
        m   = metrics.get(pmid, {})
        rcr = m.get("relative_citation_ratio") or 0.0
        if min_year and m.get("year") and m["year"] < min_year:
            rcr *= 0.8
        rcr_raw[pmid] = math.log1p(rcr)

    rcr_norm  = _minmax_norm(rcr_raw)

    # ── 3. BM25 scores ─────────────────────────────────────────────────────────
    bm25_raw  = _bm25_scores(query, articles)
    bm25_norm = _minmax_norm(bm25_raw)

    # ── 4. Combined score ──────────────────────────────────────────────────────
    combined = {
        pmid: alpha * rcr_norm.get(pmid, 0.0) + (1 - alpha) * bm25_norm.get(pmid, 0.0)
        for pmid in pmids
    }

    # ── 5. Attach scores to articles (useful for debugging / UI display) ───────
    for pmid in pmids:
        m = metrics.get(pmid, {})
        articles[pmid]["_icite_rcr"]        = m.get("relative_citation_ratio") or 0.0
        articles[pmid]["_icite_citations"]   = m.get("citation_count") or 0
        articles[pmid]["_icite_year"]        = m.get("year")
        articles[pmid]["_icite_score"]       = round(combined[pmid], 4)
        articles[pmid]["_bm25_score"]        = round(bm25_raw.get(pmid, 0.0), 4)

    # ── 6. Return top-K ────────────────────────────────────────────────────────
    ranked = sorted(pmids, key=lambda p: combined[p], reverse=True)
    return {pmid: articles[pmid] for pmid in ranked[:top_k]}


# ── Stats helper ──────────────────────────────────────────────────────────────

def icite_rerank_stats(before: Dict, after: Dict, gt: set) -> str:
    """One-line summary of reranker effect on GT coverage."""
    gt_before = len(set(before) & gt)
    gt_after  = len(set(after)  & gt)

    # Show top-5 RCR papers in the kept set
    top_rcr = sorted(
        [(pmid, art.get("_icite_rcr", 0)) for pmid, art in after.items()],
        key=lambda x: x[1], reverse=True
    )[:3]
    top_str = "  top-RCR: " + ", ".join(f"{p}(RCR={r:.1f})" for p, r in top_rcr if r > 0)

    return (
        f"iCite rerank: {len(before)} → {len(after)} papers "
        f"(GT coverage: {gt_before}/{len(gt)} → {gt_after}/{len(gt)})"
        + (f"\n  {top_str}" if top_rcr else "")
    )


# ── Standalone diagnostic ──────────────────────────────────────────────────────

def explain_scores(articles: Dict, gt: set = None, n: int = 20) -> None:
    """
    Print a ranked table of the top-n articles with their scores.
    Useful for debugging which papers got promoted/demoted.
    """
    scored = [
        (pmid, art)
        for pmid, art in articles.items()
        if "_icite_score" in art
    ]
    scored.sort(key=lambda x: x[1]["_icite_score"], reverse=True)

    print(f"\n{'Rank':<5} {'PMID':<12} {'RCR':>7} {'Cites':>7} {'BM25':>7} {'Score':>7}  GT  Title")
    print("─" * 100)
    for rank, (pmid, art) in enumerate(scored[:n], 1):
        in_gt   = "✅" if gt and pmid in gt else "  "
        title   = (art.get("title") or "")[:50]
        print(
            f"{rank:<5} {pmid:<12} "
            f"{art.get('_icite_rcr', 0):>7.1f} "
            f"{art.get('_icite_citations', 0):>7} "
            f"{art.get('_bm25_score', 0):>7.2f} "
            f"{art.get('_icite_score', 0):>7.4f}  "
            f"{in_gt}  {title}"
        )


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, ".")

    test_articles = {
        "22658127": {"title": "Safety, activity, and immune correlates of anti-PD-1 antibody in cancer", "abstract": ""},
        "25795410": {"title": "Nivolumab versus chemotherapy in patients with advanced melanoma", "abstract": ""},
        "99999999": {"title": "Some obscure 2024 preprint with zero citations", "abstract": ""},
        "11856793": {"title": "Mild therapeutic hypothermia to improve neurologic outcome", "abstract": ""},
    }
    query = "immune checkpoint inhibitor nivolumab pembrolizumab melanoma"
    gt    = {"22658127", "25795410", "11856793"}

    print("Before reranking:", list(test_articles.keys()))
    reranked = icite_rerank(query, test_articles, top_k=4, alpha=0.55)
    print("After reranking :", list(reranked.keys()))
    explain_scores(reranked, gt=gt)
