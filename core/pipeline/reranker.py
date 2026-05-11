"""
BM25-based reranker for pre-screened PubMed articles.
Scores each paper's title+abstract against the PICO query and returns
the top-K most relevant papers. Runs in milliseconds, no LLM needed.
"""
import re
from typing import Dict

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def bm25_rerank(query: str, articles: Dict, top_k: int = 100) -> Dict:
    """
    Score each article's title+abstract with BM25 against `query`,
    return the top_k highest-scoring articles (preserving original dict structure).

    Args:
        query:    The PICO question or any free-text query string.
        articles: {pmid: {"title": ..., "abstract": ..., ...}}
        top_k:    Number of top articles to keep.

    Returns:
        Filtered and ranked {pmid: article} dict (up to top_k entries).
    """
    if not articles:
        return articles

    if top_k >= len(articles):
        return articles  # nothing to trim

    pmids = list(articles.keys())
    corpus = []
    for pmid in pmids:
        art = articles[pmid]
        title    = art.get("title")    or ""
        abstract = art.get("abstract") or ""
        corpus.append(_tokenize(title + " " + abstract))

    bm25 = BM25Okapi(corpus)
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # Sort by score descending, take top_k
    ranked = sorted(zip(pmids, scores), key=lambda x: x[1], reverse=True)
    top_pmids = {pmid for pmid, _ in ranked[:top_k]}

    return {pmid: articles[pmid] for pmid in pmids if pmid in top_pmids}


def rerank_stats(before: Dict, after: Dict, gt: set) -> str:
    """Return a one-line summary of reranker effect on GT coverage."""
    gt_before = len(set(before) & gt)
    gt_after  = len(set(after)  & gt)
    return (
        f"BM25 rerank: {len(before)} → {len(after)} papers "
        f"(GT coverage: {gt_before}/{len(gt)} → {gt_after}/{len(gt)})"
    )
