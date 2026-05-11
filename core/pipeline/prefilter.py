"""
Fast keyword pre-filter to reduce PubMed results before LLM screening.
Uses the PICO term groups (from mesh_basic) to check title+abstract locally.
No LLM calls — runs in milliseconds per paper.
"""
import re
from typing import Dict


def _sig_words(terms: list, min_len: int = 8) -> set:
    """
    Extract significant words (length >= min_len) from a list of terms.
    Split on spaces only so multi-word phrases like 'colony-stimulating' stay intact.
    Used for title-level matching where we want high specificity.
    """
    words = set()
    for t in terms:
        for w in t.split():
            if len(w) >= min_len:
                words.add(w.lower())
    return words


def keyword_prefilter(articles: Dict, groups: Dict, require_groups: int = 2) -> Dict:
    """
    Two-tier pre-filter:

    Tier 1 (title check):
        Keep if any significant word (≥8 chars) from intervention-only terms
        appears in the TITLE.

    Tier 2 (abstract fallback):
        Keep if any intervention-specific significant word appears anywhere in
        title+abstract. This catches papers where the drug is mentioned only in
        the abstract but not the title.

    Both tiers use `int_sig`: significant words from intervention terms that are
    NOT shared with the comparator (e.g. "pegfilgrastim" but not "filgrastim").
    This avoids passing every paper PubMed returned (PubMed already guarantees
    population+drug match, so multi-group text matching would pass all 500 papers).

    Falls back to loose group matching only when int_sig is empty (no unique
    intervention sig words vs comparator).

    Args:
        articles:       {pmid: {"title": ..., "abstract": ..., ...}}
        groups:         {"population": [...], "intervention": [...], "comparator": [...]}
        require_groups: used only in the fallback path (default 2)

    Returns:
        Filtered {pmid: article} dict.
    """
    pop_terms = [t.lower() for t in groups.get("population", []) if t]
    int_terms = [t.lower() for t in groups.get("intervention", []) if t]
    cmp_terms = [t.lower() for t in groups.get("comparator", []) if t]

    if not int_terms and not cmp_terms:
        return articles  # nothing to filter on

    # Significant words from intervention terms that are NOT in the comparator group.
    # e.g. "pegfilgrastim", "neulasta" — but NOT "filgrastim", "granulocyte".
    cmp_sig = _sig_words(cmp_terms)
    int_sig = _sig_words(int_terms) - cmp_sig

    if not int_sig:
        # Fallback: no unique intervention sig words — use loose group matching
        def any_in(text: str, terms: list) -> bool:
            return any(t in text for t in terms)

        kept = {}
        for pmid, art in articles.items():
            title    = (art.get("title")    or "").lower()
            abstract = (art.get("abstract") or "").lower()
            text     = title + " " + abstract
            pop_match  = any_in(text, pop_terms) if pop_terms else True
            int_match  = any_in(text, int_terms)
            cmp_match  = any_in(text, cmp_terms) if cmp_terms else False
            drug_match = int_match or cmp_match
            if drug_match:
                matched = sum([pop_match, int_match, cmp_match if cmp_terms else False])
                if matched >= require_groups:
                    kept[pmid] = art
        return kept

    # Primary path: keep if any intervention-specific sig word appears in title OR abstract.
    # Papers that matched PubMed only via MeSH terms (not literal text) get filtered out.
    kept = {}
    for pmid, art in articles.items():
        title    = (art.get("title")    or "").lower()
        abstract = (art.get("abstract") or "").lower()
        text     = title + " " + abstract
        if any(w in text for w in int_sig):
            kept[pmid] = art

    return kept


def prefilter_stats(before: Dict, after: Dict, gt: set) -> str:
    """Return a one-line summary of pre-filter effect on GT coverage."""
    gt_before = len(set(before) & gt)
    gt_after  = len(set(after)  & gt)
    return (
        f"Pre-filter: {len(before)} → {len(after)} papers "
        f"(GT coverage: {gt_before}/{len(gt)} → {gt_after}/{len(gt)})"
    )
