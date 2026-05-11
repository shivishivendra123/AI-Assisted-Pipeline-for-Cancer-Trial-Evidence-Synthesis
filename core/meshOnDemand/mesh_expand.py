"""
Reference-grounded MeSH expansion (retrieval-augmented 2-stage retrieval-augmented
query construction).

Stage 1: PICO -> narrow probe -> fetch N reference papers via PubMed esearch.
Stage 2: PICO + reference paper text -> LLM emits CORE + EXPAND terms per axis.
Build:   (conditions) AND (treatments) AND (outcomes), no [MeSH Terms] qualifier.
         PubMed auto-translates free-text to MeSH where applicable.

Public API mirrors mesh_basic.generate_basic_mesh_query so it can drop into
the same call site.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from Bio import Entrez
from lxml import etree

# from agents.factory import build_vertex_agent  # medgemma via :predict endpoint
from agents.factory import build_vertex_gemini_agent  # Gemini 2.5 Flash via Vertex API
from configs.env_config import config
from pipeline.extractor.pico_extractor import _extract_vertex_content


# Configure Entrez once at import time (same pattern as efetch_utility.efetch).
Entrez.email = config.NCBI_EMAIL
Entrez.api_key = config.NCBI_API_KEY


# ── agent + prompt ────────────────────────────────────────────────────────────

def _agent():
    # Routing term-extraction through Gemini 2.5 Flash on Vertex.
    # GEMINI_LOCATION defaults to us-central1 (Gemini 2.5 Flash availability),
    # which is independent of GCP_LOCATION (us-east4) used for the medgemma
    # endpoint elsewhere in the pipeline.
    #
    # max_tokens=8000 (was 4000): DSPy diagnostic run showed 4000 truncated
    # JSON output mid-term-list — Gemini's chain-of-thought reasoning plus
    # ~24 list items + JSON formatting can exceed 4000 tokens. Truncation
    # causes the salvage parser to pick up partial axes, shrinking the
    # query and dropping recall. 8000 leaves the JSON room to close cleanly.
    # temperature is read from config (LLM_TEMPERATURE env var, default 0.0)
    # so the entire pipeline uses one consistent sampling setting. At 0 the
    # output is mostly deterministic — same query → same term lists across
    # runs (residual variance comes only from Vertex internal load
    # balancing and floating-point ties).
    return build_vertex_gemini_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GEMINI_LOCATION,
        model=config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=8000,
    )

# def _agent():  # medgemma variant — uncomment + flip imports to swap back
#     return build_vertex_agent(
#         project_id=config.GCP_PROJECT_ID,
#         location=config.GCP_LOCATION,
#         endpoint_id=config.GCP_ENDPOINT_ID,
#         dedicated_dns_or_predict_url=config.GCP_DEDICATED_DNS,
#         temperature=0.3,
#         max_tokens=4000,
#     )


def _load_prompt() -> str:
    base = Path(__file__).resolve().parent
    return (base.parent / "prompts" / "mesh_expand_prompt.txt").read_text(encoding="utf-8")


# ── stage 1: primary-term extraction → probe → fetch reference papers ───────

def _load_primary_term_prompt() -> str:
    base = Path(__file__).resolve().parent
    return (base.parent / "prompts" / "primary_term_prompt.txt").read_text(encoding="utf-8")


def _extract_primary_terms(pico: Dict[str, Any]) -> List[str]:
    """
    retrieval-augmented PRIMARY_TERM_EXTRACTION step.

    Asks the LLM for 3-4 narrow canonical terms describing the topic,
    so the reference-paper probe is `(t1) AND (t2) AND (t3)` instead of
    a verbose-PICO phrase-quoted probe. Sharper probe → more topical
    reference papers → richer vocabulary for the Stage 2 expansion call.

    Returns [] on any failure — caller falls back to verbose-PICO probe.
    """
    p = (pico.get("Population") or "").strip()
    i = (pico.get("Intervention") or "").strip()
    c = (pico.get("Comparator") or "").strip()
    o = pico.get("Outcomes", []) or []
    if not (p and i):
        return []
    if c.upper() in {"N/A", "NA", "NONE", ""}:
        c = "None"

    try:
        template = _load_primary_term_prompt()
    except Exception as e:
        print(f"[primary_term] prompt load failed: {e}")
        return []

    prompt = (template
              .replace("{{POPULATION}}",   p)
              .replace("{{INTERVENTION}}", i)
              .replace("{{COMPARATOR}}",   c)
              .replace("{{OUTCOMES}}",     json.dumps(o, ensure_ascii=False)))

    try:
        agent = _agent()
        agent.set_system(
            "You are a PubMed search-strategy expert. "
            "Output ONLY a JSON object with a 'terms' array. "
            "No markdown, no code fences, no commentary."
        )
        raw      = agent.say(prompt)
        response = _extract_vertex_content(raw) or str(raw)
        data     = json.loads(_clean_json(response))
        terms    = data.get("terms", [])
    except Exception as e:
        print(f"[primary_term] LLM call/parse failed: {e}")
        return []

    # Normalize, dedupe, cap at 4
    out, seen = [], set()
    for t in terms:
        if isinstance(t, str) and t.strip():
            k = t.strip().lower()
            if k not in seen:
                seen.add(k)
                out.append(t.strip())
    return out[:4]


def _build_probe(pico: Dict[str, Any], primary_terms: List[str] = None) -> str:
    """
    Build the primary search probe used to retrieve N reference papers.

    Preferred (retrieval-augmented): `(t1) AND (t2) AND (t3)` from LLM-extracted
        narrow terms. Each term is a single canonical phrase (e.g. "RRMM",
        "CAR-T", "BCMA"); ANDed together they surface a tight topical set.

    Fallback (used when primary-term extraction fails): the verbose-PICO
        phrase-quoted probe `("Population text") AND ("Intervention text")`.
        Less precise but always available.
    """
    if primary_terms:
        return " AND ".join(f"({t})" for t in primary_terms)

    p = (pico.get("Population") or "").strip()
    i = (pico.get("Intervention") or "").strip()
    if p and i:
        return f'("{p}") AND ("{i}")'
    if i:
        return f'"{i}"'
    if p:
        return f'"{p}"'
    return ""


def _fetch_reference_papers(pico: Dict[str, Any], n: int = 7) -> List[Dict[str, str]]:
    """Stage 1: fetch n exemplar papers (title + abstract) via Title/Abstract search.

    Now does the primary-term extraction step first — narrow LLM-extracted
    terms produce a sharper probe than the verbose-PICO phrase-quoted version.
    """
    primary_terms = _extract_primary_terms(pico)
    if primary_terms:
        print(f"[mesh_expand] primary terms: {primary_terms}")
    else:
        print("[mesh_expand] primary-term extraction empty/failed — falling back to verbose-PICO probe")

    probe = _build_probe(pico, primary_terms)
    if not probe:
        return []

    try:
        handle = Entrez.esearch(
            db="pubmed", term=probe, retmax=n,
            retmode="xml", sort="relevance", field="Title/Abstract",
        )
        root = etree.fromstring(handle.read())
        pmids = root.xpath("//Id/text()")
        if not pmids:
            return []

        handle = Entrez.efetch(
            db="pubmed", id=",".join(pmids),
            rettype="xml", retmode="xml",
        )
        root = etree.fromstring(handle.read())
        papers: List[Dict[str, str]] = []
        for art in root.xpath("//PubmedArticle"):
            title = " ".join(art.xpath(".//ArticleTitle//text()")).strip()
            abstract = " ".join(art.xpath(".//Abstract//AbstractText//text()")).strip()
            if title:
                # cap abstract length to keep prompt token budget sane
                papers.append({"title": title, "abstract": abstract[:1500]})
        return papers
    except Exception as e:
        print(f"[mesh_expand] reference-paper fetch failed: {e}")
        return []


def _format_references(papers: List[Dict[str, str]]) -> str:
    if not papers:
        return "(no reference papers retrieved — expand from PICO alone)"
    return "\n\n".join(
        f"{i+1}. {p['title']}\n   Abstract: {p['abstract']}"
        for i, p in enumerate(papers)
    )


# ── stage 2: prompt -> CORE+EXPAND terms ─────────────────────────────────────

def _prepare_prompt(pico_json: Dict[str, Any], papers: List[Dict[str, str]]) -> str:
    pico = pico_json.get("pico_valid", {})
    comparator = (pico.get("Comparator") or "").strip()
    if not comparator or comparator.upper() in {"N/A", "NA", "NONE"}:
        comparator = "None"
    return (_load_prompt()
            .replace("{{POPULATION}}",   pico.get("Population")   or "Not specified")
            .replace("{{INTERVENTION}}", pico.get("Intervention") or "Not specified")
            .replace("{{COMPARATOR}}",   comparator)
            .replace("{{OUTCOMES}}",     json.dumps(pico.get("Outcomes", []), ensure_ascii=False))
            .replace("{{REFERENCE_PAPERS}}", _format_references(papers)))


def _clean_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


# Heuristic: strip near-duplicate terms (e.g. "MM with bone marrow involvement"
# vs "MM with bone marrow infiltration") that LLM repetition loops emit.
_NEAR_DUP_NORM = re.compile(r"[^a-z0-9]+")


def _near_dup_key(s: str) -> str:
    """Token-set fingerprint, sorted — collapses word-order + punctuation diffs."""
    toks = _NEAR_DUP_NORM.split(s.lower().strip())
    return " ".join(sorted(t for t in toks if t))


def _dedupe_with_near(terms: List[str]) -> List[str]:
    """Dedupe by exact + near-duplicate fingerprint."""
    out, seen_exact, seen_near = [], set(), set()
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        kx = s.lower()
        kn = _near_dup_key(s)
        if kx in seen_exact or kn in seen_near:
            continue
        seen_exact.add(kx)
        seen_near.add(kn)
        out.append(s)
    return out


def _extract_axis_regex(text: str, key: str) -> List[str]:
    """
    Salvage extractor — pulls quoted strings from the value of `"<KEY>": [ ... ]`
    even when the JSON is truncated. Walks bracket depth so nested arrays are
    handled; falls through to end-of-text if the closing `]` was never emitted.
    """
    m = re.search(rf'"{key}"\s*:\s*\[', text)
    if not m:
        return []
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
        i += 1
    content = text[start:i]
    return [t for t in re.findall(r'"([^"]+)"', content) if t.strip()]


def _parse_axes(response: str) -> Dict[str, List[str]]:
    """
    Parse 3-step JSON; merge core+expand per axis; dedupe (exact + near-dup).
    Falls back to per-key regex salvage when JSON is truncated/malformed —
    so a half-emitted EXPAND list is still usable.
    """
    cleaned = _clean_json(response)

    # First try: strict JSON parse
    s2: Dict = {}
    s3: Dict = {}
    try:
        data = json.loads(cleaned)
        s2 = data.get("step 2", {}) or {}
        s3 = data.get("step 3", {}) or {}
    except Exception:
        # Salvage path — extract each axis with regex even if the JSON didn't close.
        # Each list is recovered independently; whichever axes the LLM finished
        # before truncation are preserved.
        for key in (
            "CORE_CONDITIONS", "CORE_TREATMENTS", "CORE_OUTCOMES",
            "EXPAND_CONDITIONS", "EXPAND_TREATMENTS", "EXPAND_OUTCOMES",
        ):
            terms = _extract_axis_regex(cleaned, key)
            if key.startswith("CORE_"):
                s2[key] = terms
            else:
                s3[key] = terms

    # Hard sharpness cap: keep the top-N most-canonical terms per axis after
    # merging CORE+EXPAND. CORE comes first in the combined list, so [:N]
    # preserves the discriminative terms and drops trailing low-IDF EXPAND
    # synonyms. LLMs tend to fill the prompt's upper bound (3-5 + 3-5 = 10
    # per axis); this enforces sharpness regardless of LLM behavior.
    #
    # Cap=8 (was 6): empirical finding from PMID 34761879 — at cap=6, CORE
    # filled 5 slots (allo-HSCT, TKI, imatinib, dasatinib, ponatinib), leaving
    # only 1 EXPAND slot. Critical drug names like "nilotinib" got cropped,
    # causing matching failures for trials whose title only mentions the
    # cropped drug. Cap=8 leaves 3 EXPAND slots — enough room for key
    # synonyms while still filtering trailing filler.
    AXIS_CAP = 8
    def merged(core_key, expand_key) -> List[str]:
        combined = (s2.get(core_key) or []) + (s3.get(expand_key) or [])
        return _dedupe_with_near(combined)[:AXIS_CAP]

    return {
        "conditions": merged("CORE_CONDITIONS", "EXPAND_CONDITIONS"),
        "treatments": merged("CORE_TREATMENTS", "EXPAND_TREATMENTS"),
        "outcomes":   merged("CORE_OUTCOMES",   "EXPAND_OUTCOMES"),
    }


# ── stage 3: build PubMed query ──────────────────────────────────────────────

def _build_query(axes: Dict[str, List[str]], include_outcomes: bool = True,
                 exclude_reviews: bool = False) -> str:
    """
    Default: 3-axis fat-OR, AND-joined: (conditions) AND (treatments) AND (outcomes).

    The outcomes axis acts as a "trial filter": ANDing in (overall survival
    OR mortality OR response rate OR ...) restricts the pool to papers that
    explicitly mention outcomes in title/abstract — overwhelmingly clinical
    trials and meta-analyses. The smaller-but-denser pool, combined with
    smaller-but-denser pool helps PubMed's relevance ranker push GT papers
    into the top-100 (which is what screening sees).

    Set include_outcomes=False to fall back to 2-axis (conditions AND
    treatments) — broader recall, weaker top-K ranking.

    No [MeSH Terms] qualifier — PubMed's automatic term mapping translates
    free-text to MeSH where it can, and falls back to all-fields matching
    where it can't (which catches recent unindexed papers).

    Terms are emitted unquoted so PubMed's automatic term mapping can do
    its job: a bare term like "breast cancer" gets mapped to BOTH
    "Breast Neoplasms"[MeSH] AND "breast cancer"[All Fields] internally.
    Phrase-quoting "breast cancer" disables this and forces exact-phrase
    matching, which is much more restrictive (we deliberately avoid quotes
    for the same reason). Terms containing characters that confuse the
    parser ('(', ')', '"', '[') are still wrapped to keep the query valid.
    """
    axes_to_use = ("conditions", "treatments")
    if include_outcomes:
        axes_to_use = ("conditions", "treatments", "outcomes")

    def _sanitize(t: str) -> str:
        """Strip characters that would force quoting; emit bare so PubMed's
        automatic term mapping can do its job.

        Removes: ( ) [ ] " — replaces them with a space and collapses runs of
        whitespace. Result is always emitted unquoted so PubMed's auto-
        translation kicks in (which is what we want for recall).

        Examples:
            'BCMA (B-cell maturation antigen)' -> 'BCMA B-cell maturation antigen'
            'pegfilgrastim "long-acting"'       -> 'pegfilgrastim long-acting'
            'breast cancer'                     -> 'breast cancer'  (unchanged)
        """
        cleaned = re.sub(r'[\(\)\[\]"]+', ' ', t)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    clauses = []
    for axis in axes_to_use:
        terms = axes.get(axis, [])
        if not terms:
            continue
        sanitized = [s for s in (_sanitize(t) for t in terms) if s]  # drop empties
        if not sanitized:
            continue
        ored = " OR ".join(sanitized)        # never quoted
        clauses.append(f"({ored})")
    query = " AND ".join(clauses)
    # Sanity guarantee: no double quotes anywhere in the final query.
    assert '"' not in query, f"Query contains double quotes: {query[:200]}"

    # Cochrane-style publication-type filter: drop reviews/meta-analyses from
    # the candidate pool BEFORE ranking, so original trials (the GT) aren't
    # outshouted by review-class papers in PubMed's relevance + iCite RCR
    # rankings. Standard SR retrieval methodology.
    if exclude_reviews and query:
        query += (" NOT (review[Publication Type] "
                  "OR meta-analysis[Publication Type] "
                  "OR systematic review[Publication Type])")
    return query


def _fallback_query(pico_json: Dict[str, Any]) -> str:
    """
    Fallback when the expand strategy can't produce a usable query (LLM error,
    parse failure with no salvageable terms, all-empty axes).

    Delegates to the basic-mesh path — that gives us a proven retrieval
    strategy instead of phrase-quoting verbose PICO text (which produced
    near-empty pools in earlier versions).
    """
    try:
        from meshOnDemand.mesh_basic import generate_basic_mesh_query
        print(f"[mesh_expand] fallback → mesh_basic")
        return generate_basic_mesh_query(pico_json)
    except Exception as e:
        # Last-resort: unquoted PICO tokens, let PubMed auto-translate.
        print(f"[mesh_expand] basic-mesh fallback also failed ({e}); using bare-token query")
        pv = pico_json.get("pico_valid", {})
        parts = []
        for k in ("Population", "Intervention", "Comparator"):
            v = (pv.get(k) or "").strip()
            if v and v.upper() not in {"N/A", "NA", "NONE"}:
                parts.append(f"({v})")
        return " AND ".join(parts) if parts else "clinical trial[Publication Type]"


# ── public entry point ───────────────────────────────────────────────────────

def generate_expand_mesh_query(pico_json: Dict[str, Any],
                              exclude_reviews: bool = False) -> str:
    """
    Reference-grounded 2-stage query generator.

    Args:
        pico_json: {qid, pico_valid: {Population, Intervention, Comparator, Outcomes}}
        exclude_reviews: if True, append a Cochrane-style publication-type
            filter that drops reviews and meta-analyses from the candidate pool.
            Lifts recall@K because original trials (the GT in TrialReviewBench)
            are no longer outranked by review-class papers in Best Match / iCite.

    Returns:
        PubMed query string of shape:
            (cond1 OR cond2 ...) AND (treat1 OR treat2 ...) AND (outcome1 OR outcome2 ...)
            [NOT (review[pt] OR meta-analysis[pt] OR systematic review[pt])]
    """
    pv  = pico_json.get("pico_valid", {})
    qid = pico_json.get("qid", "unknown")

    # --- stage 1: fetch reference papers
    papers = _fetch_reference_papers(pv, n=7)
    print(f"[mesh_expand qid={qid}] fetched {len(papers)} reference papers")

    # --- stage 2: grounded LLM expansion
    prompt = _prepare_prompt(pico_json, papers)
    agent  = _agent()
    agent.set_system(
        "You are a PubMed search-strategy expert. "
        "Output ONLY valid JSON in the exact shape requested. "
        "No markdown, no code fences, no commentary."
    )
    try:
        raw      = agent.say(prompt)
        response = _extract_vertex_content(raw) or str(raw)
    except Exception as e:
        print(f"[mesh_expand qid={qid}] LLM call failed: {e} — using fallback")
        return _fallback_query(pico_json)

    axes = _parse_axes(response)
    print(f"[mesh_expand qid={qid}] axes: "
          f"cond={len(axes['conditions'])} "
          f"treat={len(axes['treatments'])} "
          f"out={len(axes['outcomes'])}")

    # --- artifacts
    artifacts = Path(__file__).resolve().parent.parent / "artifacts_day3"
    artifacts.mkdir(exist_ok=True)
    (artifacts / f"mesh_expand_{qid}.txt").write_text(response, encoding="utf-8")

    if not any(axes.values()):
        print(f"[mesh_expand qid={qid}] empty axes — using fallback")
        return _fallback_query(pico_json)

    query = _build_query(axes, exclude_reviews=exclude_reviews)

    # Mirror the contract of mesh_basic so eval_bench can read these fields:
    pico_json["mesh_axes"]            = axes
    pico_json["mesh_groups"]          = {     # for iCite-rerank BM25 query
        "population":   axes["conditions"],
        "intervention": axes["treatments"],
        "comparator":   [],
    }
    pico_json["mesh_query_comparator"] = ""   # comparator is folded into treatments

    (artifacts / f"mesh_expand_query_{qid}.json").write_text(
        json.dumps({"qid": qid, "axes": axes, "query": query},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[mesh_expand qid={qid}] query[:200]: {query[:200]}…")
    return query


if __name__ == "__main__":
    example = {
        "qid": "test_expand_001",
        "pico_valid": {
            "Population":   "patients with relapsed/refractory multiple myeloma",
            "Intervention": "CAR-T therapy",
            "Comparator":   "None",
            "Outcomes":     ["overall response rate", "progression-free survival"],
        },
    }
    q = generate_expand_mesh_query(example)
    print("\n=== Generated query ===\n", q)
