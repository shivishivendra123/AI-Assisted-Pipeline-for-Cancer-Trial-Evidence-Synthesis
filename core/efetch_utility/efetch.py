from Bio import Entrez
from lxml import etree
import json
import time
from pathlib import Path
import re
import http.client
import urllib.error
from configs.env_config import config

# ===========================
# CONFIGURE EMAIL + API KEY
# ===========================
Entrez.email = config.NCBI_EMAIL
Entrez.api_key = config.NCBI_API_KEY


# ===========================
# GENERIC RETRY EFETCH
# ===========================
def retry_fetch(func, retries=None, wait=None, **kwargs):
    retries = retries or config.API_RETRY_ATTEMPTS
    wait = wait or config.API_RETRY_WAIT
    for i in range(retries):
        try:
            handle = func(**kwargs)
            return handle.read()
        except Exception as e:
            print(f" Retry {i+1}/{retries} for {func.__name__}: {e}")
            time.sleep(wait)
    raise RuntimeError(f" Failed after retries for: {func.__name__}")


# ===========================
# RAW ELINK → PMCID
# ===========================
def fetch_pmcid_raw(pmid):
    try:
        xml_raw = retry_fetch(
            Entrez.elink,
            dbfrom="pubmed",
            db="pmc",
            id=pmid,
            retmode="xml"
        ).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f" Error fetching ELINK for PMID {pmid}: {e}")
        return None

    try:
        root = etree.fromstring(xml_raw.encode())
    except Exception as e:
        print(f" Error parsing ELINK XML for PMID {pmid}: {e}")
        return None

    ids = root.xpath("//LinkSet/LinkSetDb/Link/Id/text()")
    return ids[0] if ids else None


# ===========================
# MAIN FUNCTION
# ===========================
def fetch_pubmed_articles(
    qid,
    search_term,
    max_results=None,
    fetch_fulltext=True,
    sort="relevance",
):
    """
    Single-sort PubMed retrieval.

    sort:
        "relevance" — PubMed's Best Match learning-to-rank (default)
        None        — omit sort param, use PubMed's default (chronological)
        any string  — passed through (e.g., "pub_date")
    """
    max_results = max_results or config.PUBMED_MAX_RESULTS

    # ---- PMID SEARCH ----
    sort_kwargs = {"sort": sort} if sort else {}
    search_xml = retry_fetch(
        Entrez.esearch,
        db="pubmed",
        term=search_term,
        retmax=max_results,
        retmode="xml",
        **sort_kwargs,
    )

    search_root = etree.fromstring(search_xml)
    pmids = search_root.xpath("//Id/text()")

    print(" Found PMIDs:", pmids)
    if not pmids:
        return

    # ---- FETCH TITLE + ABSTRACT (REAL ABSTRACT!) ----
    print("\n Fetching full abstracts...")

    efetch_xml = retry_fetch(
        Entrez.efetch,
        db="pubmed",
        id=",".join(pmids),
        rettype="xml",
        retmode="xml"
    )

    try:
        root = etree.fromstring(efetch_xml)
    except etree.XMLSyntaxError:
        # Some abstracts contain unescaped & — use recovery parser
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(efetch_xml, parser=parser)

    # metadata for ALL studies (no full XML here)
    articles = {}
    # full text XML only for those with PMCID
    fulltext_records = []

    fulltext_json = {}

    for article in root.xpath("//PubmedArticle"):
        pmid = article.xpath(".//PMID/text()")[0]

        title = " ".join(article.xpath(".//ArticleTitle//text()")).strip()

        abstract = " ".join(
            article.xpath(".//Abstract//AbstractText//text()")
        ).strip()

        # Extract NCT IDs from REAL abstract
        nct_ids = re.findall(r"NCT\d{8}", abstract, flags=re.IGNORECASE)

        articles[pmid] = {
            "qid":qid,
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "nct_ids": list(set(nct_ids)),
            "pmcid": None  # will fill if we find one
        }

    # ---- PMCID MAPPING + SEPARATE FULL PMC XML JSONL ----
    if not fetch_fulltext:
        print("\n Skipping PMCID/full-text fetch (fetch_fulltext=False)")
    else:
      print("\n Fetching PMCID and full PMC XML for each PMID with PMC...")

    for pmid in (pmids if fetch_fulltext else []):
        pmcid = fetch_pmcid_raw(pmid)
        if not pmcid:
            continue

        # update metadata record with PMCID
        if pmid in articles:
            articles[pmid]["pmcid"] = pmcid

        # Fetch full-text XML from PMC and store in SEPARATE structure
        try:
            pmc_xml_bytes = retry_fetch(
                Entrez.efetch,
                db="pmc",
                id=pmcid,
                rettype="full",
                retmode="xml"
            )
            pmc_xml_text = pmc_xml_bytes.decode("utf-8", errors="ignore")

            fulltext_records.append(
                {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "pmc_full_xml": pmc_xml_text
                }
            )

            fulltext_json[pmid] = {
                "pmcid": pmcid,
                "pmc_full_xml": pmc_xml_text
            }
        except Exception as e:
            print(f" Error fetching full PMC XML for PMCID {pmcid}: {e}")
            # just skip fulltext for this one

    # ---- SAVE JSONL FILES ----
    artifacts_dir = Path(__file__).resolve().parent.parent / config.ARTIFACTS_PUBMED_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Metadata for ALL studies (no full XML)
    metadata_file = artifacts_dir / f"all_studies_metadata_{qid}.jsonl"
    with open(metadata_file, "w", encoding="utf-8") as f:
        for a in articles.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # 2) Full-text XML ONLY for studies with PMCID
    fulltext_file = artifacts_dir / f"pmc_fulltexts_{qid}.jsonl"
    with open(fulltext_file, "w", encoding="utf-8") as f:
        for rec in fulltext_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # print(f"\n Saved metadata to: {metadata_file}")
    # print(f" Saved PMC full-text XML to: {fulltext_file}")
    return {"articles":articles
            ,"full_text":fulltext_json}


# ============================================================
# MULTI-SORT RETRIEVAL
# Issues N esearch calls with different sort orders, unions
# the deduped PMID lists, then runs the same metadata + PMC
# pipeline as fetch_pubmed_articles. Lifts old landmark papers
# that PubMed's sort=relevance buries (recency bias).
# ============================================================

# Default split: 1000 by relevance + 500 newest + 500 oldest = 2000 raw,
# typically ~1500-1900 unique after dedup. Matches the pipeline's
# RAW_RETRIEVAL_MAX budget so iCite reranking surface stays the same.
DEFAULT_SORT_STRATEGIES = [
    ("relevance",      1000),
    ("pub_date",       500),   # PubMed default for pub_date sort = descending (newest)
    ("pub_date_asc",   500),   # custom flag, mapped to ascending below
]


def _esearch_pmids(term: str, retmax: int, sort: str) -> list:
    """
    Single esearch call returning ordered PMID list.

    `sort` accepts:
      - "relevance"    → PubMed's Best Match algorithm
      - "pub_date"     → newest first (PubMed default for date sort)
      - "pub_date_asc" → oldest first (recency-bias correction)
      - any other PubMed-supported sort string is passed through.
    """
    if not term:
        return []

    # Map our convenience flag to the actual PubMed parameters.
    pubmed_sort = sort
    sort_kwargs = {}
    if sort == "pub_date_asc":
        pubmed_sort = "pub_date"
        sort_kwargs = {"sort": pubmed_sort, "sort_order": "asc"}
    else:
        sort_kwargs = {"sort": pubmed_sort}

    try:
        xml = retry_fetch(
            Entrez.esearch,
            db="pubmed",
            term=term,
            retmax=retmax,
            retmode="xml",
            **sort_kwargs,
        )
        root = etree.fromstring(xml)
        return list(root.xpath("//Id/text()"))
    except Exception as e:
        print(f"[multi-sort] esearch failed for sort={sort}: {e}")
        return []


def fetch_pubmed_articles_multi_sort(
    qid,
    search_term,
    sort_strategies=None,
    fetch_fulltext=True,
):
    """
    Multi-sort retrieval: union of multiple esearch result sets, deduped.

    Recall benefit comes from sort=pub_date_asc — pulls older landmark trials
    that sort=relevance ranks low because of PubMed's recency bias. Net pool
    is the same size budget as single-sort but more diverse across publication eras.

    Args:
        qid:             Query identifier (used in artifact filenames)
        search_term:     PubMed query string (built by mesh_basic / mesh_expand)
        sort_strategies: list of (sort_name, retmax) tuples.
                         Defaults to DEFAULT_SORT_STRATEGIES (1000/500/500).
        fetch_fulltext:  whether to elink → PMC and fetch full-text XML.

    Returns:
        Same shape as fetch_pubmed_articles:
            {"articles": {pmid: {...}}, "full_text": {pmid: {...}}}
    """
    sort_strategies = sort_strategies or DEFAULT_SORT_STRATEGIES

    # ---- 1. Multi-sort PMID search + dedup ----
    seen = set()
    union_pmids = []
    per_sort_counts = []

    for sort_name, retmax in sort_strategies:
        pmids = _esearch_pmids(search_term, retmax, sort_name)
        new_count = 0
        for p in pmids:
            if p not in seen:
                seen.add(p)
                union_pmids.append(p)
                new_count += 1
        per_sort_counts.append((sort_name, len(pmids), new_count))
        print(f"[multi-sort] sort={sort_name:<14} → {len(pmids):>4} PMIDs  "
              f"(new: {new_count})")

    print(f"[multi-sort] union (deduped) → {len(union_pmids)} unique PMIDs")

    if not union_pmids:
        return {"articles": {}, "full_text": {}}

    # ---- 2. Batched efetch on the union for title + abstract ----
    print("\n Fetching full abstracts (multi-sort union)…")
    efetch_xml = retry_fetch(
        Entrez.efetch,
        db="pubmed",
        id=",".join(union_pmids),
        rettype="xml",
        retmode="xml",
    )

    try:
        root = etree.fromstring(efetch_xml)
    except etree.XMLSyntaxError:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(efetch_xml, parser=parser)

    articles = {}
    for article in root.xpath("//PubmedArticle"):
        pmid = article.xpath(".//PMID/text()")[0]
        title = " ".join(article.xpath(".//ArticleTitle//text()")).strip()
        abstract = " ".join(article.xpath(".//Abstract//AbstractText//text()")).strip()
        nct_ids = re.findall(r"NCT\d{8}", abstract, flags=re.IGNORECASE)

        articles[pmid] = {
            "qid":      qid,
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "nct_ids":  list(set(nct_ids)),
            "pmcid":    None,
        }

    # ---- 3. PMCID + full-text (same pattern as single-sort) ----
    fulltext_records = []
    fulltext_json    = {}

    if not fetch_fulltext:
        print("\n Skipping PMCID/full-text fetch (fetch_fulltext=False)")
    else:
        print("\n Fetching PMCID and full PMC XML for each PMID with PMC...")

    for pmid in (union_pmids if fetch_fulltext else []):
        pmcid = fetch_pmcid_raw(pmid)
        if not pmcid:
            continue
        if pmid in articles:
            articles[pmid]["pmcid"] = pmcid
        try:
            pmc_xml_bytes = retry_fetch(
                Entrez.efetch, db="pmc", id=pmcid, rettype="full", retmode="xml",
            )
            pmc_xml_text = pmc_xml_bytes.decode("utf-8", errors="ignore")
            fulltext_records.append(
                {"pmid": pmid, "pmcid": pmcid, "pmc_full_xml": pmc_xml_text}
            )
            fulltext_json[pmid] = {"pmcid": pmcid, "pmc_full_xml": pmc_xml_text}
        except Exception as e:
            print(f" Error fetching full PMC XML for PMCID {pmcid}: {e}")

    # ---- 4. Save artifacts (same path scheme as single-sort) ----
    artifacts_dir = Path(__file__).resolve().parent.parent / config.ARTIFACTS_PUBMED_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / f"all_studies_metadata_{qid}.jsonl", "w", encoding="utf-8") as f:
        for a in articles.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(artifacts_dir / f"pmc_fulltexts_{qid}.jsonl", "w", encoding="utf-8") as f:
        for rec in fulltext_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"articles": articles, "full_text": fulltext_json}


# ============================================================
# DATE-STRATIFIED RETRIEVAL
# Issues N esearch calls with `mindate`/`maxdate` per-decade
# filters and sort=relevance WITHIN each bucket. Guarantees
# per-era coverage — old landmark trials don't compete with
# the volume of newer papers for retrieval slots.
# ============================================================

# 4 buckets × 500 = 2000 raw, ~1500-1900 unique after dedup.
# Lower bound 1980 = where MEDLINE indexing depth matures; pre-1980 papers
# often have sparse abstracts and rarely score well on term matching.
DEFAULT_DECADE_BUCKETS = [
    ("2020", "2030", 500),   # 2020s and later
    ("2010", "2019", 500),   # 2010s
    ("2000", "2009", 500),   # 2000s
    ("1980", "1999", 500),   # 1980s-1990s (combined — fewer indexed papers per year)
]


def _esearch_pmids_dated(term: str, mindate: str, maxdate: str,
                        retmax: int, sort: str = "relevance") -> list:
    """Single esearch call within a publication-date range. Returns ordered PMID list."""
    if not term:
        return []
    try:
        sort_kwargs = {"sort": sort} if sort else {}
        xml = retry_fetch(
            Entrez.esearch,
            db="pubmed", term=term,
            retmax=retmax, retmode="xml",
            datetype="pdat",          # filter on publication date
            mindate=mindate, maxdate=maxdate,
            **sort_kwargs,
        )
        root = etree.fromstring(xml)
        return list(root.xpath("//Id/text()"))
    except Exception as e:
        print(f"[by-decade] esearch failed for {mindate}-{maxdate}: {e}")
        return []


def fetch_pubmed_articles_by_decade(
    qid,
    search_term,
    decade_buckets=None,
    fetch_fulltext=True,
):
    """
    Date-stratified PubMed retrieval: union of per-decade esearch results.

    Each bucket runs a separate esearch with `mindate`/`maxdate` filters
    and sort=relevance WITHIN that range. So each retrieved paper is the
    most-topical paper from its era — not just the oldest by date.

    For SRs spanning multiple decades, this beats single-sort relevance
    (which buries old papers due to volume of newer matches) and beats
    multi-sort (where pub_date_asc/desc still mixes eras in the slot
    allocation). Date stratification guarantees per-era representation.

    Args:
        qid:             Query identifier (used in artifact filenames)
        search_term:     PubMed query string (built by mesh_basic / mesh_expand)
        decade_buckets:  list of (mindate, maxdate, retmax) tuples.
                         Defaults to DEFAULT_DECADE_BUCKETS (4 × 500, 1980-now).
        fetch_fulltext:  whether to elink → PMC and fetch full-text XML.

    Returns:
        Same shape as fetch_pubmed_articles:
            {"articles": {pmid: {...}}, "full_text": {pmid: {...}}}
    """
    decade_buckets = decade_buckets or DEFAULT_DECADE_BUCKETS

    # ---- 1. Per-decade esearch + dedup union ----
    seen = set()
    union_pmids = []

    for mindate, maxdate, retmax in decade_buckets:
        pmids = _esearch_pmids_dated(search_term, mindate, maxdate, retmax)
        new_count = 0
        for p in pmids:
            if p not in seen:
                seen.add(p)
                union_pmids.append(p)
                new_count += 1
        print(f"[by-decade] {mindate}-{maxdate}  → {len(pmids):>4} PMIDs  "
              f"(new: {new_count})")

    print(f"[by-decade] union (deduped) → {len(union_pmids)} unique PMIDs")

    if not union_pmids:
        return {"articles": {}, "full_text": {}}

    # ---- 2. Batched efetch on union for title + abstract ----
    print("\n Fetching full abstracts (date-stratified union)…")
    efetch_xml = retry_fetch(
        Entrez.efetch,
        db="pubmed",
        id=",".join(union_pmids),
        rettype="xml",
        retmode="xml",
    )

    try:
        root = etree.fromstring(efetch_xml)
    except etree.XMLSyntaxError:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(efetch_xml, parser=parser)

    articles = {}
    for article in root.xpath("//PubmedArticle"):
        pmid = article.xpath(".//PMID/text()")[0]
        title = " ".join(article.xpath(".//ArticleTitle//text()")).strip()
        abstract = " ".join(article.xpath(".//Abstract//AbstractText//text()")).strip()
        nct_ids = re.findall(r"NCT\d{8}", abstract, flags=re.IGNORECASE)

        articles[pmid] = {
            "qid":      qid,
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "nct_ids":  list(set(nct_ids)),
            "pmcid":    None,
        }

    # ---- 3. PMCID + full-text (same pattern as single-sort / multi-sort) ----
    fulltext_records = []
    fulltext_json    = {}

    if not fetch_fulltext:
        print("\n Skipping PMCID/full-text fetch (fetch_fulltext=False)")
    else:
        print("\n Fetching PMCID and full PMC XML for each PMID with PMC...")

    for pmid in (union_pmids if fetch_fulltext else []):
        pmcid = fetch_pmcid_raw(pmid)
        if not pmcid:
            continue
        if pmid in articles:
            articles[pmid]["pmcid"] = pmcid
        try:
            pmc_xml_bytes = retry_fetch(
                Entrez.efetch, db="pmc", id=pmcid, rettype="full", retmode="xml",
            )
            pmc_xml_text = pmc_xml_bytes.decode("utf-8", errors="ignore")
            fulltext_records.append(
                {"pmid": pmid, "pmcid": pmcid, "pmc_full_xml": pmc_xml_text}
            )
            fulltext_json[pmid] = {"pmcid": pmcid, "pmc_full_xml": pmc_xml_text}
        except Exception as e:
            print(f" Error fetching full PMC XML for PMCID {pmcid}: {e}")

    # ---- 4. Save artifacts ----
    artifacts_dir = Path(__file__).resolve().parent.parent / config.ARTIFACTS_PUBMED_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / f"all_studies_metadata_{qid}.jsonl", "w", encoding="utf-8") as f:
        for a in articles.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(artifacts_dir / f"pmc_fulltexts_{qid}.jsonl", "w", encoding="utf-8") as f:
        for rec in fulltext_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"articles": articles, "full_text": fulltext_json}


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    fetch_pubmed_articles()
