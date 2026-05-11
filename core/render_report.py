"""
Standalone CLI to (re-)render a systematic-review PDF from artifacts that
the pipeline already wrote to disk — no pipeline re-run required.

Use cases:
    1. Pipeline crashed in the synthesis stage but earlier stages saved
       their CSVs — render a partial report from what's there.
    2. You tweaked the report_generator template / styles and want to
       regenerate PDFs without re-running expensive LLM stages.
    3. Comparing PDF output across runs (different qids).

The script auto-discovers artifacts by qid:
    artifacts_day1/pico_<qid>.json                # PICO + question
    artifacts_day3/mesh_expand_query_<qid>.json   # MeSH query
    artifacts_day4/all_studies_metadata_<qid>.jsonl  # retrieved articles
    artifacts_day5/screening_results_*.csv        # newest by mtime
    artifacts_day6/study_char_*.csv               # newest by mtime
    artifacts_day6/study_outcomes_*.csv           # newest by mtime

Usage:
    python3 render_report.py                      # latest qid, latest tables
    python3 render_report.py --qid 6d09b924e073   # specific qid
    python3 render_report.py --list               # show qids found on disk
    python3 render_report.py --out /tmp/sr.pdf    # custom output path
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))

from report_generator import build_systematic_review_pdf

ART_PICO  = CORE / "artifacts_day1"
ART_MESH  = CORE / "artifacts_day3"
ART_FETCH = CORE / "artifacts_day4"
ART_SCRN  = CORE / "artifacts_day5"
ART_EXTR  = CORE / "artifacts_day6"
ART_OUT   = CORE / "artifacts_day7"


# ── discovery helpers ──────────────────────────────────────────────────────

def list_qids() -> list[tuple[str, float]]:
    """Find qids that have a PICO trace file; return (qid, mtime) sorted newest-first."""
    out = []
    if ART_PICO.exists():
        for p in ART_PICO.glob("pico_*.json"):
            qid = p.stem.removeprefix("pico_")
            out.append((qid, p.stat().st_mtime))
    out.sort(key=lambda x: -x[1])
    return out


def latest_csv(folder: Path, pattern: str) -> Path | None:
    """Return the most-recently-modified file matching `pattern` in `folder`, or None."""
    if not folder.exists():
        return None
    candidates = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


# ── artifact loaders ───────────────────────────────────────────────────────

def load_pico_trace(qid: str) -> dict:
    """Load the per-qid PICO trace (contains qid + pico_valid + question)."""
    p = ART_PICO / f"pico_{qid}.json"
    if not p.exists():
        raise FileNotFoundError(f"PICO trace not found for qid={qid}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_mesh_query(qid: str) -> str:
    """Load the executed PubMed query string for `qid`."""
    p = ART_MESH / f"mesh_expand_query_{qid}.json"
    if not p.exists():
        # Fallback to the basic-strategy mesh_query.jsonl if the expand file is missing.
        jsonl = ART_MESH / "mesh_query.jsonl"
        if jsonl.exists():
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("qid") == qid:
                        return rec.get("mesh_query", "")
        return "(query not available)"
    return json.loads(p.read_text(encoding="utf-8")).get("query", "")


def load_articles(qid: str) -> dict:
    """Reconstruct the {pmid: {title, abstract, ...}} dict from the JSONL artifact."""
    p = ART_FETCH / f"all_studies_metadata_{qid}.jsonl"
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pmid = rec.get("pmid")
            if pmid:
                out[str(pmid)] = rec
    return out


def load_pmids_relevant(csv_path: Path | None, threshold: float = 3.0) -> list[str]:
    """Read screening CSV, return PMIDs that meet the relevance threshold."""
    if not csv_path or not csv_path.exists():
        return []
    out = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if float(r.get("cumulative_score", "0")) >= threshold:
                    pid = r.get("study_id") or r.get("pmid")
                    if pid:
                        out.append(str(pid))
            except (ValueError, TypeError):
                continue
    return out


# ── synthesis-narrative discovery ───────────────────────────────────────────
# The synthesis stage doesn't currently persist its narrative to a per-qid
# file; it only returns it in-memory to the UI. If you have the synthesis
# JSON saved (e.g., the user dropped it in artifacts_day7/synthesis_<qid>.json),
# we'll pick it up. Otherwise the report shows a "narrative unavailable" note.

def load_synthesis(qid: str) -> tuple[str, str]:
    """Return (summary, markdown) for the synthesis if a JSON file exists."""
    candidates = [
        ART_OUT / f"synthesis_{qid}.json",
        ART_EXTR / f"synthesis_{qid}.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                summ = obj.get("summary", {}).get("one_sentence", "") \
                       if isinstance(obj.get("summary"), dict) \
                       else (obj.get("summary") or "")
                narr = obj.get("narrative_markdown") or obj.get("narrative") or ""
                return summ, narr
            except Exception:
                continue
    return "", ""


# ── eligibility discovery (often kept only in memory) ──────────────────────

def load_eligibility(qid: str) -> dict:
    """Look for a saved eligibility JSON; return {} if not found."""
    candidates = [
        ART_PICO  / f"eligibility_{qid}.json",
        ART_MESH  / f"eligibility_{qid}.json",
        ART_OUT   / f"eligibility_{qid}.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


# ── main render ─────────────────────────────────────────────────────────────

def render(qid: str, *,
           out_path: Path | None = None,
           csv_screened: Path | None = None,
           csv_chars:    Path | None = None,
           csv_outcomes: Path | None = None,
           threshold: float = 3.0,
           verbose: bool = True) -> Path:
    """Assemble inputs from disk and call build_systematic_review_pdf.

    Args:
        qid:           Pipeline run identifier (the hex string in artifact filenames).
        out_path:      Optional explicit output PDF path.
        csv_screened:  Override the auto-discovered screening CSV.
        csv_chars:     Override the auto-discovered characteristics CSV.
        csv_outcomes:  Override the auto-discovered outcomes CSV.
        threshold:     Cumulative-score cutoff for "included" PMIDs.
        verbose:       Print discovery progress to stdout.

    Returns:
        Path to the generated PDF.
    """
    pico_trace = load_pico_trace(qid)
    question   = pico_trace.get("question", "")
    if verbose: print(f"  pico:        qid={qid}  question={question[:60]!r}")

    mesh_query = load_mesh_query(qid)
    if verbose: print(f"  mesh_query:  {mesh_query[:80]}…")

    articles   = load_articles(qid)
    if verbose: print(f"  articles:    {len(articles)} retrieved papers")

    csv_screened = csv_screened or latest_csv(ART_SCRN,  "screening_results_*.csv")
    csv_chars    = csv_chars    or latest_csv(ART_EXTR,  "study_char_*.csv")
    csv_outcomes = csv_outcomes or latest_csv(ART_EXTR,  "study_outcomes_*.csv")
    if verbose:
        print(f"  screening:   {csv_screened.name if csv_screened else '(none found)'}")
        print(f"  chars:       {csv_chars.name    if csv_chars    else '(none found)'}")
        print(f"  outcomes:    {csv_outcomes.name if csv_outcomes else '(none found)'}")

    pmids_relevant = load_pmids_relevant(csv_screened, threshold)
    if verbose: print(f"  included:    {len(pmids_relevant)} PMIDs (score ≥ {threshold})")

    eligibility = load_eligibility(qid)
    if verbose:
        if eligibility:
            print(f"  eligibility: loaded ({sum(len(v) if isinstance(v, list) else 1 for v in eligibility.values())} criteria)")
        else:
            print(f"  eligibility: not found on disk (PDF will skip the criteria block)")

    summary, markdown = load_synthesis(qid)
    if verbose:
        print(f"  synthesis:   {'loaded' if markdown else '(none — narrative section will be empty)'}")

    pdf = build_systematic_review_pdf(
        pico_json         = pico_trace,
        question          = question,
        mesh_query        = mesh_query,
        articles          = articles,
        eligibility       = eligibility,
        csv_screened      = str(csv_screened) if csv_screened else None,
        pmids_relevant    = pmids_relevant,
        csv_chars         = str(csv_chars)    if csv_chars    else None,
        csv_outcomes      = str(csv_outcomes) if csv_outcomes else None,
        evidence_summary  = summary,
        evidence_markdown = markdown,
        score_threshold   = threshold,
        output_path       = out_path,
    )
    return Path(pdf)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qid",        default=None,
                   help="Pipeline run identifier. Defaults to the latest qid found on disk.")
    p.add_argument("--list",       action="store_true",
                   help="List available qids (newest first) and exit.")
    p.add_argument("--out",        default=None,
                   help="Output PDF path. Defaults to artifacts_day7/sr_<qid>_<ts>.pdf")
    p.add_argument("--threshold",  type=float, default=3.0,
                   help="Cumulative-score threshold for screening inclusion (default: 3.0)")
    p.add_argument("--csv-screened", default=None, help="Override screening CSV path.")
    p.add_argument("--csv-chars",    default=None, help="Override characteristics CSV path.")
    p.add_argument("--csv-outcomes", default=None, help="Override outcomes CSV path.")
    p.add_argument("--quiet",      action="store_true", help="Suppress discovery output.")
    args = p.parse_args()

    qids = list_qids()
    if args.list:
        if not qids:
            print("No qids found on disk.")
            return 0
        print(f"{len(qids)} qid(s) found (newest first):")
        for q, mt in qids:
            ts = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {q}   ({ts})")
        return 0

    if not qids:
        print("ERROR: no PICO traces found in artifacts_day1/. Run the pipeline first.",
              file=sys.stderr)
        return 1

    qid = args.qid or qids[0][0]
    if not args.quiet:
        print(f"Rendering report for qid: {qid}")
        if not args.qid:
            print(f"  (auto-selected — most recent of {len(qids)} on disk)")

    try:
        out = render(
            qid=qid,
            out_path     = Path(args.out) if args.out else None,
            csv_screened = Path(args.csv_screened) if args.csv_screened else None,
            csv_chars    = Path(args.csv_chars)    if args.csv_chars    else None,
            csv_outcomes = Path(args.csv_outcomes) if args.csv_outcomes else None,
            threshold    = args.threshold,
            verbose      = not args.quiet,
        )
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"\n✓ PDF generated: {out}")
    print(f"  Size: {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
