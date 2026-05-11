"""
Table builders for the systematic review PDF.

Two main tables modeled on Cochrane "Characteristics of included studies"
and "Summary of findings":
  - characteristics_table  (study-level metadata + design)
  - outcomes_table         (extracted outcome measures per study)

Both functions read the per-study CSVs the pipeline produces, fold them
to the included-PMID set, and emit ReportLab Table flowables with
consistent styling: bold sans-serif header, zebra striping, justified
text, hairline borders.

CSV schemas the pipeline writes (must match exactly):

  study_char_*.csv  — one row per study
      study_id, first_author, year, journal, country_or_setting,
      population_description, sample_size_total, intervention_name,
      comparator_name

  study_outcomes_*.csv — one row per study with JSON-encoded outcomes
      study_id, comparison_text,
      outcomes (JSON array of {outcome_label, effect_text, effect_type,
                                effect_value, ci_lower, ci_upper,
                                follow_up_text})

The outcomes table EXPLODES the JSON array — one PDF row per
(study × outcome) — so a study with 3 outcomes becomes 3 rows.
"""

import csv
import json
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph

from .styles import (
    NAVY, LIGHT_GRAY, TABLE_ZEBRA, build_styles,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int = 200) -> str:
    """Trim long cell values so rows don't paginate badly."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _read_csv_rows(csv_path: str | Path, included_pmids: set | None = None) -> list[dict]:
    """Load CSV and optionally filter to a set of PMIDs.

    The pipeline writes its extraction CSVs with a `study_id` (or `pmid`) column.
    """
    rows: list[dict] = []
    if not csv_path:
        return rows
    p = Path(csv_path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if included_pmids is not None:
                pmid = r.get("study_id") or r.get("pmid") or ""
                if pmid not in included_pmids:
                    continue
            rows.append(r)
    return rows


def _build_table_style(n_rows: int, header_bg=NAVY, body_color=colors.black) -> TableStyle:
    """Common table style: bold header on dark fill, zebra body, hairline grid."""
    cmds = [
        # Header
        ("BACKGROUND",       (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",        (0, 0), (-1, 0), colors.white),
        ("FONTNAME",         (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",         (0, 0), (-1, 0), 8.5),
        ("ALIGN",            (0, 0), (-1, 0), "LEFT"),
        ("VALIGN",           (0, 0), (-1, 0), "MIDDLE"),
        ("LEFTPADDING",      (0, 0), (-1, 0), 5),
        ("RIGHTPADDING",     (0, 0), (-1, 0), 5),
        ("TOPPADDING",       (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING",    (0, 0), (-1, 0), 5),
        # Body
        ("FONTNAME",         (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",         (0, 1), (-1, -1), 8.5),
        ("VALIGN",           (0, 1), (-1, -1), "TOP"),
        ("LEFTPADDING",      (0, 1), (-1, -1), 5),
        ("RIGHTPADDING",     (0, 1), (-1, -1), 5),
        ("TOPPADDING",       (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",    (0, 1), (-1, -1), 4),
        # Grid
        ("GRID",             (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
        ("LINEBELOW",        (0, 0), (-1, 0), 0.8, NAVY),
    ]
    # Zebra: shade odd body rows
    for i in range(1, n_rows):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ZEBRA))
    return TableStyle(cmds)


def _wrap(text: str, style) -> Paragraph:
    """Wrap a string in a Paragraph so long cell text wraps within the column."""
    safe = _truncate(text, 250).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


# ── 1. Characteristics of included studies ──────────────────────────────────

# (csv_column, table_label, relative_width)
# Column names match what extract_study_char_outcomes writes to disk.
CHARACTERISTICS_COLS = [
    ("study_id",                "PMID",           1.4),
    ("first_author",            "Author",         2.4),
    ("year",                    "Year",           0.9),
    ("country_or_setting",      "Setting",        2.2),
    ("population_description",  "Population",     4.0),
    ("intervention_name",       "Intervention",   2.8),
    ("comparator_name",         "Comparator",     2.4),
    ("sample_size_total",       "n",              0.8),
]


def characteristics_table(csv_path: str | Path, included_pmids: set,
                         max_rows: int | None = None,
                         page_width_cm: float = 17.0) -> Table | None:
    """Build the "Characteristics of Included Studies" table.

    Args:
        csv_path:        path to the study_char_*.csv produced by Stage 6
        included_pmids:  set of PMIDs to include (post-screening survivors)
        max_rows:        optional cap to keep large tables paginated
        page_width_cm:   total available width — column widths sum to this

    Returns:
        A ReportLab Table flowable, or None if the CSV has no rows.
    """
    rows = _read_csv_rows(csv_path, included_pmids)
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        return None

    styles = build_styles()
    header = [Paragraph(label, styles["TableHeader"]) for _, label, _ in CHARACTERISTICS_COLS]

    body: list[list] = []
    for r in rows:
        body.append([_wrap(r.get(key, ""), styles["TableCell"])
                     for key, _, _ in CHARACTERISTICS_COLS])

    # Column widths — proportional to the relative weights, summing to page_width_cm
    weights = [w for _, _, w in CHARACTERISTICS_COLS]
    total_w = sum(weights)
    col_widths = [(w / total_w) * page_width_cm * cm for w in weights]

    data = [header] + body
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(_build_table_style(len(data)))
    return tbl


# ── 2. Outcomes table ────────────────────────────────────────────────────────

# Column scheme for the FLATTENED outcomes table — one row per
# (study × outcome) pair. The CSV's `outcomes` column is a JSON array;
# we explode it so each outcome gets its own row.
OUTCOMES_COLS = [
    ("study_id",        "PMID",            1.4),
    ("comparison_text", "Comparison",      3.5),
    ("outcome_label",   "Outcome",         3.0),
    ("effect_text",     "Effect",          4.0),
    ("effect_value",    "Value",           1.4),
    ("ci_text",         "95% CI",          1.6),
    ("follow_up_text",  "Follow-up",       2.0),
]


def _parse_outcomes_json(raw: str) -> list[dict]:
    """Parse the JSON-encoded `outcomes` column into a list of dicts.

    Returns [] on any parse failure so a malformed cell doesn't crash the
    whole table.
    """
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _ci_text(o: dict) -> str:
    """Render a 95% CI from `ci_lower` / `ci_upper` columns of an outcome dict."""
    lo = o.get("ci_lower")
    hi = o.get("ci_upper")
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"≤ {hi}"
    if hi is None:
        return f"≥ {lo}"
    return f"{lo}–{hi}"


def outcomes_table(csv_path: str | Path, included_pmids: set,
                   max_rows: int | None = None,
                   page_width_cm: float = 17.0) -> Table | None:
    """Build the "Outcomes" table — flattens JSON-encoded outcomes per study.

    The pipeline writes the outcomes CSV with one row per study and a
    JSON-array `outcomes` column. We unpack each array so the PDF table
    has one row per (study × individual outcome) pair, which is the
    natural Cochrane "Summary of findings" layout.

    Returns None if no rows after filtering / explosion.
    """
    raw_rows = _read_csv_rows(csv_path, included_pmids)
    if not raw_rows:
        return None

    # Explode JSON outcomes
    flat_rows: list[dict] = []
    for r in raw_rows:
        pmid = r.get("study_id") or r.get("pmid", "")
        comparison = r.get("comparison_text", "")
        outcomes = _parse_outcomes_json(r.get("outcomes", ""))
        if not outcomes:
            # Keep one row even when there are no parseable outcomes —
            # otherwise the study disappears from the report entirely.
            flat_rows.append({
                "study_id":        pmid,
                "comparison_text": comparison,
                "outcome_label":   "(no parseable outcomes)",
                "effect_text":     "",
                "effect_value":    "",
                "ci_text":         "",
                "follow_up_text":  "",
            })
            continue
        for o in outcomes:
            flat_rows.append({
                "study_id":        pmid,
                "comparison_text": comparison,
                "outcome_label":   o.get("outcome_label", ""),
                "effect_text":     o.get("effect_text", ""),
                "effect_value":    o.get("effect_value")
                                    if o.get("effect_value") is not None else "",
                "ci_text":         _ci_text(o),
                "follow_up_text":  o.get("follow_up_text", ""),
            })

    if max_rows:
        flat_rows = flat_rows[:max_rows]
    if not flat_rows:
        return None

    styles = build_styles()
    header = [Paragraph(label, styles["TableHeader"]) for _, label, _ in OUTCOMES_COLS]

    body: list[list] = []
    for r in flat_rows:
        body.append([_wrap(r.get(key, ""), styles["TableCell"])
                     for key, _, _ in OUTCOMES_COLS])

    weights = [w for _, _, w in OUTCOMES_COLS]
    total_w = sum(weights)
    col_widths = [(w / total_w) * page_width_cm * cm for w in weights]

    data = [header] + body
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(_build_table_style(len(data)))
    return tbl


# ── 3. Compact summary table for the title page / abstract ──────────────────

def numeric_summary_table(retrieved: int, screened_in: int, included: int,
                          page_width_cm: float = 12.0) -> Table:
    """Three-cell numeric strip for executive-summary placement.

    Layout:
        [ Retrieved | Screened-in | Included ]
        [    100    |     17      |    13    ]
    """
    styles = build_styles()
    header = [
        Paragraph("Records retrieved",  styles["TableHeader"]),
        Paragraph("Passed screening",   styles["TableHeader"]),
        Paragraph("Included in synthesis", styles["TableHeader"]),
    ]
    body = [
        Paragraph(f"<b>{retrieved}</b>",  styles["TableCell"]),
        Paragraph(f"<b>{screened_in}</b>", styles["TableCell"]),
        Paragraph(f"<b>{included}</b>",   styles["TableCell"]),
    ]
    data = [header, body]

    col_w = (page_width_cm / 3) * cm
    tbl = Table(data, colWidths=[col_w, col_w, col_w])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",      (0, 1), (-1, 1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, 1), 14),
        ("TEXTCOLOR",     (0, 1), (-1, 1), NAVY),
        ("BACKGROUND",    (0, 1), (-1, 1), TABLE_ZEBRA),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.6, NAVY),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.6, NAVY),
    ]))
    return tbl
