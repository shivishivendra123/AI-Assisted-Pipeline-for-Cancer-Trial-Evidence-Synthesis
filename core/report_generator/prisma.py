"""
PRISMA flow diagram (PRISMA-2020 four-stage layout) drawn natively in
ReportLab using Drawing primitives. Renders as a vector flowchart embedded
in the PDF — no external image dependency.

Standard layout:

    ┌────────────────────────┐
    │ Records identified     │
    │ from PubMed (n = X)    │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐    ┌──────────────────────────┐
    │ Records screened       │ →  │ Records excluded         │
    │ (n = X)                │    │ (n = X)                  │
    └───────────┬────────────┘    └──────────────────────────┘
                │
                ▼
    ┌────────────────────────┐    ┌──────────────────────────┐
    │ Reports assessed       │ →  │ Reports excluded         │
    │ for eligibility (n=X)  │    │ (n = X, reasons listed)  │
    └───────────┬────────────┘    └──────────────────────────┘
                │
                ▼
    ┌────────────────────────┐
    │ Studies included in    │
    │ synthesis (n = X)      │
    └────────────────────────┘
"""

from dataclasses import dataclass

from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen.canvas import Canvas


@dataclass
class PrismaCounts:
    """Counts that drive each box of the PRISMA flow diagram."""
    identified:        int   # Stage 3 retrieval count (PubMed top-K)
    screened:          int   # Stage 5 input pool (= identified for our pipeline)
    screened_excluded: int   # Failed screening threshold
    assessed:          int   # Passed screening, entered extraction
    assessed_excluded: int   # Excluded during extraction (typically 0 — we extract everything passed)
    included:          int   # Final included for synthesis


# ── visual constants ────────────────────────────────────────────────────────

BOX_W       = 5.4 * cm
BOX_H       = 1.6 * cm
H_GAP       = 0.9 * cm   # horizontal gap between main column and sidebar
V_GAP       = 1.1 * cm   # vertical gap between row centers

NAVY        = colors.HexColor("#1A365D")
LIGHT_NAVY  = colors.HexColor("#E6EAF1")
COOL_GRAY   = colors.HexColor("#4A5568")
LINE_GRAY   = colors.HexColor("#718096")


def _box(d: Drawing, x: float, y: float, lines: list[str],
         fill=LIGHT_NAVY, stroke=NAVY) -> tuple:
    """Draw a labeled box on the diagram.

    Args:
        d: Drawing object to render onto
        x, y: bottom-left corner of the box
        lines: list of text lines (each line drawn vertically stacked)
        fill, stroke: box fill + border colors

    Returns:
        (x_center, y_top, y_bottom) — anchors used to wire arrows.
    """
    d.add(Rect(x, y, BOX_W, BOX_H,
              fillColor=fill, strokeColor=stroke, strokeWidth=0.8))

    # vertical centering: stack lines around the box midline
    line_h = 11
    total = len(lines) * line_h
    start = y + (BOX_H - total) / 2 + line_h * 0.7
    for i, txt in enumerate(lines):
        d.add(String(x + BOX_W / 2, start + (len(lines) - 1 - i) * line_h, txt,
                    fontName="Helvetica", fontSize=8.5, fillColor=NAVY,
                    textAnchor="middle"))

    return (x + BOX_W / 2, y + BOX_H, y)


def _arrow_down(d: Drawing, x: float, y_from: float, y_to: float):
    """Vertical arrow from (x, y_from) downward to (x, y_to). Adds a small triangle head."""
    d.add(Line(x, y_from, x, y_to + 4, strokeColor=LINE_GRAY, strokeWidth=0.7))
    d.add(Polygon([x - 3, y_to + 4, x + 3, y_to + 4, x, y_to],
                  fillColor=LINE_GRAY, strokeColor=LINE_GRAY))


def _arrow_right(d: Drawing, x_from: float, x_to: float, y: float):
    """Horizontal arrow from (x_from, y) rightward to (x_to, y)."""
    d.add(Line(x_from, y, x_to - 4, y, strokeColor=LINE_GRAY, strokeWidth=0.7))
    d.add(Polygon([x_to - 4, y - 3, x_to - 4, y + 3, x_to, y],
                  fillColor=LINE_GRAY, strokeColor=LINE_GRAY))


def build_prisma_diagram(counts: PrismaCounts, source_label: str = "PubMed") -> Drawing:
    """Build a PRISMA-2020 flow diagram as a ReportLab Drawing flowable.

    Args:
        counts: PrismaCounts populated from pipeline stage outputs
        source_label: name of the database queried (default "PubMed")

    Returns:
        A Drawing flowable that can be added directly to a Story.
    """
    # Diagram canvas size
    width  = 2 * BOX_W + H_GAP + 0.2 * cm
    rows   = 4                                # identified, screened, assessed, included
    height = rows * BOX_H + (rows - 1) * V_GAP + 0.4 * cm

    d = Drawing(width, height)

    # Main column (left) and sidebar column (right)
    main_x = 0.1 * cm
    side_x = main_x + BOX_W + H_GAP

    # Row Y coords (top row first, count downward)
    row_top_y = lambda i: height - (i + 1) * BOX_H - i * V_GAP - 0.2 * cm

    # --- Row 1: Identification ---
    cx1, top1, bot1 = _box(d, main_x, row_top_y(0),
        [f"Records identified through {source_label} search",
         f"(n = {counts.identified})"])

    # --- Row 2: Screening ---
    cx2, top2, bot2 = _box(d, main_x, row_top_y(1),
        [f"Records screened (n = {counts.screened})"])
    _box(d, side_x, row_top_y(1),
        [f"Records excluded by", f"LLM screening (n = {counts.screened_excluded})"],
        fill=colors.white)

    # --- Row 3: Eligibility ---
    cx3, top3, bot3 = _box(d, main_x, row_top_y(2),
        [f"Reports assessed for eligibility",
         f"(n = {counts.assessed})"])
    if counts.assessed_excluded > 0:
        _box(d, side_x, row_top_y(2),
            [f"Reports excluded (n = {counts.assessed_excluded})"],
            fill=colors.white)

    # --- Row 4: Included ---
    cx4, top4, bot4 = _box(d, main_x, row_top_y(3),
        [f"Studies included in synthesis",
         f"(n = {counts.included})"])

    # --- Arrows: vertical between rows ---
    _arrow_down(d, cx1, bot1, top2)
    _arrow_down(d, cx2, bot2, top3)
    _arrow_down(d, cx3, bot3, top4)

    # --- Arrows: lateral exclusions ---
    side_y_2 = row_top_y(1) + BOX_H / 2
    _arrow_right(d, main_x + BOX_W, side_x, side_y_2)
    if counts.assessed_excluded > 0:
        side_y_3 = row_top_y(2) + BOX_H / 2
        _arrow_right(d, main_x + BOX_W, side_x, side_y_3)

    return d


def estimate_prisma_height(counts: PrismaCounts) -> float:
    """Return approximate diagram height in points — useful for layout planning."""
    rows = 4
    return rows * BOX_H + (rows - 1) * V_GAP + 0.4 * cm
