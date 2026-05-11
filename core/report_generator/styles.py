"""
Typography, colors, and paragraph styles for journal-style systematic
review PDFs. Models the visual language of BMC Medicine / BMJ /
Cochrane reviews: serif body, sans-serif headings, two-column body
layout, structured abstract, numbered references.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import cm

# ── Page geometry ────────────────────────────────────────────────────────────

PAGE_SIZE       = LETTER
MARGIN_TOP      = 1.8 * cm
MARGIN_BOTTOM   = 1.8 * cm
MARGIN_LEFT     = 1.6 * cm
MARGIN_RIGHT    = 1.6 * cm
COLUMN_GAP      = 0.6 * cm

# ── Color palette (academic) ─────────────────────────────────────────────────

NAVY        = colors.HexColor("#1A365D")
BMC_BLUE    = colors.HexColor("#0084B5")    # accent for journal banner
SLATE       = colors.HexColor("#2D3748")
COOL_GRAY   = colors.HexColor("#4A5568")
LIGHT_GRAY  = colors.HexColor("#E2E8F0")
TABLE_ZEBRA = colors.HexColor("#F7FAFC")
LINK_BLUE   = colors.HexColor("#2B6CB0")
RULE_BLACK  = colors.HexColor("#000000")


# ── Paragraph styles ─────────────────────────────────────────────────────────

def build_styles():
    """Return a dict of named ParagraphStyle objects used throughout the report."""
    base = getSampleStyleSheet()
    s: dict = {}

    # ── Page 1 / front-matter ────────────────────────────────────────────────

    # Journal-style banner labels: "RESEARCH ARTICLE" and "Open Access"
    s["Banner"] = ParagraphStyle(
        "Banner", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=12,
        textColor=SLATE, alignment=TA_LEFT,
    )
    s["BannerRight"] = ParagraphStyle(
        "BannerRight", parent=s["Banner"],
        textColor=BMC_BLUE, alignment=TA_RIGHT,
    )

    # Article title — large, bold, single-column, full width
    s["ArticleTitle"] = ParagraphStyle(
        "ArticleTitle", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=SLATE, alignment=TA_LEFT, spaceBefore=8, spaceAfter=10,
    )

    # Author line — small, with superscripted markers
    s["Authors"] = ParagraphStyle(
        "Authors", parent=base["Normal"],
        fontName="Helvetica", fontSize=10, leading=13,
        textColor=SLATE, alignment=TA_LEFT, spaceAfter=6,
    )

    # Abstract heading and structured-abstract labels
    s["AbstractHeading"] = ParagraphStyle(
        "AbstractHeading", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=SLATE, alignment=TA_LEFT, spaceBefore=10, spaceAfter=4,
    )
    s["AbstractText"] = ParagraphStyle(
        "AbstractText", parent=base["Normal"],
        fontName="Times-Roman", fontSize=9, leading=12,
        textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=4,
    )
    # Inline-bold leader for structured abstract paragraphs:
    # rendered via <b>Background:</b> within AbstractText
    s["AbstractLabel"] = ParagraphStyle(
        "AbstractLabel", parent=s["AbstractText"],
        fontName="Times-Bold",
    )

    # Keywords / registration / correspondence — small caption-like text
    s["MetaLine"] = ParagraphStyle(
        "MetaLine", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=11,
        textColor=COOL_GRAY, alignment=TA_LEFT, spaceAfter=2,
    )
    s["License"] = ParagraphStyle(
        "License", parent=base["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=COOL_GRAY, alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["Affiliation"] = ParagraphStyle(
        "Affiliation", parent=base["Normal"],
        fontName="Helvetica", fontSize=7.5, leading=9.5,
        textColor=COOL_GRAY, alignment=TA_LEFT, spaceAfter=2,
    )
    s["CitationFooter"] = ParagraphStyle(
        "CitationFooter", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=SLATE, alignment=TA_LEFT, spaceBefore=4,
    )

    # ── Body (two-column) ──────────────────────────────────────────────────

    s["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=12, leading=15,
        textColor=SLATE, spaceBefore=12, spaceAfter=4,
        keepWithNext=True,
    )
    # Subsection headings — italic, slightly smaller, no spaceBefore so they
    # tuck under H1 cleanly (matches BMC style)
    s["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontName="Helvetica-BoldOblique", fontSize=10, leading=13,
        textColor=SLATE, spaceBefore=8, spaceAfter=2,
        keepWithNext=True,
    )

    # Body text — Times Roman 9pt, justified, tight leading for two columns
    s["Body"] = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontName="Times-Roman", fontSize=9.5, leading=12.5,
        textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["Bullet"] = ParagraphStyle(
        "Bullet", parent=s["Body"],
        leftIndent=0.5 * cm, bulletIndent=0.15 * cm,
        spaceAfter=2, alignment=TA_LEFT,
    )

    # Table styles
    s["TableHeader"] = ParagraphStyle(
        "TableHeader", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=8, leading=10,
        textColor=colors.white, alignment=TA_LEFT,
    )
    s["TableCell"] = ParagraphStyle(
        "TableCell", parent=base["Normal"],
        fontName="Times-Roman", fontSize=8, leading=10,
        textColor=SLATE, alignment=TA_LEFT,
    )
    s["TableCaption"] = ParagraphStyle(
        "TableCaption", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=11,
        textColor=SLATE, alignment=TA_LEFT,
        spaceBefore=4, spaceAfter=2,
    )
    s["FigureCaption"] = ParagraphStyle(
        "FigureCaption", parent=s["TableCaption"],
    )

    # Reference list — Vancouver-style, hanging indent
    s["Reference"] = ParagraphStyle(
        "Reference", parent=base["Normal"],
        fontName="Times-Roman", fontSize=8.5, leading=10.5,
        textColor=SLATE, alignment=TA_LEFT,
        spaceAfter=2, leftIndent=0.7 * cm, firstLineIndent=-0.7 * cm,
    )

    # End-matter blocks (Acknowledgments, Funding, etc.) — slightly smaller body
    s["EndMatter"] = ParagraphStyle(
        "EndMatter", parent=base["Normal"],
        fontName="Times-Roman", fontSize=8.5, leading=11,
        textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=3,
    )
    # Bold inline label for end-matter blocks (Funding:, Authors' contributions:, ...)
    s["EndMatterHeading"] = ParagraphStyle(
        "EndMatterHeading", parent=s["H1"],
        fontSize=10, leading=12, spaceBefore=8, spaceAfter=2,
    )

    # Code / monospace blocks (e.g. the search query verbatim)
    s["Code"] = ParagraphStyle(
        "Code", parent=base["Normal"],
        fontName="Courier", fontSize=7.5, leading=10,
        textColor=SLATE, leftIndent=0.3 * cm, rightIndent=0.3 * cm,
        spaceBefore=2, spaceAfter=4,
    )

    # Inline footer style (page numbers / running header)
    s["Footer"] = ParagraphStyle(
        "Footer", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=COOL_GRAY,
    )

    return s
