"""
Public entry point for journal-style systematic-review PDF generation.

Layout mirrors BMC Medicine / Cochrane Review conventions:
  - Page 1 (single column):
      "RESEARCH ARTICLE / Open Access" banner
      Article title
      Author block + affiliation markers
      Structured abstract (Background / Methods / Results / Conclusions)
      Keywords
      License notice
      Correspondence + affiliations footer
      Citation tail with synthetic DOI
  - Page 2 onward (two columns):
      Background, Methods (with subsections), Results (with subsections,
      tables, PRISMA flow), Discussion (with subsections), Conclusions,
      Abbreviations, end-matter (Acknowledgments, Funding, Authors'
      contributions, Ethics, Competing interests), References, Appendix.

Section headings, subheadings, inline numbered citations [1], full Vancouver-
style references, and standard journal end-matter blocks are produced
automatically from the pipeline outputs.

Public API:
    from report_generator import build_systematic_review_pdf
    pdf_path = build_systematic_review_pdf(...)
"""

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, PageBreak, KeepTogether,
    Table, TableStyle, NextPageTemplate,
)
from reportlab.platypus.flowables import HRFlowable

from .styles import (
    PAGE_SIZE, MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT,
    COLUMN_GAP,
    NAVY, BMC_BLUE, SLATE, COOL_GRAY, LIGHT_GRAY, LINK_BLUE, RULE_BLACK,
    build_styles,
)
from .prisma import PrismaCounts, build_prisma_diagram
from .tables import characteristics_table, outcomes_table


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

def _safe(s: Any) -> str:
    """XML-escape a value so it's safe to drop into a Paragraph."""
    if s is None:
        return ""
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_paragraphs(text: str) -> list[str]:
    """Break a multi-paragraph string into separate prose paragraphs."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _read_pmids_from_csv(csv_path: Path | str, threshold: float = 3.0) -> list[str]:
    """Pull PMIDs that pass `cumulative_score >= threshold` from a screening CSV."""
    if not csv_path:
        return []
    p = Path(csv_path)
    if not p.exists():
        return []
    out: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if float(r.get("cumulative_score", "0")) >= threshold:
                    pid = r.get("study_id") or r.get("pmid")
                    if pid:
                        out.append(str(pid))
            except (ValueError, TypeError):
                continue
    return out


def _capitalize_first(s: str) -> str:
    """Capitalize the first letter only — preserve everything else verbatim.

    Unlike str.capitalize() which lowercases the rest of the string (and
    would corrupt acronyms like PICO, CAR-T, BCMA), this only flips the
    very first character. Acronyms in the middle of the title stay intact.
    """
    if not s:
        return s
    return s[0].upper() + s[1:]


def _generate_title(pico: dict) -> str:
    """Build a journal-style review title from PICO with sentence-case start."""
    p = (pico.get("Population")   or "").strip()
    i = (pico.get("Intervention") or "").strip()
    c = (pico.get("Comparator")   or "").strip()

    has_comp = bool(c) and c.lower() not in ("none", "n/a", "na")
    if i and p and has_comp:
        title = f"{i} versus {c} in {p}: a systematic review"
    elif i and p:
        title = f"{i} for {p}: a systematic review"
    elif i:
        title = f"{i}: a systematic review"
    else:
        title = "Systematic Review"
    return _capitalize_first(title)


# ────────────────────────────────────────────────────────────────────────────
# Inline citation numbering — assigns [1], [2], ... in order of appearance
# of PMIDs in the synthesis narrative, then emits the ordered reference list
# at the end. Matches the BMC-style inline citations seen in the example.
# ────────────────────────────────────────────────────────────────────────────

class CitationRegistry:
    """Track PMIDs as they appear in narrative, assign sequential [N] numbers.

    Usage:
        reg = CitationRegistry()
        text_with_numbers = reg.linkify(narrative_markdown)   # replaces 8-digit
                                                              # PMIDs with [N]
        ordered_pmids     = reg.ordered_pmids()
        # Then emit references in this order at the end of the document.
    """

    def __init__(self):
        self._pmid_to_num: dict[str, int] = {}
        self._order: list[str] = []

    def _assign(self, pmid: str) -> int:
        if pmid not in self._pmid_to_num:
            self._pmid_to_num[pmid] = len(self._order) + 1
            self._order.append(pmid)
        return self._pmid_to_num[pmid]

    def linkify(self, text: str) -> str:
        """Replace 8-digit PMIDs in `text` with [N] superscript-style citations.

        The returned text has citations rendered as <super>[N]</super> ReportLab
        markup so they appear superscripted in the final PDF.
        """
        if not text:
            return text

        def _replace(match: re.Match) -> str:
            pmid = match.group(1)
            n = self._assign(pmid)
            return f'<super>[{n}]</super>'

        return re.sub(r"\b(\d{8})\b", _replace, text)

    def ordered_pmids(self) -> list[str]:
        return list(self._order)

    def empty(self) -> bool:
        return len(self._order) == 0


# ════════════════════════════════════════════════════════════════════════════
# Page templates
# ════════════════════════════════════════════════════════════════════════════

def _make_doc(output_path: Path, title: str, journal: str, pub_year: int,
              run_id: str):
    """Build a BaseDocTemplate with three page templates:
       - 'page1':       single column (title + abstract + license)
       - 'body':        two-column body
       - 'tables_full': single column for full-page tables that span both columns
    """
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        title=title, author="REASON Pipeline",
    )

    # Single full-width frame
    full_frame = Frame(
        MARGIN_LEFT, MARGIN_BOTTOM,
        doc.width, doc.height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0,
    )

    # Two-column frames for body pages
    col_w = (doc.width - COLUMN_GAP) / 2.0
    left_frame = Frame(
        MARGIN_LEFT, MARGIN_BOTTOM, col_w, doc.height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0, id="left",
    )
    right_frame = Frame(
        MARGIN_LEFT + col_w + COLUMN_GAP, MARGIN_BOTTOM, col_w, doc.height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0, id="right",
    )

    short_title = title if len(title) < 65 else title[:62] + "…"
    citation_tail = f"{run_id} {journal} ({pub_year})"

    def _draw_footer(canvas, doc_):
        """Page footer: 'Author et al. Journal (Year) Vol:Iss   Page X of Y'."""
        canvas.saveState()
        # Page footer line (mid-bottom margin)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(COOL_GRAY)
        # Left side: short citation
        canvas.drawString(MARGIN_LEFT, MARGIN_BOTTOM / 2 - 2, citation_tail)
        # Right side: page number
        page_num = canvas.getPageNumber()
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN_RIGHT,
                              MARGIN_BOTTOM / 2 - 2,
                              f"Page {page_num}")
        # Hairline rule above footer text
        canvas.setStrokeColor(LIGHT_GRAY)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN_LEFT, MARGIN_BOTTOM / 2 + 8,
                    PAGE_SIZE[0] - MARGIN_RIGHT, MARGIN_BOTTOM / 2 + 8)
        canvas.restoreState()

    def _draw_page1(canvas, doc_):
        """Page 1 has its own footer (citation only — no page number)."""
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(COOL_GRAY)
        canvas.drawString(MARGIN_LEFT, MARGIN_BOTTOM / 2 - 2, citation_tail)
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN_RIGHT,
                              MARGIN_BOTTOM / 2 - 2, "Page 1")
        canvas.setStrokeColor(LIGHT_GRAY)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN_LEFT, MARGIN_BOTTOM / 2 + 8,
                    PAGE_SIZE[0] - MARGIN_RIGHT, MARGIN_BOTTOM / 2 + 8)
        canvas.restoreState()

    page1 = PageTemplate(id="page1", frames=[full_frame], onPage=_draw_page1)
    body  = PageTemplate(id="body",
                         frames=[left_frame, right_frame],
                         onPage=_draw_footer)
    tables_full = PageTemplate(id="tables_full",
                              frames=[full_frame],
                              onPage=_draw_footer)
    doc.addPageTemplates([page1, body, tables_full])
    return doc


# ════════════════════════════════════════════════════════════════════════════
# Page 1 — banner, title, authors, structured abstract, license
# ════════════════════════════════════════════════════════════════════════════

def _build_page1(story: list, S: dict, *,
                 title: str, pico: dict, n_retrieved: int, n_included: int,
                 evidence_summary: str, evidence_markdown: str,
                 question: str, run_id: str, journal: str, pub_year: int):
    """Compose page 1: banner, title, authors, structured abstract, license."""

    # ── Banner row: "RESEARCH ARTICLE" left, "Open Access" right ────────────
    banner = Table(
        [[Paragraph("RESEARCH ARTICLE", S["Banner"]),
          Paragraph("Open Access", S["BannerRight"])]],
        colWidths=[8 * cm, 8 * cm],
    )
    banner.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LINEBELOW",    (0, 0), (-1, -1), 1.0, RULE_BLACK),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.5 * cm))

    # ── Article title ───────────────────────────────────────────────────────
    story.append(Paragraph(_safe(title), S["ArticleTitle"]))

    # ── Author byline (placeholder authors with daggers/asterisks) ──────────
    author_html = (
        "REASON Automated Synthesis<super>1†</super>, "
        "Pipeline Generator<super>1†</super>, "
        "and Reviewer<super>2*</super>"
    )
    story.append(Paragraph(author_html, S["Authors"]))
    story.append(Spacer(1, 0.3 * cm))

    # ── Structured abstract ─────────────────────────────────────────────────
    story.append(Paragraph("Abstract", S["AbstractHeading"]))

    intervention = _safe(pico.get("Intervention", "the intervention"))
    population   = _safe(pico.get("Population", "the population of interest"))
    comparator   = _safe(pico.get("Comparator", "comparator")) or "comparator"
    outcomes     = _safe(pico.get("Outcomes", "the specified outcomes"))

    bg = (f"Multiple studies have examined {intervention} in {population}, "
          f"yet a synthesised view of the evidence is needed to inform "
          f"clinical decision-making. The relationship between the "
          f"intervention and patient-relevant outcomes "
          f"({outcomes}) remains incompletely characterised.")

    methods = (f"This systematic review followed a PRISMA-aligned "
               f"protocol. PubMed was searched using a retrieval-augmented "
               f"MeSH expansion strategy: an LLM extracted narrow primary "
               f"terms from the PICO question, fetched seven reference "
               f"papers, and emitted a three-axis Boolean query "
               f"(conditions AND treatments AND outcomes). The top "
               f"{n_retrieved} records by PubMed Best Match relevance were "
               f"screened by an LLM-driven agent against PICO-derived "
               f"eligibility criteria. Studies meeting the relevance "
               f"threshold underwent automated study-characteristics and "
               f"outcome extraction, and were synthesised qualitatively.")

    results = (f"Of {n_retrieved} records retrieved, "
               f"<b>{n_included} studies</b> met the eligibility criteria "
               f"and contributed to the synthesis. ")
    if evidence_summary:
        results += _safe(evidence_summary)

    conclusions = ("This automated synthesis suggests directions for "
                   "further investigation. Findings should be triangulated "
                   "with expert-curated systematic reviews and confirmed "
                   "in primary studies where possible.")

    abstract_blocks = [
        ("Background",  bg),
        ("Methods",     methods),
        ("Results",     results),
        ("Conclusions", conclusions),
    ]
    for label, body in abstract_blocks:
        para = Paragraph(f"<b>{label}:</b> {body}", S["AbstractText"])
        story.append(para)

    # ── Registration + Keywords ─────────────────────────────────────────────
    story.append(Paragraph(
        f"<b>Run ID:</b> {_safe(run_id)} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d')}",
        S["MetaLine"]))

    keywords = ", ".join([
        s.strip() for s in [
            pico.get("Population", "").strip(),
            pico.get("Intervention", "").strip(),
            pico.get("Comparator", "").strip() or "",
        ] if s and s.lower() not in ("none", "n/a")
    ])
    story.append(Paragraph(
        f"<b>Keywords:</b> {_safe(keywords or 'systematic review, evidence synthesis')}",
        S["MetaLine"]))

    story.append(Spacer(1, 0.3 * cm))

    # ── License & correspondence block (small print) ────────────────────────
    license_text = (
        "© The Author(s). Open Access. This article is licensed under a "
        "Creative Commons Attribution 4.0 International License. To view a "
        "copy of this licence, visit "
        '<link href="http://creativecommons.org/licenses/by/4.0/" '
        'color="#2B6CB0">http://creativecommons.org/licenses/by/4.0/</link>. '
        "The Creative Commons Public Domain Dedication waiver applies to "
        "the data made available in this article, unless otherwise stated "
        "in a credit line to the data."
    )
    story.append(Paragraph(license_text, S["License"]))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph(
        "<b>Correspondence:</b> automated synthesis — no human correspondence "
        "available for this report. For the original research question see "
        "the Run ID above.",
        S["License"]))

    story.append(Paragraph(
        "<sup>1</sup>REASON Pipeline, Department of Automated Evidence "
        "Synthesis. <sup>2</sup>Reviewer (placeholder).",
        S["Affiliation"]))

    # ── Citation footer ─────────────────────────────────────────────────────
    citation_tail = (f"REASON Synthesis. {journal} ({pub_year}) {run_id}<br/>"
                     f'<link href="https://reason.example/{_safe(run_id)}" '
                     f'color="#2B6CB0">reason.example/{_safe(run_id)}</link>')
    story.append(HRFlowable(width="100%", thickness=0.4, color=LIGHT_GRAY,
                          spaceBefore=8, spaceAfter=4))
    story.append(Paragraph(citation_tail, S["CitationFooter"]))


# ════════════════════════════════════════════════════════════════════════════
# Body sections (two-column)
# ════════════════════════════════════════════════════════════════════════════

def _build_background(story: list, S: dict, pico: dict, question: str,
                     citations: CitationRegistry):
    """Background: 2-3 paragraphs of clinical context, with placeholder cites."""
    story.append(Paragraph("Background", S["H1"]))

    p1 = (
        f"{_safe(pico.get('Population', 'The condition under review'))} "
        f"presents a substantial clinical and public-health burden. "
        f"Existing therapeutic strategies vary in their efficacy and "
        f"safety profiles, and high-quality syntheses of the available "
        f"evidence are critical for clinical decision-making, guideline "
        f"development, and the prioritisation of further research."
    )

    p2 = (
        f"The role of "
        f"{_safe(pico.get('Intervention', 'the intervention of interest'))} "
        f"in this population has been the focus of multiple primary "
        f"studies, but findings have not always been concordant. A "
        f"comprehensive synthesis is therefore required to characterise "
        f"the consistency, magnitude, and applicability of the evidence."
    )

    p3 = (
        f"The clinical question motivating this review is: "
        f"&ldquo;{_safe(question)}&rdquo;. The PICO framework "
        f"(Population, Intervention, Comparator, Outcomes) was used to "
        f"structure both the search strategy and the eligibility criteria, "
        f"as detailed in the Methods."
    )

    for para in (p1, p2, p3):
        story.append(Paragraph(para, S["Body"]))


def _build_methods(story: list, S: dict, pico: dict, mesh_query: str,
                  eligibility: dict, threshold: float, n_retrieved: int):
    """Methods section with subsections matching the BMC layout."""
    story.append(Paragraph("Methods", S["H1"]))
    story.append(Paragraph(
        "This systematic review was conducted under the guidance of the "
        "Preferred Reporting Items for Systematic Reviews and Meta-analyses "
        "(PRISMA) statement. The retrieval, screening, and synthesis stages "
        "were executed by the REASON automated pipeline; methods detail "
        "follows.",
        S["Body"]))

    # 3.1 Literature search strategy
    story.append(Paragraph("Literature search strategy", S["H2"]))
    story.append(Paragraph(
        "Search terms were generated using a retrieval-augmented MeSH "
        "expansion strategy. An initial language-model call extracted "
        "three to four narrow primary terms from the PICO question; these "
        "terms were used to fetch seven on-topic reference papers from "
        "PubMed via the NCBI E-utilities API. A second language-model call, "
        "grounded in the titles and abstracts of those reference papers, "
        "generated an expanded term list across three axes (conditions, "
        "treatments, outcomes). The final query is constructed as a "
        "fat-OR within each axis joined by Boolean AND across axes, with "
        "no explicit field qualifier so that PubMed's automatic term "
        "mapping engages. The exact query string appears in the Appendix.",
        S["Body"]))
    story.append(Paragraph(
        "PubMed/MEDLINE was queried via the NCBI E-utilities `esearch` "
        f"endpoint with `sort=relevance` and `retmax={n_retrieved}`. No "
        "explicit date or language restrictions were applied at the search "
        "step.",
        S["Body"]))

    # 3.2 Inclusion / exclusion criteria
    story.append(Paragraph("Inclusion and exclusion criteria", S["H2"]))
    story.append(Paragraph(
        "Inclusion and exclusion criteria were generated automatically "
        "from the PICO framework and applied during the screening stage. "
        "The criteria below were used:",
        S["Body"]))

    if isinstance(eligibility, dict):
        inc = eligibility.get("inclusion_criteria") or eligibility.get("Inclusion") or []
        exc = eligibility.get("exclusion_criteria") or eligibility.get("Exclusion") or []
        if inc:
            story.append(Paragraph("<i>Inclusion criteria:</i>", S["Body"]))
            items = inc if isinstance(inc, list) else [inc]
            for c in items:
                story.append(Paragraph("&bull; " + _safe(c), S["Bullet"]))
        if exc:
            story.append(Paragraph("<i>Exclusion criteria:</i>", S["Body"]))
            items = exc if isinstance(exc, list) else [exc]
            for c in items:
                story.append(Paragraph("&bull; " + _safe(c), S["Bullet"]))

    # 3.3 Data collection and quality assessment
    story.append(Paragraph("Data collection and quality assessment", S["H2"]))
    story.append(Paragraph(
        "Study-level characteristics (study name, design, year, "
        "geographical setting, population, intervention, comparator, "
        "sample size, age) and per-arm outcome data were extracted "
        "automatically using language-model extraction agents. Extraction "
        "was performed independently from screening to limit confirmation "
        "bias on the part of the agent. Per-study extracted records are "
        "preserved as CSV artefacts and reproduced in Tables 1 and 2.",
        S["Body"]))
    story.append(Paragraph(
        "Quality of evidence was not formally graded by the automated "
        "pipeline; the reader is referred to standard tools (e.g., the "
        "Newcastle-Ottawa scale or Cochrane Risk of Bias 2) for follow-on "
        "assessment.",
        S["Body"]))

    # 3.4 Data analyses / synthesis methods
    story.append(Paragraph("Data analyses", S["H2"]))
    story.append(Paragraph(
        f"Records with cumulative screening score ≥ {threshold:g} "
        "were considered eligible. Given likely heterogeneity in study "
        "designs, populations, and outcome definitions, a narrative "
        "synthesis was performed; no formal meta-analysis was conducted. "
        "The synthesis agent integrated study-level characteristics and "
        "extracted outcomes into a structured discussion.",
        S["Body"]))


def _build_results(story: list, S: dict,
                  pmids_relevant: list[str], n_retrieved: int,
                  csv_chars: str, csv_outcomes: str,
                  evidence_markdown: str,
                  citations: CitationRegistry,
                  doc_pagewidth_cm: float):
    """Results section: literature search → characteristics → outcomes → synthesis."""
    story.append(Paragraph("Results", S["H1"]))

    n_included = len(pmids_relevant)
    n_screened_excluded = n_retrieved - n_included

    # 4.1 Literature search results
    story.append(Paragraph("Literature search results", S["H2"]))
    story.append(Paragraph(
        f"The literature search retrieved {n_retrieved} records. After "
        f"automated screening against the predefined eligibility criteria, "
        f"{n_screened_excluded} records were excluded and "
        f"<b>{n_included} studies were included in the synthesis</b>. "
        f"The selection process is summarised in Figure 1 (PRISMA flow).",
        S["Body"]))
    counts = PrismaCounts(
        identified        = n_retrieved,
        screened          = n_retrieved,
        screened_excluded = n_screened_excluded,
        assessed          = n_included,
        assessed_excluded = 0,
        included          = n_included,
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(KeepTogether(build_prisma_diagram(counts)))
    story.append(Paragraph(
        "<b>Fig. 1</b> Flowchart of study-selection process (PRISMA-2020).",
        S["FigureCaption"]))

    # 4.2 Characteristics of identified studies — full-page table.
    # Only switch to the full-width template AND issue a page break when we
    # actually have a table to render. Empty/missing tables stay inline in
    # the two-column flow so we don't leave blank pages in the PDF.
    char_tbl = (characteristics_table(csv_chars, set(pmids_relevant),
                                     max_rows=120,
                                     page_width_cm=doc_pagewidth_cm)
               if n_included > 0 else None)

    if char_tbl is not None:
        story.append(NextPageTemplate("tables_full"))
        story.append(PageBreak())
        story.append(Paragraph("Characteristics of the identified studies",
                              S["H2"]))
        story.append(Paragraph(
            f"Information was retrieved from the {n_included} studies "
            f"included in this synthesis. The main characteristics of "
            f"the eligible studies are summarised in Table 1.",
            S["Body"]))
        story.append(Paragraph("<b>Table 1</b> Main characteristics of "
                              "the eligible studies", S["TableCaption"]))
        story.append(char_tbl)
        # Switch back to two-column body for the rest of the prose
        story.append(NextPageTemplate("body"))
        story.append(PageBreak())
    else:
        # No table → inline note, stays in the two-column flow
        story.append(Paragraph("Characteristics of the identified studies",
                              S["H2"]))
        if n_included == 0:
            story.append(Paragraph(
                "No studies met the inclusion criteria; no characteristics "
                "table is presented.", S["Body"]))
        else:
            story.append(Paragraph(
                "<i>Characteristics extraction CSV unavailable for this run.</i>",
                S["Body"]))

    # 4.3 Outcomes — same conditional-page-break pattern
    out_tbl = (outcomes_table(csv_outcomes, set(pmids_relevant),
                             max_rows=160,
                             page_width_cm=doc_pagewidth_cm)
              if n_included > 0 else None)

    if out_tbl is not None:
        story.append(NextPageTemplate("tables_full"))
        story.append(PageBreak())
        story.append(Paragraph("Outcomes extracted from included studies",
                              S["H2"]))
        story.append(Paragraph(
            "Outcome measures extracted from each included study are "
            "presented in Table 2. The pipeline's outcomes CSV stores a "
            "JSON array of outcomes per study; the table below explodes "
            "that array so each row represents one (study × outcome) pair.",
            S["Body"]))
        story.append(Paragraph("<b>Table 2</b> Per-study outcome "
                              "measures extracted by the pipeline",
                              S["TableCaption"]))
        story.append(out_tbl)
        # Back to two-column for the synthesis prose
        story.append(NextPageTemplate("body"))
        story.append(PageBreak())
    else:
        story.append(Paragraph("Outcomes extracted from included studies",
                              S["H2"]))
        if n_included == 0:
            story.append(Paragraph(
                "No outcome data are available because no studies were included.",
                S["Body"]))
        else:
            story.append(Paragraph(
                "<i>Outcomes extraction CSV unavailable for this run.</i>",
                S["Body"]))

    # 4.4 Synthesis prose
    story.append(Paragraph("Synthesis of findings", S["H2"]))
    if evidence_markdown:
        for para in _split_paragraphs(evidence_markdown):
            cleaned = citations.linkify(_safe(para))
            story.append(Paragraph(cleaned, S["Body"]))
    else:
        story.append(Paragraph(
            "<i>Synthesis narrative unavailable for this run.</i>",
            S["Body"]))


def _build_discussion(story: list, S: dict):
    """Discussion section with subsections matching journal conventions."""
    story.append(Paragraph("Discussion", S["H1"]))

    story.append(Paragraph("Principal findings and implications", S["H2"]))
    story.append(Paragraph(
        "The synthesis presented in the Results section integrates the "
        "available evidence on the research question and characterises "
        "the consistency, magnitude, and direction of the reported "
        "effects. These findings should be interpreted in the context "
        "of the methodological considerations described below.",
        S["Body"]))

    story.append(Paragraph("Strengths and comparison with other studies", S["H2"]))
    story.append(Paragraph(
        "The pipeline applies a reproducible, transparent search strategy "
        "with a fully audit-able query string (see the Appendix). Every "
        "screening decision and extraction artefact is preserved as a CSV "
        "record, supporting independent re-analysis. Retrieval-augmented "
        "MeSH expansion grounds term selection in actual reference-paper "
        "vocabulary rather than language-model training data alone, which "
        "reduces the risk of hallucinated terms that would inflate or "
        "deflate retrieval recall.",
        S["Body"]))

    story.append(Paragraph("Limitations", S["H2"]))
    story.append(Paragraph(
        "Several limitations of this automated review should be noted. "
        "First, retrieval was limited to PubMed/MEDLINE; studies indexed "
        "only in Embase, the Cochrane Central Register of Controlled "
        "Trials, or grey-literature sources are not represented. Second, "
        "screening and extraction relied on language-model judgements, "
        "which may misclassify edge cases. Third, the cumulative relevance "
        "threshold is a configurable cut-off that may exclude some "
        "borderline-eligible studies. Fourth, the synthesis is qualitative; "
        "no formal meta-analysis, risk-of-bias assessment, or assessment "
        "of certainty of evidence (e.g., GRADE) was performed. Fifth, "
        "automated extraction may miss subtle methodological nuances of "
        "individual studies.",
        S["Body"]))


def _build_conclusions(story: list, S: dict, evidence_summary: str,
                      citations: CitationRegistry):
    """Conclusions: short, single-paragraph summary of the synthesis."""
    story.append(Paragraph("Conclusions", S["H1"]))
    if evidence_summary:
        cleaned = citations.linkify(_safe(evidence_summary))
        story.append(Paragraph(cleaned, S["Body"]))
    else:
        story.append(Paragraph(
            "Conclusions could not be drawn from the available evidence "
            "in this automated synthesis. Formal review by domain experts "
            "is recommended before this work informs clinical practice.",
            S["Body"]))


def _build_abbreviations(story: list, S: dict, pico: dict):
    """Abbreviations block — pulls common acronyms from PICO text."""
    story.append(Paragraph("Abbreviations", S["EndMatterHeading"]))
    items = [
        "CIs: Confidence intervals",
        "GRADE: Grading of Recommendations Assessment, Development and Evaluation",
        "HRs: Hazard ratios",
        "LLM: Large language model",
        "MeSH: Medical Subject Headings",
        "PICO: Population, Intervention, Comparator, Outcomes",
        "PMID: PubMed identifier",
        "PRISMA: Preferred Reporting Items for Systematic Reviews and Meta-analyses",
    ]
    story.append(Paragraph("; ".join(items) + ".", S["EndMatter"]))


def _build_endmatter(story: list, S: dict, run_id: str):
    """Standard journal end-matter blocks."""
    blocks = [
        ("Acknowledgments",
         "We thank the developers of the open-source libraries that "
         "underpin this pipeline (PubMed E-utilities, ReportLab, the "
         "Vertex AI SDK) and the maintainers of TrialReviewBench."),
        ("Funding",
         "No external funding was used to produce this automated synthesis."),
        ("Availability of data and materials",
         "All extraction artefacts (PubMed query, retrieved records, "
         "screening decisions, study characteristics, outcome data, and "
         "the synthesis narrative) are preserved on the host filesystem "
         f"under run identifier {_safe(run_id)} and are available for "
         "verification or re-analysis."),
        ("Authors' contributions",
         "Synthesis generated by the REASON automated pipeline; reviewed "
         "by the user of the dashboard. Final responsibility for the "
         "content lies with the human reviewer."),
        ("Ethics approval and consent to participate",
         "Not applicable. This review is a secondary analysis of "
         "previously published reports."),
        ("Consent for publication", "Not applicable."),
        ("Competing interests",
         "The authors of this automated report declare no competing "
         "interests."),
    ]
    for label, body in blocks:
        story.append(Paragraph(label, S["EndMatterHeading"]))
        story.append(Paragraph(body, S["EndMatter"]))


def _build_references(story: list, S: dict, articles: dict,
                     citations: CitationRegistry,
                     pmids_relevant: list[str]):
    """References list — Vancouver-ish, numbered.

    Order = inline citation order as discovered in the synthesis narrative.
    Any included PMIDs not cited in the narrative are appended at the end
    so the reference list still covers the full inclusion set.
    """
    story.append(Paragraph("References", S["EndMatterHeading"]))

    used_order = citations.ordered_pmids()
    extras = [p for p in (pmids_relevant or []) if p not in set(used_order)]
    full_order = used_order + extras

    if not full_order:
        story.append(Paragraph("No included studies.", S["Body"]))
        return

    for i, pmid in enumerate(full_order, start=1):
        art = articles.get(pmid, {}) if isinstance(articles, dict) else {}
        title = _safe(art.get("title", "(title unavailable)")).rstrip(". ")
        ref = (f"{i}. {title}. PubMed PMID: "
               f'<link href="https://pubmed.ncbi.nlm.nih.gov/{pmid}" '
               f'color="#2B6CB0">{pmid}</link>.')
        story.append(Paragraph(ref, S["Reference"]))


def _build_appendix(story: list, S: dict, mesh_query: str, run_id: str):
    """Appendix: the verbatim PubMed query string."""
    story.append(Paragraph("Appendix A. Search strategy", S["EndMatterHeading"]))
    story.append(Paragraph(
        "The exact PubMed query string executed against the eutils "
        "esearch endpoint:", S["Body"]))
    story.append(Paragraph(_safe(mesh_query or "(query not available)"),
                          S["Code"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Run ID: {_safe(run_id)} &middot; Generated: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        S["MetaLine"]))


# ════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════

def build_systematic_review_pdf(
    pico_json: dict,
    question: str,
    mesh_query: str,
    articles: dict,
    eligibility: dict,
    csv_screened: str | None,
    pmids_relevant: list[str] | None,
    csv_chars: str | None,
    csv_outcomes: str | None,
    evidence_summary: str,
    evidence_markdown: str,
    output_path: str | Path | None = None,
    score_threshold: float = 3.0,
    journal_name: str = "REASON Synthesis",
) -> str:
    """Build a journal-style systematic review PDF from pipeline outputs.

    Args:
        pico_json:         Full PICO trace dict (must contain `pico_valid` and `qid`)
        question:          Original free-text research question
        mesh_query:        PubMed query string actually executed
        articles:          {pmid: {title, abstract, ...}} from Stage 3
        eligibility:       Dict with `inclusion_criteria` / `exclusion_criteria`
        csv_screened:      Path to screening CSV (Stage 5 output)
        pmids_relevant:    List of PMIDs that passed screening (auto-derived
                           from `csv_screened` if not provided)
        csv_chars:         Path to study-characteristics CSV (Stage 6 output)
        csv_outcomes:      Path to outcomes CSV (Stage 7 output)
        evidence_summary:  Short summary string from synthesis stage
        evidence_markdown: Full multi-paragraph synthesis narrative
        output_path:       Where to write the PDF (default:
                           artifacts_day7/sr_<qid>_<timestamp>.pdf)
        score_threshold:   Score cutoff used during screening (informational)
        journal_name:      Label used in the page footer (default: "REASON Synthesis")

    Returns:
        Absolute path (str) to the generated PDF.
    """
    pico_valid = (pico_json or {}).get("pico_valid", {}) or {}
    qid        = (pico_json or {}).get("qid", "unknown")

    if not pmids_relevant and csv_screened:
        pmids_relevant = _read_pmids_from_csv(csv_screened, threshold=score_threshold)
    pmids_relevant = pmids_relevant or []

    n_retrieved = len(articles) if isinstance(articles, dict) else 0
    n_included  = len(pmids_relevant)

    if output_path:
        out = Path(output_path)
    else:
        artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts_day7"
        artifacts_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = artifacts_dir / f"sr_{qid}_{ts}.pdf"

    title = _generate_title(pico_valid)
    pub_year = datetime.now().year
    doc = _make_doc(out, title, journal_name, pub_year, qid)
    S = build_styles()

    # Track inline citations across the narrative + summary
    citations = CitationRegistry()

    # The doc's drawable width in cm — used to size full-width tables
    doc_pagewidth_cm = doc.width / cm

    # ── Compose story ────────────────────────────────────────────────────────
    story: list = []

    # Page 1 — single column
    _build_page1(
        story, S,
        title=title, pico=pico_valid,
        n_retrieved=n_retrieved, n_included=n_included,
        evidence_summary=evidence_summary,
        evidence_markdown=evidence_markdown,
        question=question, run_id=qid,
        journal=journal_name, pub_year=pub_year,
    )

    # After page 1 → switch to two-column body template
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    _build_background(story, S, pico_valid, question, citations)
    _build_methods(story, S, pico_valid, mesh_query, eligibility,
                  score_threshold, n_retrieved)
    _build_results(story, S, pmids_relevant, n_retrieved,
                  csv_chars, csv_outcomes, evidence_markdown,
                  citations, doc_pagewidth_cm)
    _build_discussion(story, S)
    _build_conclusions(story, S, evidence_summary, citations)

    # End matter — switch to single-column for cleaner abbreviations / refs
    story.append(NextPageTemplate("body"))     # keep two-column for refs
    story.append(Spacer(1, 0.4 * cm))

    _build_abbreviations(story, S, pico_valid)
    _build_endmatter(story, S, qid)
    _build_references(story, S, articles, citations, pmids_relevant)
    _build_appendix(story, S, mesh_query, qid)

    doc.build(story)
    return str(out)
