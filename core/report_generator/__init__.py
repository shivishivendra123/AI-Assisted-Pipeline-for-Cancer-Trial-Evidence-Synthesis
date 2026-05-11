"""
report_generator: produce a journal-quality systematic-review PDF from
the REASON pipeline's stage outputs.

Public API:
    from report_generator import build_systematic_review_pdf

    pdf_path = build_systematic_review_pdf(
        pico_json=...,
        question=...,
        mesh_query=...,
        articles=...,
        eligibility=...,
        csv_screened=...,
        pmids_relevant=...,
        csv_chars=...,
        csv_outcomes=...,
        evidence_summary=...,
        evidence_markdown=...,
    )
"""

from .build_report import build_systematic_review_pdf

__all__ = ["build_systematic_review_pdf"]
