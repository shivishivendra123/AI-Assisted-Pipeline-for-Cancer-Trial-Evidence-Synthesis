"""
REASON — Evidence Synthesis Dashboard
Redesigned UI: left control panel + tabbed results, paginated tables,
download-first output design. Core pipeline logic preserved from app_modern.py.
"""

import csv
import json
import traceback
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd

from pipeline.pico import pico_pipe
from meshOnDemand.mesh_expand import generate_expand_mesh_query
from efetch_utility.efetch import fetch_pubmed_articles
from eligibility_builder.built_eligibility import build_eligibility
from screening.screening import screen_studies
from extraction.study_char_outcome_ext import extract_study_char_outcomes
from extraction.outcome_extract import extract_study_outcomes
from basic_sythesis.synthesis import run_evidence_synthesis
from report_generator import build_systematic_review_pdf
from configs.env_config import config

# ============================================================
# Pipeline architecture (2026 simplification)
#
# 8-stage flow, single retrieval strategy:
#   1. PICO Extraction       — LLM extracts P/I/C/O from the user's question
#   2. MeSH Discovery        — Retrieval-augmented term expansion (always
#                              active; no UI-selectable strategies). A
#                              primary-term LLM call drives a narrow probe,
#                              7 reference papers ground a second LLM call
#                              that emits 3-axis fat-OR query terms.
#   3. PubMed Retrieval      — Single esearch with sort=relevance,
#                              retmax=100. The retrieved pool feeds
#                              screening directly — no reranker, no
#                              filter stage.
#   4. Eligibility Criteria  — LLM builds inclusion/exclusion rules
#   5. Screening             — LLM rates each of the 100 papers
#   6. Study Characteristics — LLM extracts study-level fields
#   7. Outcome Extraction    — LLM extracts outcome measures
#   8. Evidence Synthesis    — LLM produces narrative summary
#
# Removed in this simplification:
#   - Mesh-mode dropdown (basic / advanced / api / expand options)
#   - BM25 + iCite reranking stage
#   - Multi-sort retrieval (relevance + newest + oldest union)
#   - Multi-sort and rerank were contributing no papers to the
#     screening pool past rank 100 anyway, so removing them simplifies
#     the architecture without changing what reaches screening.
# ============================================================


# ============================================================
# HELPERS
# ============================================================

def get_valid_studies(csv_path, threshold: float = 3.0) -> list:
    """Return PMIDs from a screening CSV that meet the score threshold."""
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)
    pmids = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if float(row.get("cumulative_score", "")) >= threshold:
                    pmid = row.get("study_id")
                    if pmid:
                        pmids.append(pmid)
            except ValueError:
                continue
    return pmids


def json_pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def stage_label(stage: int, total: int, name: str, status: str = "running") -> str:
    icons = {"running": "🔄", "complete": "✅", "pending": "⏳", "error": "❌"}
    return f"{icons.get(status, '•')} Stage {stage}/{total}: {name}"


def flatten_articles(articles: dict) -> pd.DataFrame:
    """
    Convert the articles dict into a flat DataFrame with readable scalar columns.
    Handles nested lists/dicts so every article becomes exactly one row.
    """
    rows = []
    for pmid, art in articles.items():
        if not isinstance(art, dict):
            art = {"raw": str(art)}

        def _scalar(v):
            if isinstance(v, list):
                return "; ".join(str(i) for i in v)
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return v

        row = {"pmid": pmid}
        for k, v in art.items():
            row[k] = _scalar(v)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Truncate abstract for display (full text is in the CSV)
    if "abstract" in df.columns:
        df["abstract"] = df["abstract"].astype(str).str[:200] + "…"

    # Put most useful columns first
    priority = ["pmid", "title", "authors", "year", "journal", "abstract"]
    ordered = [c for c in priority if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def save_articles_csv(articles: dict, artifacts_dir: str) -> str:
    """Save full (un-truncated) articles to CSV and return the file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(artifacts_dir) / f"search_results_{ts}.csv"
    try:
        flatten_articles(articles).to_csv(path, index=False)
        return str(path)
    except Exception:
        return None


# ============================================================
# PIPELINE
# Returns 20 values for the dashboard UI.
#
# Return order:
#   pipeline_status, pico_text, mesh_text,
#   articles_status, articles_df, articles_file,
#   elig_text,
#   screening_status, screening_file, screening_df, pmids_text,
#   chars_status, chars_file, chars_df,
#   outcomes_status, outcomes_file, outcomes_df,
#   evidence_summary, evidence_markdown,
#   logs_text
# ============================================================

def run_pipeline_dashboard(
    question: str,
    score_threshold: float = None,
    max_workers: int = None,
    fetch_fulltext: bool = False,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """End-to-end evidence-synthesis pipeline runner.

    Drives all 8 stages in sequence and returns the 20 outputs the Gradio
    dashboard expects (status messages, per-stage data frames, file paths,
    a final synthesis narrative, and accumulated logs).

    Args:
        question:        Free-text clinical question (typed by the user).
        score_threshold: Minimum screening score for a study to be
                         considered relevant (default: config-driven).
        max_workers:     Number of concurrent screening threads.
        fetch_fulltext:  If True, run elink → PMC fetch for each retrieved
                         PMID and feed full-text XML into screening + extraction.
                         Adds ~30-90s per case but enables richer downstream
                         analysis. If False (default), screening + extraction
                         operate on title + abstract only — faster, cheaper,
                         still adequate for most use cases.
        progress:        Gradio progress reporter.

    Returns:
        Tuple of 20 values feeding the dashboard's right panel.
    """
    score_threshold = score_threshold or config.DEFAULT_SCORE_THRESHOLD
    max_workers     = max_workers     or config.DEFAULT_MAX_WORKERS

    logs: list = []

    def log(msg: str):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(entry)
        logs.append(entry)

    def all_logs() -> str:
        return "\n".join(logs)

    # Artifacts directories — each stage writes to a stage-appropriate folder.
    # Note the split: PubMed search-results CSV lives in the augmented/retrieval
    # bucket (artifacts_day2), while the screening + downstream stage outputs
    # live in artifacts_day5. Keeping them separated means a single artifact
    # browser can show retrieval-vs-screening side by side without mixing.
    artifacts_dir          = config.ARTIFACTS_SCREENING_DIR  # day5 — screening + extraction
    search_artifacts_dir   = config.ARTIFACTS_AUGMENTED_DIR  # day2 — search results CSV
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(search_artifacts_dir).mkdir(parents=True, exist_ok=True)

    _empty_df = pd.DataFrame()

    # Pipeline depth knobs (fixed, not exposed in the UI):
    #   Stage 3 retrieves the top RETRIEVE_TOP_K papers from PubMed by
    #   sort=relevance (Best Match). All retrieved papers feed screening
    #   directly — no separate filter or rerank stage.
    TOTAL_STAGES        = 8
    RETRIEVE_TOP_K      = 100
    max_studies         = RETRIEVE_TOP_K   # screen the full retrieved pool

    try:
        # ── Stage 1: PICO Extraction ─────────────────────────────────────
        progress(0.05, desc=stage_label(1, TOTAL_STAGES, "PICO Extraction"))
        log(f"Stage 1/{TOTAL_STAGES}: PICO Extraction…")

        pico_json  = pico_pipe(question)
        pico_valid = pico_json.get("pico_valid", {})
        pico_text  = "\n".join(f"**{k}:** {v}" for k, v in pico_valid.items())
        log(f"✅ Stage 1/{TOTAL_STAGES} complete")

        # ── Stage 2: MeSH Discovery (retrieval-augmented term extraction) ──
        # Single fixed strategy:
        #   1. LLM call A extracts 3-4 narrow primary terms from PICO
        #   2. Those terms drive a PubMed probe that fetches 7 reference papers
        #   3. LLM call B sees PICO + those 7 papers and emits CORE+EXPAND
        #      term lists across (conditions, treatments, outcomes) axes
        #   4. Final query is fat-OR within each axis, AND across axes,
        #      with no `[MeSH Terms]` lock-in (PubMed auto-translates terms)
        progress(0.18, desc=stage_label(2, TOTAL_STAGES, "MeSH Discovery"))
        log(f"Stage 2/{TOTAL_STAGES}: MeSH Discovery (retrieval-augmented expansion)…")

        # exclude_reviews=True appends a Cochrane-style publication-type
        # filter to the PubMed query that drops reviews and meta-analyses.
        # Without this, PubMed Best Match's top-100 is dominated by review-
        # class papers (which the screening agent rejects), starving the
        # screening pool of original trials. Filter restores screening
        # yield (the rerank stage previously did this implicitly via RCR).
        mesh_query = generate_expand_mesh_query(pico_json, exclude_reviews=True)
        mesh_text  = mesh_query if isinstance(mesh_query, str) else json_pretty(mesh_query)
        log(f"✅ Stage 2/{TOTAL_STAGES} complete")

        # ── Stage 3: PubMed Retrieval (top-100 by relevance) ──────────────
        # Single esearch with sort=relevance, retmax=100. The retrieved
        # pool feeds screening directly — no rerank, no filter stage.
        progress(0.35, desc=stage_label(3, TOTAL_STAGES, "PubMed Retrieval"))
        ft_label = "with PMC full-text" if fetch_fulltext else "abstracts only"
        log(f"Stage 3/{TOTAL_STAGES}: Fetching top-{RETRIEVE_TOP_K} PubMed "
            f"articles ({ft_label}) by relevance…")

        data       = fetch_pubmed_articles(
            pico_json["qid"], mesh_query,
            max_results=RETRIEVE_TOP_K,
            sort="relevance",
            fetch_fulltext=fetch_fulltext,
        )
        articles   = data["articles"]
        full_text  = data["full_text"]
        n_articles = len(articles)

        try:
            articles_df = flatten_articles(articles).head(10)
        except Exception:
            articles_df = _empty_df

        articles_file   = save_articles_csv(articles, search_artifacts_dir)
        articles_status = (
            f"Retrieved **{n_articles}** articles from PubMed (top-{RETRIEVE_TOP_K} by relevance) — "
            f"showing first 10 rows · download CSV for full results"
        )
        log(f"✅ Stage 3/{TOTAL_STAGES} complete — {n_articles} articles")

        # ── Stage 5: Eligibility Criteria ─────────────────────────────────
        progress(0.45, desc=stage_label(4, TOTAL_STAGES, "Eligibility Criteria"))
        log(f"Stage 4/{TOTAL_STAGES}: Building eligibility criteria…")

        eligibility = build_eligibility(
            pico_json["question"],
            pico_json["pico_valid"],
            pico_json["qid"],
        )
        elig_text = json_pretty(eligibility)
        log(f"✅ Stage 4/{TOTAL_STAGES} complete")

        # ── Stage 5: Screening ────────────────────────────────────────────
        progress(0.58, desc=stage_label(5, TOTAL_STAGES, "Screening Studies"))
        log(f"Stage 5/{TOTAL_STAGES}: Screening ({max_workers} workers, max {max_studies})…")

        csv_screened   = screen_studies(
            articles, full_text, eligibility, pico_json,
            max_studies, score_threshold,
            ftmode=fetch_fulltext,    # use full text only if it was fetched
            max_workers=max_workers,
        )
        pmids_relevant = get_valid_studies(csv_screened, score_threshold)
        n_relevant     = len(pmids_relevant)

        try:
            screening_df = pd.read_csv(csv_screened).head(10)
        except Exception:
            screening_df = _empty_df

        screening_status = (
            f"Screened **{len(articles)}** filtered studies — **{n_relevant}** passed "
            f"(score ≥ {score_threshold}) · showing first 10 rows · download CSV for full results"
        )
        pmids_text = "\n".join(pmids_relevant) if pmids_relevant else "No studies met threshold"
        log(f"✅ Stage 5/{TOTAL_STAGES} complete — {n_relevant} relevant studies")

        # ── Stage 6: Study Characteristics ───────────────────────────────
        progress(0.72, desc=stage_label(6, TOTAL_STAGES, "Study Characteristics"))
        log(f"Stage 6/{TOTAL_STAGES}: Extracting characteristics ({n_relevant} studies)…")

        csv_chars    = extract_study_char_outcomes(articles, full_text, pico_json, pmids_relevant)
        chars_status = (
            f"Characteristics extracted for **{n_relevant}** studies — "
            f"showing first 10 rows · download CSV for full results"
        )

        try:
            chars_df = pd.read_csv(csv_chars).head(10) if csv_chars else _empty_df
        except Exception:
            chars_df = _empty_df

        log(f"✅ Stage 6/{TOTAL_STAGES} complete")

        # ── Stage 7: Outcome Extraction ───────────────────────────────────
        progress(0.84, desc=stage_label(7, TOTAL_STAGES, "Outcome Extraction"))
        log(f"Stage 7/{TOTAL_STAGES}: Extracting outcomes ({n_relevant} studies)…")

        csv_outcomes    = extract_study_outcomes(articles, full_text, pico_json, pmids_relevant)
        outcomes_status = (
            f"Outcomes extracted for **{n_relevant}** studies — "
            f"showing first 10 rows · download CSV for full results"
        )

        try:
            outcomes_df = pd.read_csv(csv_outcomes).head(10) if csv_outcomes else _empty_df
        except Exception:
            outcomes_df = _empty_df

        log(f"✅ Stage 7/{TOTAL_STAGES} complete")

        # ── Stage 8: Evidence Synthesis ───────────────────────────────────
        progress(0.95, desc=stage_label(8, TOTAL_STAGES, "Evidence Synthesis"))
        log(f"Stage 8/{TOTAL_STAGES}: Evidence synthesis…")

        evidence_markdown = ""
        evidence_summary  = ""
        try:
            es = run_evidence_synthesis(
                pico_json=pico_json,
                chars_csv_path=csv_chars,
                outcomes_csv_path=csv_outcomes,
            )
            if isinstance(es, str):
                try:
                    es = json.loads(es)
                except json.JSONDecodeError:
                    es = {}
            evidence_markdown = es.get("narrative_markdown", "")
            evidence_summary  = es.get("summary", {}).get("one_sentence", "")
        except Exception as e:
            log(f"❌ Synthesis error: {e}")
            evidence_markdown = f"Error during synthesis: {e}"

        log(f"✅ Stage 8/{TOTAL_STAGES} complete")

        # ── Collect state for on-demand PDF report generation ─────────────
        # PDF generation is no longer in the critical path of the pipeline.
        # We bundle everything build_systematic_review_pdf() needs into a
        # plain dict and emit it via a gr.State output. The Report tab has
        # a "Generate Report PDF" button that consumes this state when the
        # user is ready — keeps pipeline UI responsive and lets the user
        # iterate on the synthesis before committing to a final PDF.
        pdf_state = {
            "pico_json":         pico_json,
            "question":          question,
            "mesh_query":        mesh_query if isinstance(mesh_query, str) else mesh_text,
            "articles":          articles,
            "eligibility":       eligibility,
            "csv_screened":      str(csv_screened) if csv_screened else None,
            "pmids_relevant":    pmids_relevant,
            "csv_chars":         str(csv_chars)    if csv_chars    else None,
            "csv_outcomes":      str(csv_outcomes) if csv_outcomes else None,
            "evidence_summary":  evidence_summary,
            "evidence_markdown": evidence_markdown,
            "score_threshold":   score_threshold,
        }

        progress(1.0, desc=" Pipeline complete!")
        log("🎉 All stages complete!")

        pipeline_status = (
            f"**Pipeline complete** — {n_articles} articles retrieved, "
            f"{n_relevant} studies included in synthesis"
        )

        return (
            pipeline_status,
            pico_text,
            mesh_text,
            articles_status,
            articles_df,
            articles_file,
            elig_text,
            screening_status,
            str(csv_screened) if csv_screened else None,
            screening_df,
            pmids_text,
            chars_status,
            str(csv_chars)    if csv_chars    else None,
            chars_df,
            outcomes_status,
            str(csv_outcomes) if csv_outcomes else None,
            outcomes_df,
            evidence_summary,
            evidence_markdown,
            pdf_state,                # 20th output: gr.State holding PDF kwargs
            all_logs(),                # 21st output: full log tail
        )

    except Exception as e:
        log(f"❌ Pipeline error: {e}")
        log(traceback.format_exc())
        return (
            f"❌ **Error:** {e}",
            "", "", "", _empty_df, None,
            "", "", None, _empty_df, "",
            "", None, _empty_df,
            "", None, _empty_df,
            "", "",
            None,                     # pdf_state slot on error
            all_logs(),
        )


def generate_report_pdf(state: dict | None,
                       progress: gr.Progress = gr.Progress(track_tqdm=True)):
    """Button-handler for the 'Generate Report PDF' control on the Report tab.

    Reads the gr.State populated at pipeline completion, calls
    build_systematic_review_pdf, and returns:
        (status_markdown, pdf_file_value)

    The status string drives the markdown component above the download
    link; pdf_file_value is the path that Gradio's gr.File component
    surfaces as a download link in the UI.

    `progress` is included so Gradio renders its built-in spinner /
    progress bar on the button row while the PDF is being built —
    user feedback that the request is in flight.
    """
    if not state:
        return ("_⚠️ Run the pipeline first — there's no synthesis data to "
                "render into a PDF yet._", None)

    progress(0.10, desc="Preparing PDF report…")
    try:
        progress(0.30, desc="Building title page + abstract…")
        pdf_path = build_systematic_review_pdf(**state)
        progress(0.95, desc="Finalising download…")
        pdf_name = Path(pdf_path).name
        return (f"✅ **Report ready:** `{pdf_name}` — click below to download.",
                pdf_path)
    except Exception as e:
        import traceback
        print(f"[report_pdf] {traceback.format_exc()}")
        return (f"❌ **PDF generation failed:** `{type(e).__name__}: {e}`",
                None)


def clear_outputs():
    """Reset all output components to their blank state.

    Must return the same number of values as `run_pipeline_dashboard`:
    21 named outputs + the logs tail = 21 items total.
    """
    _e = pd.DataFrame()
    return (
        "_Waiting for pipeline run…_",
        "", "", "", _e, None,
        "", "", None, _e, "",
        "", None, _e,
        "", None, _e,
        "", "",
        None,        # pdf_file slot
        "",          # logs_text
    )


# ============================================================
# UI: LEFT CONTROL PANEL
# ============================================================

def build_left_panel() -> dict:
    c = {}

    gr.HTML('<p class="panel-section-label">Research Question</p>')
    c["question"] = gr.Textbox(
        placeholder=(
            "e.g. In adults with early-stage HER2-positive breast cancer, does "
            "adjuvant trastuzumab + chemotherapy vs chemotherapy alone reduce "
            "5-year disease-free survival events?"
        ),
        lines=5,
        show_label=False,
        elem_classes=["compact-text"],
    )

    gr.HTML('<p class="panel-section-label">Search & Screening</p>')
    # Note: raw retrieval is fixed at 2000 papers (PubMed) and screening
    # is the same 100 papers (no separate filter or rerank stage). Both are
    # fixed in the pipeline, not exposed in the UI.
    c["threshold"] = gr.Slider(
        label="Relevance Threshold (score ≥)",
        minimum=0, maximum=15, step=0.5,
        value=config.DEFAULT_SCORE_THRESHOLD,
    )
    c["max_workers"] = gr.Slider(
        label="Parallel Workers",
        minimum=1, maximum=20, step=1,
        value=config.DEFAULT_MAX_WORKERS,
    )
    # Toggle to enable PMC full-text fetch + downstream full-text use.
    # OFF by default — keeps the pipeline fast (skip elink+efetch on PMC,
    # ~30-90s saved per run) at the cost of richer downstream evidence.
    # When ON, screening + extraction + synthesis can use the article body
    # text, not just title + abstract.
    c["fetch_fulltext"] = gr.Checkbox(
        label="Fetch PMC full-text (slower but richer extraction)",
        value=False,
    )

    # MeSH Generation strategy is fixed (retrieval-augmented expansion);
    # no UI selection. The dropdown was removed in the 2026 simplification
    # because empirical testing showed the expand path consistently
    # outperformed the other strategies on TrialReviewBench.

    with gr.Row():
        c["run_btn"] = gr.Button(
            "▶  Run Pipeline",
            variant="primary",
            size="lg",
            elem_id="run-btn",
            scale=3,
        )
        c["clear_btn"] = gr.Button(
            "✕  Clear",
            variant="secondary",
            size="lg",
            elem_id="clear-btn",
            scale=1,
        )

    return c


# ============================================================
# UI: RIGHT RESULTS TABS
# ============================================================

def _dataframe(label: str) -> gr.Dataframe:
    """Shared settings for all result tables."""
    return gr.Dataframe(
        label=label,
        interactive=False,
        wrap=True,
        elem_classes=["result-table"],
    )


def build_results_tabs() -> dict:
    c = {}

    with gr.Tabs(elem_id="results-tabs"):

        # ── Overview / How to Use ─────────────────────────────────────────
        with gr.Tab("How to Use"):
            gr.Markdown(
                """
### Welcome to REASON

**REASON** is an 8-stage automated pipeline for systematic evidence synthesis from PubMed literature.

---

#### Quick Start

1. **Type your clinical research question** in the left panel.
   Use a structured PICO-style question for best results.
   > *Example: "In adults with type 2 diabetes, does SGLT-2 inhibitor therapy vs placebo reduce major adverse cardiovascular events?"*

2. **Adjust settings** as needed (all have sensible defaults):
   - **Relevance Threshold** — minimum score for a study to be included (0–15)
   - **Parallel Workers** — concurrent threads for screening; higher = faster
   - **Fetch PMC full-text** — when ON, the pipeline pulls full-text XML from PubMed Central (when available) and feeds it into screening + extraction. Adds ~30-90s per run; produces richer evidence. OFF by default (uses title + abstract only).

   *Retrieval depth (top 100 PubMed papers by relevance) and screening pool size (the same 100) are fixed in the pipeline. MeSH term generation uses a retrieval-augmented expansion strategy (LLM-driven, grounded in 7 reference papers).*

3. **Click ▶ Run Pipeline** and monitor the Status badge on the left.

4. **Navigate the tabs** (right panel) to explore each stage's output.

5. **Download results** using the download buttons in each tab.
                """
            )

        # ── PICO ──────────────────────────────────────────────────────────
        with gr.Tab("PICO"):
            gr.Markdown(
                "### Stage 1 — PICO Extraction\n"
                "_Population · Intervention · Comparator · Outcomes_"
            )
            c["pico"] = gr.Markdown(value="")

        # ── MeSH ──────────────────────────────────────────────────────────
        with gr.Tab("MeSH"):
            gr.Markdown(
                "### Stage 2 — MeSH Term Discovery\n"
                "_Medical Subject Headings generated for the PubMed query._"
            )
            c["mesh"] = gr.Textbox(
                lines=14, interactive=False, show_label=False,
                elem_classes=["compact-text", "mono-text"],
            )

        # ── Search ────────────────────────────────────────────────────────
        with gr.Tab("Search"):
            gr.Markdown("### Stage 3 — PubMed Article Retrieval")
            c["articles_status"] = gr.Markdown(value="")
            with gr.Row():
                c["articles_file"] = gr.File(
                    label="Download Search Results (CSV)",
                    interactive=False,
                    elem_classes=["dl-file"],
                )
            c["articles_df"] = _dataframe("Retrieved Articles")

        # ── Eligibility ───────────────────────────────────────────────────
        with gr.Tab("Eligibility"):
            gr.Markdown(
                "### Stage 4 — Eligibility Criteria\n"
                "_Inclusion and exclusion criteria derived from PICO._"
            )
            c["eligibility"] = gr.Textbox(
                lines=18, interactive=False, show_label=False,
                elem_classes=["compact-text", "mono-text"],
            )

        # ── Screening ─────────────────────────────────────────────────────
        with gr.Tab("Screening"):
            gr.Markdown("### Stage 5 — Study Screening")
            c["screening_status"] = gr.Markdown(value="")
            with gr.Accordion("Included PMIDs", open=False):
                c["pmids"] = gr.Textbox(
                    lines=4, interactive=False, show_label=False,
                    elem_classes=["compact-text", "mono-text"],
                )
            with gr.Row():
                c["screening_file"] = gr.File(
                    label="Download Screening Results (CSV)",
                    interactive=False,
                    elem_classes=["dl-file"],
                )
            c["screening_df"] = _dataframe("Screening Decisions")

        # ── Characteristics ───────────────────────────────────────────────
        with gr.Tab("Characteristics"):
            gr.Markdown(
                "### Stage 6 — Study Characteristics  _(Table 1)_\n"
                "_Study design, population, intervention details._"
            )
            c["chars_status"] = gr.Markdown(value="")
            with gr.Row():
                c["chars_file"] = gr.File(
                    label="Download Table 1 (CSV)",
                    interactive=False,
                    elem_classes=["dl-file"],
                )
            c["chars_df"] = _dataframe("Study Characteristics")

        # ── Outcomes ──────────────────────────────────────────────────────
        with gr.Tab("Outcomes"):
            gr.Markdown(
                "### Stage 7 — Study Outcomes  _(Table 2)_\n"
                "_Measured outcomes and quantitative results per study._"
            )
            c["outcomes_status"] = gr.Markdown(value="")
            with gr.Row():
                c["outcomes_file"] = gr.File(
                    label="Download Table 2 (CSV)",
                    interactive=False,
                    elem_classes=["dl-file"],
                )
            c["outcomes_df"] = _dataframe("Study Outcomes")

        # ── Synthesis ─────────────────────────────────────────────────────
        with gr.Tab("Synthesis"):
            gr.Markdown("### Stage 8 — Evidence Synthesis")
            gr.Markdown("**One-sentence summary**")
            c["evidence_summary"] = gr.Textbox(
                lines=2, interactive=False, show_label=False,
                elem_classes=["compact-text"],
            )
            gr.Markdown("**Narrative synthesis**")
            c["evidence_markdown"] = gr.Markdown(
                value="",
                elem_id="synthesis-output",
            )

        # ── Final Report (PDF) — on-demand generation ──────────────────────
        # Pipeline state is captured into a hidden gr.State at the end of
        # run_pipeline_dashboard. The user clicks "Generate Report PDF"
        # when ready; the button handler reads that state and renders a
        # journal-quality systematic-review PDF (title page, structured
        # abstract, methods, PRISMA flow, characteristics + outcomes
        # tables, synthesis narrative, references). Generation is
        # decoupled from the main pipeline so the UI stays responsive
        # and the user can iterate on the synthesis before committing
        # to a final document.
        with gr.Tab("Report"):
            gr.Markdown("### Systematic Review PDF")
            gr.Markdown(
                "Once the pipeline finishes, click **Generate Report PDF** "
                "below to assemble a journal-style systematic-review document "
                "from every stage's output (title page, structured abstract, "
                "methods, PRISMA flow diagram, characteristics & outcomes "
                "tables, synthesis narrative, references). Generation takes "
                "~5–10 seconds and runs on demand — re-run after any change "
                "to regenerate."
            )
            c["report_btn"] = gr.Button(
                "📄  Generate Report PDF",
                variant="primary",
                size="lg",
            )
            c["pdf_status"] = gr.Markdown(
                "_Run the pipeline first, then click the button above._"
            )
            c["pdf_file"] = gr.File(
                label="Download Systematic Review PDF",
                interactive=False,
            )
            # Hidden state that holds the kwargs for build_systematic_review_pdf.
            # Populated by the pipeline's final return value; consumed by the
            # report-button click handler.
            c["pdf_state"] = gr.State(value=None)

        # ── Logs ──────────────────────────────────────────────────────────
        with gr.Tab("Logs"):
            gr.Markdown("### Execution Log")
            c["logs"] = gr.Textbox(
                lines=22, max_lines=40, interactive=False,
                show_label=False, elem_id="logs-box",
            )

    return c


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
/* ── Force light mode ───────────────────────────────────── */
:root,
:root[data-theme="dark"],
:root[data-theme="system"] {
    color-scheme: light !important;
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #e2e8f0 !important;
    --color-accent: #16a34a !important;
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #64748b !important;
    --block-title-text-color: #1e293b !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --input-placeholder-color: #94a3b8 !important;
    --shadow-drop: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* ── Base ───────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 100% !important;
    background: #ffffff !important;
    color: #1e293b !important;
}

/* ── Kill ALL dark/black Gradio containers ──────────────── */
.block,
.form,
.gradio-group,
.gradio-accordion,
.gradio-accordion > .label-wrap,
.gradio-accordion > div,
div[data-testid="block"],
div.svelte-1gfkn6j,
fieldset,
.gap,
.wrap,
.panel,
.tabs,
.tabitem,
.tab-nav,
.tab-content,
.prose,
label.block,
.block.padded,
.block.border-focus,
.contain,
.stretch {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #e2e8f0 !important;
    color: #1e293b !important;
    box-shadow: none !important;
}

/* Accordion header specifically */
.gradio-accordion > .label-wrap button,
details > summary {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
}

/* Group borders */
.gradio-group {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* Textareas and inputs — ensure white fill */
textarea,
input[type="text"],
input[type="number"],
select {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* File upload / download blocks */
.upload-container,
.file-preview,
[data-testid="file-upload"] {
    background: #f0fdf4 !important;
    border-color: #86efac !important;
    color: #1e293b !important;
}

/* Dataframe table wrapper */
.result-table,
.result-table > div,
.result-table table,
[data-testid="dataframe"],
[data-testid="dataframe"] > div {
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
}

/* ── App Header ─────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
    padding: 14px 22px;
    border-radius: 10px;
    margin-bottom: 14px;
}
#app-header h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #ffffff !important;
}
#app-header .subtitle {
    margin: 3px 0 0;
    font-size: 12px;
    color: #bbf7d0 !important;
}

/* ── Left Panel ─────────────────────────────────────────── */
#left-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px !important;
    color: #1e293b !important;
}

.panel-section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #94a3b8 !important;
    margin: 10px 0 4px !important;
}

/* ── Run Button — Green ─────────────────────────────────── */
#run-btn {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
#run-btn:hover {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(22, 163, 74, 0.40) !important;
}

/* ── Clear Button — Red ─────────────────────────────────── */
#clear-btn {
    background: #ffffff !important;
    border: 1.5px solid #ef4444 !important;
    color: #dc2626 !important;
    font-weight: 600 !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
#clear-btn:hover {
    background: #fef2f2 !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25) !important;
}

/* ── Pipeline Status ────────────────────────────────────── */
#pipeline-status {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    color: #166534 !important;
    min-height: 38px;
}

/* ── Tabs ───────────────────────────────────────────────── */
#results-tabs .tab-nav button {
    font-size: 12px !important;
    padding: 7px 12px !important;
    font-weight: 500 !important;
    color: #475569 !important;
}
#results-tabs .tab-nav button.selected {
    font-weight: 700 !important;
    color: #16a34a !important;
    border-bottom: 2px solid #16a34a !important;
}

/* ── Compact & mono text ────────────────────────────────── */
.compact-text textarea,
.compact-text input {
    font-size: 13px !important;
    line-height: 1.55 !important;
    color: #1e293b !important;
}
.mono-text textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 12px !important;
    color: #1e293b !important;
}

/* ── Execution log ──────────────────────────────────────── */
#logs-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 11.5px !important;
    background: #f8fafc !important;
    color: #334155 !important;
    line-height: 1.6 !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Evidence synthesis ─────────────────────────────────── */
#synthesis-output {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #1e293b !important;
}

/* ── Result tables — aggressive light-mode whitelist ────── */

/* Scrollable outer wrapper */
.result-table {
    overflow-x: auto !important;
    overflow-y: auto !important;
    max-height: 340px !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
.result-table::-webkit-scrollbar { height: 6px; width: 6px; }
.result-table::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 3px; }
.result-table::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
.result-table::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* All wrappers and containers */
.result-table > *,
.result-table > * > *,
[data-testid="dataframe"],
[data-testid="dataframe"] > *,
[data-testid="dataframe"] > * > *,
.table-wrap,
.table-wrap > *,
.cell-wrap,
.cell-wrap > *,
.svelte-scroll-container,
.svelte-scroll-container > * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
    box-shadow: none !important;
}

/* Table element itself */
.result-table table,
[data-testid="dataframe"] table {
    border-collapse: collapse !important;
    background: #ffffff !important;
    width: 100% !important;
    font-size: 12px !important;
}

/* Header cells */
.result-table th,
[data-testid="dataframe"] th,
.result-table thead td,
[data-testid="dataframe"] thead td {
    background: #f1f5f9 !important;
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 7px 10px !important;
    border-bottom: 2px solid #e2e8f0 !important;
    white-space: nowrap !important;
}

/* Body cells */
.result-table td,
[data-testid="dataframe"] td {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    padding: 6px 10px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    vertical-align: top !important;
    font-size: 12px !important;
}

/* Zebra stripe */
.result-table tr:nth-child(even) td,
[data-testid="dataframe"] tbody tr:nth-child(even) td {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
}

/* Row hover */
.result-table tr:hover td,
[data-testid="dataframe"] tbody tr:hover td {
    background: #f0fdf4 !important;
    background-color: #f0fdf4 !important;
}

/* Cells that are editable inputs inside dataframe */
.result-table td input,
.result-table td textarea,
[data-testid="dataframe"] td input,
[data-testid="dataframe"] td textarea {
    background: #ffffff !important;
    color: #1e293b !important;
    border: none !important;
    font-size: 12px !important;
}

/* ── Download file components — full whitelist ──────────── */

/* Every possible wrapper Gradio uses for gr.File */
.dl-file,
.dl-file > *,
.dl-file > * > *,
[data-testid="file"],
[data-testid="file"] > *,
[data-testid="file-upload"],
[data-testid="file-upload"] > *,
.file-preview,
.file-preview > *,
.upload-container,
.upload-container > *,
.download-link,
.file,
.file > *,
.generating {
    background:       #f0fdf4 !important;
    background-color: #f0fdf4 !important;
    color:            #15803d !important;
    border-color:     #86efac !important;
    box-shadow:       none    !important;
}

/* The filename text pill / span */
.dl-file .file-name,
.file-preview .file-name,
[data-testid="file"] .file-name,
.download-link span,
.file span {
    color:       #15803d !important;
    font-weight: 500     !important;
    background:  transparent !important;
}

/* The dashed border on the outer box */
.dl-file {
    border:        1.5px dashed #86efac !important;
    border-radius: 8px           !important;
}

/* Icon inside the file box if present */
.dl-file svg,
[data-testid="file"] svg,
.file-preview svg {
    color:  #16a34a !important;
    stroke: #16a34a !important;
    fill:   none    !important;
}

/* ── Slider accent ──────────────────────────────────────── */
input[type=range]::-webkit-slider-thumb {
    background: #16a34a !important;
}
input[type=range]::-webkit-slider-runnable-track {
    background: #bbf7d0 !important;
}

/* ── Progress bar — lift off the very bottom ────────────── */
/* Gradio renders the progress toast as position:fixed at the bottom.
   Nudge it up so the timer text and stage label are fully visible. */
.progress-bar-wrap,
.generating,
.progress-level,
[data-testid="progress-bar"],
.wrap.progress-bar {
    bottom: 60px !important;
}

/* The inner progress track */
.progress-bar-wrap .progress-bar,
.progress-level .progress-level-inner {
    background: #16a34a !important;
    border-radius: 4px !important;
}

/* Stage label / desc text above the bar */
.progress-bar-wrap .progress-text,
.progress-level .meta-text,
.progress-level .progress-text,
.eta-bar .meta-text {
    color:       #1e293b  !important;
    font-size:   13px     !important;
    font-weight: 500      !important;
    background:  #ffffff  !important;
    padding:     2px 6px  !important;
    border-radius: 4px    !important;
    margin-bottom: 4px    !important;
}

/* Timer / ETA — hide every possible selector across Gradio versions */
.meta-text,
.meta-text span,
.meta-text-center,
.progress-text,
.progress-text.meta-text-center,
[class*="meta-text"],
[class*="progress-text"],
.eta-bar,
.eta-bar *,
.progress-eta,
.progress-time,
.time,
.duration,
.countdown,
.progress-bar-wrap .time,
.progress-bar-wrap .eta,
.progress-bar-wrap .meta-text,
.progress-level .meta-text,
.progress-level .time,
span.meta-text,
div.meta-text,
p.meta-text {
    display:    none !important;
    visibility: hidden !important;
    opacity:    0 !important;
    width:      0 !important;
    height:     0 !important;
    overflow:   hidden !important;
}
"""


# ============================================================
# APP ASSEMBLY
# ============================================================

def _all_outputs(pipeline_status: gr.Markdown, right: dict) -> list:
    """Ordered list of output components matching pipeline return values.

    The order MUST match `run_pipeline_dashboard`'s return tuple
    position-for-position. Adding a new pipeline output requires updating
    BOTH this list and the return tuple in run_pipeline_dashboard.
    """
    return [
        pipeline_status,
        right["pico"],
        right["mesh"],
        right["articles_status"],
        right["articles_df"],
        right["articles_file"],
        right["eligibility"],
        right["screening_status"],
        right["screening_file"],
        right["screening_df"],
        right["pmids"],
        right["chars_status"],
        right["chars_file"],
        right["chars_df"],
        right["outcomes_status"],
        right["outcomes_file"],
        right["outcomes_df"],
        right["evidence_summary"],
        right["evidence_markdown"],
        right["pdf_state"],         # gr.State holding kwargs for the PDF builder
        right["logs"],
    ]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="REASON — Evidence Synthesis", css=CUSTOM_CSS) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            """
            <div id="app-header">
              <h1>REASON</h1>
              <span class="subtitle">
                Retrieval-Enhanced Evidence Assessment with Synthesis and Organized Narration
              </span>
            </div>
            """
        )

        # ── Main layout ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3, elem_id="left-panel", min_width=280):
                left = build_left_panel()
            with gr.Column(scale=7, min_width=500):
                pipeline_status = gr.Markdown(
                    value="_Waiting for pipeline run…_",
                    elem_id="pipeline-status",
                )
                right = build_results_tabs()

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML(
            """
            <div style="
                text-align:center; color:#94a3b8; font-size:11px;
                margin-top:10px; padding-top:8px; border-top:1px solid #e2e8f0;
            ">
                MeSH &nbsp;·&nbsp;
                <b>Basic</b>: simple OR &nbsp;|&nbsp;
                <b>Advanced</b>: PICO + synonyms AND/OR &nbsp;|&nbsp;
                <b>API</b>: meshb.nlm.nih.gov
                &nbsp;&nbsp;·&nbsp;&nbsp;
                All artifacts saved to <code>artifacts/</code>
            </div>
            """
        )

        # ── Wiring ────────────────────────────────────────────────────────
        outputs = _all_outputs(pipeline_status, right)

        left["run_btn"].click(
            fn=run_pipeline_dashboard,
            inputs=[
                left["question"],
                left["threshold"],
                left["max_workers"],
                left["fetch_fulltext"],
            ],
            outputs=outputs,
        )

        left["clear_btn"].click(
            fn=clear_outputs,
            inputs=[],
            outputs=outputs,
        )

        # Report-tab button — generates the SR PDF on demand from the
        # state populated at pipeline completion. Gradio's built-in
        # progress indicator drives the spinner on the status component
        # and the file output while the PDF is being built (~5-10s).
        right["report_btn"].click(
            fn=generate_report_pdf,
            inputs=[right["pdf_state"]],
            outputs=[right["pdf_status"], right["pdf_file"]],
            show_progress="full",
        )

    return demo


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("🚀 Starting REASON Dashboard…")
    print(f"📁 Artifacts: {config.ARTIFACTS_SCREENING_DIR}")

    try:
        theme = gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
        theme.set(
            body_background_fill="white",
            body_background_fill_dark="white",
            body_text_color="#1e293b",
            body_text_color_dark="#1e293b",
            body_text_color_subdued="#64748b",
            body_text_color_subdued_dark="#64748b",
            background_fill_primary="white",
            background_fill_primary_dark="white",
            background_fill_secondary="#f8fafc",
            background_fill_secondary_dark="#f8fafc",
            border_color_primary="#e2e8f0",
            border_color_primary_dark="#e2e8f0",
            input_background_fill="white",
            input_background_fill_dark="white",
            block_title_text_color="#1e293b",
            block_title_text_color_dark="#1e293b",
            block_label_text_color="#475569",
            block_label_text_color_dark="#475569",
        )
    except Exception:
        theme = None

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
    )
