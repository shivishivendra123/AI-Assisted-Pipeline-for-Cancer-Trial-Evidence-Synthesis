"""
Recall Evaluation UI — two-stage:
  Stage 1: Fetch PubMed results → check if GT PMIDs are retrieved
  Stage 2: Screen all retrieved → compute true screening recall

Run from evsy/core/:
    python eval_ui.py
"""

import csv
import json
import traceback
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.pico import pico_pipe
from pipeline.prefilter import keyword_prefilter, prefilter_stats
from pipeline.reranker import bm25_rerank, rerank_stats
from pipeline.icite_reranker import icite_rerank, icite_rerank_stats
from pipeline.augment.term_expander import expand_terms_with_llm
from schemas.pico import PICO
from meshOnDemand.mesh_query import query_mesh_api
from meshOnDemand.mesh_generator_agent import generate_mesh_terms_with_agent
from meshOnDemand.mesh_basic import generate_basic_mesh_query
from efetch_utility.efetch import fetch_pubmed_articles
from eligibility_builder.built_eligibility import build_eligibility
from screening.screening import screen_studies
from configs.env_config import config

TESTCASE_PATH = Path(__file__).parent.parent.parent / "testcase.json"


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_testcase():
    with open(TESTCASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def gt_pmids_from_testcase(tc: dict) -> set:
    return {c["pmid"] for c in tc.get("Involved_Citations", []) if c.get("pmid")}

def gt_titles_from_testcase(tc: dict) -> dict:
    """PMID → title for ground-truth citations."""
    return {c["pmid"]: c.get("title", "") for c in tc.get("Involved_Citations", []) if c.get("pmid")}

def pico_from_testcase(tc: dict) -> dict:
    """
    Build a pico_json dict from the testcase's pre-extracted PICO (P/I/C/O keys),
    bypassing the LLM-based pico_pipe. Returns a dict in the same shape that
    generate_basic_mesh_query and downstream steps expect.
    """
    import hashlib
    pico = tc.get("PICO", {})
    population   = pico.get("P", "") or ""
    intervention = pico.get("I", "") or ""
    comparator   = pico.get("C", "") or ""
    outcomes_raw = pico.get("O", "") or ""
    # Outcomes may be a string in testcase; wrap in list for consistency
    outcomes = outcomes_raw if isinstance(outcomes_raw, list) else ([outcomes_raw] if outcomes_raw else [])

    qid = hashlib.md5(tc.get("Title", "").encode()).hexdigest()[:12]
    return {
        "qid": qid,
        "question": tc.get("Title", ""),
        "pico_valid": {
            "Population":   population,
            "Intervention": intervention,
            "Comparator":   comparator,
            "Outcomes":     outcomes,
        },
    }

def get_valid_studies(csv_path, threshold: float = 3.0) -> list:
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

def recall(pipeline: set, gt: set) -> float:
    return len(pipeline & gt) / len(gt) if gt else 0.0

def json_pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_recall_chart(r_ret: float, r_pre: float | None, r_scr: float | None,
                       ret_hits: int, pre_hits: int | None, scr_hits: int | None, total: int):
    stages, values, colors = [], [], []
    stages.append(f"Retrieval\n({ret_hits}/{total})")
    values.append(r_ret * 100)
    colors.append("#4C9BE8")
    if r_pre is not None:
        stages.append(f"Pre-filter\n({pre_hits}/{total})")
        values.append(r_pre * 100)
        colors.append("#F59E0B")
    if r_scr is not None:
        stages.append(f"Screening\n({scr_hits}/{total})")
        values.append(r_scr * 100)
        colors.append("#2ECC71")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stages, values, color=colors, width=0.45, zorder=3)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Pipeline Recall vs Ground-Truth PMIDs", fontsize=13, fontweight="bold")
    ax.axhline(100, color="#94a3b8", linestyle="--", linewidth=0.8, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def build_retrieval_table(retrieved: set, gt: set, gt_titles: dict, articles: dict) -> pd.DataFrame:
    """Table of ALL ground-truth PMIDs showing whether they were retrieved."""
    rows = []
    for pmid in sorted(gt):
        art = articles.get(pmid, {})
        rows.append({
            "PMID": pmid,
            "GT Title (expected)": gt_titles.get(pmid, "")[:80] + "…",
            "Retrieved": "✅" if pmid in retrieved else "❌",
            "Pipeline Title": (art.get("title", "") or "")[:80] + ("…" if art.get("title") else ""),
        })
    return pd.DataFrame(rows)


def build_screening_table(retrieved: set, screened: set, gt: set,
                           gt_titles: dict, score_map: dict) -> pd.DataFrame:
    """Full PMID table with retrieval + screening status."""
    rows = []
    for pmid in sorted(gt):
        score, dec = score_map.get(pmid, ("-", "-"))
        rows.append({
            "PMID": pmid,
            "GT Title": gt_titles.get(pmid, "")[:70] + "…",
            "Retrieved": "✅" if pmid in retrieved else "❌",
            "Screened (pass)": "✅" if pmid in screened else "❌",
            "Score": score,
            "Decision": dec,
        })
    # pipeline-only hits
    for pmid in sorted((retrieved | screened) - gt):
        score, dec = score_map.get(pmid, ("-", "-"))
        rows.append({
            "PMID": pmid,
            "GT Title": "— not in ground truth —",
            "Retrieved": "✅" if pmid in retrieved else "—",
            "Screened (pass)": "✅" if pmid in screened else "—",
            "Score": score,
            "Decision": dec,
        })
    return pd.DataFrame(rows)


# ─── Stage 1: Fetch ───────────────────────────────────────────────────────────

def stage1_fetch(max_retrieve: int, mesh_mode: str, progress=gr.Progress(track_tqdm=True)):
    tc        = load_testcase()
    query     = tc["Title"]
    gt        = gt_pmids_from_testcase(tc)
    gt_titles = gt_titles_from_testcase(tc)

    logs = []
    def log(msg):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(entry)
        logs.append(entry)

    try:
        progress(0.10, desc="Stage 1: PICO (from testcase)…")
        log("Stage 1: PICO (from testcase — skipping LLM extraction)…")
        pico_json = pico_from_testcase(tc)
        log(f"  PICO: {pico_json.get('pico_valid', {})}")

        progress(0.40, desc=f"Stage 2: MeSH Discovery ({mesh_mode})…")
        log(f"Stage 2: MeSH Discovery (mode={mesh_mode})…")
        if mesh_mode == "basic":
            mesh_query = generate_basic_mesh_query(pico_json)
        elif mesh_mode == "advanced":
            try:
                syn_prompt = Path("prompts/synonym_prompt.txt").read_text(encoding="utf-8")
                pico_obj   = PICO.model_validate(pico_json["pico_valid"])
                augmented  = expand_terms_with_llm(pico_obj, syn_prompt)
                pico_json["augmented"] = json.loads(augmented.model_dump_json())
            except Exception as e:
                log(f"  ⚠️  Augmentation failed ({e}) — continuing with basic PICO")
            mesh_query = generate_mesh_terms_with_agent(pico_json)
        else:
            mesh_query = query_mesh_api(pico_json)
        cmp_q_preview = pico_json.get("mesh_query_comparator", "")
        log(f"  MeSH query (intervention): {mesh_query}")
        if cmp_q_preview:
            log(f"  MeSH query (comparator):   {cmp_q_preview}")

        progress(0.70, desc=f"Stage 3: PubMed Fetch (max {max_retrieve})…")
        log(f"Stage 3: PubMed Fetch (max_retrieve={max_retrieve})…")
        data = fetch_pubmed_articles(pico_json["qid"], mesh_query, max_results=max_retrieve, fetch_fulltext=False)
        articles = data["articles"] if data else {}

        # Fetch comparator query separately and merge (avoids ranking conflict)
        cmp_query = pico_json.get("mesh_query_comparator", "")
        if cmp_query:
            data2 = fetch_pubmed_articles(pico_json["qid"], cmp_query, max_results=max_retrieve, fetch_fulltext=False)
            if data2 and data2.get("articles"):
                before = len(articles)
                # Intervention results take precedence on conflict
                merged = {**data2["articles"], **articles}
                articles = merged
                log(f"  Comparator query added {len(articles) - before} new PMIDs "
                    f"(total {len(articles)})")

        retrieved = set(articles.keys())
        log(f"  Retrieved {len(retrieved)} PMIDs")

        ret_hits = len(retrieved & gt)
        r_ret    = recall(retrieved, gt)
        log(f"  Retrieval recall: {r_ret:.1%} ({ret_hits}/{len(gt)})")
        progress(1.0, desc="Done!")
        log("✅ Fetch complete — ready to screen.")

    except Exception as e:
        log(f"❌ {e}\n{traceback.format_exc()}")
        empty_fig, _ = plt.subplots(); plt.close()
        return (
            "\n".join(logs), empty_fig,
            "", pd.DataFrame({"Error": [str(e)]}),
            "—", "—",
            gr.update(interactive=False),
            {},
        )

    fig   = build_recall_chart(r_ret, None, None, ret_hits, None, None, len(gt))
    table = build_retrieval_table(retrieved, gt, gt_titles, articles)

    state = {
        "pico_json":  pico_json,
        "articles":   articles,
        "full_text":  {},
        "retrieved":  list(retrieved),
        "gt":         list(gt),
        "gt_titles":  gt_titles,
    }

    display_query = mesh_query
    if pico_json.get("mesh_query_comparator"):
        display_query += (
            "\n\n--- Comparator query (fetched separately & merged) ---\n"
            + pico_json["mesh_query_comparator"]
        )

    return (
        "\n".join(logs),
        fig,
        display_query,
        table,
        f"{r_ret:.1%}  ({ret_hits}/{len(gt)})",
        f"{len(retrieved)} articles fetched",
        gr.update(interactive=True),
        state,
    )


# ─── Stage 2: Screen ──────────────────────────────────────────────────────────

def stage2_screen(score_threshold: float, max_workers: int, use_prefilter: bool,
                  use_icite: bool, icite_alpha: float,
                  top_k: int, state: dict, progress=gr.Progress(track_tqdm=True)):
    if not state or not state.get("articles"):
        return (
            "No fetched data — run Stage 1 first.",
            gr.update(), pd.DataFrame(), "—", "—", "—", "",
        )

    pico_json  = state["pico_json"]
    articles   = state["articles"]
    full_text  = state["full_text"]
    retrieved  = set(state["retrieved"])
    gt         = set(state["gt"])
    gt_titles  = state["gt_titles"]

    logs = []
    def log(msg):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(entry)
        logs.append(entry)

    try:
        # ── Optional keyword pre-filter ─────────────────────────────────────
        articles_to_screen = articles
        if use_prefilter:
            groups = pico_json.get("mesh_groups", {})
            if groups:
                progress(0.05, desc="Pre-filtering articles…")
                articles_to_screen = keyword_prefilter(articles, groups, require_groups=2)
                stats = prefilter_stats(articles, articles_to_screen, gt)
                log(f"  {stats}")
            else:
                log("  ⚠️  No mesh_groups in state — skipping pre-filter")

        # ── Reranking → keep top-K ───────────────────────────────────────────
        groups = pico_json.get("mesh_groups", {})
        bm25_query = " ".join(
            groups.get("population", []) +
            groups.get("intervention", []) +
            groups.get("comparator", [])
        ) or pico_json.get("question", "")

        if use_icite and articles_to_screen:
            progress(0.10, desc=f"iCite reranking → top {top_k if top_k > 0 else 'all'}…")
            log(f"  iCite reranking (α={icite_alpha:.2f})…")
            effective_k = top_k if top_k > 0 else len(articles_to_screen)
            articles_to_screen = icite_rerank(
                bm25_query, articles_to_screen,
                top_k=effective_k, alpha=icite_alpha,
            )
            stats = icite_rerank_stats(articles, articles_to_screen, gt)
            log(f"  {stats}")
        elif top_k > 0 and top_k < len(articles_to_screen):
            progress(0.10, desc=f"BM25 reranking → top {top_k}…")
            articles_to_screen = bm25_rerank(bm25_query, articles_to_screen, top_k=top_k)
            stats = rerank_stats(articles, articles_to_screen, gt)
            log(f"  {stats}")

        progress(0.15, desc="Building eligibility criteria…")
        log("Stage 4: Building eligibility criteria…")
        eligibility = build_eligibility(
            pico_json["question"], pico_json["pico_valid"], pico_json["qid"]
        )

        progress(0.35, desc=f"Screening {len(articles_to_screen)} articles…")
        log(f"Stage 5: Screening {len(articles_to_screen)} articles (workers={max_workers})…")
        csv_path = screen_studies(
            articles_to_screen, full_text, eligibility, pico_json,
            max_studies=len(articles),          # screen everything
            score_threshold=score_threshold,
            ftmode=False,
            max_workers=max_workers,
        )

        pmids_relevant = get_valid_studies(csv_path, threshold=score_threshold)
        screened       = set(pmids_relevant)

        # Build score map from CSV
        score_map = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pmid = row.get("study_id", "").strip()
                if pmid:
                    score_map[pmid] = (row.get("cumulative_score", "-"), row.get("decision", "-"))

        scr_hits = len(screened & gt)
        r_scr    = recall(screened, gt)
        ret_hits = len(retrieved & gt)
        r_ret    = recall(retrieved, gt)

        # Pre-filter recall (if used)
        pre_filtered = set(articles_to_screen.keys())
        pre_hits = len(pre_filtered & gt) if use_prefilter and articles_to_screen is not articles else None
        r_pre    = recall(pre_filtered, gt) if pre_hits is not None else None

        log(f"  Screening recall: {r_scr:.1%} ({scr_hits}/{len(gt)})")
        progress(1.0, desc="Done!")
        log("✅ Screening complete.")

    except Exception as e:
        log(f"❌ {e}\n{traceback.format_exc()}")
        return (
            "\n".join(logs), gr.update(),
            pd.DataFrame({"Error": [str(e)]}),
            "—", "—", "—", "",
        )

    fig   = build_recall_chart(r_ret, r_pre, r_scr, ret_hits, pre_hits, scr_hits, len(gt))
    table = build_screening_table(retrieved, screened, gt, gt_titles, score_map)

    pre_recall_str = (
        f"{r_pre:.1%}  ({pre_hits}/{len(gt)})  [{len(articles_to_screen)} papers"
        f" from {len(articles)}]"
        if r_pre is not None else "—  (pre-filter off)"
    )

    return (
        "\n".join(logs),
        fig,
        table,
        pre_recall_str,
        f"{r_scr:.1%}  ({scr_hits}/{len(gt)})",
        f"{len(screened)} passed (score ≥ {score_threshold})",
        str(csv_path) if csv_path else "",
    )


# ─── testcase display ─────────────────────────────────────────────────────────

def get_testcase_display():
    tc   = load_testcase()
    gt   = gt_pmids_from_testcase(tc)
    pico = tc.get("PICO", {})
    pico_str = (
        f"P: {pico.get('P','—')}\n"
        f"I: {pico.get('I','—')}\n"
        f"C: {pico.get('C','—')}\n"
        f"O: {pico.get('O','—')}"
    )
    citations = "\n".join(
        f"  {i+1}. PMID {c['pmid']}  —  {c['title'][:70]}…"
        for i, c in enumerate(tc.get("Involved_Citations", []))
    )
    return (
        f"{tc['Title']}\n\n{pico_str}",
        f"{len(gt)} ground-truth PMIDs:\n{citations}",
    )


# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root { color-scheme: light !important; --color-accent: #16a34a !important; }
html, body { background: #fff !important; }
.gradio-container { font-family: 'Inter', sans-serif !important; background: #fff !important; }

/* ── Force textareas/selects to white (NOT checkboxes/radios) ── */
textarea, select,
.block, .wrap, .gap,
.svelte-1gfkn6j, .svelte-s1r2yt,
[class*="container"], [class*="block"],
.prose, .label-wrap,
.scroll-hide { background: #fff !important; color: #1e293b !important; }

textarea { background: #fff !important; color: #1e293b !important; border: 1px solid #e2e8f0 !important; }
input[type="text"], input[type="number"], input[type="search"] { background: #fff !important; color: #1e293b !important; }

/* ── Restore native checkbox/radio appearance ── */
input[type="checkbox"], input[type="radio"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    background-color: revert !important;
    background: revert !important;
    accent-color: #16a34a !important;
    width: 16px !important;
    height: 16px !important;
    cursor: pointer !important;
    border: revert !important;
}
input[type="checkbox"]:checked {
    background-color: #16a34a !important;
    background: #16a34a !important;
    accent-color: #16a34a !important;
}

/* ── Header ── */
#app-header { background: linear-gradient(135deg,#14532d,#16a34a); padding:14px 22px; border-radius:10px; margin-bottom:14px; }
#app-header h1 { margin:0; font-size:20px; font-weight:700; color:#fff !important; }
#app-header .subtitle { margin:3px 0 0; font-size:12px; color:#bbf7d0 !important; }

/* ── Left panel ── */
#left-panel { background:#f8fafc !important; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px !important; }
.panel-lbl { font-size:10px !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:.08em !important; color:#94a3b8 !important; margin:10px 0 4px !important; }

/* ── Buttons ── */
#fetch-btn { background:linear-gradient(135deg,#2563eb,#1d4ed8) !important; border:none !important; color:#fff !important; font-weight:600 !important; }
#fetch-btn:hover { box-shadow:0 6px 16px rgba(37,99,235,.4) !important; }
#screen-btn { background:linear-gradient(135deg,#16a34a,#15803d) !important; border:none !important; color:#fff !important; font-weight:600 !important; }
#screen-btn:hover { box-shadow:0 6px 16px rgba(22,163,74,.4) !important; }

/* ── Tabs ── */
#results-tabs .tab-nav button { font-size:12px !important; padding:7px 12px !important; background:#fff !important; color:#1e293b !important; }
#results-tabs .tab-nav button.selected { font-weight:700 !important; color:#16a34a !important; border-bottom:2px solid #16a34a !important; }

/* ── Logs box ── */
#logs-box textarea { font-family:monospace !important; font-size:11.5px !important; background:#f8fafc !important; color:#1e293b !important; }
"""


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="Pipeline Recall Evaluator") as demo:

        pipeline_state = gr.State({})

        gr.HTML("""
        <div id="app-header">
          <h1>Pipeline Recall Evaluator</h1>
          <span class="subtitle">
            Stage 1 — Fetch PubMed results &amp; check GT coverage &nbsp;|&nbsp;
            Stage 2 — Screen all results &amp; compute true recall
          </span>
        </div>
        """)

        with gr.Row():

            # ── Left panel ────────────────────────────────────────────────────
            with gr.Column(scale=3, elem_id="left-panel", min_width=280):
                gr.HTML('<p class="panel-lbl">Testcase Query</p>')
                query_box = gr.Textbox(label="Query", interactive=False, lines=4)

                gr.HTML('<p class="panel-lbl">Ground-Truth Citations</p>')
                gt_box = gr.Textbox(label="Involved Citations", interactive=False, lines=10)

                gr.HTML('<p class="panel-lbl">Stage 1 — PubMed Fetch</p>')
                max_retrieve = gr.Slider(10, 500, value=500, step=10, label="Max PubMed results")
                mesh_mode = gr.Dropdown(
                    choices=[
                        ("Basic — PICO only, simple OR", "basic"),
                        ("Advanced — PICO + synonyms", "advanced"),
                        ("External MeSH API", "api"),
                    ],
                    value="basic", show_label=False,
                )
                fetch_btn = gr.Button("🔍  Fetch & Check Coverage", variant="primary", elem_id="fetch-btn")

                gr.HTML('<p class="panel-lbl">Stage 2 — Screening</p>')
                score_thresh   = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Score threshold")
                max_workers    = gr.Slider(1, 20, value=config.DEFAULT_MAX_WORKERS, step=1, label="Parallel workers")
                use_prefilter  = gr.Checkbox(value=True, label="Fast keyword pre-filter before screening")
                use_icite      = gr.Checkbox(value=False, label="iCite rerank (promotes landmark papers)")
                icite_alpha    = gr.Slider(0.0, 1.0, value=0.55, step=0.05,
                                           label="iCite weight α (0=BM25 only, 1=citations only)")
                top_k_slider   = gr.Slider(0, 500, value=100, step=10,
                                           label="Rerank → screen top-K (0 = screen all)")
                screen_btn     = gr.Button("📋  Screen All & Compute Recall", variant="primary",
                                           elem_id="screen-btn", interactive=False)

            # ── Right panel ───────────────────────────────────────────────────
            with gr.Column(scale=7, min_width=500):

                with gr.Row():
                    ret_recall_box = gr.Textbox(label="Retrieval Recall",   interactive=False)
                    pre_recall_box = gr.Textbox(label="Pre-filter Recall",  interactive=False)
                    scr_recall_box = gr.Textbox(label="Screening Recall",   interactive=False)
                with gr.Row():
                    fetch_stat_box = gr.Textbox(label="Fetch Status",       interactive=False)
                    scr_stat_box   = gr.Textbox(label="Screen Status",      interactive=False)

                with gr.Tabs(elem_id="results-tabs"):

                    with gr.Tab("Recall Chart"):
                        recall_chart = gr.Plot(label="Recall Chart")

                    with gr.Tab("MeSH Query"):
                        mesh_query_box = gr.Textbox(
                            label="Generated MeSH Query", lines=10,
                            interactive=False,
                        )

                    with gr.Tab("Coverage Table"):
                        gr.Markdown("Ground-truth PMIDs — ✅ retrieved / ❌ missed (updates after Stage 1 & Stage 2)")
                        pmid_table = gr.Dataframe(interactive=False, wrap=True)

                    with gr.Tab("Screening CSV"):
                        screening_file = gr.Textbox(label="CSV path", interactive=False)

                    with gr.Tab("Logs"):
                        log_box = gr.Textbox(lines=24, interactive=False,
                                             show_label=False, elem_id="logs-box")

        demo.load(fn=get_testcase_display, outputs=[query_box, gt_box])

        # Stage 1 outputs
        s1_outputs = [
            log_box, recall_chart,
            mesh_query_box, pmid_table,
            ret_recall_box, fetch_stat_box,
            screen_btn,
            pipeline_state,
        ]

        fetch_btn.click(
            fn=stage1_fetch,
            inputs=[max_retrieve, mesh_mode],
            outputs=s1_outputs,
        )

        # Stage 2 outputs
        s2_outputs = [
            log_box, recall_chart, pmid_table,
            pre_recall_box, scr_recall_box, scr_stat_box,
            screening_file,
        ]

        screen_btn.click(
            fn=stage2_screen,
            inputs=[score_thresh, max_workers, use_prefilter,
                    use_icite, icite_alpha, top_k_slider, pipeline_state],
            outputs=s2_outputs,
        )

    return demo


if __name__ == "__main__":
    try:
        theme = gr.themes.Soft(
            primary_hue="green", secondary_hue="emerald", neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
        theme.set(
            body_background_fill="white",
            body_background_fill_dark="white",
            body_text_color="#1e293b",
            body_text_color_dark="#1e293b",
            background_fill_primary="white",
            background_fill_primary_dark="white",
            background_fill_secondary="white",
            background_fill_secondary_dark="white",
            input_background_fill="white",
            input_background_fill_dark="white",
            block_background_fill="white",
            block_background_fill_dark="white",
            block_border_color="#e2e8f0",
            block_label_background_fill="white",
            block_label_background_fill_dark="white",
            table_even_background_fill="white",
            table_odd_background_fill="#f8fafc",
        )
    except Exception:
        theme = None

    app = build_app()
    app.launch(server_port=7861, share=False, show_error=True,
               theme=theme, css=CUSTOM_CSS)
