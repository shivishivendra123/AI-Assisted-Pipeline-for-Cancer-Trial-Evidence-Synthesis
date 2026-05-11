"""
Benchmark Recall Evaluator UI

Runs PICO → MeSH → dual-fetch across all TrialReviewBench test cases and
displays per-case recall + aggregate statistics.

Run from evsy/core/:
    python eval_bench_ui.py
"""

import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

BENCH_PATH = (
    Path(__file__).parent.parent.parent
    / "TrialReviewBench"
    / "TrialReviewBench-study-search-screening.jsonl"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts_day5"


# ── data helpers ──────────────────────────────────────────────────────────────

def load_bench(path: Path = BENCH_PATH):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def gt_pmids(tc: dict) -> set:
    return {c["pmid"] for c in tc.get("Involved_Citations", []) if c.get("pmid")}


def recall(retrieved: set, gt: set) -> float:
    return len(retrieved & gt) / len(gt) if gt else 0.0


# ── per-case runner ───────────────────────────────────────────────────────────

def run_one(
    tc: dict, max_retrieve: int,
    pico_pipe, generate_basic_mesh_query, fetch_pubmed_articles,
    use_icite: bool = False, icite_top_k: int = 200, icite_alpha: float = 0.55,
) -> dict:
    gt = gt_pmids(tc)
    base = {
        "pmid":              tc.get("PMID", ""),
        "title":             tc["Title"][:80],
        "topic":             tc.get("Topic", ""),
        "gt_count":          len(gt),
        "retrieved":         0,
        "hits":              0,
        "recall":            0.0,
        "recall_pre_rerank": 0.0,
        "missed_pmids":      "",
        "error":             "",
    }

    if not gt:
        base["error"] = "no GT PMIDs"
        return base

    try:
        pico_json  = pico_pipe(tc["Title"])
        mesh_query = generate_basic_mesh_query(pico_json)

        data = fetch_pubmed_articles(
            pico_json["qid"], mesh_query,
            max_results=max_retrieve, fetch_fulltext=False,
        )
        articles = data["articles"] if data else {}

        cmp_query = pico_json.get("mesh_query_comparator", "")
        if cmp_query:
            data2 = fetch_pubmed_articles(
                pico_json["qid"], cmp_query,
                max_results=max_retrieve, fetch_fulltext=False,
            )
            if data2 and data2.get("articles"):
                articles = {**data2["articles"], **articles}

        # Pre-rerank recall
        pre_retrieved = set(articles.keys())
        base["recall_pre_rerank"] = round(recall(pre_retrieved, gt), 4)

        # Optional iCite reranking
        if use_icite and articles:
            from pipeline.icite_reranker import icite_rerank
            groups = pico_json.get("mesh_groups", {})
            bm25_q = " ".join(
                groups.get("population", []) +
                groups.get("intervention", []) +
                groups.get("comparator", [])
            ) or tc["Title"]
            articles = icite_rerank(bm25_q, articles, top_k=icite_top_k, alpha=icite_alpha)

        retrieved = set(articles.keys())
        hits      = retrieved & gt
        missed    = gt - retrieved

        base["retrieved"]    = len(retrieved)
        base["hits"]         = len(hits)
        base["recall"]       = round(recall(retrieved, gt), 4)
        base["missed_pmids"] = "|".join(sorted(missed))

    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"

    return base


# ── chart builders ────────────────────────────────────────────────────────────

def build_distribution_chart(results: list):
    valid = [r for r in results if not r["error"] and r["gt_count"] > 0]
    if not valid:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        return fig

    recalls = [r["recall"] * 100 for r in valid]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 101]
    labels = ["0–10", "10–20", "20–30", "30–40", "40–50",
              "50–60", "60–70", "70–80", "80–90", "90–100", "100"]

    counts = [0] * len(labels)
    for r in recalls:
        for i in range(len(bins) - 1):
            if bins[i] <= r < bins[i + 1]:
                counts[i] += 1
                break

    colors = ["#ef4444" if i < 5 else "#f59e0b" if i < 8 else "#22c55e"
              for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.7, zorder=3)
    ax.set_xlabel("Retrieval Recall (%)", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title(f"Recall Distribution  —  {len(valid)} cases  |  mean {sum(recalls)/len(recalls):.1f}%",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, cnt in zip(bars, counts):
        if cnt:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def build_bucket_chart(results: list):
    valid = [r for r in results if not r["error"] and r["gt_count"] > 0]
    if not valid:
        fig, ax = plt.subplots()
        return fig

    buckets = {
        "Easy\n(1–5 GT)":    [],
        "Medium\n(6–15 GT)": [],
        "Hard\n(16–30 GT)":  [],
        "V.Hard\n(31+ GT)":  [],
    }
    for r in valid:
        gt_c = r["gt_count"]
        if gt_c <= 5:
            buckets["Easy\n(1–5 GT)"].append(r["recall"] * 100)
        elif gt_c <= 15:
            buckets["Medium\n(6–15 GT)"].append(r["recall"] * 100)
        elif gt_c <= 30:
            buckets["Hard\n(16–30 GT)"].append(r["recall"] * 100)
        else:
            buckets["V.Hard\n(31+ GT)"].append(r["recall"] * 100)

    labels  = [k for k, v in buckets.items() if v]
    means   = [sum(v) / len(v) for k, v in buckets.items() if v]
    medians = [sorted(v)[len(v) // 2] for k, v in buckets.items() if v]
    counts  = [len(v) for k, v in buckets.items() if v]
    colors  = ["#4C9BE8", "#F59E0B", "#EF4444", "#9333EA"][:len(labels)]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w/2, means,   w, label="Mean",   color=colors, alpha=0.9, zorder=3)
    b2 = ax.bar(x + w/2, medians, w, label="Median", color=colors, alpha=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Recall by GT Difficulty Bucket", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, cnt in zip(b1, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"n={cnt}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


# ── results table ─────────────────────────────────────────────────────────────

def build_results_df(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        recall_pct = f"{r['recall']:.1%}" if not r["error"] else "—"
        rows.append({
            "PMID":     r["pmid"],
            "GT":       r["gt_count"],
            "Retrieved": r["retrieved"],
            "Hits":     r["hits"],
            "Recall":   recall_pct,
            "Status":   "❌ " + r["error"][:40] if r["error"] else "✅",
            "Title":    r["title"],
        })
    return pd.DataFrame(rows)


def summary_text(results: list) -> str:
    valid   = [r for r in results if not r["error"] and r["gt_count"] > 0]
    errors  = [r for r in results if r["error"]]
    pending = [r for r in results if r["gt_count"] == 0 and not r["error"]]

    if not valid:
        return f"Evaluated: {len(results)}  |  Errors: {len(errors)}  |  No valid results yet"

    recalls = sorted(r["recall"] for r in valid)
    n = len(recalls)
    mean   = sum(recalls) / n
    median = recalls[n // 2]
    mn, mx = recalls[0], recalls[-1]

    at80  = sum(1 for r in recalls if r >= 0.80)
    at60  = sum(1 for r in recalls if r >= 0.60)
    at100 = sum(1 for r in recalls if r >= 1.00)

    return (
        f"Cases evaluated: {len(results)}   ✅ valid: {n}   ⚠️ errors: {len(errors)}\n"
        f"Mean recall:   {mean:.1%}   |   Median: {median:.1%}   |   Min: {mn:.1%}   Max: {mx:.1%}\n"
        f"≥80% recall:  {at80}/{n} ({at80/n:.0%})   |   ≥60%: {at60}/{n} ({at60/n:.0%})   |   100%: {at100}/{n} ({at100/n:.0%})"
    )


# ── main eval function (called by Gradio) ─────────────────────────────────────

def run_benchmark(max_retrieve: int, limit_n: int, use_icite: bool, icite_top_k: int, icite_alpha: float, progress=gr.Progress(track_tqdm=True)):
    from pipeline.pico import pico_pipe
    from meshOnDemand.mesh_basic import generate_basic_mesh_query
    from efetch_utility.efetch import fetch_pubmed_articles

    cases = load_bench()
    if limit_n > 0:
        cases = cases[:limit_n]

    total = len(cases)
    results = []

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ARTIFACTS_DIR / f"bench_recall_{ts}.csv"
    fieldnames = ["pmid", "title", "topic", "gt_count", "retrieved", "hits",
                  "recall", "missed_pmids", "error"]

    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for i, tc in enumerate(cases, 1):
            progress(i / total, desc=f"Case {i}/{total}: {tc['Title'][:50]}…")

            r = run_one(tc, max_retrieve, pico_pipe, generate_basic_mesh_query, fetch_pubmed_articles,
                        use_icite=use_icite, icite_top_k=icite_top_k, icite_alpha=icite_alpha)
            results.append(r)
            writer.writerow(r)
            csvf.flush()

            # Yield incremental updates every case
            dist_fig   = build_distribution_chart(results)
            bucket_fig = build_bucket_chart(results)
            df         = build_results_df(results)
            summ       = summary_text(results)

            yield summ, dist_fig, bucket_fig, df, str(out_path)

    # Final yield
    dist_fig   = build_distribution_chart(results)
    bucket_fig = build_bucket_chart(results)
    df         = build_results_df(results)
    summ       = summary_text(results)
    yield summ, dist_fig, bucket_fig, df, str(out_path)


# ── load existing CSV ─────────────────────────────────────────────────────────

def load_existing_csv(csv_path: str):
    if not csv_path or not Path(csv_path).exists():
        return "File not found.", None, None, pd.DataFrame(), ""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append({
                "pmid":        row.get("pmid", ""),
                "title":       row.get("title", ""),
                "topic":       row.get("topic", ""),
                "gt_count":    int(row.get("gt_count", 0)),
                "retrieved":   int(row.get("retrieved", 0)),
                "hits":        int(row.get("hits", 0)),
                "recall":      float(row.get("recall", 0)),
                "missed_pmids": row.get("missed_pmids", ""),
                "error":       row.get("error", ""),
            })
    return (
        summary_text(results),
        build_distribution_chart(results),
        build_bucket_chart(results),
        build_results_df(results),
        csv_path,
    )


def list_existing_runs() -> list:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    files = sorted(ARTIFACTS_DIR.glob("bench_recall_*.csv"), reverse=True)
    return [str(f) for f in files]


# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root { color-scheme: light !important; --color-accent: #16a34a !important; }
html, body { background: #fff !important; }
.gradio-container { font-family: 'Inter', sans-serif !important; background: #fff !important; }
textarea, select, .block, .wrap, .gap, .prose, .label-wrap, .scroll-hide {
    background: #fff !important; color: #1e293b !important; }
textarea { border: 1px solid #e2e8f0 !important; }
input[type="checkbox"], input[type="radio"] {
    appearance: auto !important; -webkit-appearance: checkbox !important;
    background: revert !important; accent-color: #16a34a !important;
    width: 16px !important; height: 16px !important; }
#app-header { background: linear-gradient(135deg,#14532d,#16a34a);
    padding:14px 22px; border-radius:10px; margin-bottom:14px; }
#app-header h1 { margin:0; font-size:20px; font-weight:700; color:#fff !important; }
#app-header .subtitle { margin:3px 0 0; font-size:12px; color:#bbf7d0 !important; }
#run-btn { background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    border:none !important; color:#fff !important; font-weight:600 !important; }
#run-btn:hover { box-shadow:0 6px 16px rgba(37,99,235,.4) !important; }
#load-btn { background:linear-gradient(135deg,#16a34a,#15803d) !important;
    border:none !important; color:#fff !important; font-weight:600 !important; }
#summary-box textarea { font-family: monospace !important; font-size: 13px !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="Benchmark Recall Evaluator", css=CUSTOM_CSS) as demo:

        gr.HTML("""
        <div id="app-header">
          <h1>Benchmark Recall Evaluator</h1>
          <span class="subtitle">
            Runs PICO → MeSH → dual-fetch across all TrialReviewBench test cases
            and reports per-case retrieval recall
          </span>
        </div>
        """)

        with gr.Row():
            # ── Left panel ────────────────────────────────────────────────────
            with gr.Column(scale=3, min_width=260):
                gr.Markdown("### Run Settings")
                max_retrieve = gr.Slider(50, 500, value=500, step=50,
                                         label="Max PubMed results per case")
                limit_n = gr.Slider(0, 100, value=0, step=5,
                                    label="Limit cases (0 = all 100)")
                use_icite   = gr.Checkbox(value=False, label="iCite rerank (promotes landmark papers)")
                icite_top_k = gr.Slider(50, 500, value=200, step=25,
                                        label="Keep top-K after reranking")
                icite_alpha = gr.Slider(0.0, 1.0, value=0.55, step=0.05,
                                        label="iCite weight α (0=BM25 only, 1=citations only)")
                run_btn = gr.Button("▶  Run Benchmark", variant="primary", elem_id="run-btn")

                gr.Markdown("---")
                gr.Markdown("### Load Existing Run")
                existing_runs = gr.Dropdown(
                    choices=list_existing_runs(),
                    label="Previous result CSVs",
                    value=None,
                )
                refresh_btn = gr.Button("↻  Refresh list", size="sm")
                load_btn    = gr.Button("📂  Load", variant="secondary", elem_id="load-btn")

                gr.Markdown("---")
                csv_path_box = gr.Textbox(label="Saved CSV", interactive=False)

            # ── Right panel ───────────────────────────────────────────────────
            with gr.Column(scale=7, min_width=500):
                summary_box = gr.Textbox(
                    label="Summary", lines=3, interactive=False, elem_id="summary-box"
                )

                with gr.Tabs():
                    with gr.Tab("Distribution"):
                        dist_chart = gr.Plot(label="Recall Distribution")

                    with gr.Tab("By Difficulty Bucket"):
                        bucket_chart = gr.Plot(label="Recall by GT Bucket")

                    with gr.Tab("Per-Case Results"):
                        gr.Markdown("All evaluated cases. Sortable by clicking column headers.")
                        results_table = gr.Dataframe(
                            interactive=False,
                            wrap=True,
                            column_widths=["8%","5%","8%","6%","7%","8%","58%"],
                        )

        # ── wire up ───────────────────────────────────────────────────────────

        run_btn.click(
            fn=run_benchmark,
            inputs=[max_retrieve, limit_n, use_icite, icite_top_k, icite_alpha],
            outputs=[summary_box, dist_chart, bucket_chart, results_table, csv_path_box],
        )

        load_btn.click(
            fn=load_existing_csv,
            inputs=[existing_runs],
            outputs=[summary_box, dist_chart, bucket_chart, results_table, csv_path_box],
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=list_existing_runs()),
            outputs=[existing_runs],
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
    app.launch(server_port=7862, share=False, show_error=True,
               theme=theme)
