# AI-Assisted Pipeline for Cancer Trial Evidence Synthesis

**Retrieval-enhanced Evidence Assessment with Synthesis and Organized Narration** — an end-to-end agent-based LLM pipeline that takes a free-text clinical question and produces a PRISMA-2020-aligned systematic-review PDF.

The retrieval stage uses a two-call language-model strategy: a *primary-term* call drives a narrow seven-paper PubMed probe whose results ground a *second expansion* call that emits a three-axis Boolean query — `(conditions) AND (treatments) AND (outcomes)`. On TrialReviewBench, this strategy lifts mean recall@2000 from **34.9 % → 52.8 %** over a basic MeSH-only baseline (n = 100), and reaches **55.6 %** on a 67-case oncology subset with only **2 / 67** cases retaining zero relevant trials in the top-2000 candidate pool.

---

## Headline results

Evaluated on **TrialReviewBench** (100 published systematic reviews with gold-standard PMID lists) and a **67-case oncology subset**.

| Strategy | N | recall@100 | recall@500 | recall@2000 | zero@2000 |
|---|---|---|---|---|---|
| Basic MeSH-only (MedGemma)                | 100 | 2.2 % | 18.3 % | 34.9 % | 14 |
| Gemini 2.5 Flash, bare terms              | 100 | 2.5 % | 25.8 % | 48.2 % | 10 |
| 3-axis expand (full bench)                | 100 | 4.4 % | 31.2 % | 52.8 % | 12 |
| **3-axis expand (cancer-only)**           |  67 | **7.1 %** | **35.6 %** | **55.6 %** | **2** |
| 3-axis expand + pub-type filter (cancer-only) |  67 | 9.4 % | 30.8 % | 46.8 % | — |

Source CSVs live in [`core/evals/`](core/evals/); reproducible via `core/eval_bench.py` (see [Benchmarking](#benchmarking)).

### Per-axis ablation (recall@2000, 67-case oncology subset)

| Configuration                  | recall@2000 | Δ vs full |
|---|---|---|
| Full 3-axis (C ∧ T ∧ O)        | 55.6 %      | --         |
| Drop T (C ∧ O)                 | 52.9 %      | −2.7 pp    |
| Single-axis (C only)           | 45.2 %      | −10.4 pp   |

Dropping the **treatments** axis is the single largest contributor to recall loss, consistent with the observation that the most discriminative PubMed token in oncology trials is typically the drug INN (international nonproprietary name) or device name.

---

## Novel contributions

1. **Two-call retrieval-augmented MeSH expansion.** A *primary-term* probe fetches seven on-topic reference papers from PubMed; their titles and abstracts ground a second LLM call that emits CORE + EXPAND term lists per axis. This grounds language-model term selection in the actual vocabulary of the literature rather than relying on the model's parametric memory alone. Implementation: [`core/meshOnDemand/mesh_expand.py`](core/meshOnDemand/mesh_expand.py).
2. **Three-axis Boolean query structure.** Final query is `(conditions OR …) AND (treatments OR …) AND (outcomes OR …)` rather than a flat OR. The outcomes axis acts as an implicit trial-filter — papers reporting trial outcomes in their title/abstract are overwhelmingly clinical trials.
3. **Empirical decomposition of the retrieval-recall gap** into independent, individually attributable interventions: LLM substrate (+13.3 pp at recall@2000), three-axis structure (+4.6 pp at recall@2000), publication-type filter (+2.3 pp at recall@100 on the oncology subset). Each gain was measured incrementally on the same benchmark using the same harness.
4. **Multi-backend agent orchestration.** Single `Agent` interface over both private Vertex AI endpoints (MedGemma via `:predict`) and Vertex Gemini (via `GenerativeModel`), letting individual pipeline stages pick the model that best suits the task. Implementation: [`core/agents/`](core/agents/).
5. **Stand-alone PRISMA-2020 PDF generator.** Reads pipeline artifacts from disk and produces a journal-style PDF with title page, structured abstract, methods, PRISMA flow diagram, characteristics/outcomes tables, narrative synthesis with inline numbered citations, and Vancouver-style references — without re-running any LLM stage. Implementation: [`core/render_report.py`](core/render_report.py) + [`core/report_generator/`](core/report_generator/).
6. **Reproducible evaluation harness with rank-recall**, measuring not just final-pool recall but PubMed's rank-ordered recall@K for K ∈ {100, 500, 2000} on the primary query alone, so query-formulation failures and rank-ordering failures can be distinguished. Implementation: [`core/eval_bench.py`](core/eval_bench.py).

---

## Architecture (eight stages)

```
free-text clinical question
        │
        ▼
[1] PICO extraction                        →  artifacts_day1/pico_<qid>.json
[2] Retrieval-augmented MeSH expansion     →  artifacts_day3/mesh_expand_query_<qid>.json
        │   ├── primary-term call → 7-paper probe
        │   └── grounded expansion → CORE + EXPAND terms per axis
[3] PubMed retrieval (esearch + efetch)    →  artifacts_day4/all_studies_metadata_<qid>.jsonl
[4] Eligibility-criteria generation        →  artifacts_day5/eligibility_<qid>.json
[5] Agent-based title/abstract screening   →  artifacts_day5/screening_results_<ts>.csv
[6] Study-characteristics extraction       →  artifacts_day6/study_char_<ts>.csv
[7] Outcome extraction                     →  artifacts_day6/study_outcomes_<ts>.csv
[8] Narrative synthesis + PDF generation   →  artifacts_day7/sr_<qid>_<ts>.pdf
```

Each stage emits durable JSON / CSV / PDF artefacts to disk, so partial runs are recoverable and the PDF can be regenerated without re-running expensive LLM stages.

---

## Repository layout

```
ES_Pipeline/
├── core/
│   ├── agents/                  Multi-backend Agent abstraction (MedGemma, Gemini, OpenAI)
│   ├── pipeline/                PICO extraction, prefilter, reranker
│   ├── meshOnDemand/            MeSH query builders (basic + retrieval-augmented expand)
│   ├── efetch_utility/          PubMed esearch/efetch with multi-sort + by-decade fallbacks
│   ├── pubmedSearch/            PubMed API search wrapper
│   ├── eligibility_builder/     LLM-generated eligibility criteria from PICO
│   ├── screening/               Agent-based title/abstract screening
│   ├── extraction/              Study characteristics + outcome extraction
│   ├── basic_sythesis/          Narrative evidence synthesis
│   ├── report_generator/        PRISMA-2020 PDF assembler
│   ├── render_report.py         Stand-alone PDF re-render from artefacts
│   ├── app_ui.py                Gradio dashboard for the full pipeline
│   ├── eval_bench.py            Batch recall evaluator vs TrialReviewBench
│   ├── eval_ui.py / eval_recall.py  Interactive recall evaluators
│   ├── prompts/                 LLM prompt templates
│   ├── configs/env_config.py    Env-var-driven configuration (no secrets in source)
│   ├── evals/                   Bench CSVs (generated)
│   ├── archived_rsults/         Historical artifact snapshots
│   └── artifacts_day{1..7}/     Per-stage artefact directories (generated)
├── TrialReviewBench/
│   ├── TrialReviewBench-study-search-screening.jsonl   100-case full bench
│   └── TrialReviewBench-cancer-only.jsonl              67-case oncology subset
├── figures/                     Result figures (PNG) + regen_figs.py
├── requirements.txt
├── test_config.py               Sanity-check that .env loaded correctly
└── README.md
```

---

## Quickstart

See [SETUP.md](SETUP.md) for step-by-step NCBI and Google Cloud setup.

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure secrets (none in source)

Copy your credentials into a local `.env` file in the repo root (already in `.gitignore`):

```bash
# GCP / Vertex AI (MedGemma private endpoint)
GCP_PROJECT_ID=<your-project-id>
GCP_LOCATION=us-east4
GCP_ENDPOINT_ID=<your-endpoint-id>
GCP_DEDICATED_DNS=<your-endpoint>.<region>-<id>.prediction.vertexai.goog

# GCP / Vertex AI (Gemini family, used by mesh_expand)
GEMINI_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash

# NCBI / PubMed
NCBI_EMAIL=you@example.com
NCBI_API_KEY=<your-key>

LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=800
```

Authenticate to Vertex once per machine:

```bash
gcloud auth application-default login
```

Verify everything loaded:

```bash
python test_config.py
```

### 3. Run the full pipeline (Gradio UI)

```bash
cd core
python app_ui.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`). Paste a clinical question and step through the eight stages. Example question:

> *In adult patients with breast cancer receiving cytotoxic chemotherapy, does pegfilgrastim compared with filgrastim reduce the incidence of febrile neutropenia?*

Final PDF is emitted to `core/artifacts_day7/sr_<qid>_<timestamp>.pdf`.

### 4. Re-render an existing report without re-running LLM stages

```bash
cd core
python render_report.py --list                # list available qids
python render_report.py --qid 6d09b924e073    # render that one
python render_report.py --out /tmp/sr.pdf
```

---

## Benchmarking

The evaluator iterates every test case in `TrialReviewBench`, runs `PICO → MeSH → dual-fetch → (optional iCite rerank)` for each, and records per-case **rank-recall@{100, 500, 2000}** (PubMed's ranked-order recall on the primary query alone) plus the **final-pool recall** (after the dual-fetch and any rerank).

### Full benchmark on the cancer-only subset

```bash
cd core
python eval_bench.py \
  --bench ../TrialReviewBench/TrialReviewBench-cancer-only.jsonl \
  --strategy expand \
  --terms-per-axis 15 \
  --exclude-reviews
```

Output: `core/evals/bench_recall_expand_noreviews_<timestamp>.csv` + console summary.

### Smoke test (≈ 30 seconds)

```bash
python eval_bench.py --strategy expand --limit 3
```

### Available flags

| Flag | Purpose |
|---|---|
| `--strategy {basic,expand}` | basic = 2-axis P-AND-I; expand = 3-axis retrieval-augmented |
| `--bench PATH` | Custom benchmark JSONL |
| `--limit N` | Eval only first N cases |
| `--pmids 30854085,12137670` | Eval only listed PMIDs |
| `--worst PATH:N` | Re-eval N lowest-recall cases from a prior CSV |
| `--terms-per-axis N` | Per-axis cap (default 8); set 15 for fat-OR |
| `--exclude-reviews` | Cochrane-style pub-type filter (expand only) |
| `--multi-sort` | Union of relevance + newest + oldest sorts |
| `--by-decade` | 4 × 500 PMIDs stratified by era |
| `--no-sort` | Drop `sort=relevance` (test PubMed default ordering) |
| `--icite --top-k N --alpha F` | iCite RCR + BM25 rerank on the retrieved pool |

---

## Failure mode taxonomy

Manual examination of the eight worst-performing cases on the 67-case oncology subset surfaced three recurring patterns:

1. **Adverse-event / predictive-biomarker reviews** (≥ 4 / 8 cases) — e.g. nephrotoxicity from anti-PD-1/PD-L1 antibodies, endocrine dysfunction from ICIs, tumor mutation burden as ICI-efficacy predictor. The query expands around disease + treatment, but trial papers index their AE or biomarker findings only as secondary results — so query-relevant tokens appear in trial methods/results rather than title/abstract, and PubMed Best Match instead promotes review-class papers that name the AE in the title.
2. **Combination-regimen reviews** (3 / 8) — e.g. trastuzumab-containing regimens, intravesical chemo + hyperthermia, thalidomide/dexamethasone. The canonical drug INNs are captured by the expansion, but the ground-truth papers use regimen-level naming conventions ("HIVEC", "Vel-Dex") that the expansion misses.
3. **Rare cell-therapy reviews** (1 / 8) — autologous CD19 CAR-T in a specific lymphoma subset, where the candidate pool is itself shallow and any one axis using a non-canonical synonym loses a substantial fraction of the GT.

Concrete remediation paths: outcome-axis boosting for AE-anchored reviews; regimen-abbreviation expansion for combination-regimen cases; CHSSS-style hand-validated topic clauses as a fallback when the primary probe returns fewer than seven on-topic papers.

---

## Limitations

1. **Single-database retrieval.** Studies indexed only in Embase, the Cochrane Central Register of Controlled Trials, ClinicalTrials.gov, or grey literature are by definition not in the PMID-keyed ground truth and therefore not measured here. The true multi-database coverage gap is not directly observable from this benchmark.
2. **Residual query-formulation gap.** Within the PubMed-indexed ground truth, 3 % of cases on the oncology subset (12 % on the full bench) retain zero recall@2000 — a query-formulation gap, not a database-coverage ceiling.
3. **Qualitative synthesis.** No formal meta-analysis (random- or fixed-effects pooling), risk-of-bias assessment, or GRADE certainty rating is performed.
4. **Extraction granularity.** Automated extraction may miss methodological nuances that a human extractor would capture, particularly for studies with non-standard reporting or content buried in supplementary appendices.
5. **Configurable thresholds.** The cumulative relevance threshold `τ` is a knob that may exclude borderline-eligible studies; sensitivity analyses across τ ∈ {2.0, 2.5, 3.0, 3.5} show yield variations of ±3–5 included studies for typical queries.
6. **LLM nondeterminism.** Pipeline reproducibility is bounded by language-model nondeterminism; `LLM_TEMPERATURE=0.0` reduces but does not eliminate it.

---

## Citation

If you use REASON in your work, please cite:

```
@misc{reason2026,
  title  = {AI-Assisted Pipeline for Cancer Trial Evidence Synthesis: Retrieval-enhanced Evidence Assessment with Synthesis and Organized Narration},
  author = {Gupta, Shivendra and Alluri, Eesha Reddy},
  year   = {2026},
  note   = {AIM5008 — AI-Assisted Pipeline for Cancer Trial Evidence Synthesis}
}
```

The benchmark used for evaluation is **TrialReviewBench** (Wang et al.), which we redistribute under its original license in [`TrialReviewBench/`](TrialReviewBench/).

---


