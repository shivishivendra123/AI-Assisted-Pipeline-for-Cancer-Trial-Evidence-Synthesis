# Setup Guide

Step-by-step instructions for the two external services REASON depends on:

1. **NCBI** — for PubMed retrieval (`esearch` / `efetch`)
2. **Google Cloud (Vertex AI)** — for the LLM backends (MedGemma + Gemini)

Total setup time: ~15 minutes if you already have a Google account; ~25 minutes from a clean slate.

---

## 1. Generate an NCBI API key

PubMed lets unauthenticated callers issue ~3 requests per second. With an API key you get **10 requests/second** and your `efetch` calls won't be throttled mid-run. The pipeline assumes a key is present.

### Steps

1. Go to **https://www.ncbi.nlm.nih.gov/account/** and either sign in or click **"Sign up"**. NCBI accepts ORCID, Google, Microsoft, and email-based logins.
2. After signing in, click your username in the top-right corner and choose **"Account settings"**.
3. Scroll to the **"API Key Management"** section.
4. Click **"Create an API Key"**. A 36-character hex string appears. Copy it now — NCBI lets you regenerate but does not show the key again on subsequent visits.
5. (Optional, recommended) Set a name on the key so you can identify it later if you ever issue multiple keys.

### What goes in `.env`

```bash
NCBI_EMAIL=you@example.com           # the email NCBI has on file for the account
NCBI_API_KEY=<the 36-char hex you just copied>
```

`NCBI_EMAIL` must match the account that owns the API key — NCBI logs usage against the email, and a mismatch triggers throttling.

### Verifying

```bash
python -c "
from Bio import Entrez
import os
Entrez.email   = os.environ['NCBI_EMAIL']
Entrez.api_key = os.environ['NCBI_API_KEY']
h = Entrez.esearch(db='pubmed', term='pegfilgrastim breast cancer', retmax=5)
print(Entrez.read(h)['IdList'])
"
```

You should see a list of five PMIDs. If you get an HTTP 429 or `bad_request`, the key or email is wrong.

### If you need to rotate a leaked key

If you ever commit a key to a public repo (or believe it has leaked), go straight to **API Key Management** → click **"Revoke"** next to the affected key → click **"Create an API Key"** to issue a fresh one. The old key stops working immediately.

---

## 2. Connect to Google Cloud (Vertex AI)

REASON uses two Vertex AI surfaces:

- **Gemini family** (`gemini-2.5-flash` / `gemini-2.5-pro`) via `GenerativeModel` — used by the MeSH-expansion stage. Available to every GCP project with billing enabled.
- **MedGemma private endpoint** (`mg-endpoint-<uuid>...vertexai.goog`) via `aiplatform.predict` — used by the legacy PICO-extraction and screening stages. This is a one-click deploy from the Model Garden; not strictly required if you switch every stage to Gemini.

### Prerequisite

A Google Cloud account with billing enabled. If you don't have one, start at **https://console.cloud.google.com/** and follow the new-account flow. Google's $300 free trial credit is usually more than enough to run the full bench several times.

### 2a. Create or select a project

1. Open the **GCP Console** at https://console.cloud.google.com.
2. Click the project-picker dropdown at the top (next to "Google Cloud").
3. Click **"NEW PROJECT"** → name it (e.g. `reason-evidence-synthesis`) → **"CREATE"**.
4. Copy the **Project ID** that appears (it's distinct from the project *name* — it usually looks like `reason-evidence-synthesis-415203`). Put this in `.env` as `GCP_PROJECT_ID`.

### 2b. Enable the Vertex AI API

1. Console → search bar → type **"Vertex AI API"** → click the result.
2. Click **"ENABLE"**. Takes ~30 seconds.

Alternatively, from the command line (after step 2c installs `gcloud`):

```bash
gcloud services enable aiplatform.googleapis.com
```

### 2c. Install the gcloud CLI

macOS (Homebrew):
```bash
brew install --cask google-cloud-sdk
```

Linux / WSL:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

Windows: download the installer from **https://cloud.google.com/sdk/docs/install**.

Verify:
```bash
gcloud version
```

### 2d. Authenticate (two separate logins, both needed)

```bash
# (1) Authenticate the `gcloud` CLI itself
gcloud auth login

# (2) Authenticate Application Default Credentials — this is what
#     Python SDKs (google-cloud-aiplatform, etc.) actually read.
gcloud auth application-default login
```

The second command opens a browser, you grant consent, and the resulting credentials are written to `~/.config/gcloud/application_default_credentials.json`. The Python `vertexai` and `aiplatform` SDKs pick them up automatically.

### 2e. Set the active project

```bash
gcloud config set project <your-project-id>
```

This is what `gcloud` uses for command-line operations. Python code reads the project ID from `.env` (`GCP_PROJECT_ID`), so both must point to the same project.

### 2f. Verify Gemini access

```bash
python -c "
import os, vertexai
from vertexai.generative_models import GenerativeModel
vertexai.init(project=os.environ['GCP_PROJECT_ID'], location='us-central1')
print(GenerativeModel('gemini-2.5-flash').generate_content('Say hi in one word.').text)
"
```

You should see a one-word reply. If you get `PermissionDenied`, re-run `gcloud auth application-default login`. If you get `NotFound: model gemini-2.5-flash`, check that the location is `us-central1` (Gemini availability varies by region — `us-east4` does not host every model).

### 2g. (Optional) Deploy a MedGemma endpoint

Only needed if you want to use MedGemma for any pipeline stage. The codebase already provides Gemini overrides (e.g. the eval harness routes everything through Gemini), so you can skip this section if cost or complexity is a concern.

1. Open **Model Garden** → https://console.cloud.google.com/vertex-ai/model-garden.
2. Search **"MedGemma"** → click the variant you want (typically `medgemma-27b-text-it`).
3. Click **"DEPLOY"** → accept defaults → pick a region (e.g. `us-east4`) → **"DEPLOY"**.
4. Deployment takes ~10–15 minutes. When it finishes, the **"Online prediction"** tab shows:
   - **Endpoint ID** (a UUID like `mg-endpoint-fc4b2334-...`)
   - **Dedicated DNS** (looks like `mg-endpoint-<uuid>.<region>-<project-number>.prediction.vertexai.goog`)
5. Copy both into `.env`:

```bash
GCP_ENDPOINT_ID=mg-endpoint-<your-uuid>
GCP_DEDICATED_DNS=mg-endpoint-<your-uuid>.<region>-<project-number>.prediction.vertexai.goog
GCP_LOCATION=us-east4   # whatever region you deployed in
```

> **Cost warning.** A deployed MedGemma 27B endpoint with the smallest GPU (`g2-standard-12`, 1× L4) bills at roughly **$1.10 / hour idle** in `us-east4`. Undeploy it from the same page when you're not using it.

### 2h. Final `.env` file

```bash
# GCP / Vertex AI — MedGemma private endpoint (optional)
GCP_PROJECT_ID=reason-evidence-synthesis-415203
GCP_LOCATION=us-east4
GCP_ENDPOINT_ID=mg-endpoint-fc4b2334-585c-4e64-9c0d-df7caee0cf01
GCP_DEDICATED_DNS=mg-endpoint-fc4b2334-585c-4e64-9c0d-df7caee0cf01.us-east4-132817493282.prediction.vertexai.goog

# GCP / Vertex AI — Gemini family (required)
GEMINI_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash

# NCBI / PubMed (required)
NCBI_EMAIL=you@example.com
NCBI_API_KEY=<your-36-char-hex>

# LLM defaults
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=800
```

`.env` lives in the repo root and is already in `.gitignore` — it never gets committed.

---

## 3. End-to-end verification

Once both services are configured, run the bundled sanity check:

```bash
python test_config.py
```

It prints the loaded configuration (with the NCBI key masked) and confirms that all required fields are non-empty.

Then run a 3-case smoke test of the recall benchmark, which exercises both PubMed and Vertex AI Gemini in a single command:

```bash
cd core
python eval_bench.py --strategy expand --limit 3
```

Expected output:
- Three lines like `[1/3] PMID 33746596 GT=23 …` followed by `→ recall=…%`.
- A CSV in `core/evals/bench_recall_expand_<timestamp>.csv`.
- No `gaierror`, `PermissionDenied`, or HTTP 429 messages.

If all three cases complete with non-zero recall, your setup is working end-to-end and you can proceed to `python core/app_ui.py` for the full Gradio pipeline.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `socket.gaierror: nodename nor servname provided` from `vertex_predict.py` | `GCP_DEDICATED_DNS` is unset or points to a non-existent endpoint | Either deploy a MedGemma endpoint (§ 2g) or use the Gemini override (already wired in `eval_bench.py` for evaluation) |
| `google.auth.exceptions.DefaultCredentialsError` | `gcloud auth application-default login` was never run | Run it (§ 2d, step 2) |
| `404 Not Found: gemini-2.5-flash` | `GEMINI_LOCATION` is set to a region that doesn't host this model | Set `GEMINI_LOCATION=us-central1` in `.env` |
| `urllib.error.HTTPError: HTTP Error 429` from `Entrez` | NCBI rate-limit hit | Confirm `NCBI_API_KEY` is set and `NCBI_EMAIL` matches the key's owner |
| `[ERROR] Benchmark file not found` | Path mismatch | The default expects `TrialReviewBench/TrialReviewBench-study-search-screening.jsonl` next to `core/`; override with `--bench` if your layout differs |
| `dotenv: could not load .env` | `.env` is in the wrong location | It must be at the repo root, next to `requirements.txt` — not inside `core/` |

For other issues, run `python test_config.py` first; it usually tells you which environment variable is missing or has a stale value.
