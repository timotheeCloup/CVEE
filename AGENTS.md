# CVEE (CV Embedding Engine)

AI-powered job matching: upload CV (PDF) → semantic + full-text hybrid search → ranked job offers.

> **Scope**: this file is **project-specific guidance for CVEE only**. General usage rules — zero-billing, conventions, autonomy, response style — live in the global `~/.config/opencode/AGENTS.md`.

---

## ⚠️ RULE #1: everything in `us-east1` (pipeline)

The GCP Always Free Tier **Storage** only covers 3 US regions: `us-east1`, `us-west1`, `us-central1`. A bucket in Europe is billed (even a few cents). All pipeline infrastructure (CFs, Scheduler, GCS) therefore runs in `us-east1`.

| Component | Region | Reason |
|-----------|--------|--------|
| Cloud Functions + Scheduler + GCS | `us-east1` | Always Free Storage |
| Cloud Run (API + UI) | `europe-west1` | User latency, no storage quota |
| Databricks | external | Reads GCS via `gs://` |
| Supabase | external | Own free tier |

### ⚠️ ABSOLUTE RULE: nothing must generate billing

- **No call to a paid API** (Google Translate, Vertex AI, etc.) may be in the code without explicit validation.
- Every service used must be in the GCP **Always Free Tier** or have a third-party free tier (Supabase, GitHub).
- Instructions to run a paid job/script must be given to the user, who runs them **manually** after validation.


---

## GCP Always Free Tier services — full table

Source: [Google Cloud Free Tier](https://cloud.google.com/free/docs/gcp-free-tier).  
Unless stated otherwise, limits are **monthly** (reset on the 1st of the month).

### Storage & Data

| Service | Free limit/month | Use in CVEE | Overage risk |
|---------|------------------|-------------|--------------|
| **Cloud Storage** | 5 GB-months Standard | Parquet raw/silver/gold (~50 MB max) | ✅ Zero risk — **⚠️ bucket must be in `us-east1`** (no free tier in Europe) |
| **BigQuery** | 1 TiB queries + 10 GiB storage | Optional (replaces Databricks) | ✅ Zero risk |
| **Artifact Registry** | 0.5 GiB storage | Docker images for CFs/Cloud Run | ⚠️ Can exceed with many builds → run `gcloud artifacts docker images delete` regularly |

### Compute

| Service | Free limit/month | Use in CVEE | Overage risk |
|---------|------------------|-------------|--------------|
| **Cloud Run** | 2M requests + 360,000 GiB·s memory + 180,000 vCPU·s + 1 GiB egress | API (FastAPI) + UI (Streamlit) + pipeline job | ✅ Large margin (a few hundred requests/day max) |
| **Cloud Run functions** | 2M invocations | `api-to-gcs-cf`, `pipeline-cf`, `ingest-db-cf`, `billing-guard` | ✅ 4 invocations/day → ~120/month |
| **Compute Engine** | 1 e2-micro VM (us-central1/us-east1/us-west1) | Unused | — |

### Network & Messaging

| Service | Free limit/month | Use in CVEE | Overage risk |
|---------|------------------|-------------|--------------|
| **Pub/Sub** | 10 GB messages | Budget alerts → `billing-guard` | ✅ A few KB/month |
| **Cloud Scheduler** | 3 jobs | 3 jobs (api-to-gcs, pipeline, ingest-db) | ✅ OK |

### CI/CD

| Service | Free limit/month | Use in CVEE | Overage risk |
|---------|------------------|-------------|--------------|
| **Cloud Build** | 2,500 build-minutes (e2-standard-2 machine) | CF + Cloud Run deployments | ✅ Each build ~2-3 min |
| **Cloud Deploy** | 1 active pipeline per billing account | Optional (CD) | ✅ OK |

### Security & Observability

| Service | Free limit/month | Use in CVEE | Overage risk |
|---------|------------------|-------------|--------------|
| **Secret Manager** | 6 secret versions (all secrets combined) | 1 secret `cvee-secrets`, ~3-4 versions/year | ✅ |
| **Cloud KMS** | 100 active keys + 10,000 operations (Autokey) | Unused | — |
| **Cloud Logging** (ex-Operations) | 50 GiB logs per project | CF + Cloud Run logs | ✅ Almost no volume |
| **Cloud Monitoring** | 1M time series API read calls + free GCP metrics | Unused | — |

### **NOT free** services — avoid absolutely

| Service | Cost | Present in CVEE? | Action |
|---------|------|------------------|--------|
| **Cloud Translation API** | $20/million chars | Was in `pipeline.py` + `embed_cv_search.py` | ✅ Removed — local multilingual model |
| **Vertex AI** | ~$0.05-5 per LLM call | No | — |
| **Cloud SQL** | ~$7-50/month minimum | No (we use Supabase free tier) | — |
| **Dataproc** (Spark) | ~$0.01/vCPU·h + cluster fees | Was in Databricks | ✅ Abandoned — replaced by local Python pipeline |
| **Cloud Vision / NLP API** | $1.50/1000 units | No | — |

### External services (outside GCP)

| Service | Free tier | Use in CVEE | Risk |
|---------|-----------|-------------|------|
| **Supabase** | Up to 500 MB DB, 2 projects, 5 GB bandwidth | PostgreSQL + pgvector database | ✅ Large margin |
| **GitHub Actions** | 2000 minutes/month (free) | CI/CD | ✅ |
| **GitHub** (code hosting) | Free | Source code | ✅ |
| **Hugging Face** (models) | Free (download) | Embedding model | ✅ ~120 MB model, downloaded once |

---

## 🔒 Billing Guard — automatic billing cutoff (✅ tested & validated)

A safety mechanism automatically cuts billing when the budget is exceeded.

```
GCP budget (€5) → Pub/Sub → Cloud Function billing-guard → unlink billing
```

**How it works:**
1. A GCP **budget** of €5/month (free service) sends Pub/Sub notifications at 50%, 90% and 100% of the threshold
2. At 100% of the threshold, the `billing-guard` **Cloud Function** calls the Cloud Billing API to disable billing
3. The project keeps existing but no paid service can run — resources stop
4. To re-enable: `gcloud billing projects link cvee-20260208 --billing-account=016979-43CAAA-865F35`
5. **Emails**: GCP Budgets automatically emails the Billing Admin (you) at each threshold

**⚠️ Current threshold for June 2026**: **€250** (instead of €5, since €242 already consumed). Reset to €5 on July 1st.

**Technical stack:**
- CF written in **pure Python stdlib** (urllib + metadata server, zero external dependencies)
- 128 MiB memory, 30s timeout, 1 instance max
- Dedicated service account: `billing-guard@cvee-20260208.iam.gserviceaccount.com`

**IAM prerequisites (all verified ✅):**
- `roles/billing.admin` on billing account 016979-43CAAA-865F35
- `roles/billing.projectManager` on project cvee-20260208
- `roles/run.invoker` (allUsers) on the `billing-guard` Cloud Run service

**Deployment:**
```bash
./infra/setup_billing_guard.sh
# Then manually: IAM billing.admin on the billing account
# And: gcloud run services add-iam-policy-binding billing-guard --member=allUsers --role=roles/run.invoker
```

**Real test (June 21, 2026 ✅):**
```bash
gcloud pubsub topics publish budget-alerts --project=cvee-20260208 \
  --message='{"budgetDisplayName":"cvee-budget-guard","costAmount":250,"budgetAmount":5,"alertThresholdExceeded":1.0,"currencyCode":"EUR"}'
# → billingEnabled = False (confirmed)
```

---

## Project Layout

```
CVEE/
├── pyproject.toml              # Root workspace + dev deps (ruff, pytest)
├── uv.lock
├── docker-compose.yml          # Local dev (PG + API + UI)
├── api/                        # FastAPI on Cloud Run
│   ├── pyproject.toml
│   ├── app.py, embed_cv_search.py, utils.py, stopwords.json
│   └── Dockerfile
├── ui/                         # Streamlit on Cloud Run
│   ├── pyproject.toml
│   ├── app.py
│   └── Dockerfile
├── pipeline/                   # Local ETL fallback (if no Databricks)
│   ├── pyproject.toml
│   ├── pipeline.py             # Bronze→Silver→Gold
│   └── init_db.py
├── databricks/                 # Spark/Delta Lake ETL (workspace notebooks)
│   ├── pyproject.toml               # pyspark + delta-spark deps (local test only)
│   ├── common.py                    # Shared utilities (clean_html, find_latest_raw)
│   ├── silver.py                    # GCS raw → Delta silver (Spark)
│   ├── gold.py                      # Delta silver → embeddings → Delta gold (Pandas UDF)
│   ├── export.py                    # Delta → GCS Parquet (timestamped)
│   └── run_all.py                   # Full pipeline orchestrator (%run silver → gold → export)
├── functions/                  # Cloud Functions (2nd gen)
│   ├── api-to-gcs/             # FT API → GCS (nightly 21:00)
│   │   ├── pyproject.toml, main.py, ft_client.py
│   ├── pipeline/               # Bronze→Silver→Gold (nightly 21:30)
│   │   ├── pyproject.toml, main.py, core.py
│   ├── ingest-db/              # GCS → Supabase + cleanup (nightly 23:30)
│   │   ├── pyproject.toml, main.py, gcs_sync.py, cleanup.py
│   └── billing-guard/          # Auto-disable billing
│       ├── pyproject.toml, main.py
├── infra/                       # Terraform IaC
│   ├── main.tf                  # Provider, backend GCS
│   ├── variables.tf             # Input variables (sensitive handled)
│   ├── outputs.tf               # CF URLs, bucket name
│   ├── cloud_functions.tf       # 3 CFs gen2 + zip archives
│   ├── scheduler.tf             # 3 Cloud Scheduler jobs
│   ├── storage.tf               # GCS bucket
│   ├── secrets.tf               # Secret Manager (single JSON)
│   └── terraform.tfvars         # Values (gitignored)
├── scripts/                     # Utility scripts
│   ├── deploy.sh                # DEPRECATED — use Terraform
│   └── backfill.py              # Historical backfill (month by month)
├── tests/                       # pytest
├── .github/workflows/ci.yaml   # Ruff + pytest + Docker build
└── assets/                     # Diagrams, demo media
```

**Cutting-edge tooling:** `uv` (astral.sh), `ruff` (lint + format, 0 errors), `pytest`, GitHub Actions CI.

## Deploy + Backfill

```bash
# Prerequisite: create the Terraform state bucket (once)
gsutil mb -l us-east1 gs://cvee-20260208-tfstate

# Deploy the infrastructure
cd infra && terraform init && terraform apply

# Historical backfill
uv run python scripts/backfill.py --date-min 2026-01-01 --date-max 2026-06-30

# Manual pipeline (reprocesses the last 7 days of raw)
curl -X POST "$(terraform output -raw pipeline_url)?days=7"
```

## Cleanup Legacy GCP Resources

```bash
# Legacy CFs + Scheduler (S3)
gcloud functions delete api-to-s3-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete api-to-s3-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
gcloud storage rm -r gs://cvee-bucket-eu-north-1

# europe-west1 duplicates (replaced by us-east1)
gcloud functions delete api-to-gcs-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud functions delete ingest-db-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete api-to-gcs-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete ingest-db-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
```

**Dev quickstart:**
```bash
uv sync --group dev      # install everything
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest tests/     # test
uv run python api/app.py # run API
```

**Legacy dirs** (`src/`, `infra/`, `docker/`) — kept until the migration is complete.

---

## 🔀 Pipeline: GCS vs Databricks — analysis

The pipeline must transform `jobs_raw/` (bronze) → `jobs_silver/` + `jobs_gold/`.

### Option A: Databricks (current)

```
FT API → api-to-gcs-cf → GCS raw → Databricks notebooks → GCS silver/gold → ingest-db-cf → Supabase
```

| ✅ Pros | ❌ Cons |
|---------|--------|
| Delta Lake (versioning, time travel, schema evolution) | **Paid** — DBU + cluster, no free tier |
| Spark: parallelism, scale on large volumes | Cluster = cost even when idle |
| 3 battle-tested notebooks, already migrated S3→GCS | Requires a cloud provider backend (AWS/Azure/GCP) |
| Includes `dbutils.fs.ls` to list GCS files | No usable free tier in production |
| | Limited scheduling in the Standard version |

### Option B: pipeline.py (local Python, zero cost)

```
FT API → api-to-gcs-cf → GCS raw → pipeline.py (Cloud Run Job) → GCS silver/gold → ingest-db-cf → Supabase
```

| ✅ Pros | ❌ Cons |
|---------|--------|
| **100% free** (Cloud Run free tier) | No Delta Lake (plain Parquet, no versioning) |
| Same code as Databricks, without Spark | Single-machine, no parallelism |
| No external dependency | 3000 offers = ~5 min CPU (acceptable in a nightly batch) |
| Unified deployment via deploy.sh (3 scheduler jobs) | Less "clean" than Delta tables |
| `antoinelouis/french-me5-small` model (36M params) ideal as it is light | |

### Target embedding model: `antoinelouis/french-me5-small`

| Model | Params | Size | Dims | Languages | Note |
|-------|--------|------|------|-----------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` (current) | 118M | 120 MB | 384 | 50+ (incl. FR) | Multilingual, decent for FR |
| `antoinelouis/french-me5-small` (target) | **36M** | **~40 MB** | 384 | FR only | 70% lighter, FR-optimized, pruned from `multilingual-e5-small` |

### Recommendation

**→ Option B chosen**: `pipeline-cf` (Cloud Function, 2 GB, CPU embeddings) + 3 scheduler jobs = €0, 3 free-tier jobs.

Databricks stays available as a fallback (notebooks up to date with `gs://` and `french-me5-small`), but the GCP pipeline runs autonomously without an external dependency.

Reminder of monthly pipeline cost: ~28,800 GiB·s → **8%** of the free tier quota (360,000 GiB·s).

---

## 🐛 Known bugs

### France Travail API: `publieeDepuis` > 1 broken

The `publieeDepuis` parameter (which filters offers published in the last N days) is buggy on the France Travail API side: **only `publieeDepuis=1` works**. Any value > 1 returns 0 results, regardless of the real volume of offers.

**Workaround**: use `minDateCreation` + `maxDateCreation` (explicit dates) which work correctly.

**Impact**:
- "daily" mode (`publiee_depuis=1`, no params) works and fetches offers from the last 24h
- To fetch more than the last 24h, you **must** go through `date_min`/`date_max`
- For a quick test without a precise date range, use `?max_results=N` which automatically computes a 30-day window
