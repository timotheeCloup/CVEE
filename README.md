# CVEE: CV-Embedding Engine

Traditional job boards match you based on keywords. CVEE analyzes your entire CV to understand your profile, then finds the roles that truly align with your skills and experience using semantic search — in seconds.

**CVEE** is an AI-powered job matching system that combines semantic embeddings and full-text search to find the most relevant job opportunities from the France Travail API.

**Try it now:** [CV Match Engine](https://cvee-ui-1081304882492.europe-west1.run.app/)

[![CI](https://github.com/timotheeCloup/CVEE/actions/workflows/ci.yaml/badge.svg)](https://github.com/timotheeCloup/CVEE/actions/workflows/ci.yaml)

---

<p align="center">
  <img src="./assets/demo.gif" alt="demo">
  <br>
  <b>End-to-end CV parsing and real-time hybrid search</b>
</p>

---

## Features

- **Hybrid Semantic Search** — Cosine similarity (pgvector) + full-text search (PostgreSQL `ts_rank`) + title matching, combined via Reciprocal Rank Fusion (RRF)
- **Real-Time Matching** — Upload your CV, get ranked results with keyword highlighting in seconds
- **Multilingual Embeddings** — [`antoinelouis/french-me5-small`](https://huggingface.co/antoinelouis/french-me5-small) (384-dim)
- **Automated ETL Pipeline** — Nightly ingestion, cleaning, deduplication, embedding generation, and database sync via Cloud Workflows
- **Dead Link Detection** — Async HTTP verification of job posting links, prunes expired offers from the database

---

## Architecture

<p align="center">
  <img src="./assets/CVEE.drawio.svg" alt="Data pipelines architecture">
  <br>
  <b>Data Pipeline and System Architecture</b>
</p>

### System Components

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **API Service** | FastAPI + Cloud Run | CV parsing, embedding generation, hybrid search with RRF |
| **Web UI** | Streamlit + Cloud Run | CV upload, job cards with scores, keyword highlighting |
| **Vector Database** | Supabase (PostgreSQL 16 + pgvector) | Structured job data and 384-dimensional embeddings |
| **Data Processing** | Databricks (PySpark/Delta Lake) + Polars (fallback) | ETL pipeline: HTML cleaning, JSON aggregation, embedding generation |
| **Cloud Storage** | Google Cloud Storage | Data lake: Bronze/Silver/Gold layers in Parquet format |
| **Orchestration** | Cloud Workflows + Cloud Scheduler | Nightly trigger: fetch → transform → ingest, with automatic retry |
| **Infrastructure** | Terraform | Full IaC for all GCP resources, secrets stored in Secret Manager |

### Data Pipeline

1. **Fetch** (`api-to-gcs-cf`) — France Travail API → GCS Bronze layer (Parquet)
2. **Transform** — Bronze → Silver (HTML cleaning, JSON aggregation) → Gold (384-dim embeddings)
   - **Primary:** Databricks (PySpark + Delta Lake)
   - **Fallback:** Cloud Function `pipeline-cf` (Polars), triggered if Databricks job has failed
3. **Ingest** (`ingest-db-cf`) — GCS Silver + Gold → Supabase (upsert), dead job cleanup
4. **Search** — CV upload → FastAPI embedding → hybrid pgvector + FTS + RRF → ranked results

### Hybrid Search Algorithm

The matching engine combines two ranking signals :

- **Semantic Similarity** — Cosine distance between CV and job embeddings via pgvector `<->` operator
- **Full-Text Search** — PostgreSQL `ts_rank` on weighted tsvector (title, description, competences), French stopwords removed

---

## Technology Stack

### Backend & API
- **FastAPI** — Async REST API, OpenAPI auto-generated
- **Python 3.12** — Primary language, strict type checking via mypy
- **Pydantic** — Request/response models, `pydantic-settings` for config
- **structlog** — Structured JSON logging across all services
- **slowapi** — Rate limiting (5 req/min on `/embed-cv`)
- **Prometheus** — `/metrics` endpoint via `prometheus-fastapi-instrumentator`

### Data & Storage
- **Supabase (PostgreSQL 16 + pgvector)** — Vector database with HNSW index
- **Google Cloud Storage** — Data lake (Bronze/Silver/Gold Parquet layers)
- **Databricks** — PySpark/Delta lake
- **Polars** — Lazy, multi-threaded ETL
- **DuckDB** — On-the-fly OLAP analytics on GCS Parquet files

### ML & Embeddings
- **Sentence Transformers** — `antoinelouis/french-me5-small` (36M params, 384-dim)
- **PyTorch** — CPU-only inference
- **Batch encoding** — Configurable batch size with progress tracking

### Infrastructure & DevOps
- **Terraform** — Full IaC: Cloud Run, Cloud Functions, GCS, Scheduler, Workflows, Secret Manager, Artifact Registry
- **Cloud Run** — Serverless containers for API and UI
- **Cloud Functions (2nd gen)** — Event-driven ETL steps
- **Cloud Workflows** — Orchestration with YAML-based DAG, automatic retry
- **Cloud Scheduler** — Cron trigger for nightly pipeline
- **Docker** — Multi-stage builds, non-root user, HEALTHCHECK

### CI/CD & Quality
- **GitHub Actions** — CI (ruff, mypy, bandit, pytest+cov, Docker build, Terraform validate) + CD (auto-deploy to Cloud Run via WIF)
- **ruff** — Linter + formatter, strict ruleset
- **mypy** — Strict type checking on `api/`
- **bandit** — Security linting
- **pytest** — 54 tests, 88% coverage (unit + integration + E2E/Playwright)
- **pre-commit** — ruff + gitleaks hooks
- **uv** — Package management, workspace monorepo

---

## Project Structure

```
CVEE/
├── api/                          # FastAPI search service
│   ├── app.py                    # Endpoints: /health, /embed-cv, /metrics
│   ├── embed_cv_search.py        # Hybrid search: embeddings + RRF
│   ├── utils.py                  # PDF extraction, DB queries, keyword highlight
│   ├── models.py                 # Pydantic models
│   ├── config.py                 # pydantic-settings
│   ├── stopwords.json            # French stopwords for FTS
│   ├── pyproject.toml
│   └── Dockerfile
│
├── ui/                           # Streamlit frontend
│   ├── app.py                    # Upload, progress bars, job cards
│   ├── pyproject.toml
│   └── Dockerfile
│
├── functions/                    # Cloud Functions (2nd gen)
│   ├── api-to-gcs/               # France Travail API → GCS (Bronze)
│   │   ├── main.py               # CF entry point
│   │   ├── ft_client.py          # OAuth2, pagination, export
│   │   └── pyproject.toml
│   ├── pipeline/                 # Bronze → Silver → Gold ETL
│   │   ├── main.py               # CF entry point
│   │   ├── core.py               # Polars ETL: HTML clean, embeddings, dedup
│   │   └── pyproject.toml
│   ├── ingest-db/                # GCS → Supabase + cleanup
│   │   ├── main.py               # CF entry point
│   │   ├── gcs_sync.py           # Silver + Gold ingestion (upsert)
│   │   ├── cleanup.py            # Dead job verification + deletion
│   │   └── pyproject.toml
│   ├── billing-guard/            # Auto-disable GCP billing (safety net)
│   │   ├── main.py
│   │   └── pyproject.toml
│   └── shared/                   # Shared config (Secret Manager)
│       └── config.py
│
├── databricks/                   # Spark/Delta Lake pipeline 
│   ├── silver.py                 # Bronze → Silver (HTML clean, JSON aggregation)
│   ├── gold.py                   # Silver → Gold (embeddings)
│   ├── export.py                 # Delta → GCS Parquet export
│   └── cleanup.py                # Dead job cleanup
│
├── infra/                        # Terraform
│   ├── main.tf                   # Provider, backend
│   ├── cloud_functions.tf        # 4 CFs gen2
│   ├── cloud_run.tf              # API + UI Cloud Run services
│   ├── storage.tf                # GCS bucket + lifecycle rules
│   ├── scheduler.tf              # Cloud Scheduler → Workflows
│   ├── workflows.tf              # ETL orchestration DAG
│   ├── secrets.tf                # Secret Manager
│   └── terraform.tfvars.example
│
├── pipeline/                     # DB migrations & init
│   ├── init_db.py               # Alembic runner
│   └── migrations/              # Alembic versions
│
├── tests/                        # Test suite
│   ├── conftest.py              # Fixtures, mocks
│   ├── test_api.py              # API endpoints
│   ├── test_embed_cv_search.py  # Embeddings + FTS + link verification
│   ├── test_utils.py            # PDF extraction, keywords, hybrid search
│   ├── test_pipeline_core.py    # Pipeline ETL logic
│   └── e2e/                     # Playwright E2E tests
│       └── test_upload_flow.py  # Upload → results verification
│
├── scripts/                      # Utilities
│   ├── backfill.py              # Historical data backfill
│   ├── analytics.py             # DuckDB OLAP on GCS Parquet
│   └── update_secrets.py        # Secret Manager helper
│
├── docker-compose.yml           # Local dev: PostgreSQL + API + UI
├── pyproject.toml               # Root workspace (uv, ruff, mypy, pytest)
├── justfile                     # Task runner: just lint, just test, just deploy
├── alembic.ini                  # Alembic configuration
└── .github/workflows/
    ├── ci.yaml                  # CI: lint, typecheck, test, Docker build, Terraform
    └── deploy.yaml              # CD: auto-deploy to Cloud Run via WIF
```

---

## License

MIT License — See [LICENSE](LICENSE) file for details.
