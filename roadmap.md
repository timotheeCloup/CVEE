# CVEE — Roadmap & Objectifs

## Objectif principal : Minimiser les coûts

Le projet vise le **zéro facturation** en exploitant les **Always Free Tiers** GCP. Des compromis sont acceptables (moins de confort, plus de code) si ça réduit la facture.

## Objectif professionnel : Vitrine Data Engineer

Ce projet sert de **portfolio technique** pour démontrer la maîtrise des technologies modernes attendues d'un Data Engineer :

| Domaine | Technologies |
|---------|-------------|
| **Infrastructure as Code** | Terraform (HCL), GCP provider |
| **Orchestration** | Cloud Scheduler, Cloud Functions gen2 |
| **ETL/ELT** | Python, pandas, PyArrow, Parquet |
| **Vector Search** | sentence-transformers, pgvector, embeddings français |
| **Stockage** | GCS (Data Lake), Supabase (PostgreSQL) |
| **CI/CD** | GitHub Actions, ruff, pytest |
| **Tooling** | uv, pyproject.toml (PEP 621), ruff |
| **Monitoring** | Billing guard, Pub/Sub, budget alerts |

---

## Roadmap (progression actuelle)

### ✅ Étape 0 — Billing Guard & Sécurité facturation
- ✅ Budget `cvee-budget-guard` 5€ → Pub/Sub → CF `billing-guard` → coupe la facturation
- ✅ Seuil abaissé de 300€ → 5€ (alerte 50%/90%/100%)
- ✅ 2 budgets redondants supprimés
- ✅ Test réel : coupure confirmée (`billingEnabled=False`)
- ✅ Cloud Translation API désactivée (prévention coûts)
- ✅ La CF est 100% Python stdlib (zéro dépendance externe, 128 MiB)

### ✅ Étape 1 — Secrets consolidés + Migration GCP Storage (S3 → GCS)
- ✅ 12 secrets individuels → 1 secret JSON `cvee-secrets`
- ✅ AWS S3 (`boto3`/`s3fs`) → GCP Cloud Storage (`gcsfs`, ADC natif)
- ✅ Cloud Functions `api-to-gcs-cf` + `ingest-db-cf` déployées et testées
- ✅ Scheduler jobs redirigés vers les nouvelles CF
- ✅ Bucket `gs://cvee-20260208` créé

### ✅ Étape 2 — Databricks pipeline migré S3→GCS + embedding multilingue
- ✅ Databricks notebooks mis à jour : S3 → GCS
- ✅ Suppression de `ai_translate` (API payante Google Translate) → modèle local
- ✅ `pipeline/pipeline.py` conservé comme fallback local (sans Spark)
- ✅ Backfill historique : `scripts/backfill.py` + paramètres `date_min`/`date_max`

### ✅ Étape 3 — uv + pyproject.toml + restructure (cutting-edge tooling)
- ✅ `uv` remplace pip (lockfile reproductible)
- ✅ 7 `pyproject.toml` (1 root + 6 sub-projects)
- ✅ Restructure : `functions/` (3 CFs), `pipeline/`, `api/`, `ui/`
- ✅ Ruff (lint + format) configuré — 0 erreurs
- ✅ CI GitHub Actions : Ruff + pytest + build Docker
- ✅ `docker-compose.yml` déplacé à la racine
- ✅ Python 3.12 standardisé

### ✅ Étape 4 — Migration us-east1 + nettoyage legacy
- ✅ Pipeline (CFs + Scheduler + GCS) migré en `us-east1` (Always Free Storage)
- ✅ Nettoyage legacy GCP : doublons europe-west1, bucket eu-north-1
- ✅ `pyproject.toml` CFs au format PEP 621 (compatible Cloud Functions buildpack)

### ✅ Étape 5 — Backfill historique
- ✅ `scripts/backfill.py` : import par plage de dates, chunking mensuel
- ✅ `api-to-gcs-cf` supporte `?date_min=&date_max=` en HTTP
- ✅ Timeout sur toutes les requêtes HTTP FT API
- ✅ Test réel : 3000 offres en 2 minutes

### ✅ Étape 6 — Terraform IaC + Pipeline GCP
- ✅ Infrastructure as Code complète (CFs, Scheduler, GCS, Secrets, Cloud Run, Artifact Registry)
- ✅ `functions/pipeline/` : CF `pipeline-cf` (bronze→silver→gold) à 21:30
- ✅ Modèle `antoinelouis/french-me5-small` (36M params, FR, 384-dim)
- ✅ Déduplication auto par `id` sur les reruns/backfills
- ✅ 3 scheduler jobs : 21:00 → 21:30 → 22:00

### ✅ Étape 7 — Optimisation coûts & fiabilisation (06/2026)
- ✅ `cvee-api` : CPU 2→1 vCPU (always free tier compatible)
- ✅ Cleanup policies Artifact Registry : keepCount=1 sur les 3 repos
- ✅ Suppression manuelle des 34 anciennes images Docker → stockage ÷ 3
- ✅ GCS lifecycle rule : suppression auto après 180 jours sur jobs_raw/silver/gold
- ✅ `ingest-db` : purge auto des offres >30j (déjà en place dans `gcs_sync.py`)
- ✅ `pipeline-cf` timeout 600→3600s (gros datasets)
- ✅ UI : barre de progression centrée (cold start + analyse)
- ✅ API : log du temps de chargement modèle (cold start)
- ✅ Nettoyage : 13 fichiers/dossiers deprecated supprimés (`src/cf-api-to-s3/`, `src/cf-ingest-db/`, `docker/`, etc.)
- ✅ Hybrid search scoring amélioré (FTS_WEIGHT, linear mapping 0.15-0.85)

---

## À faire — Tier 1 : Sécurité & Corrections

- [ ] **Rotation credentials `.env`** — FT_CLIENT_ID, FT_CLIENT_SECRET, SB_PASSWORD en clair sur le disque
- [ ] **Créer `.env.example`** avec placeholders, supprimer `.env` du dépôt local
- [ ] **Ajouter `gitleaks` pre-commit hook** — scan anti-fuite de secrets
- [ ] **`model.encode()` → `asyncio.to_thread()`** — bloquant dans l'event loop FastAPI
- [ ] **`init_db.py` DDL destructif** — `DROP TABLE CASCADE` derrière un flag `--force`
- [ ] **Corriger incohérence scheduler** — `scheduler.tf` dit 22:00, `deploy.sh` dit 23:30
- [ ] **Fixer `db_password` non marqué `sensitive`** dans `variables.tf`
- [ ] **Fixer VSCode `typeCheckingMode: off`** → `basic`

## À faire — Tier 2 : Qualité code (vitrine)

- [ ] **Mypy `strict=true`** — zéro typing statique aujourd'hui, 🚩 critique pour portfolio
- [ ] **Pydantic models** — remplacer les `dict` bruts par des `BaseModel` (API request/response, config)
- [ ] **Dataclasses / TypedDict** — remplacer les `dict` internes (`verify_job_link`, résultats de recherche)
- [ ] **`pydantic-settings`** — remplacer `load_dotenv()` + `os.getenv()` éparpillés
- [ ] **`psycopg` v3 async** → remplacer `psycopg2` synchrone (ou `asyncpg`)
- [ ] **DRY `get_config()`** — dupliqué 4 fois (api-to-gcs, pipeline, ingest-db, billing-guard) → module partagé
- [ ] **DRY pipeline** — `pipeline/pipeline.py` duplique 80% de `functions/pipeline/core.py`
- [ ] **Structured logging** — `structlog` avec request ID, trace context
- [ ] **`.pre-commit-config.yaml`** — hooks ruff + mypy + gitleaks

## À faire — Tier 3 : Tests

- [ ] **Tests API** — `TestClient` FastAPI pour `/health` et `/embed-cv`
- [ ] **Tests unitaires** — `extract_text_from_pdf`, `clean_text_for_fts`, `linear_mapping`, `clean_html`, déduplication
- [ ] **Tests pipeline** — `_extract_field`, `serialize_json_col`, `_deduplicate`
- [ ] **`conftest.py`** — fixtures DB, mock GCS, mock FT API, sample PDF
- [ ] **Coverage threshold** — 80%+ avec `pytest-cov` + `fail_under` dans CI
- [ ] **Tests E2E** — Playwright sur l'UI Streamlit (upload PDF → résultats)

## À faire — Tier 4 : DevOps & Docker

- [ ] **Docker multi-stage builds** — séparer build deps / runtime
- [ ] **Docker non-root user** — `USER app` partout
- [ ] **Docker HEALTHCHECK** — curl `/health` dans les Dockerfiles
- [ ] **`.dockerignore` UI** — inexistant
- [ ] **CI : ajouter terraform validate + fmt check**
- [ ] **CI : ajouter mypy, pytest-cov, bandit/trivy**
- [ ] **CI : build Docker UI en plus de l'API**
- [ ] **CD : `gcloud run deploy` automatique sur push main**

## À faire — Tier 5 : Documentation & Polish

- [ ] **README.md** — obsolète (parle de S3, Databricks, Python 3.11) → réécrire architecture actuelle
- [ ] **Docstring** — systématiser Args/Returns sur toutes les fonctions publiques
- [ ] **Alembic migrations** — remplacer `init_db.py` DDL hardcodé
- [ ] **`expose_metrics` endpoint** — Prometheus / OpenTelemetry
- [ ] **Rate limiting** sur `/embed-cv` (coûteux en compute)
- [ ] **i18n** — externaliser les strings français du UI

---

## Stack technique — État des lieux

| Domaine | Actuel | Cible |
|---------|--------|-------|
| Package manager | `uv` ✅ | — |
| Linter/formatter | `ruff` ✅ | + règles `T`, `ANN`, `RUF` |
| Type checker | ❌ aucun | `mypy --strict` |
| API models | `dict` brut | Pydantic `BaseModel` |
| DB driver | `psycopg2` (sync) | `psycopg` v3 / `asyncpg` |
| Config | `os.getenv()` | `pydantic-settings` |
| Logging | `logging` basique | `structlog` |
| Migrations DB | `init_db.py` DDL | Alembic |
| Tests | 0 vrais tests | 80%+ coverage |
| CI | ruff + pytest + docker | + mypy + terraform + security scan |
| CD | manuel | auto sur push main |
