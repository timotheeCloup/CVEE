# CVEE (CV Embedding Engine)

AI-powered job matching: upload CV (PDF) → semantic + full-text hybrid search → ranked job offers.

---

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

### ⚠️ RÈGLE #1 : Tout en `us-east1` (pipeline)

Le Always Free Tier **Storage** de GCP ne couvre que 3 régions US : `us-east1`, `us-west1`, `us-central1`. Un bucket en Europe est payant (même si quelques centimes). Toute l'infrastructure pipeline (CFs, Scheduler, GCS) tourne donc en `us-east1`.

| Composant | Région | Raison |
|-----------|--------|--------|
| Cloud Functions + Scheduler + GCS | `us-east1` | Always Free Storage |
| Cloud Run (API + UI) | `europe-west1` | Latence utilisateur, pas de quota storage |
| Databricks | externe | Lit GCS via `gs://` |
| Supabase | externe | Free tier propre |

### ⚠️ RÈGLE ABSOLUE : Rien ne doit générer de facturation

- **Aucun appel à une API payante** (Google Translate, Vertex AI, etc.) ne doit être dans le code sans validation explicite.
- Tout service utilisé doit être dans l'**Always Free Tier** GCP ou avoir un free tier tiers (Supabase, GitHub).
- Les instructions pour lancer un job/script payant doivent être données à l'utilisateur, qui les exécute **manuellement** après validation.

## Historique des incidents de facturation
| Date | Service | Montant | Cause | Correction |
|------|---------|---------|-------|------------|
| 2026-06-21 | Google Cloud Translation API | 242€ | `pipeline.py` + `embed_cv_search.py` traduisaient FR→EN via API payante (~2100 textes × 2000 chars) | ✅ API désactivée dans GCP + code migré vers modèle multilingue local |
| 2026-06-21 | — | 0€ | Billing Guard déployé et testé avec succès : billing coupé automatiquement | ✅ Garde-fou actif |

---

## Services GCP Always Free Tier — le tableau complet

Source : [Google Cloud Free Tier](https://cloud.google.com/free/docs/gcp-free-tier).  
Sauf indication contraire, les limites sont **mensuelles** (reset au 1er du mois).

### Stockage & Données

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Cloud Storage** | 5 GB-months Standard | Parquet raw/silver/gold (~50 Mo max) | ✅ Zéro risque — **⚠️ bucket en `us-east1` obligatoire** (pas de free tier en Europe) |
| **BigQuery** | 1 TiB requêtes + 10 GiB stockage | Optionnel (remplace Databricks) | ✅ Zéro risque |
| **Artifact Registry** | 0,5 GiB stockage | Images Docker des CF/Cloud Run | ⚠️ Peut dépasser si beaucoup de builds → `gcloud artifacts docker images delete` régulier |

### Compute

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Cloud Run** | 2M requêtes + 360 000 GiB·s mémoire + 180 000 vCPU·s + 1 GiB egress | API (FastAPI) + UI (Streamlit) + pipeline job | ✅ Large marge (qqs centaines de requêtes/jour max) |
| **Cloud Run functions** | 2M invocations | `api-to-gcs-cf`, `pipeline-cf`, `ingest-db-cf`, `billing-guard` | ✅ 4 invocations/jour → ~120/mois |
| **Compute Engine** | 1 VM e2-micro (us-central1/us-east1/us-west1) | Non utilisé | — |

### Réseau & Messaging

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Pub/Sub** | 10 Go messages | Budget alerts → `billing-guard` | ✅ Quelques Ko/mois |
| **Cloud Scheduler** | 3 jobs | 3 jobs (api-to-gcs, pipeline, ingest-db) | ✅ OK |

### CI/CD

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Cloud Build** | 2 500 build-minutes (machine e2-standard-2) | Déploiements CF + Cloud Run | ✅ Chaque build ~2-3 min |
| **Cloud Deploy** | 1 pipeline actif par compte de facturation | Optionnel (CD) | ✅ OK |

### Sécurité & Observabilité

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Secret Manager** | 6 versions de secrets (tous secrets confondus) | 1 secret `cvee-secrets`, ~3-4 versions/an | ✅ |
| **Cloud KMS** | 100 clés actives + 10 000 opérations (Autokey) | Non utilisé | — |
| **Cloud Logging** (ex-Operations) | 50 GiB logs par projet | Logs CF + Cloud Run | ✅ Quasi-aucun volume |
| **Cloud Monitoring** | 1M time series API read calls + métriques gratuites GCP | Non utilisé | — |

### Services **NON gratuits** — à éviter absolument

| Service | Coût | Présent dans CVEE ? | Action |
|---------|------|---------------------|--------|
| **Cloud Translation API** | 20$/million caractères | Était dans `pipeline.py` + `embed_cv_search.py` | ✅ Supprimé — modèle multilingue local |
| **Vertex AI** | ~0,05-5$/appel LLM | Non | — |
| **Cloud SQL** | ~7-50$/mois minimum | Non (on utilise Supabase free tier) | — |
| **Dataproc** (Spark) | ~0,01$/vCPU·h + frais cluster | Était dans Databricks | ✅ Abandonné — remplacé par pipeline Python local |
| **Cloud Vision / NLP API** | 1,50$/1000 unités | Non | — |

### Services externes (hors GCP)

| Service | Free Tier | Utilisation dans CVEE | Risque |
|---------|-----------|-----------------------|--------|
| **Supabase** | Jusqu'à 500 Mo DB, 2 projets, 5 Go bande passante | Base PostgreSQL + pgvector | ✅ Large marge |
| **GitHub Actions** | 2000 minutes/mois (gratuit) | CI/CD | ✅ |
| **GitHub** (hébergement code) | Gratuit | Code source | ✅ |
| **Hugging Face** (modèles) | Gratuit (téléchargement) | Modèle d'embedding | ✅ Modèle ~120 Mo, téléchargé 1x |

---

## 🔒 Billing Guard — Coupure automatique de facturation (✅ testé & validé)

Un mécanisme de sécurité coupe automatiquement la facturation si le budget est dépassé.

```
Budget GCP (5€) → Pub/Sub → Cloud Function billing-guard → unlink billing
```

**Fonctionnement complet :**
1. Un **budget** GCP de 5€/mois (service gratuit) envoie des notifications Pub/Sub à 50%, 90% et 100% du seuil
2. À 100% du seuil, la **Cloud Function** `billing-guard` appelle l'API Cloud Billing pour désactiver la facturation
3. Le projet continue d'exister mais plus aucun service payant peut tourner — les ressources s'arrêtent
4. Pour réactiver : `gcloud billing projects link cvee-20260208 --billing-account=016979-43CAAA-865F35`
5. **Emails** : GCP Budgets envoie automatiquement un email au Billing Admin (toi) à chaque seuil

**⚠️ Seuil actuel pour Juin 2026** : **250€** (au lieu de 5€, car déjà 242€ consommés). À remettre à 5€ le 1er juillet.

**Stack technique :**
- CF écrite en **Python stdlib pur** (urllib + metadata server, zéro dépendance externe)
- 128 MiB mémoire, timeout 30s, 1 instance max
- Service account dédié : `billing-guard@cvee-20260208.iam.gserviceaccount.com`

**Prérequis IAM (tous vérifiés ✅) :**
- `roles/billing.admin` sur le compte de facturation 016979-43CAAA-865F35
- `roles/billing.projectManager` sur le projet cvee-20260208
- `roles/run.invoker` (allUsers) sur le service Cloud Run `billing-guard`

**Déploiement :**
```bash
./infra/setup_billing_guard.sh
# Puis manuellement : IAM billing.admin sur le compte de facturation
# Et : gcloud run services add-iam-policy-binding billing-guard --member=allUsers --role=roles/run.invoker
```

**Test réel (21 juin 2026 ✅) :**
```bash
gcloud pubsub topics publish budget-alerts --project=cvee-20260208 \
  --message='{"budgetDisplayName":"cvee-budget-guard","costAmount":250,"budgetAmount":5,"alertThresholdExceeded":1.0,"currencyCode":"EUR"}'
# → billingEnabled = False (confirmé)

---

## Project Layout (`refacto`)

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
├── pipeline/                   # Fallback local ETL (si pas de Databricks)
│   ├── pyproject.toml
│   ├── pipeline.py             # Bronze→Silver→Gold
│   └── init_db.py
├── src/                        # Databricks notebooks (S3→GCS migré)
│   ├── jobs_ingestion_silver.ipynb   # raw → Delta silver
│   ├── jobs_embeddings_gold.ipynb    # silver → embeddings → Delta gold
│   └── jobs_export.ipynb             # Delta → GCS Parquet
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
├── scripts/                     # Scripts utilitaires
│   ├── deploy.sh                # DEPRECATED — use Terraform
│   └── backfill.py              # Historical backfill (month by month)
├── tests/                       # pytest
├── .github/workflows/ci.yaml   # Ruff + pytest + Docker build
└── assets/                     # Diagrams, demo media
```

**Tooling cutting-edge :** `uv` (astral.sh), `ruff` (lint + format, 0 errors), `pytest`, GitHub Actions CI.

**Quickstart dev :**
## Deploy + Backfill

```bash
# Prérequis : créer le bucket de state Terraform (une seule fois)
gsutil mb -l us-east1 gs://cvee-20260208-tfstate

# Déployer l'infrastructure
cd infra && terraform init && terraform apply

# Backfill historique
uv run python scripts/backfill.py --date-min 2026-01-01 --date-max 2026-06-30

# Pipeline manuel (retraite les 7 derniers jours de raw)
curl -X POST "$(terraform output -raw pipeline_url)?days=7"
```

## Cleanup Legacy GCP Resources

```bash
# CFs + Scheduler legacy (S3)
gcloud functions delete api-to-s3-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete api-to-s3-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
gcloud storage rm -r gs://cvee-bucket-eu-north-1

# Doublons europe-west1 (remplacés par us-east1)
gcloud functions delete api-to-gcs-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud functions delete ingest-db-cf --region=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete api-to-gcs-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
gcloud scheduler jobs delete ingest-db-scheduler --location=europe-west1 --project=cvee-20260208 --quiet
```

**Quickstart dev :**
```bash
uv sync --group dev      # install everything
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest tests/     # test
uv run python api/app.py # run API
```

**Legacy dirs** (`src/`, `infra/`, `docker/`) — conservés le temps de la migration complète.

---

## Roadmap (progression actuelle)

### ✅ Étape 0 — Billing Guard & Sécurité facturation
- ✅ Budget `cvee-budget-guard` 5€ → Pub/Sub → CF `billing-guard` → coupe la facturation
- ✅ 2 budgets redondants supprimés (`budget-stop-signal`, `budget_stop_signal_06-26`)
- ✅ Seuil remonté à 250€ pour le mois en cours (juin, 242€ déjà consommés)
- ✅ Test réel : coupure confirmée (`billingEnabled=False`)
- ✅ Cloud Translation API désactivée (prévention coûts)
- ✅ La CF est 100% Python stdlib (zéro dépendance externe, 128 MiB)

### ✅ Étape 1 — Secrets consolidés + Migration GCP Storage (S3 → GCS)
- ✅ 12 secrets individuels → 1 secret JSON `cvee-secrets` (6 versions free = large)
- ✅ AWS S3 (`boto3`/`s3fs`) → GCP Cloud Storage (`gcsfs`, ADC natif)
- ✅ Cloud Functions `api-to-gcs-cf` + `ingest-db-cf` déployées et testées
- ✅ Scheduler jobs redirigés vers les nouvelles CF
- ✅ Bucket `gs://cvee-20260208` créé, données raw déjà présentes

### ✅ Étape 2 — Databricks pipeline migré S3→GCS + embedding multilingue
- ✅ Databricks notebooks mis à jour : S3 → GCS (`gs://cvee-20260208`)
- ✅ Suppression de `ai_translate` (API payante Google Translate) → modèle multilingue local
- ✅ Modèle `paraphrase-multilingual-MiniLM-L12-v2` (120 Mo, 384-dim, CPU, FR inclus)
- ✅ `pipeline/pipeline.py` conservé comme fallback local (sans Spark)
- ✅ Backfill historique : `scripts/backfill.py` + paramètres `date_min`/`date_max` dans la CF
- ✅ 2 Scheduler jobs : 21:00 (api-to-gcs-cf) + 23:30 (ingest-db-cf), Databricks entre les deux

### ✅ Étape 3 — uv + pyproject.toml + restructure (cutting-edge tooling)
- ✅ `uv` remplace pip (17x plus rapide, lockfile reproductible)
- ✅ 7 `pyproject.toml` (1 root + 6 sub-projects), plus de `requirements.txt`
- ✅ Restructure : `functions/` (3 CFs), `pipeline/`, `api/`, `ui/`
- ✅ Ruff (lint + format) configuré — **0 erreurs**
- ✅ CI GitHub Actions : Ruff + pytest + build Docker
- ✅ `docker-compose.yml` déplacé à la racine
- ✅ `tests/` directory + pytest config
- ✅ Python 3.12 standardisé

### ✅ Étape 4 — Migration us-east1 + nettoyage legacy
- ✅ Pipeline (CFs + Scheduler + GCS) migré en `us-east1` (Always Free Storage)
- ✅ Nettoyage legacy GCP : `api-to-s3-cf`, `api-to-s3-scheduler`, `cvee-bucket-eu-north-1`, doublons europe-west1
- ✅ `deploy.sh` upload réellement le secret (plus juste un echo) et filtre les clés `AWS_*`
- ✅ `.env` nettoyé des credentials AWS
- ✅ `pyproject.toml` CFs au format PEP 621 (compatible Cloud Functions buildpack)
- ✅ `GCS_BUCKET_NAME` uniformisé (plus de confusion `GCP_BUCKET_NAME`)

### ✅ Étape 5 — Backfill historique
- ✅ `scripts/backfill.py` : import par plage de dates, chunking mensuel
- ✅ `api-to-gcs-cf` supporte `?date_min=&date_max=` en HTTP
- ✅ `ft_client.py` : deux modes (daily `publieeDepuis` / backfill `minDateCreation`)
- ✅ Timeout sur toutes les requêtes HTTP FT API
- ✅ Test réel : 3000 offres en 2 minutes (19-20 juin 2026)

### ✅ Étape 6 — Terraform IaC + Pipeline GCP
- ✅ Infrastructure as Code : 6 fichiers Terraform (CFs, Scheduler, GCS, Secrets)
- ✅ `functions/pipeline/` : CF `pipeline-cf` (bronze→silver→gold) à 21:30
- ✅ Modèle `antoinelouis/french-me5-small` (36M params, FR, ~40 Mo, 384-dim)
- ✅ Déduplication auto par `id` sur les reruns/backfills (`?days=N`)
- ✅ Hash-based deploy : le bash deploy.sh skip les CFs inchangées (fallback, déprécié)
- ✅ `terraform.tfvars` pour les secrets, `.gitignore` pour pas les commit
- ✅ 3 scheduler jobs : 21:00 → 21:30 → 23:30

### 📋 Étape 7 — Optimiser les Dockerfiles
- Base `python:3.12-slim`, multi-stage, .dockerignore

### 📋 Étape 8 — Tests E2E Playwright
- Scénario UI Streamlit automatisé

### 📋 Étape 9 — CD : Déploiement automatique
- `gcloud run deploy` + `gcloud run jobs deploy` sur push main

---

## 🔀 Pipeline : GCS vs Databricks — analyse

Le pipeline doit transformer `jobs_raw/` (bronze) → `jobs_silver/` + `jobs_gold/`.

### Option A : Databricks (actuel)

```
FT API → api-to-gcs-cf → GCS raw → Databricks notebooks → GCS silver/gold → ingest-db-cf → Supabase
```

| ✅ Pour | ❌ Contre |
|---------|----------|
| Delta Lake (versioning, time travel, schema evolution) | **Payant** — DBU + cluster, pas de free tier |
| Spark : parallélisme, passage à l'échelle sur gros volumes | Cluster = coût même à vide |
| 3 notebooks rodés, déjà migrés S3→GCS | Obligation d'avoir un cloud provider backend (AWS/Azure/GCP) |
| intègre `dbutils.fs.ls` pour lister les fichiers GCS | Pas de version gratuite utilisable en prod |
| | Scheduling limité en version Standard |

### Option B : pipeline.py (Python local, zéro coût)

```
FT API → api-to-gcs-cf → GCS raw → pipeline.py (Cloud Run Job) → GCS silver/gold → ingest-db-cf → Supabase
```

| ✅ Pour | ❌ Contre |
|---------|----------|
| **100% gratuit** (Cloud Run free tier) | Pas de Delta Lake (Parquet simple, pas de versioning) |
| Même code que Databricks, sans Spark | Single-machine, pas de parallélisme |
| Pas de dépendance externe | 3000 offres = ~5 min CPU (acceptable en batch nocturne) |
| Déploiement unifié via deploy.sh (3 scheduler jobs) | Moins "propre" que Delta tables |
| Modèle `antoinelouis/french-me5-small` (36M params) idéal car léger | |

### Modèle d'embedding cible : `antoinelouis/french-me5-small`

| Modèle | Params | Taille | Dims | Langues | Note |
|--------|--------|--------|------|---------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` (actuel) | 118M | 120 Mo | 384 | 50+ (dont FR) | Multilingue, correct pour FR |
| `antoinelouis/french-me5-small` (cible) | **36M** | **~40 Mo** | 384 | FR uniquement | 70% plus léger, FR optimisé, pruned de `multilingual-e5-small` |

### Recommandation

**→ Option B retenue** : `pipeline-cf` (Cloud Function, 2 Go, embeddings CPU) + 3 scheduler jobs = 0€, 3 jobs free tier.

Databricks reste dispo en fallback (notebooks à jour avec `gs://` et `french-me5-small`), mais le pipeline GCP tourne en autonome sans dépendance externe.

Rappel coût pipeline mensuel : ~28 800 GiB·s → **8%** du quota free tier (360 000 GiB·s).
