# CVEE (CV Embedding Engine)

AI-powered job matching: upload CV (PDF) → semantic + full-text hybrid search → ranked job offers.

---

## Objectif principal : Minimiser les coûts

Le projet vise le **zéro facturation** en exploitant les **Always Free Tiers** GCP. Des compromis sont acceptables (moins de confort, plus de code) si ça réduit la facture.

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
| **Cloud Storage** | 5 GB-months Standard | Parquet raw/silver/gold (~50 Mo max) | ✅ Zéro risque |
| **BigQuery** | 1 TiB requêtes + 10 GiB stockage | Optionnel (remplace Databricks) | ✅ Zéro risque |
| **Artifact Registry** | 0,5 GiB stockage | Images Docker des CF/Cloud Run | ⚠️ Peut dépasser si beaucoup de builds → `gcloud artifacts docker images delete` régulier |

### Compute

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Cloud Run** | 2M requêtes + 360 000 GiB·s mémoire + 180 000 vCPU·s + 1 GiB egress | API (FastAPI) + UI (Streamlit) + pipeline job | ✅ Large marge (qqs centaines de requêtes/jour max) |
| **Cloud Run functions** | 2M invocations | `api-to-gcs-cf`, `ingest-db-cf`, `billing-guard` | ✅ 3 invocations/jour → ~90/mois |
| **Compute Engine** | 1 VM e2-micro (us-central1/us-east1/us-west1) | Non utilisé | — |

### Réseau & Messaging

| Service | Limite free/mois | Utilisation dans CVEE | Risque dépassement |
|---------|------------------|-----------------------|---------------------|
| **Pub/Sub** | 10 Go messages | Budget alerts → `billing-guard` | ✅ Quelques Ko/mois |
| **Cloud Scheduler** | 3 jobs | 2 jobs (api-to-gcs, ingest-db) | ✅ OK |

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
├── pipeline/                   # Cloud Run Job (batch ETL)
│   ├── pyproject.toml
│   ├── pipeline.py             # Bronze→Silver→Gold
│   └── init_db.py
├── functions/                  # Cloud Functions (2nd gen)
│   ├── api-to-gcs/             # FT API → GCS (nightly)
│   │   ├── pyproject.toml, main.py, ft_client.py
│   ├── ingest-db/              # GCS → Supabase + cleanup
│   │   ├── pyproject.toml, main.py, gcs_sync.py, cleanup.py
│   └── billing-guard/          # Auto-disable billing
│       ├── pyproject.toml, main.py
├── tests/                      # pytest
├── .github/workflows/ci.yaml   # Ruff + pytest + Docker build
└── assets/                     # Diagrams, demo media
```

**Tooling cutting-edge :** `uv` (astral.sh), `ruff` (lint + format, 0 errors), `pytest`, GitHub Actions CI.

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

### 🚧 Étape 2 — Pipeline Python remplaçant Databricks (en cours)
- 🚧 `pipeline/pipeline.py` créé (Bronze→Silver→Gold en Python pur)
- 🚧 Problème à fixer : embedding lent (~458s) + crash serialisation JSON
- 📋 À déployer comme **Cloud Run Job** déclenché par Cloud Scheduler
- 📋 Adapter au **modèle d'embedding multilingue** (plus besoin de traduction → zéro coût)
- 📋 2 Cloud Functions restent en backup

### ✅ Étape 3 — uv + pyproject.toml + restructure (cutting-edge tooling)
- ✅ `uv` remplace pip (17x plus rapide, lockfile reproductible)
- ✅ 7 `pyproject.toml` (1 root + 6 sub-projects), plus de `requirements.txt`
- ✅ Restructure : `functions/` (3 CFs), `pipeline/`, `api/`, `ui/`
- ✅ Ruff (lint + format) configuré — **0 erreurs**
- ✅ CI GitHub Actions : Ruff + pytest + build Docker
- ✅ `docker-compose.yml` déplacé à la racine
- ✅ `tests/` directory + pytest config
- ✅ Python 3.12 standardisé

### 📋 Étape 4 — Optimiser les Dockerfiles
- Base `python:3.12-slim`, multi-stage, .dockerignore

### 📋 Étape 5 — Tests E2E Playwright
- Scénario UI Streamlit automatisé

### 📋 Étape 6 — CD : Déploiement automatique
- `gcloud run deploy` + `gcloud run jobs deploy` sur push main
