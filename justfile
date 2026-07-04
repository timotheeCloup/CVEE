# CVEE project commands — run `just` or `just --list`
set shell := ["bash", "-e", "-u", "-o", "pipefail", "-c"]

PROJECT := "cvee-20260208"
REGION := "us-east1"
CR_REGION := "europe-west1"
AR := "europe-west1-docker.pkg.dev/" + PROJECT + "/cvee"

# ── Cloud Run ──

# Deploy API to Cloud Run (Docker build + push + deploy)
deploy-api:
    docker build -t {{AR}}/cvee-api:latest -f api/Dockerfile api/
    docker push {{AR}}/cvee-api:latest
    gcloud run deploy cvee-api --image {{AR}}/cvee-api:latest --region {{CR_REGION}} --project {{PROJECT}} --allow-unauthenticated

# Deploy UI to Cloud Run (Docker build + push + deploy)
deploy-ui:
    docker build -t {{AR}}/cvee-ui:latest -f ui/Dockerfile ui/
    docker push {{AR}}/cvee-ui:latest
    gcloud run deploy cvee-ui --image {{AR}}/cvee-ui:latest --region {{CR_REGION}} --project {{PROJECT}} --allow-unauthenticated

# ── Cloud Functions (quick dev redeploy) ──

# Copy shared module into each CF dir (required before deploy)
cf-prep:
    rsync -r --delete functions/shared/ functions/api-to-gcs/shared/
    rsync -r --delete functions/shared/ functions/pipeline/shared/
    rsync -r --delete functions/shared/ functions/ingest-db/shared/

# Deploy api-to-gcs-cf
cf-api:
    gcloud functions deploy api-to-gcs-cf \
        --gen2 --region={{REGION}} --project={{PROJECT}} \
        --runtime=python312 --entry-point=api_to_gcs_cf \
        --trigger-http --allow-unauthenticated \
        --memory=512M --timeout=540s --max-instances=1 \
        --source=functions/api-to-gcs

# Deploy pipeline-cf
cf-pipeline:
    gcloud functions deploy pipeline-cf \
        --gen2 --region={{REGION}} --project={{PROJECT}} \
        --runtime=python312 --entry-point=pipeline_cf \
        --trigger-http --allow-unauthenticated \
        --memory=2048M --timeout=3600s --max-instances=1 \
        --source=functions/pipeline

# Deploy ingest-db-cf
cf-ingest:
    gcloud functions deploy ingest-db-cf \
        --gen2 --region={{REGION}} --project={{PROJECT}} \
        --runtime=python312 --entry-point=ingest_db_cf \
        --trigger-http --allow-unauthenticated \
        --memory=1024M --timeout=3600s --max-instances=1 \
        --source=functions/ingest-db

# Deploy all 3 CFs
cf-all: cf-prep cf-api cf-pipeline cf-ingest

# ── ETL Pipeline ──

# Run pipeline-cf on last N days (default: 1)
pipeline days="1":
    curl -X POST "https://{{REGION}}-{{PROJECT}}.cloudfunctions.net/pipeline-cf?days={{days}}"

# Manually trigger each CF independently (for testing):
# Step 1 — fetch FT API → GCS raw
etl-fetch:
    curl -X POST "https://{{REGION}}-{{PROJECT}}.cloudfunctions.net/api-to-gcs-cf?max_results=100"

# Step 2 — run bronze→silver→gold (bypasses Databricks check)
etl-process days="1":
    curl -X POST "https://{{REGION}}-{{PROJECT}}.cloudfunctions.net/pipeline-cf?days={{days}}"

# Step 3 — sync GCS → Supabase
etl-ingest:
    curl -X POST "https://{{REGION}}-{{PROJECT}}.cloudfunctions.net/ingest-db-cf"

# All 3 steps sequentially (sans sleep, sans Databricks check)
etl-full:
    just etl-fetch
    just etl-process
    just etl-ingest

# Check if Databricks produced today's silver+gold in GCS
check-databricks:
    @today=$$(date +%Y%m%d); \
    silver=$$(gsutil ls "gs://cvee-20260208/jobs_silver/jobs_silver_$${today}*" 2>/dev/null | wc -l); \
    gold=$$(gsutil ls "gs://cvee-20260208/jobs_gold/jobs_gold_$${today}*" 2>/dev/null | wc -l); \
    echo "Databricks output today:"; \
    echo "  Silver files: $$silver"; \
    echo "  Gold files:   $$gold"; \
    [ "$$silver" -gt 0 ] && [ "$$gold" -gt 0 ] && echo "  → Status: DONE" || echo "  → Status: not yet"
# Databricks cron should be set to 21:01 UTC (runs during the 30min sleep).
# If Databricks wrote silver+gold to GCS, pipeline-cf skips automatically.
# Trigger the full ETL Cloud Workflow (api-to-gcs → wait 30min → pipeline → ingest-db)
# Databricks cron should be set to 21:01 UTC (runs during the 30min sleep).
# If Databricks wrote silver+gold to GCS, pipeline-cf skips automatically.
workflow:
    gcloud workflows executions run cvee-etl-pipeline --location={{REGION}} --project={{PROJECT}}

# Backfill historical data
backfill date_min date_max:
    uv run python scripts/backfill.py --date-min {{date_min}} --date-max {{date_max}}

# ── Dev ──

# Ruff lint
lint:
    uv run ruff check .

# Ruff format
fmt:
    uv run ruff format .

# Run tests
test:
    uv run pytest tests/

# Lint + format check + test (CI)
check:
    uv run ruff check .
    uv run ruff format . --check
    uv run pytest tests/

# Test API health
health:
    curl https://cvee-api-4ihvpv7gha-ew.a.run.app/health

# ── Logs ──

# Read CF logs (just logs api 20)
logs name limit="20":
    gcloud functions logs read {{name}}-cf --region={{REGION}} --project={{PROJECT}} --limit={{limit}}

# ── Secrets ──

# Update Secret Manager from .env
secrets:
    uv run python scripts/update_secrets.py
    gcloud secrets versions add cvee-secrets --project={{PROJECT}} --data-file=/tmp/cvee-secrets.json --quiet

# ── Terraform ──

# Terraform plan
tf-plan:
    cd infra && terraform plan

# Terraform apply
tf-apply:
    cd infra && terraform apply

# Terraform outputs
tf-output:
    cd infra && terraform output
