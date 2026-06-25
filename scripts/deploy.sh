#!/bin/bash
# DEPRECATED — use Terraform (cd infra && terraform apply)
# Kept as fallback for quick manual deploys during development.
# Usage:
#   bash scripts/deploy.sh              # deploy only changed CFs
#   bash scripts/deploy.sh --force      # force redeploy all
#   bash scripts/deploy.sh <name>       # deploy specific CF (api-to-gcs|pipeline|ingest-db)
set -e

FORCE=false
TARGET=""

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        api-to-gcs|pipeline|ingest-db) TARGET="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ID="cvee-20260208"
REGION="us-east1"
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"
SRC_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== CVEE Deployment ==="
echo "Project: $PROJECT_ID  |  Region: $REGION"
echo ""
echo "Architecture:"
echo "  Scheduler 21:00  →  api-to-gcs-cf     →  GCS jobs_raw/"
echo "  Scheduler 21:30  →  pipeline-cf       →  GCS jobs_silver/ + jobs_gold/"
echo "  Scheduler 23:30  →  ingest-db-cf      →  Supabase + cleanup"
echo ""

# ------------------------------------------------------------------
# Step 1: Single secret
# ------------------------------------------------------------------
echo "Step 1: Setting up cvee-secrets (single JSON)..."
SECRET_ID="cvee-secrets"

if gcloud secrets describe "$SECRET_ID" --project="$PROJECT_ID" &>/dev/null; then
    echo "  ✓ Secret '$SECRET_ID' already exists"
else
    echo "  Creating secret '$SECRET_ID'..."
    gcloud secrets create "$SECRET_ID" --project="$PROJECT_ID"
fi

# Add new version with current .env values (if .env exists)
if [ -f "$SRC_BASE/.env" ]; then
    echo "  Updating secret version from .env file..."
    SECRET_JSON=$(python3 - "$SRC_BASE/.env" << 'PYEOF'
import sys, json
from pathlib import Path

env_file = Path(sys.argv[1])
config = {}
for line in env_file.read_text().strip().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, val = line.partition("=")
    key, val = key.strip(), val.strip().strip("'\"")
    # Filter out legacy AWS keys
    if key.startswith("AWS_"):
        continue
    config[key] = val
print(json.dumps(config, ensure_ascii=False))
PYEOF
    )
    TMP_FILE=$(mktemp)
    echo "$SECRET_JSON" > "$TMP_FILE"
    gcloud secrets versions add "$SECRET_ID" \
        --project="$PROJECT_ID" \
        --data-file="$TMP_FILE" \
        --quiet
    rm -f "$TMP_FILE"
    echo "  ✓ Secret updated"
fi

gcloud secrets add-iam-policy-binding "$SECRET_ID" \
    --project="$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet 2>/dev/null || true
echo "  ✓ Permissions granted"
echo ""

# ------------------------------------------------------------------
# Helper: deploy CF only if source changed
# ------------------------------------------------------------------
deploy_if_changed() {
    local name="$1"
    local source_dir="$2"
    local entry_point="$3"
    local memory="${4:-512MB}"
    local timeout="${5:-540s}"

    # Skip if targeting specific CF and this isn't it
    if [ -n "$TARGET" ] && [ "$TARGET" != "${name%-cf}" ] && [ "$TARGET" != "$name" ]; then
        echo "  - $name skipped (target: $TARGET)"
        return 0
    fi

    local hash_file="/tmp/cvee_deploy_${name}.hash"
    local current_hash=$(find "$source_dir" -type f -exec sha256sum {} \; | sort | sha256sum | awk '{print $1}')

    if [ "$FORCE" = false ] && [ -f "$hash_file" ]; then
        local previous_hash=$(cat "$hash_file")
        if [ "$current_hash" = "$previous_hash" ]; then
            echo "  ✓ $name unchanged — skipping (use --force to override)"
            return 0
        fi
    fi

    echo "  Deploying $name..."
    gcloud functions deploy "$name" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --runtime=python312 \
        --entry-point="$entry_point" \
        --trigger-http \
        --allow-unauthenticated \
        --memory="$memory" \
        --timeout="$timeout" \
        --max-instances=1 \
        --source="$source_dir" \
        --quiet
    echo "$current_hash" > "$hash_file"
    echo "  ✓ $name deployed"
}

# ------------------------------------------------------------------
# Step 2: Deploy Cloud Functions
# ------------------------------------------------------------------
echo "Step 2: Deploying Cloud Functions..."

deploy_if_changed api-to-gcs-cf "$SRC_BASE/functions/api-to-gcs" \
    api_to_gcs_cf 512MB 540s

deploy_if_changed ingest-db-cf "$SRC_BASE/functions/ingest-db" \
    ingest_db_cf 1024MB 3600s

deploy_if_changed pipeline-cf "$SRC_BASE/functions/pipeline" \
    pipeline_cf 2048MB 1800s

echo ""

# ------------------------------------------------------------------
# Step 3: Create Cloud Scheduler jobs
# ------------------------------------------------------------------
echo "Step 3: Setting up Cloud Scheduler..."

API_TO_GCS_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/api-to-gcs-cf"
PIPELINE_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/pipeline-cf"
INGEST_DB_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/ingest-db-cf"

if [ -z "$TARGET" ] || [ "$TARGET" = "api-to-gcs" ]; then
    if gcloud scheduler jobs describe api-to-gcs-scheduler \
        --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
        echo "  ✓ api-to-gcs-scheduler already exists"
    else
        gcloud scheduler jobs create http api-to-gcs-scheduler \
            --project="$PROJECT_ID" --location="$REGION" \
            --schedule="0 21 * * *" --time-zone=UTC \
            --http-method=POST --uri="$API_TO_GCS_URL" \
            --oidc-service-account-email="$SERVICE_ACCOUNT" \
            --oidc-token-audience="$API_TO_GCS_URL" \
            --quiet
        echo "  ✓ api-to-gcs-scheduler created"
    fi
fi

if [ -z "$TARGET" ] || [ "$TARGET" = "pipeline" ]; then
    if gcloud scheduler jobs describe pipeline-scheduler \
        --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
        echo "  ✓ pipeline-scheduler already exists"
    else
        gcloud scheduler jobs create http pipeline-scheduler \
            --project="$PROJECT_ID" --location="$REGION" \
            --schedule="30 21 * * *" --time-zone=UTC \
            --http-method=POST --uri="$PIPELINE_URL" \
            --oidc-service-account-email="$SERVICE_ACCOUNT" \
            --oidc-token-audience="$PIPELINE_URL" \
            --quiet
        echo "  ✓ pipeline-scheduler created"
    fi
fi

if [ -z "$TARGET" ] || [ "$TARGET" = "ingest-db" ]; then
    if gcloud scheduler jobs describe ingest-db-scheduler \
        --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
    gcloud scheduler jobs create http ingest-db-scheduler \
        --project="$PROJECT_ID" --location="$REGION" \
        --schedule="30 23 * * *" --time-zone=UTC \
        --http-method=POST --uri="$INGEST_DB_URL" \
        --oidc-service-account-email="$SERVICE_ACCOUNT" \
        --oidc-token-audience="$INGEST_DB_URL" \
        --quiet
    echo "  ✓ ingest-db-scheduler created"
fi
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Usage:"
echo "  bash scripts/deploy.sh                 # deploy only changed CFs"
echo "  bash scripts/deploy.sh --force          # force redeploy all"
echo "  bash scripts/deploy.sh pipeline         # deploy only pipeline-cf"
echo ""
echo "Functions:"
echo "  • api-to-gcs-cf:  $API_TO_GCS_URL"
echo "  • pipeline-cf:    $PIPELINE_URL"
echo "  • ingest-db-cf:   $INGEST_DB_URL"
echo ""
echo "Scheduler (UTC):"
echo "  • api-to-gcs at 21:00  (FT API → GCS raw)"
echo "  • pipeline  at 21:30  (raw → silver + gold, embeddings)"
echo "  • ingest-db at 23:30  (GCS → Supabase + cleanup)"
echo ""
echo "Manual trigger:"
echo "  gcloud scheduler jobs run api-to-gcs-scheduler --location=$REGION"
echo "  gcloud scheduler jobs run pipeline-scheduler  --location=$REGION"
echo "  gcloud scheduler jobs run ingest-db-scheduler  --location=$REGION"
echo ""
echo "Pipeline manual backfill (last N days, deduplicates):"
echo "  curl -X POST \"${PIPELINE_URL}?days=7\""
echo ""
echo "Full historical backfill:"
echo "  python scripts/backfill.py --date-min 2026-01-01 --date-max 2026-01-31"
echo "  curl -X POST \"${PIPELINE_URL}?days=30\""
echo "  curl -X POST \"${API_TO_GCS_URL}?date_min=2026-01-01&date_max=2026-01-31\""
echo ""
echo "View logs:"
echo "  gcloud functions logs read api-to-gcs-cf --limit 50"
echo "  gcloud functions logs read pipeline-cf --limit 50"
echo "  gcloud functions logs read ingest-db-cf  --limit 50"
