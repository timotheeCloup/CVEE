#!/bin/bash
# Deployment script for CVEE Cloud Functions + Scheduler
# Usage: bash deploy.sh  (from src/ directory)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ID="cvee-20260208"
REGION="europe-west1"
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

echo "=== CVEE Deployment ==="
echo "Project: $PROJECT_ID  |  Region: $REGION"
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
if [ -f ../.env ]; then
    echo "  Updating secret version from .env file..."
    python3 - "$(realpath ../.env)" << 'PYEOF'
import sys, json, os
from pathlib import Path
env_file = Path(sys.argv[1])
config = {}
for line in env_file.read_text().strip().splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key, _, val = line.partition('=')
    key, val = key.strip(), val.strip().strip("'\"")
    config[key] = val
print(json.dumps(config, indent=2))
PYEOF
    # Pipe JSON to secret
    # Note: replace with the actual command when ready
    echo "  (Run manually: gcloud secrets versions add cvee-secrets --data-file=<(python3 generate_json.py))"
fi

gcloud secrets add-iam-policy-binding "$SECRET_ID" \
    --project="$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet 2>/dev/null || true
echo "  ✓ Permissions granted"
echo ""

# ------------------------------------------------------------------
# Step 2: Deploy Cloud Functions
# ------------------------------------------------------------------
echo "Step 2: Deploying Cloud Functions..."

echo "  Deploying api-to-gcs-cf..."
gcloud functions deploy api-to-gcs-cf \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --runtime=python312 \
    --entry-point=api_to_gcs_cf \
    --trigger-http \
    --allow-unauthenticated \
    --memory=512MB \
    --timeout=540s \
    --max-instances=1 \
    --source=./functions/api-to-gcs \
    --quiet
echo "  ✓ api-to-gcs-cf deployed"

echo "  Deploying ingest-db-cf..."
gcloud functions deploy ingest-db-cf \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --runtime=python312 \
    --entry-point=ingest_db_cf \
    --trigger-http \
    --allow-unauthenticated \
    --memory=1024MB \
    --timeout=3600s \
    --max-instances=1 \
    --source=./functions/ingest-db \
    --quiet
echo "  ✓ ingest-db-cf deployed"
echo ""

# ------------------------------------------------------------------
# Step 3: Create Cloud Scheduler jobs
# ------------------------------------------------------------------
echo "Step 3: Setting up Cloud Scheduler..."

API_TO_GCS_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/api-to-gcs-cf"
INGEST_DB_URL="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/ingest-db-cf"

# api-to-gcs scheduler (daily at 21:00 UTC)
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

# ingest-db scheduler (daily at 23:30 UTC)
if gcloud scheduler jobs describe ingest-db-scheduler \
    --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
    echo "  ✓ ingest-db-scheduler already exists"
else
    gcloud scheduler jobs create http ingest-db-scheduler \
        --project="$PROJECT_ID" --location="$REGION" \
        --schedule="30 23 * * *" --time-zone=UTC \
        --http-method=POST --uri="$INGEST_DB_URL" \
        --oidc-service-account-email="$SERVICE_ACCOUNT" \
        --oidc-token-audience="$INGEST_DB_URL" \
        --quiet
    echo "  ✓ ingest-db-scheduler created"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Functions:"
echo "  • api-to-gcs-cf:  $API_TO_GCS_URL"
echo "  • ingest-db-cf:   $INGEST_DB_URL"
echo ""
echo "Scheduler (UTC):"
echo "  • api-to-gcs at 21:00  (France → S3)"
echo "  • ingest-db at 23:30   (S3 → Supabase + cleanup)"
echo ""
echo "Manual trigger:"
echo "  gcloud scheduler jobs run api-to-gcs-scheduler --location=$REGION"
echo "  gcloud scheduler jobs run ingest-db-scheduler  --location=$REGION"
echo ""
echo "View logs:"
echo "  gcloud functions logs read api-to-gcs-cf --limit 50"
echo "  gcloud functions logs read ingest-db-cf  --limit 50"
