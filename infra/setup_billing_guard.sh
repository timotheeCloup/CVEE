#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CVEE — Billing Guard Setup
# Creates a Cloud Function that automatically disables billing
# when the GCP budget threshold is reached (e.g. 5€).
# ============================================================
#
# This script does:
#   1. Enable required APIs
#   2. Creates a dedicated service account
#   3. Creates a Pub/Sub topic for budget alerts
#   4. Deploys the Cloud Function (triggered by Pub/Sub)
#   5. Creates a budget alert at 5€
#   6. Disables unused APIs to prevent accidental billing
#
# ⚠️  ONE MANUAL STEP REQUIRED (see end of script):
#   Grant the service account the role 'billing.admin' on your
#   billing account. This cannot be done via project-level IAM.
# ============================================================

PROJECT_ID="${GCP_PROJECT_ID:-cvee-20260208}"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
REGION="europe-west1"
SA_NAME="billing-guard"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
TOPIC_NAME="budget-alerts"
FUNCTION_NAME="billing-guard"
BUDGET_AMOUNT="${BUDGET_AMOUNT:-5EUR}"

echo "============================================"
echo " CVEE — Billing Guard Setup"
echo " Project:  $PROJECT_ID ($PROJECT_NUMBER)"
echo " Budget:   $BUDGET_AMOUNT"
echo " Region:   $REGION"
echo "============================================"
echo ""

# ---- 0. Enable required APIs ----
echo "[0/6] Enabling required APIs..."

gcloud services enable \
    billingbudgets.googleapis.com \
    cloudbilling.googleapis.com \
    pubsub.googleapis.com \
    cloudfunctions.googleapis.com \
    cloudbuild.googleapis.com \
    eventarc.googleapis.com \
    cloudscheduler.googleapis.com \
    --project="$PROJECT_ID"

echo "       Done."

# ---- 1. Service Account ----
echo "[1/6] Creating service account: $SA_EMAIL"

if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    echo "       Already exists, skipping."
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="Billing Guard — stops billing on budget overshoot" \
        --project="$PROJECT_ID"
    echo "       Created."
fi

# Grant the SA rights to read/write billing info on the PROJECT
echo "       Granting roles/billing.projectManager on project..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/billing.projectManager" \
    --condition=None &>/dev/null || true

# ---- 2. Pub/Sub topic ----
echo "[2/6] Creating Pub/Sub topic: $TOPIC_NAME"

if gcloud pubsub topics describe "$TOPIC_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "       Already exists, skipping."
else
    gcloud pubsub topics create "$TOPIC_NAME" --project="$PROJECT_ID"
    echo "       Created."
fi

# Grant Pub/Sub publisher role to Google's budget service account
# (budget alerts are published by a Google-owned account)
echo "       Granting Pub/Sub publisher to billing-notifications..."
gcloud pubsub topics add-iam-policy-binding "$TOPIC_NAME" \
    --member="serviceAccount:billing-notifications@system.gserviceaccount.com" \
    --role="roles/pubsub.publisher" \
    --project="$PROJECT_ID" &>/dev/null || true

# ---- 3. Deploy Cloud Function ----
echo "[3/6] Deploying Cloud Function: $FUNCTION_NAME"

gcloud functions deploy "$FUNCTION_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --runtime="python311" \
    --trigger-topic="$TOPIC_NAME" \
    --entry-point="stop_billing" \
    --service-account="$SA_EMAIL" \
    --source="infra/billing-guard" \
    --memory="128Mi" \
    --timeout="30s" \
    --max-instances="1" \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID"

echo "       Deployed."

# ---- 4. Create Budget ----
echo "[4/6] Creating budget: ${BUDGET_AMOUNT}"

# Get the billing account ID linked to this project
BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" \
    --format="value(billingAccountName)" | sed 's|billingAccounts/||')

if [ -z "$BILLING_ACCOUNT" ]; then
    echo "⚠️  No billing account found for project $PROJECT_ID."
    echo "   Set up billing at: https://console.cloud.google.com/billing"
    echo "   Then re-run this script."
    exit 1
fi

echo "       Billing account: $BILLING_ACCOUNT"

# Delete old budget with same name if it exists
EXISTING_BUDGET=$(gcloud billing budgets list \
    --billing-account="$BILLING_ACCOUNT" \
    --filter="displayName=cvee-budget-guard" \
    --format="value(name)" 2>/dev/null || true)

if [ -n "$EXISTING_BUDGET" ]; then
    echo "       Removing existing budget..."
    gcloud billing budgets delete "$EXISTING_BUDGET" \
        --billing-account="$BILLING_ACCOUNT" --quiet 2>/dev/null || true
fi

FULL_TOPIC="projects/${PROJECT_ID}/topics/${TOPIC_NAME}"

gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT" \
    --display-name="cvee-budget-guard" \
    --budget-amount="$BUDGET_AMOUNT" \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0 \
    --notifications-rule-pubsub-topic="$FULL_TOPIC" \
    --project="$PROJECT_ID"

echo "       Budget created (alerts at 50%, 90%, 100% of ${BUDGET_AMOUNT})."

# ---- 5. Lock down: disable unused APIs ----
echo "[5/6] Disabling APIs we never use (prevents accidental billing)..."

DISABLE_APIS=(
    compute.googleapis.com          # Compute Engine (VMs)
    sqladmin.googleapis.com         # Cloud SQL
    aiplatform.googleapis.com       # Vertex AI
    translate.googleapis.com        # Cloud Translation
    dataproc.googleapis.com         # Dataproc (Spark)
    dataflow.googleapis.com         # Dataflow
    ml.googleapis.com               # AI Platform
    automl.googleapis.com           # AutoML
    firestore.googleapis.com        # Firestore
    redis.googleapis.com            # Memorystore
    servicenetworking.googleapis.com # Private networking
)

for api in "${DISABLE_APIS[@]}"; do
    if gcloud services list --enabled --project="$PROJECT_ID" --filter="name:$api" 2>/dev/null | grep -q "$api"; then
        echo "       Disabling $api..."
        gcloud services disable "$api" --project="$PROJECT_ID" --force 2>/dev/null || true
    fi
done

echo "       Done (unused APIs locked)."

# ---- 6. Manual step ----
echo ""
echo "============================================"
echo " ⚠️  DERNIÈRE ÉTAPE (MANUELLE [6/6]) — 2 min"
echo "============================================"
echo ""
echo "La Cloud Function a besoin du rôle 'billing.admin' sur"
echo "le compte de facturation pour pouvoir le DÉSACTIVER."
echo "Cette permission ne peut pas être donnée via gcloud en"
echo "ligne de commande."
echo ""
echo "👉 Ouvre ce lien :"
echo "   https://console.cloud.google.com/billing/${BILLING_ACCOUNT}/iam?project=${PROJECT_ID}"
echo ""
echo "Puis :"
echo "   1. Clique sur '+ GRANT ACCESS' (ou 'AJOUTER UN ACCÈS')"
echo "   2. Dans 'New principals', colle :"
echo "      $SA_EMAIL"
echo "   3. Dans 'Select a role', cherche :"
echo "      Billing > Billing Account Administrator"
echo "      (ou tape 'billing.admin')"
echo "   4. Clique 'SAVE'"
echo ""
echo "============================================"
echo " ✅ Setup terminé !"
echo ""
echo "   Budget actif :  ${BUDGET_AMOUNT}"
echo "   Alertes à    :  50%, 90%, 100%"
echo "   Coupure auto :  à 100% (${BUDGET_AMOUNT})"
echo ""
echo "   Pour tester :  gcloud pubsub topics publish $TOPIC_NAME \\"
echo "                    --message='{\"budgetDisplayName\":\"test\",\"costAmount\":10,\"budgetAmount\":5,\"alertThresholdExceeded\":1.0,\"currencyCode\":\"EUR\"}'"
echo "============================================"
