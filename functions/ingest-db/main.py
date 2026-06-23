import asyncio
import json
import os

import functions_framework
from cleanup import cleanup_dead_jobs_main
from gcs_sync import main as ingest_db_main
from google.cloud import secretmanager

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cvee-20260208")


def get_config():
    """Load the single cvee-config secret from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{PROJECT_ID}/secrets/cvee-secrets/versions/latest"
    response = client.access_secret_version(request={"name": resource_name})
    return json.loads(response.payload.data.decode("UTF-8"))


@functions_framework.http
def ingest_db_cf(request):
    """
    Cloud Function: ingest new jobs from GCS → Supabase + cleanup dead offers.
    Triggered by Cloud Scheduler (nightly).
    """
    try:
        print("Starting ingest-db Cloud Function")
        config = get_config()

        # Step 1: Ingest new jobs from GCS
        print("\n### STEP 1: INGESTION ###")
        ingest_db_main(
            bucket_name=config["GCS_BUCKET_NAME"],
            sb_host=config["SB_HOST"],
            sb_port=int(config["SB_PORT"]),
            sb_user=config["SB_USER"],
            sb_password=config["SB_PASSWORD"],
            sb_name=config["SB_NAME"],
        )

        # Step 2: Cleanup dead jobs
        print("\n### STEP 2: CLEANUP ###")
        cleanup_result = asyncio.run(cleanup_dead_jobs_main(config))

        print("ingest-db Cloud Function completed successfully")
        return {"status": "success", "cleanup_result": cleanup_result}, 200

    except Exception as e:
        error_msg = f"Error in ingest-db Cloud Function: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}, 500
