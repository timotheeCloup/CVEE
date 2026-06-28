import json
import os

import functions_framework
from core import run_pipeline
from google.cloud import secretmanager

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cvee-20260208")


def get_config():
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{PROJECT_ID}/secrets/cvee-secrets/versions/latest"
    response = client.access_secret_version(request={"name": resource_name})
    return json.loads(response.payload.data.decode("UTF-8"))


@functions_framework.http
def pipeline_cf(request):
    """
    Cloud Function: Bronze (raw) → Silver + Gold.
    Triggered by Cloud Scheduler (nightly, after api-to-gcs-cf).

    Query params:
    - ?days=N      → process last N days of raw files (deduplicated)
    - ?max_jobs=N  → limit jobs processed (for fast tests)
    - no params    → process latest raw file only (daily mode)
    """
    try:
        print("Starting pipeline Cloud Function")
        config = get_config()
        bucket_name = config["GCS_BUCKET_NAME"]

        days = request.args.get("days")
        if days:
            days = int(days)
            print(f"Backfill mode: last {days} days")
        else:
            print("Daily mode: latest raw file only")

        max_jobs = request.args.get("max_jobs")
        if max_jobs:
            max_jobs = int(max_jobs)
            print(f"Max jobs limit: {max_jobs}")

        silver_path, gold_path = run_pipeline(bucket_name, days=days, max_jobs=max_jobs)

        if silver_path is None:
            print("Pipeline produced no output.")
            return {"status": "no_data"}, 200

        print(f"Pipeline completed successfully: {silver_path}, {gold_path}")
        return {"status": "success", "silver": silver_path, "gold": gold_path}, 200

    except Exception as e:
        error_msg = f"Error in pipeline Cloud Function: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}, 500
