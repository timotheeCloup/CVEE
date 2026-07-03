import json
import os
from datetime import datetime, timedelta

import functions_framework
from ft_client import main as fetch_and_store
from google.cloud import secretmanager

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cvee-20260208")


def get_config():
    """Load the single cvee-config secret from Secret Manager (1 version = free)"""
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{PROJECT_ID}/secrets/cvee-secrets/versions/latest"
    response = client.access_secret_version(request={"name": resource_name})
    return json.loads(response.payload.data.decode("UTF-8"))


@functions_framework.http
def api_to_gcs_cf(request):
    """
    Cloud Function: fetch France Travail jobs → store in GCS.

    Query params:
    - No params            → daily mode (publieeDepuis=1, last 24h)
    - ?max_results=N       → quick test: fetch up to N jobs from last 30 days
    - ?date_min=YYYY-MM-DD&date_max=YYYY-MM-DD            → backfill by date range
    - ?date_min=YYYY-MM-DD&date_max=YYYY-MM-DD&max_results=N  → backfill + limit

    Note: publieeDepuis > 1 is broken on FT API side, only =1 works.
    """
    try:
        print("Starting api-to-gcs Cloud Function")
        config = get_config()

        date_min = request.args.get("date_min")
        date_max = request.args.get("date_max")
        max_results = request.args.get("max_results")
        if max_results:
            max_results = int(max_results)

        if date_min and date_max:
            print(f"Backfill mode: {date_min} → {date_max}")
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                date_min=date_min,
                date_max=date_max,
                max_results=max_results,
            )
        elif max_results:
            today = datetime.now().strftime("%Y-%m-%d")
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            print(
                f"Quick test mode: last 30 days ({thirty_days_ago} → {today}), max {max_results} jobs"
            )
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                date_min=thirty_days_ago,
                date_max=today,
                max_results=max_results,
            )
        else:
            print("Daily mode: publiee_depuis=1")
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                publiee_depuis=1,
                max_results=max_results,
            )

        print("api-to-gcs Cloud Function completed successfully")
        return "OK", 200

    except Exception as e:
        error_msg = f"Error in api-to-gcs Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
