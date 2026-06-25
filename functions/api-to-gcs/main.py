import json
import os

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

    Two modes via query parameters:
    - Daily (default): no params → fetches last 24h (publieeDepuis=1)
    - Backfill: ?date_min=2026-01-01&date_max=2026-01-31 → fetches historical range

    Also accepts:
    - ?publiee_depuis=7 → fetch jobs from last 7 days
    """
    try:
        print("Starting api-to-gcs Cloud Function")
        config = get_config()

        date_min = request.args.get("date_min")
        date_max = request.args.get("date_max")
        publiee_depuis = int(request.args.get("publiee_depuis", 1))

        if date_min and date_max:
            print(f"Backfill mode: {date_min} → {date_max}")
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                date_min=date_min,
                date_max=date_max,
            )
        else:
            print(f"Daily mode: publiee_depuis={publiee_depuis}")
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                publiee_depuis=publiee_depuis,
            )

        print("api-to-gcs Cloud Function completed successfully")
        return "OK", 200

    except Exception as e:
        error_msg = f"Error in api-to-gcs Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
