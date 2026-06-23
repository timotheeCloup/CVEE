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
    Triggered by Cloud Scheduler (nightly).
    """
    try:
        print("Starting api-to-gcs Cloud Function")
        config = get_config()

        fetch_and_store(
            ft_client_id=config["FT_CLIENT_ID"],
            ft_client_secret=config["FT_CLIENT_SECRET"],
            bucket_name=config["GCS_BUCKET_NAME"],
        )

        print("api-to-gcs Cloud Function completed successfully")
        return "OK", 200

    except Exception as e:
        error_msg = f"Error in api-to-gcs Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
