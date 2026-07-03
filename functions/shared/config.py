import json
import os

from google.cloud import secretmanager


def get_config():
    """Load cvee-secrets from GCP Secret Manager (free tier: 1 version).

    Returns a dict with all project configuration keys:
        FT_CLIENT_ID, FT_CLIENT_SECRET, GCS_BUCKET_NAME, GCP_PROJECT_ID,
        SB_HOST, SB_PORT, SB_NAME, SB_USER, SB_PASSWORD,
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_API_URL
    """
    project_id = os.getenv("GCP_PROJECT_ID", "cvee-20260208")
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{project_id}/secrets/cvee-secrets/versions/latest"
    response = client.access_secret_version(request={"name": resource_name})
    return json.loads(response.payload.data.decode("UTF-8"))
