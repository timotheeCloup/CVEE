import functions_framework
import json
from google.cloud import secretmanager
from sync_s3_to_supabase import main as ingest_db_main

PROJECT_ID = "cvee-20260208"


def get_secret(secret_id, version_id="latest"):
    """Retrieve secret from GCP Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": resource_name})
    return response.payload.data.decode("UTF-8")


def get_all_secrets():
    """Load all secrets required for the job"""
    try:
        # Try to get secrets from Secret Manager (production)
        secrets = {
            "AWS_ACCESS_KEY_ID": get_secret("aws-access-key-id"),
            "AWS_SECRET_ACCESS_KEY": get_secret("aws-secret-access-key"),
            "AWS_S3_BUCKET_NAME": get_secret("aws-s3-bucket-name"),
            "SB_HOST": get_secret("sb-host"),
            "SB_PORT": get_secret("sb-port"),
            "SB_USER": get_secret("sb-user"),
            "SB_PASSWORD": get_secret("sb-password"),
            "SB_NAME": get_secret("sb-name"),
        }
        return secrets
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        raise


@functions_framework.http
def ingest_db_cf(request):
    """
    HTTP Cloud Function to run S3 to Supabase ingestion job.
    Triggered by Cloud Scheduler.
    """
    try:
        print("Starting ingest-db Cloud Function")
        
        # Get secrets from GCP Secret Manager
        secrets = get_all_secrets()
        
        # Run the main job with secrets
        ingest_db_main(
            aws_access_key_id=secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=secrets["AWS_SECRET_ACCESS_KEY"],
            aws_s3_bucket_name=secrets["AWS_S3_BUCKET_NAME"],
            sb_host=secrets["SB_HOST"],
            sb_port=secrets["SB_PORT"],
            sb_user=secrets["SB_USER"],
            sb_password=secrets["SB_PASSWORD"],
            sb_name=secrets["SB_NAME"],
        )
        
        print("ingest-db Cloud Function completed successfully")
        return "OK", 200
    
    except Exception as e:
        error_msg = f"Error in ingest-db Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
