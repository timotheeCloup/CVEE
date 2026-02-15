import functions_framework
import json
from google.cloud import secretmanager
from api_to_s3_loader import main as api_to_s3_main

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
            "FT_CLIENT_ID": get_secret("ft-client-id"),
            "FT_CLIENT_SECRET": get_secret("ft-client-secret"),
            "AWS_ACCESS_KEY_ID": get_secret("aws-access-key-id"),
            "AWS_SECRET_ACCESS_KEY": get_secret("aws-secret-access-key"),
            "AWS_S3_BUCKET_NAME": get_secret("aws-s3-bucket-name"),
        }
        return secrets
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        raise


@functions_framework.http
def api_to_s3_cf(request):
    """
    HTTP Cloud Function to run API to S3 job.
    Triggered by Cloud Scheduler.
    """
    try:
        print("Starting api-to-s3 Cloud Function")
        
        # Get secrets from GCP Secret Manager
        secrets = get_all_secrets()
        
        # Run the main job with secrets
        api_to_s3_main(
            ft_client_id=secrets["FT_CLIENT_ID"],
            ft_client_secret=secrets["FT_CLIENT_SECRET"],
            aws_access_key_id=secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=secrets["AWS_SECRET_ACCESS_KEY"],
            aws_s3_bucket_name=secrets["AWS_S3_BUCKET_NAME"],
        )
        
        print("api-to-s3 Cloud Function completed successfully")
        return "OK", 200
    
    except Exception as e:
        error_msg = f"Error in api-to-s3 Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
