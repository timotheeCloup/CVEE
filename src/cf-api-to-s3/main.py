import functions_framework
import json
import requests
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
            "DATABRICKS_HOST": get_secret("databricks-host"),
            "DATABRICKS_TOKEN": get_secret("databricks-token"),
        }
        return secrets
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        raise


def ping_databricks(databricks_host, databricks_token):
    """Ping Databricks to keep the workspace active (keep-alive)"""
    try:
        headers = {
            "Authorization": f"Bearer {databricks_token}",
            "Content-Type": "application/json"
        }
        url = f"{databricks_host}/api/2.0/clusters/list"
        
        # Set a short timeout to avoid blocking the function
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(" Databricks keep-alive ping successful")
            return True
        else:
            print(f" Databricks ping returned status {response.status_code}")
            return False
            
    except requests.Timeout:
        print(" Databricks ping timed out (continuing anyway)")
        return False
    except Exception as e:
        print(f" Databricks keep-alive ping failed: {e} (continuing anyway)")
        return False


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
        
        # Ping Databricks to keep workspace active (keep-alive)
        print("Sending Databricks keep-alive ping...")
        ping_databricks(
            databricks_host=secrets["DATABRICKS_HOST"],
            databricks_token=secrets["DATABRICKS_TOKEN"]
        )
        
        print("api-to-s3 Cloud Function completed successfully")
        return "OK", 200
    
    except Exception as e:
        error_msg = f"Error in api-to-s3 Cloud Function: {str(e)}"
        print(error_msg)
        return error_msg, 500
