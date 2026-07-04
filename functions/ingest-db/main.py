import asyncio

import functions_framework
from cleanup import cleanup_dead_jobs_main
from gcs_sync import main as ingest_db_main
from shared.config import get_config


@functions_framework.http
def ingest_db_cf(request):
    """Cloud Function: sync gold Parquet from GCS → Supabase PostgreSQL.

    Args:
        request: Flask request object. No query params required.

    Returns:
        Tuple (response_body, status_code).
    """
    """
    Cloud Function: ingest silver + gold from GCS → Supabase + cleanup dead offers.
    Triggered by Cloud Scheduler (nightly, after Databricks pipeline).
    """
    try:
        print("Starting ingest-db Cloud Function")
        config = get_config()

        # Step 1: Ingest silver + gold from GCS into Supabase
        print("\n### STEP 1: INGESTION (GCS → Supabase) ###")
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
