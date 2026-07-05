import asyncio

import functions_framework
import structlog
from cleanup import cleanup_dead_jobs_main
from gcs_sync import main as ingest_db_main
from shared.config import get_config

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


@functions_framework.http
def ingest_db_cf(request):
    """Cloud Function: sync gold Parquet from GCS → Supabase PostgreSQL.

    Args:
        request: Flask request object. No query params required.

    Returns:
        Tuple (response_body, status_code).
    """
    try:
        logger.info("starting_ingest_db")
        config = get_config()

        logger.info("step_ingestion")
        ingest_db_main(
            bucket_name=config["GCS_BUCKET_NAME"],
            sb_host=config["SB_HOST"],
            sb_port=int(config["SB_PORT"]),
            sb_user=config["SB_USER"],
            sb_password=config["SB_PASSWORD"],
            sb_name=config["SB_NAME"],
        )

        logger.info("step_cleanup")
        cleanup_result = asyncio.run(cleanup_dead_jobs_main(config))

        logger.info("ingest_db_completed", deleted_count=cleanup_result.get("deleted_count"))
        return {"status": "success", "cleanup_result": cleanup_result}, 200

    except Exception:
        logger.exception("ingest_db_failed")
        return {"status": "error", "message": "Internal server error"}, 500
