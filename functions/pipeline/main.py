import functions_framework
import structlog
from core import run_pipeline
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
def pipeline_cf(request):
    """Cloud Function: run ETL pipeline bronze → silver → gold.

    Args:
        request: Flask request object. Query params: ``days`` (int, default 1).

    Returns:
        Tuple (response_body, status_code).
    """
    try:
        logger.info("starting_pipeline_cf")
        config = get_config()
        bucket_name = config["GCS_BUCKET_NAME"]

        days = request.args.get("days")
        if days:
            days = int(days)
            logger.info("backfill_mode", days=days)
        else:
            logger.info("daily_mode")

        max_jobs = request.args.get("max_jobs")
        if max_jobs:
            max_jobs = int(max_jobs)
            logger.info("max_jobs_limit", max_jobs=max_jobs)

        silver_path, gold_path = run_pipeline(bucket_name, days=days, max_jobs=max_jobs)

        if silver_path is None:
            logger.info("pipeline_no_output")
            return {"status": "no_data"}, 200

        logger.info("pipeline_completed", silver=silver_path, gold=gold_path)
        return {"status": "success", "silver": silver_path, "gold": gold_path}, 200

    except Exception:
        logger.exception("pipeline_cf_failed")
        return {"status": "error", "message": "Internal server error"}, 500
