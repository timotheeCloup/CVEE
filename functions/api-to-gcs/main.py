from datetime import datetime, timedelta

import functions_framework
import structlog
from ft_client import main as fetch_and_store
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
def api_to_gcs_cf(request):
    """Cloud Function: fetch job offers from France Travail API → store in GCS.

    Args:
        request: Flask request object. Query params: ``date_min``, ``date_max``,
            ``max_results``, ``publiee_depuis``.

    Returns:
        Tuple (response_body, status_code).
    """
    try:
        logger.info("starting_api_to_gcs")
        config = get_config()

        date_min = request.args.get("date_min")
        date_max = request.args.get("date_max")
        max_results = request.args.get("max_results")
        if max_results:
            max_results = int(max_results)

        if date_min and date_max:
            logger.info("backfill_mode", date_min=date_min, date_max=date_max)
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
            logger.info(
                "quick_test_mode", date_min=thirty_days_ago, date_max=today, max_results=max_results
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
            logger.info("daily_mode", publiee_depuis=1)
            fetch_and_store(
                ft_client_id=config["FT_CLIENT_ID"],
                ft_client_secret=config["FT_CLIENT_SECRET"],
                bucket_name=config["GCS_BUCKET_NAME"],
                publiee_depuis=1,
                max_results=max_results,
            )

        logger.info("api_to_gcs_completed")
        return "OK", 200

    except Exception:
        logger.exception("api_to_gcs_failed")
        return "Internal server error", 500
