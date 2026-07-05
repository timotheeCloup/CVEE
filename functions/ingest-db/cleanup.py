import asyncio
import time

import aiohttp
import psycopg2
import structlog

logger = structlog.get_logger()


async def verify_job_link(job_id: str, timeout: float = 0.5) -> dict:
    """
    Verify if a job offer link is still available on France Travail.
    France Travail returns 404 for deleted/expired offers.

    Args:
        job_id: Job ID from database
        timeout: Timeout in seconds (default 500ms)

    Returns:
        Dict with job_id and alive status
    """
    job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job_id}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.head(
                job_url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True
            ) as resp,
        ):
            # 200 = alive, 404 = dead, timeouts = assume alive (prudent)
            is_alive = resp.status == 200
            return {"job_id": job_id, "alive": is_alive, "status": resp.status}
    except TimeoutError:
        # Timeout = assume alive (prudent, avoid false positives)
        return {"job_id": job_id, "alive": True, "status": "timeout"}
    except Exception:
        # Any other error = assume alive (prudent)
        return {"job_id": job_id, "alive": True, "status": "error"}


async def batch_verify_all_jobs(job_ids: list, max_concurrent: int = 10) -> dict:
    """
    Verify all job offers in parallel using aggressive timeout.

    Args:
        job_ids: List of all job IDs from database
        max_concurrent: Max concurrent requests (default 10)

    Returns:
        Dict with alive_ids (set) and dead_ids (set)
    """
    if not job_ids:
        return {"alive_ids": set(), "dead_ids": set()}

    t_start = time.time()

    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(job_id):
        async with semaphore:
            return await verify_job_link(job_id)

    # Check all jobs in parallel
    verification_results = await asyncio.gather(
        *[check_with_semaphore(job_id) for job_id in job_ids], return_exceptions=True
    )

    # Build sets of alive and dead job IDs
    alive_ids = set()
    dead_ids = set()

    for result in verification_results:
        if isinstance(result, Exception):
            # On exception, assume alive (prudent)
            continue
        if result.get("alive", True):
            alive_ids.add(result["job_id"])
        else:
            dead_ids.add(result["job_id"])

    t_end = time.time()

    return {
        "alive_ids": alive_ids,
        "dead_ids": dead_ids,
        "duration": t_end - t_start,
        "total_checked": len(job_ids),
        "dead_count": len(dead_ids),
    }


def get_all_job_ids(db_host, db_port, db_user, db_password, db_name):
    """
    Retrieve all job IDs from jobs_gold table.

    Args:
        DB connection parameters

    Returns:
        List of job_ids
    """
    try:
        conn = psycopg2.connect(
            host=db_host, database=db_name, user=db_user, password=db_password, port=db_port
        )
        cur = conn.cursor()

        # Get all job IDs
        cur.execute("SELECT job_id FROM jobs_gold;")
        results = cur.fetchall()
        job_ids = [row[0] for row in results]

        cur.close()
        conn.close()

        return job_ids
    except Exception as e:
        logger.error("db_retrieve_error", error=str(e))
        raise


def delete_dead_jobs(dead_ids: set, db_host, db_port, db_user, db_password, db_name):
    """
    Delete dead job offers from jobs_silver table.
    This will cascade delete from jobs_gold automatically due to ON DELETE CASCADE constraint.

    Args:
        dead_ids: Set of job IDs to delete
        DB connection parameters

    Returns:
        Number of deleted rows
    """
    if not dead_ids:
        logger.info("no_dead_jobs")
        return 0

    try:
        conn = psycopg2.connect(
            host=db_host, database=db_name, user=db_user, password=db_password, port=db_port
        )
        cur = conn.cursor()

        # Delete dead jobs from jobs_silver (parent table)
        # ON DELETE CASCADE will automatically delete from jobs_gold (child table)
        dead_ids_list = list(dead_ids)
        placeholders = ",".join(["%s"] * len(dead_ids_list))
        delete_sql = f"DELETE FROM jobs_silver WHERE job_id IN ({placeholders});"  # nosec B608 -- parameterized via %s

        cur.execute(delete_sql, dead_ids_list)
        deleted_count = cur.rowcount

        conn.commit()
        cur.close()
        conn.close()

        return deleted_count
    except Exception as e:
        logger.error("db_delete_error", error=str(e))
        raise


async def cleanup_dead_jobs_main(secrets):
    """
    Main cleanup function: retrieve all jobs, verify links, delete dead ones.

    Args:
        secrets: Dict with DB credentials (SB_HOST, SB_PORT, SB_USER, SB_PASSWORD, SB_NAME)
    """
    db_host = secrets.get("SB_HOST")
    db_port = int(secrets.get("SB_PORT", "5432"))
    db_user = secrets.get("SB_USER")
    db_password = secrets.get("SB_PASSWORD")
    db_name = secrets.get("SB_NAME")

    logger.info("cleanup_started")

    # Step 1: Retrieve all job IDs from database
    t0 = time.time()
    logger.info("cleanup_step_retrieve_ids")
    job_ids = get_all_job_ids(db_host, db_port, db_user, db_password, db_name)
    t1 = time.time()
    logger.info("cleanup_retrieved_ids", count=len(job_ids), duration=f"{t1 - t0:.2f}s")

    if not job_ids:
        logger.info("cleanup_no_jobs")
        return {"status": "success", "deleted_count": 0}

    # Step 2: Batch verify all jobs
    logger.info("cleanup_step_verify_links")
    verification_result = await batch_verify_all_jobs(job_ids, max_concurrent=10)

    logger.info(
        "cleanup_verified",
        total_checked=verification_result["total_checked"],
        duration=f"{verification_result['duration']:.2f}s",
        dead_count=verification_result["dead_count"],
    )

    # Step 3: Delete dead jobs
    dead_ids = verification_result["dead_ids"]
    if dead_ids:
        logger.info("cleanup_step_delete", dead_count=len(dead_ids))
        deleted_count = delete_dead_jobs(dead_ids, db_host, db_port, db_user, db_password, db_name)
        logger.info("cleanup_deleted", count=deleted_count)
    else:
        logger.info("cleanup_no_dead_jobs")
        deleted_count = 0

    logger.info("cleanup_completed")

    return {
        "status": "success",
        "total_checked": verification_result["total_checked"],
        "dead_count": verification_result["dead_count"],
        "deleted_count": deleted_count,
        "duration": verification_result["duration"],
    }
