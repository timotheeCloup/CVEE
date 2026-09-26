"""Storage-budget retention for the Supabase free tier (500 MB).

Why budget the *live payload* instead of ``pg_database_size``
-------------------------------------------------------------
PostgreSQL never returns freed pages to the operating system after a plain
``DELETE`` + ``VACUUM``: the file keeps its high-water mark and only reuses the
pages on the next inserts. Only ``VACUUM FULL`` shrinks the file, and it needs
roughly twice the live size temporarily — impossible next to the 500 MB hard
limit.

So a retention rule driven by ``pg_database_size`` would trigger on every run
once the watermark crosses the threshold, and would keep deleting the whole
corpus trying (in vain) to make the size drop: an erosion spiral.

We therefore drive the hysteresis on the *live payload*
(``sum(pg_column_size(...))`` over ``jobs_silver`` + ``jobs_gold``), which does
decrease deterministically when rows are deleted. The physical size tracks it
with a stable overhead ratio, so the thresholds are still expressed in the
physical MB the user reasons about and converted here via
``STORAGE_OVERHEAD_FACTOR``.

Behaviour: fill up to ``STORAGE_HIGH_MB``; when that is reached, delete the
oldest offers in batches until back down to ``STORAGE_LOW_MB``, leaving room for
the next ingestions.
"""

import os

import structlog

logger = structlog.get_logger()

# Physical database size (MB) at which we start trimming.
STORAGE_HIGH_MB = float(os.getenv("STORAGE_HIGH_MB", "470"))
# Physical database size (MB) we trim back down to.
STORAGE_LOW_MB = float(os.getenv("STORAGE_LOW_MB", "420"))
# Measured physical/live ratio on the production corpus (~2.06, little bloat yet).
# Deliberately conservative: with delete-driven churn the ratio can rise, so a
# higher factor keeps the physical size safely under the 500 MB hard limit.
# Recalibrate from prod: pg_database_size / (sum(pg_column_size) over both tables).
STORAGE_OVERHEAD_FACTOR = float(os.getenv("STORAGE_OVERHEAD_FACTOR", "2.2"))
# Offers removed per statement, to keep locks short.
STORAGE_DELETE_BATCH = int(os.getenv("STORAGE_DELETE_BATCH", "500"))
# Safety net: hard stop on the delete loop even if sizes behave unexpectedly.
STORAGE_MAX_BATCHES = int(os.getenv("STORAGE_MAX_BATCHES", "1000"))

_MB = 1024 * 1024

# Sum of the live logical bytes of both tables. Overcounts toasted values
# relative to the heap, but is monotonic with deletions (what we need) and
# stable run to run (what the calibration assumes).
_CONTENT_BYTES_SQL = """
    SELECT COALESCE((SELECT sum(pg_column_size(t.*)) FROM jobs_silver t), 0)
         + COALESCE((SELECT sum(pg_column_size(g.*)) FROM jobs_gold g), 0);
"""


def budget_from_config(
    high_mb: float = STORAGE_HIGH_MB,
    low_mb: float = STORAGE_LOW_MB,
    overhead: float = STORAGE_OVERHEAD_FACTOR,
) -> tuple[int, int]:
    """Convert the physical-MB thresholds into live-payload byte budgets.

    Returns ``(high_bytes, low_bytes)`` in live-payload bytes.
    """
    if low_mb > high_mb:
        raise ValueError("STORAGE_LOW_MB must be <= STORAGE_HIGH_MB")
    return int(high_mb * _MB / overhead), int(low_mb * _MB / overhead)


def get_content_bytes(cur) -> int:
    """Live payload size (bytes) of jobs_silver + jobs_gold."""
    cur.execute(_CONTENT_BYTES_SQL)
    return int(cur.fetchone()[0])


def get_physical_bytes(cur) -> int:
    """Physical database size (bytes), the metric Supabase bills against."""
    cur.execute("SELECT pg_database_size(current_database());")
    return int(cur.fetchone()[0])


def delete_oldest_batch(cur, batch_size: int) -> int:
    """Delete the ``batch_size`` oldest offers (by ingest date, then id).

    ``jobs_gold`` is removed by the ``ON DELETE CASCADE`` foreign key. Returns
    the number of ``jobs_silver`` rows deleted.
    """
    cur.execute(
        """
        DELETE FROM jobs_silver
        WHERE job_id IN (
            SELECT job_id FROM jobs_silver
            ORDER BY ingestion_date ASC NULLS FIRST, job_id ASC
            LIMIT %s
        );
        """,
        (batch_size,),
    )
    return cur.rowcount


def enforce_storage_budget(
    cur,
    high_bytes: int,
    low_bytes: int,
    batch_size: int = STORAGE_DELETE_BATCH,
    max_batches: int = STORAGE_MAX_BATCHES,
) -> dict:
    """Trim the oldest offers while the live payload exceeds ``high_bytes``.

    Deletes oldest-first in batches until the payload is back under
    ``low_bytes`` (or the table is empty). No-op when already under the high
    watermark. Returns a summary dict for logging.
    """
    content = get_content_bytes(cur)
    physical = get_physical_bytes(cur)
    logger.info(
        "storage_budget_check",
        content_mb=round(content / _MB, 1),
        physical_mb=round(physical / _MB, 1),
        high_mb=round(high_bytes / _MB, 1),
        low_mb=round(low_bytes / _MB, 1),
    )

    if content <= high_bytes:
        return {
            "triggered": False,
            "deleted_count": 0,
            "content_mb": round(content / _MB, 1),
            "physical_mb": round(physical / _MB, 1),
        }

    deleted_count = 0
    batches = 0
    while content > low_bytes and batches < max_batches:
        batch_deleted = delete_oldest_batch(cur, batch_size)
        if batch_deleted == 0:
            break  # table empty
        deleted_count += batch_deleted
        batches += 1
        content = get_content_bytes(cur)

    physical = get_physical_bytes(cur)
    logger.info(
        "storage_budget_enforced",
        deleted_count=deleted_count,
        batches=batches,
        content_mb=round(content / _MB, 1),
        physical_mb=round(physical / _MB, 1),
    )
    return {
        "triggered": True,
        "deleted_count": deleted_count,
        "batches": batches,
        "content_mb": round(content / _MB, 1),
        "physical_mb": round(physical / _MB, 1),
    }
