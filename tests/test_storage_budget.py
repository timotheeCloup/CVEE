"""Tests for the storage-budget retention (functions/ingest-db/storage_budget.py).

Hermetic: a stateful fake cursor simulates the corpus so the hysteresis loop is
exercised without a live database.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "functions" / "ingest-db"))

import storage_budget  # noqa: E402
from storage_budget import (  # noqa: E402
    budget_from_config,
    delete_oldest_batch,
    enforce_storage_budget,
    get_content_bytes,
    get_physical_bytes,
)

_MB = 1024 * 1024


class FakeCursor:
    """Simulates jobs_silver/jobs_gold: an ordered list of (job_id, bytes)."""

    def __init__(self, rows: list[tuple[str, int]], physical_bytes: int = 0):
        self.rows = list(rows)
        self.physical_bytes = physical_bytes
        self.deletes: list[tuple] = []
        self._result = None

    def execute(self, sql: str, params=None):
        normalized = " ".join(sql.split())
        if "pg_database_size" in normalized:
            self._result = [(self.physical_bytes,)]
        elif "pg_column_size" in normalized:
            self._result = [(sum(size for _, size in self.rows),)]
        elif normalized.startswith("DELETE FROM jobs_silver"):
            batch = params[0]
            self.deletes.append(params)
            n = min(batch, len(self.rows))
            del self.rows[:n]
            self.rowcount = n
        else:
            raise AssertionError(f"unexpected SQL: {normalized}")

    def fetchone(self):
        return self._result[0]


def test_budget_from_config_converts_physical_to_live():
    high, low = budget_from_config(high_mb=470, low_mb=420, overhead=2.0)
    assert high == int(470 * _MB / 2.0)
    assert low == int(420 * _MB / 2.0)


def test_budget_from_config_uses_defaults():
    high, low = budget_from_config()
    assert high > low > 0


def test_budget_from_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        budget_from_config(high_mb=100, low_mb=200)


def test_get_content_bytes_sums_both_tables():
    cur = FakeCursor([("A", 100), ("B", 250)])
    assert get_content_bytes(cur) == 350


def test_no_op_when_under_high_watermark():
    cur = FakeCursor([("A", 100), ("B", 100)])
    result = enforce_storage_budget(cur, high_bytes=1000, low_bytes=500, batch_size=10)
    assert result["triggered"] is False
    assert result["deleted_count"] == 0
    assert cur.deletes == []


def test_trims_only_down_to_low_watermark():
    rows = [(f"J{i:03d}", 100) for i in range(20)]  # 2000 bytes total
    cur = FakeCursor(rows, physical_bytes=999 * _MB)
    # high=1500, low=800 -> must delete until <= 800 bytes.
    result = enforce_storage_budget(cur, high_bytes=1500, low_bytes=800, batch_size=3)
    assert result["triggered"] is True
    assert sum(size for _, size in cur.rows) <= 800
    assert result["deleted_count"] == 1200 // 100  # 2000 -> 800 = 12 rows
    assert cur.deletes and all(p == (3,) for p in cur.deletes)


def test_deletes_oldest_first_batches():
    rows = [("oldest", 100), ("middle", 100), ("newest", 100)]
    cur = FakeCursor(rows)
    deleted = delete_oldest_batch(cur, batch_size=1)
    assert deleted == 1
    assert [job_id for job_id, _ in cur.rows] == ["middle", "newest"]


def test_stops_when_table_empty():
    cur = FakeCursor([("A", 100), ("B", 100)])
    result = enforce_storage_budget(cur, high_bytes=50, low_bytes=10, batch_size=5)
    assert cur.rows == []
    assert result["deleted_count"] == 2


def test_respects_max_batches_safety():
    rows = [(f"J{i:03d}", 100) for i in range(50)]  # 5000 bytes
    cur = FakeCursor(rows)
    # low unreachable within max_batches: guarantee the loop is bounded.
    result = enforce_storage_budget(cur, high_bytes=4000, low_bytes=0, batch_size=1, max_batches=3)
    assert result["batches"] == 3
    assert result["deleted_count"] == 3


def test_get_physical_bytes_reads_database_size():
    cur = FakeCursor([], physical_bytes=123 * _MB)
    assert get_physical_bytes(cur) == 123 * _MB


def test_module_defaults_are_ordered():
    assert storage_budget.STORAGE_LOW_MB < storage_budget.STORAGE_HIGH_MB
    assert storage_budget.STORAGE_HIGH_MB < 500
