from unittest.mock import MagicMock, patch


def _fake_fs(files):
    fs = MagicMock()
    fs.ls = MagicMock(return_value=files)
    return fs


def test_latest_batch_returns_only_most_recent_day() -> None:
    files = [
        {
            "name": "gs://b/jobs_silver/jobs_silver_20260918_210000.parquet",
            "updated": "2026-09-18T21:00:00Z",
        },
        {
            "name": "gs://b/jobs_silver/jobs_silver_20260919_210000.parquet",
            "updated": "2026-09-19T21:00:00Z",
        },
    ]
    with patch("gcs_sync.gcsfs.GCSFileSystem", return_value=_fake_fs(files)):
        from gcs_sync import get_latest_batch_parquet_files

        assert get_latest_batch_parquet_files("b", "jobs_silver/") == [
            "gs://b/jobs_silver/jobs_silver_20260919_210000.parquet"
        ]


def test_date_range_returns_every_file_in_range() -> None:
    files = [
        {"name": "gs://b/jobs_silver/a.parquet", "updated": "2026-09-10T21:00:00Z"},
        {"name": "gs://b/jobs_silver/b.parquet", "updated": "2026-09-15T21:00:00Z"},
        {"name": "gs://b/jobs_silver/c.parquet", "updated": "2026-09-20T21:00:00Z"},
    ]
    with patch("gcs_sync.gcsfs.GCSFileSystem", return_value=_fake_fs(files)):
        from gcs_sync import get_latest_batch_parquet_files

        assert get_latest_batch_parquet_files(
            "b", "jobs_silver/", date_min="2026-09-12", date_max="2026-09-18"
        ) == ["gs://b/jobs_silver/b.parquet"]


def test_no_files_returns_empty() -> None:
    with patch("gcs_sync.gcsfs.GCSFileSystem", return_value=_fake_fs([])):
        from gcs_sync import get_latest_batch_parquet_files

        assert get_latest_batch_parquet_files("b", "jobs_silver/") == []


def test_silver_upsert_updates_changed_columns() -> None:
    from gcs_sync import _silver_upsert_sql

    sql = _silver_upsert_sql(["job_id", "intitule", "ingestion_date"])
    assert "ON CONFLICT (job_id) DO UPDATE SET" in sql
    assert "intitule = EXCLUDED.intitule" in sql
    # job_id is the conflict key and ingestion_date keeps its first-seen value.
    assert "job_id = EXCLUDED.job_id" not in sql
    assert "ingestion_date = EXCLUDED.ingestion_date" not in sql
    # Only rewrite rows whose values actually changed (idempotent re-runs).
    assert "jobs_silver.intitule IS DISTINCT FROM EXCLUDED.intitule" in sql


def test_gold_upsert_refreshes_embedding() -> None:
    from gcs_sync import _gold_upsert_sql

    sql = _gold_upsert_sql()
    assert "ON CONFLICT (job_id) DO UPDATE SET embedding = EXCLUDED.embedding" in sql
    assert "jobs_gold.embedding IS DISTINCT FROM EXCLUDED.embedding" in sql
