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
