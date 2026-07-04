import json
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


@pytest.mark.asyncio
async def test_clean_html_strips_tags() -> None:
    from core import clean_html

    assert clean_html("<p>Hello <b>World</b></p>") == "Hello World"


@pytest.mark.asyncio
async def test_clean_html_handles_entities() -> None:
    from core import clean_html

    assert "&amp;" not in clean_html("Dev &amp; Ops")
    assert "&lt;" not in clean_html("x &lt; 5")


@pytest.mark.asyncio
async def test_clean_html_handles_none() -> None:
    from core import clean_html

    assert clean_html(None) == ""
    assert clean_html(float("nan")) == ""


@pytest.mark.asyncio
async def test_clean_html_collapses_whitespace() -> None:
    from core import clean_html

    result = clean_html("Hello   World\n\nTest\tTab")
    assert "  " not in result
    assert "\n" not in result
    assert "\t" not in result


@pytest.mark.asyncio
async def test_extract_field_libelle() -> None:
    from core import _extract_field

    data = [{"libelle": "Python"}, {"libelle": "GCP"}, {"libelle": "FastAPI"}]
    result = _extract_field(data, "libelle")
    assert result == "Python GCP FastAPI"


@pytest.mark.asyncio
async def test_extract_field_json_string() -> None:
    from core import _extract_field

    json_str = json.dumps([{"libelle": "Python"}, {"libelle": "GCP"}])
    result = _extract_field(json_str, "libelle")
    assert result == "Python GCP"


@pytest.mark.asyncio
async def test_extract_field_none() -> None:
    from core import _extract_field

    assert _extract_field(None) == ""
    assert _extract_field(float("nan")) == ""


@pytest.mark.asyncio
async def test_extract_field_simple_dict() -> None:
    from core import _extract_field

    result = _extract_field({"libelle": "CDI", "code": "01"}, "libelle")
    assert result == "CDI"


@pytest.mark.asyncio
async def test_extract_field_custom_field() -> None:
    from core import _extract_field

    data = [{"domaineLibelle": "Informatique"}, {"domaineLibelle": "Data"}]
    result = _extract_field(data, "domaineLibelle")
    assert result == "Informatique Data"


@pytest.mark.asyncio
async def test_extract_field_plain_string() -> None:
    from core import _extract_field

    result = _extract_field("simple string", "libelle")
    assert result == "simple string"


@pytest.mark.asyncio
async def test_extract_field_numpy_array() -> None:
    from core import _extract_field

    arr = np.array([{"libelle": "Python"}, {"libelle": "R"}])
    result = _extract_field(arr, "libelle")
    assert result == "Python R"


@pytest.mark.asyncio
async def test_serialize_json_col_none() -> None:
    from core import serialize_json_col

    assert serialize_json_col(None) is None
    assert serialize_json_col(float("nan")) is None


@pytest.mark.asyncio
async def test_serialize_json_col_already_string() -> None:
    from core import serialize_json_col

    assert serialize_json_col('{"key":"value"}') == '{"key":"value"}'


@pytest.mark.asyncio
async def test_serialize_json_col_converts_list() -> None:
    from core import serialize_json_col

    data = [{"libelle": "Python"}, {"code": "CDI"}]
    result = serialize_json_col(data)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == data


@pytest.mark.asyncio
async def test_serialize_json_col_converts_dict() -> None:
    from core import serialize_json_col

    data = {"libelle": "Paris", "code": "75"}
    result = serialize_json_col(data)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == data


@pytest.mark.asyncio
async def test_serialize_json_col_converts_numpy() -> None:
    from core import serialize_json_col

    data = [np.int64(1), np.float64(2.5), np.bool_(True)]
    result = serialize_json_col(data)
    parsed = json.loads(result)
    assert parsed == [1, 2.5, True]


@pytest.mark.asyncio
async def test_deduplicate_removes_duplicates() -> None:
    from core import _deduplicate

    df = pl.DataFrame({"id": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]})
    result = _deduplicate(df)
    assert len(result) == 3
    assert list(result["id"]) == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_deduplicate_no_id_column() -> None:
    from core import _deduplicate

    df = pl.DataFrame({"other_col": [1, 2, 3]})
    result = _deduplicate(df)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_list_raw_files_no_files() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(side_effect=FileNotFoundError)

    with patch("core.gcsfs.GCSFileSystem", return_value=mock_fs):
        from core import _list_raw_files

        result = _list_raw_files("test-bucket")
        assert result == []


@pytest.mark.asyncio
async def test_list_raw_files_empty() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(return_value=[])

    with patch("core.gcsfs.GCSFileSystem", return_value=mock_fs):
        from core import _list_raw_files

        result = _list_raw_files("test-bucket")
        assert result == []


@pytest.mark.asyncio
async def test_list_raw_files_latest_only() -> None:
    files = [
        "gs://bucket/jobs_raw/job_20250601.parquet",
        "gs://bucket/jobs_raw/job_20250602.parquet",
    ]
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(return_value=files)

    with patch("core.gcsfs.GCSFileSystem", return_value=mock_fs):
        from core import _list_raw_files

        result = _list_raw_files("bucket", days=None)
        assert len(result) == 1
        assert result[0] == files[-1]


@pytest.mark.asyncio
async def test_run_pipeline_basic() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(return_value=["gs://bucket/jobs_raw/test.parquet"])
    mock_fs.info = MagicMock(return_value={"updated": "2025-06-01T00:00:00Z"})

    test_df = pl.DataFrame(
        {
            "id": ["J1", "J2"],
            "intitule": ["Dev Python", "Data Engineer"],
            "description": ["<p>Python dev</p>", "<b>Data engineer</b>"],
            "competences": [None, None],
            "formations": [None, None],
            "qualitesProfessionnelles": [None, None],
        }
    )

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.array([[0.1] * 384, [0.2] * 384]))

    with (
        patch("core.gcsfs.GCSFileSystem", return_value=mock_fs),
        patch("core.pl.read_parquet", return_value=test_df),
        patch("core.SentenceTransformer", return_value=mock_model),
        patch.object(pl.DataFrame, "write_parquet") as mock_write,
    ):
        from core import run_pipeline

        silver, gold = run_pipeline("bucket", days=None)
        assert silver is not None
        assert gold is not None
        assert mock_model.encode.called
        assert mock_write.call_count == 2


@pytest.mark.asyncio
async def test_run_pipeline_no_files() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(side_effect=FileNotFoundError)

    with patch("core.gcsfs.GCSFileSystem", return_value=mock_fs):
        from core import run_pipeline

        silver, gold = run_pipeline("bucket")
        assert silver is None
        assert gold is None


@pytest.mark.asyncio
async def test_run_pipeline_with_duplicates() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(return_value=["gs://bucket/jobs_raw/test.parquet"])
    mock_fs.info = MagicMock(return_value={"updated": "2025-06-01T00:00:00Z"})

    test_df = pl.DataFrame(
        {
            "id": ["J1", "J1", "J2"],
            "intitule": ["Dev", "Dev", "Data"],
            "description": ["a", "b", "c"],
            "competences": [None, None, None],
            "formations": [None, None, None],
            "qualitesProfessionnelles": [None, None, None],
        }
    )

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.array([[0.1] * 384, [0.2] * 384]))

    with (
        patch("core.gcsfs.GCSFileSystem", return_value=mock_fs),
        patch("core.pl.read_parquet", return_value=test_df),
        patch("core.SentenceTransformer", return_value=mock_model),
        patch.object(pl.DataFrame, "write_parquet"),
    ):
        from core import run_pipeline

        silver, gold = run_pipeline("bucket", days=None)
        assert silver is not None
        assert gold is not None
        args = mock_model.encode.call_args[0][0]
        assert len(args) == 2


@pytest.mark.asyncio
async def test_run_pipeline_max_jobs() -> None:
    mock_fs = MagicMock()
    mock_fs.glob = MagicMock(return_value=["gs://bucket/jobs_raw/test.parquet"])
    mock_fs.info = MagicMock(return_value={"updated": "2025-06-01T00:00:00Z"})

    test_df = pl.DataFrame(
        {
            "id": [f"J{i}" for i in range(10)],
            "intitule": ["Dev"] * 10,
            "description": ["a"] * 10,
            "competences": [None] * 10,
            "formations": [None] * 10,
            "qualitesProfessionnelles": [None] * 10,
        }
    )

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.array([[0.1] * 384] * 3))

    with (
        patch("core.gcsfs.GCSFileSystem", return_value=mock_fs),
        patch("core.pl.read_parquet", return_value=test_df),
        patch("core.SentenceTransformer", return_value=mock_model),
        patch.object(pl.DataFrame, "write_parquet"),
    ):
        from core import run_pipeline

        silver, gold = run_pipeline("bucket", days=None, max_jobs=3)
        assert silver is not None
        assert len(mock_model.encode.call_args[0][0]) == 3
