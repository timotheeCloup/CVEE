from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_extract_text_from_pdf_with_text(sample_pdf_bytes: bytes) -> None:
    from utils import extract_text_from_pdf

    result = extract_text_from_pdf(sample_pdf_bytes)
    assert result == "Développeur Python backend"


@pytest.mark.asyncio
async def test_extract_text_from_pdf_empty(pdf_with_no_text: bytes) -> None:
    from utils import extract_text_from_pdf

    result = extract_text_from_pdf(pdf_with_no_text)
    assert result == ""


@pytest.mark.asyncio
async def test_extract_text_from_pdf_invalid() -> None:
    from pypdf.errors import PdfReadError
    from utils import extract_text_from_pdf

    with pytest.raises(PdfReadError):
        extract_text_from_pdf(b"not-a-valid-pdf")


@pytest.mark.asyncio
async def test_search_jobs_vector_hybrid_returns_results() -> None:
    mock_row = (
        "123ABC",
        0.72,
        0.08,
        0.30,
        0.50,
        "Développeur Python",
        "TechCorp",
        "Paris",
        "CDI",
        "2025-06-01T00:00:00Z",
        ["python", "fastapi"],
        "48.8566",
        "2.3522",
        None,
        None,
    )

    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[mock_row])
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection = MagicMock(return_value=mock_conn)
    mock_pool.__aenter__ = AsyncMock(return_value=mock_pool)
    mock_pool.__aexit__ = AsyncMock(return_value=None)

    with patch("utils._get_pool", AsyncMock(return_value=mock_pool)):
        from utils import search_jobs_vector_hybrid

        embedding = [0.1] * 384
        results = await search_jobs_vector_hybrid(
            embedding=embedding,
            cv_text_fts="développeur python",
            cv_text_orig="Développeur Python expérimenté",
        )
        assert len(results) == 1
        assert results[0]["job_id"] == "123ABC"
        assert "similarity_score" in results[0]
        assert results[0]["matching_terms"] == ["python", "fastapi"]
        assert results[0]["latitude"] == 48.8566
        assert results[0]["longitude"] == 2.3522

        # No filters -> both filter params are NULL (no corpus restriction).
        sql_params = mock_cursor.execute.call_args_list[1].args[1]
        assert sql_params["departements"] is None
        assert sql_params["types_contrat"] is None


@pytest.mark.asyncio
async def test_search_jobs_vector_hybrid_forwards_filters() -> None:
    mock_row = (
        "123ABC",
        0.72,
        0.08,
        0.30,
        0.50,
        "Développeur Python",
        "TechCorp",
        "Lyon",
        "CDI",
        "2025-06-01T00:00:00Z",
        ["python"],
        None,
        None,
        "69123",
        None,
    )

    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(return_value=[mock_row])
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection = MagicMock(return_value=mock_conn)

    with patch("utils._get_pool", AsyncMock(return_value=mock_pool)):
        from utils import search_jobs_vector_hybrid

        results = await search_jobs_vector_hybrid(
            embedding=[0.1] * 384,
            cv_text_fts="développeur python",
            cv_text_orig="Développeur Python expérimenté",
            departements=["69"],
            types_contrat=["CDI"],
        )
        assert len(results) == 1

        sql_params = mock_cursor.execute.call_args_list[1].args[1]
        assert sql_params["departements"] == ["69"]
        assert sql_params["types_contrat"] == ["CDI"]

        # No coordinates in the offer -> resolved from the INSEE code.
        assert results[0]["latitude"] == 45.758
        assert results[0]["longitude"] == 4.8351


def test_resolve_coordinates_prefers_insee_then_postal_then_department() -> None:
    from utils import resolve_coordinates

    assert resolve_coordinates("69123", "69001", "69 - Lyon") == (45.758, 4.8351)
    assert resolve_coordinates(None, "69001", "69 - Lyon") == (45.7701, 4.8264)
    assert resolve_coordinates(None, None, "69 - Rhône") == (45.8507, 4.6695)
    assert resolve_coordinates(None, None, "75 - Paris") == (48.8602, 2.3442)


def test_resolve_coordinates_returns_none_for_unmappable() -> None:
    from utils import resolve_coordinates

    assert resolve_coordinates(None, None, "France") == (None, None)
    assert resolve_coordinates(None, None, "Île-de-France") == (None, None)
    assert resolve_coordinates(None, None, None) == (None, None)
    assert resolve_coordinates(None, "99999", " (FRONTALIER)") == (None, None)


def test_to_float_handles_invalid_values() -> None:
    from utils import _to_float

    assert _to_float(None) is None
    assert _to_float("") is None
    assert _to_float("abc") is None
    assert _to_float("48.8566") == 48.8566
    assert _to_float(2.3522) == 2.3522
