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
async def test_extract_french_keywords_from_headline() -> None:
    from utils import extract_french_keywords_from_headline

    headline = "Développeur <b>Python</b> avec expérience en <b>FastAPI</b> et <b>GCP</b>"
    result = extract_french_keywords_from_headline(headline)
    assert result == ["python", "fastapi", "gcp"]


@pytest.mark.asyncio
async def test_extract_french_keywords_from_empty() -> None:
    from utils import extract_french_keywords_from_headline

    assert extract_french_keywords_from_headline("") == []
    assert extract_french_keywords_from_headline(None) == []


@pytest.mark.asyncio
async def test_extract_french_keywords_short_words_filtered() -> None:
    from utils import extract_french_keywords_from_headline

    headline = "<b>de</b> <b>la</b> <b>Big Data</b>"
    result = extract_french_keywords_from_headline(headline)
    assert "de" not in result
    assert "la" not in result


@pytest.mark.asyncio
async def test_extract_french_keywords_dedup() -> None:
    from utils import extract_french_keywords_from_headline

    headline = "<b>Python</b> and <b>python</b> and <b>PYTHON</b>"
    result = extract_french_keywords_from_headline(headline)
    assert len(result) == 1
    assert "python" in result


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
        "Développeur <b>Python</b> avec expérience en <b>FastAPI</b>",
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
        assert "matching_terms" in results[0]

        # No filters -> both filter params are NULL (no corpus restriction).
        sql_params = mock_cursor.execute.call_args_list[1].args[1]
        assert None in sql_params


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
        "Développeur <b>Python</b>",
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
        assert ["69"] in sql_params
        assert ["CDI"] in sql_params
