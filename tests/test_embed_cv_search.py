from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_clean_text_for_fts_removes_stopwords() -> None:
    from embed_cv_search import clean_text_for_fts

    result = clean_text_for_fts("Développeur Python avec de la expérience data")
    words = result.split()
    assert "de" not in words
    assert "la" not in words
    assert "avec" not in words


@pytest.mark.asyncio
async def test_clean_text_for_fts_preserves_meaningful() -> None:
    from embed_cv_search import clean_text_for_fts

    result = clean_text_for_fts("Développeur Python GCP FastAPI")
    words = result.split()
    assert "développeur" in words
    assert "python" in words
    assert "gcp" in words
    assert "fastapi" in words


@pytest.mark.asyncio
async def test_clean_text_for_fts_removes_short_words() -> None:
    from embed_cv_search import clean_text_for_fts

    result = clean_text_for_fts("a b c x y z python")
    words = result.split()
    assert "a" not in words
    assert "b" not in words
    assert "python" in words


@pytest.mark.asyncio
async def test_clean_text_for_fts_strips_special_chars() -> None:
    from embed_cv_search import clean_text_for_fts

    result = clean_text_for_fts("Développeur@Python!GCP")
    assert "@" not in result
    assert "!" not in result


@pytest.mark.asyncio
async def test_clean_text_for_fts_empty() -> None:
    from embed_cv_search import clean_text_for_fts

    result = clean_text_for_fts("")
    assert result == ""

    result = clean_text_for_fts("de la le les un une des")
    assert result == ""


@pytest.mark.asyncio
async def test_verify_job_link_alive() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_session = AsyncMock()
    mock_session.head = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        from embed_cv_search import verify_job_link

        result = await verify_job_link("123ABC")
        assert result["alive"] is True
        assert result["status"] == 200


@pytest.mark.asyncio
async def test_verify_job_link_dead() -> None:
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_session = AsyncMock()
    mock_session.head = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        from embed_cv_search import verify_job_link

        result = await verify_job_link("456DEF")
        assert result["alive"] is False
        assert result["status"] == 404


@pytest.mark.asyncio
async def test_verify_job_link_timeout() -> None:
    import asyncio

    mock_session = AsyncMock()
    mock_session.head = MagicMock(side_effect=asyncio.TimeoutError)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        from embed_cv_search import verify_job_link

        result = await verify_job_link("789GHI")
        assert result["alive"] is True
        assert result["status"] == "timeout"


@pytest.mark.asyncio
async def test_filter_dead_jobs_removes_dead() -> None:
    from embed_cv_search import filter_dead_jobs

    jobs = [
        {"job_id": "A", "intitule": "Job A"},
        {"job_id": "B", "intitule": "Job B"},
        {"job_id": "C", "intitule": "Job C"},
    ]

    async def mock_verify(job_id: str, timeout: float = 0.2) -> dict:
        results = {
            "A": {"job_id": "A", "alive": True, "status": 200},
            "B": {"job_id": "B", "alive": False, "status": 404},
            "C": {"job_id": "C", "alive": True, "status": 200},
        }
        return results[job_id]

    with patch("embed_cv_search.verify_job_link", side_effect=mock_verify):
        result = await filter_dead_jobs(jobs)
        assert len(result) == 2
        assert all(j["job_id"] in ("A", "C") for j in result)


@pytest.mark.asyncio
async def test_filter_dead_jobs_empty() -> None:
    from embed_cv_search import filter_dead_jobs

    result = await filter_dead_jobs([])
    assert result == []


@pytest.mark.asyncio
async def test_verify_job_link_unexpected_error() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(side_effect=RuntimeError("unexpected"))
    mock_session = AsyncMock()
    mock_session.head = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        from embed_cv_search import verify_job_link

        result = await verify_job_link("ERROR")
        assert result["alive"] is True
        assert result["status"] == "error"


@pytest.mark.asyncio
async def test_filter_dead_jobs_handles_exceptions() -> None:
    from embed_cv_search import filter_dead_jobs

    jobs = [
        {"job_id": "A", "intitule": "Job A"},
        {"job_id": "B", "intitule": "Job B"},
    ]

    async def mock_verify(job_id: str, timeout: float = 0.2) -> dict:
        if job_id == "A":
            raise RuntimeError("network error")
        return {"job_id": job_id, "alive": True, "status": 200}

    with patch("embed_cv_search.verify_job_link", side_effect=mock_verify):
        result = await filter_dead_jobs(jobs)
        assert len(result) == 1
        assert result[0]["job_id"] == "B"


@pytest.mark.asyncio
async def test_load_french_stopwords_file_not_found() -> None:
    with patch("builtins.open", side_effect=FileNotFoundError):
        import importlib

        import embed_cv_search

        importlib.reload(embed_cv_search)
        assert embed_cv_search.load_french_stopwords() == set()
        importlib.reload(embed_cv_search)


@pytest.mark.asyncio
async def test_embed_cv_and_search_returns_results() -> None:
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.1] * 384))

    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)
    mock_cursor.execute = AsyncMock()
    mock_cursor.fetchall = AsyncMock(
        return_value=[
            (
                "123ABC",
                0.72,
                0.08,
                0.02,
                "Développeur Python",
                "TechCorp",
                "Paris",
                "CDI",
                "2025-06-01",
                "Développeur <b>Python</b>",
            )
        ]
    )
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection = MagicMock(return_value=mock_conn)

    with (
        patch("embed_cv_search._get_model", return_value=mock_model),
        patch("utils._get_pool", AsyncMock(return_value=mock_pool)),
    ):
        from embed_cv_search import embed_cv_and_search

        results = await embed_cv_and_search("Développeur Python expérimenté")
        assert len(results) == 1
        assert results[0]["job_id"] == "123ABC"
