from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from app import app

    return TestClient(app)


@pytest.mark.asyncio
async def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_embed_cv_pdf_success(
    client: TestClient, mock_search_results: list[dict], sample_pdf_bytes: bytes
) -> None:
    with patch("app.embed_cv_and_search_async", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_search_results

        response = client.post(
            "/embed-cv",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "top_jobs" in data
        assert len(data["top_jobs"]) == 2
        assert data["top_jobs"][0]["job_id"] == "123ABC"


@pytest.mark.asyncio
async def test_embed_cv_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/embed-cv",
        files={"file": ("test.txt", b"not-a-pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_embed_cv_empty_results(client: TestClient, sample_pdf_bytes: bytes) -> None:
    with patch("app.embed_cv_and_search_async", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        response = client.post(
            "/embed-cv",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json() == {"top_jobs": []}
