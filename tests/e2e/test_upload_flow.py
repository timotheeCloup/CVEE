import pytest


@pytest.mark.e2e
def test_page_loads_title(page, streamlit_url: str) -> None:
    page.goto(streamlit_url, wait_until="domcontentloaded")
    page.locator("text=CV Match Engine").wait_for(state="visible", timeout=15000)
    assert page.locator("text=CV Match Engine").is_visible()


@pytest.mark.e2e
def test_file_uploader_visible(page, streamlit_url: str) -> None:
    page.goto(streamlit_url, wait_until="domcontentloaded")
    uploader = page.locator("input[type='file']")
    uploader.wait_for(state="visible", timeout=15000)
    assert uploader.is_visible()


@pytest.mark.e2e
def test_upload_shows_results(page, streamlit_url: str, sample_pdf_bytes: bytes) -> None:
    page.goto(streamlit_url, wait_until="domcontentloaded")

    uploader = page.locator("input[type='file']")
    uploader.wait_for(state="visible", timeout=15000)
    uploader.set_input_files(
        [{"name": "cv.pdf", "mimeType": "application/pdf", "buffer": sample_pdf_bytes}]
    )

    page.locator("text=Développeur Python Senior").wait_for(state="visible", timeout=30000)
    page.locator("text=TechCorp").wait_for(state="visible", timeout=5000)
    page.locator("text=Data Engineer").wait_for(state="visible", timeout=5000)

    score_elements = page.locator("[data-testid='stMetricValue']")
    assert score_elements.count() >= 2


@pytest.mark.e2e
def test_upload_shows_match_scores(page, streamlit_url: str, sample_pdf_bytes: bytes) -> None:
    page.goto(streamlit_url, wait_until="domcontentloaded")

    uploader = page.locator("input[type='file']")
    uploader.wait_for(state="visible", timeout=15000)
    uploader.set_input_files(
        [{"name": "cv.pdf", "mimeType": "application/pdf", "buffer": sample_pdf_bytes}]
    )

    page.locator("text=85%").wait_for(state="visible", timeout=30000)
    page.locator("text=72%").wait_for(state="visible", timeout=5000)
