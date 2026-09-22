import pytest
from playwright.sync_api import expect


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

    # Metrics render slightly after the job titles: poll instead of counting once.
    expect(page.locator("[data-testid='stMetricValue']")).to_have_count(2, timeout=15000)


@pytest.mark.e2e
def test_upload_shows_offers_map(page, streamlit_url: str, sample_pdf_bytes: bytes) -> None:
    page.goto(streamlit_url, wait_until="domcontentloaded")

    uploader = page.locator("input[type='file']")
    uploader.wait_for(state="visible", timeout=15000)
    uploader.set_input_files(
        [{"name": "cv.pdf", "mimeType": "application/pdf", "buffer": sample_pdf_bytes}]
    )

    page.locator("text=Développeur Python Senior").wait_for(state="visible", timeout=30000)

    # The map is hidden by default; the "Carte" button reveals it.
    assert page.locator("[data-testid='stDeckGlJsonChart']").count() == 0
    page.get_by_role("button", name="Carte", exact=True).click()
    deck = page.locator("[data-testid='stDeckGlJsonChart']")
    deck.wait_for(state="visible", timeout=15000)
    assert deck.count() == 1


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
