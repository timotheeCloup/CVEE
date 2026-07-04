import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF containing French text for testing."""

    def _make_pdf_with_text(text: str) -> bytes:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        stream_bytes = stream.encode("latin-1", errors="replace")

        header = b"%PDF-1.4\n"
        obj1 = b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        obj2 = b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        obj3 = (
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
            b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>"
            b"/Contents 4 0 R>>endobj\n"
        )
        obj4 = (
            b"4 0 obj<</Length "
            + str(len(stream_bytes)).encode()
            + b">>stream\n"
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

        parts = [header]
        offsets = []
        for obj in [obj1, obj2, obj3, obj4]:
            offsets.append(len(b"".join(parts)))
            parts.append(obj)

        xref_start = len(b"".join(parts))
        xref = b"xref\n0 5\n0000000000 65535 f \n"
        for off in offsets:
            xref += f"{off:010d} 00000 n \n".encode()
        parts.append(xref)
        parts.append(f"trailer<</Size 5/Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF".encode())

        return b"".join(parts)

    return _make_pdf_with_text("Développeur Python backend")


@pytest.fixture
def pdf_with_no_text() -> bytes:
    """A valid PDF with no text content."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def mock_search_results() -> list[dict]:
    """Sample search results mimicking the API response."""
    return [
        {
            "job_id": "123ABC",
            "similarity_score": 0.85,
            "embedding_score": 0.72,
            "fts_score": 0.08,
            "combined_score": 0.80,
            "intitule": "Développeur Python Senior",
            "entreprise": "TechCorp",
            "lieu": "Paris",
            "type_contrat": "CDI",
            "date_creation": "2025-06-01T00:00:00Z",
            "matching_terms": ["python", "développeur", "senior"],
        },
        {
            "job_id": "456DEF",
            "similarity_score": 0.72,
            "embedding_score": 0.60,
            "fts_score": 0.05,
            "combined_score": 0.65,
            "intitule": "Data Engineer",
            "entreprise": "DataInc",
            "lieu": "Lyon",
            "type_contrat": "CDI",
            "date_creation": "2025-05-15T00:00:00Z",
            "matching_terms": ["data", "python", "engineer"],
        },
    ]


@pytest.fixture(autouse=True)
def _mock_api_module_deps():
    """Auto-mock heavy dependencies so api module can be imported without real infra."""
    with (
        patch("embed_cv_search._get_model", return_value=MagicMock()),
        patch("utils._get_pool", new_callable=AsyncMock),
    ):
        yield


@pytest.fixture(autouse=True)
def _add_src_to_path():
    """Add api/ and functions/pipeline/ to sys.path for test imports."""
    from pathlib import Path

    root = Path(__file__).parent.parent
    for subdir in ("api", "functions/pipeline"):
        p = root / subdir
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
