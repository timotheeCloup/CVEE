import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def pytest_configure(config):  # type: ignore[no-untyped-def]
    if not config.option.browser:
        config.option.browser = ["chromium"]


class MockAPIHandler(BaseHTTPRequestHandler):
    MOCK_RESULTS = [
        {
            "job_id": "123ABC",
            "similarity_score": 0.85,
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
            "intitule": "Data Engineer",
            "entreprise": "DataInc",
            "lieu": "Lyon",
            "type_contrat": "CDI",
            "date_creation": "2025-05-15T00:00:00Z",
            "matching_terms": ["data", "python", "engineer"],
        },
    ]

    def _json(self, data: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "healthy"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/embed-cv":
            self._json({"top_jobs": self.MOCK_RESULTS})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, format, *args):  # type: ignore[override]
        pass


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_api_server() -> str:
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), MockAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    time.sleep(0.2)
    yield url
    server.shutdown()


@pytest.fixture(scope="module")
def streamlit_url(mock_api_server: str) -> str:
    port = _find_free_port()
    env = os.environ.copy()
    env["API_URL"] = f"{mock_api_server}/embed-cv"
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--server.runOnSave",
            "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}"
    for _ in range(30):
        try:
            import requests

            requests.get(f"{url}/healthz", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Streamlit did not start in time")
    yield url
    proc.terminate()
    proc.wait(timeout=10)
