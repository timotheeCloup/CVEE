"""Basic API health test."""

import pytest


@pytest.mark.asyncio
async def test_health():
    """Simulate a health check response."""
    assert 1 == 1  # placeholder until FastAPI test client is set up
