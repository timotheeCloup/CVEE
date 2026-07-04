import os
import time
from typing import Any

import structlog
from embed_cv_search import embed_cv_and_search_async
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.util import get_remote_address
from utils import extract_text_from_pdf

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

app = FastAPI(title="CV-Embedding Engine API")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


@app.exception_handler(429)
async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Max 10 requests/minute."},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}


@app.post("/embed-cv")
@limiter.limit("5/minute")
async def embed_cv(request: Request, file: UploadFile = File(...)) -> dict[str, list[Any]]:
    """Extract text from uploaded PDF, generate embedding, and return matching jobs.

    Rate-limited to 5 requests/minute (costly compute: embedding model + DB search).

    Args:
        request: FastAPI request object (required by slowapi).
        file: PDF file upload (max 5MB).

    Returns:
        Dict with ``top_jobs`` key containing a list of matching job results,
        each with job_id, similarity_score, intitule, entreprise, lieu, etc.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")

    t_start = time.time()

    file_bytes = await file.read()
    t_read = time.time()
    logger.info("file_read", duration=round(t_read - t_start, 2))

    text = extract_text_from_pdf(file_bytes)
    t_extract = time.time()
    logger.info("pdf_extract", duration=round(t_extract - t_read, 2))

    top_jobs = await embed_cv_and_search_async(text, t_api_start=t_start)
    t_end = time.time()
    logger.info("request_complete", total_duration=round(t_end - t_start, 2))

    return {"top_jobs": top_jobs}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104 -- intended for container
