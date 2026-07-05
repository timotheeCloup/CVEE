import os
import time
import traceback

import structlog
from config import settings
from embed_cv_search import embed_cv_and_search_async
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from models import EmbedResponse, HealthResponse
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint for Cloud Run"""
    return HealthResponse(status="healthy")


@app.post("/embed-cv", response_model=EmbedResponse)
@limiter.limit("5/minute")
async def embed_cv(request: Request, file: UploadFile = File(...)) -> EmbedResponse:
    """Extract text from uploaded PDF, generate embedding, and return matching jobs.

    Rate-limited to 5 requests/minute (costly compute: embedding model + DB search).
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")

    t_start = time.time()

    file_bytes = await file.read()
    t_read = time.time()
    logger.info(
        "file_read",
        duration=round(t_read - t_start, 2),
        filename=file.filename,
        size_bytes=len(file_bytes),
    )

    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        logger.error("pdf_extract_error", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}") from e

    t_extract = time.time()
    logger.info("pdf_extract", duration=round(t_extract - t_read, 2), text_chars=len(text))

    if not text.strip():
        logger.warning("empty_cv_text", text_chars=len(text))
        return EmbedResponse(top_jobs=[])

    try:
        top_jobs = await embed_cv_and_search_async(text, t_api_start=t_start)
    except Exception as e:
        logger.error(
            "embed_search_error",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            elapsed=round(time.time() - t_start, 2),
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e

    t_end = time.time()
    logger.info(
        "request_complete",
        total_duration=round(t_end - t_start, 2),
        job_count=len(top_jobs),
    )

    try:
        return EmbedResponse(top_jobs=top_jobs)
    except Exception as e:
        logger.error(
            "response_validation_error",
            error=str(e),
            error_type=type(e).__name__,
            job_count=len(top_jobs),
            first_job=top_jobs[0] if top_jobs else None,
            traceback=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=f"Response validation failed: {e}") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(settings.port)))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104 -- intended for container
