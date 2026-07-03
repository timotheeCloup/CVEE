import os
import time
from typing import Any

import structlog
from embed_cv_search import embed_cv_and_search_async
from fastapi import FastAPI, File, HTTPException, UploadFile
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


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}


@app.post("/embed-cv")
async def embed_cv(file: UploadFile = File(...)) -> dict[str, list[Any]]:
    if not file.filename.endswith(".pdf"):
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
    uvicorn.run(app, host="0.0.0.0", port=port)
