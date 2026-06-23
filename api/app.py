import logging
import os
import time

from embed_cv_search import embed_cv_and_search_async
from fastapi import FastAPI, File, HTTPException, UploadFile
from utils import extract_text_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV-Embedding Engine API")


@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}


@app.post("/embed-cv")
async def embed_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")

    t_start = time.time()

    file_bytes = await file.read()
    t_read = time.time()
    logger.info("File read: %.2fs", t_read - t_start)

    text = extract_text_from_pdf(file_bytes)
    t_extract = time.time()
    logger.info("PDF extract: %.2fs", t_extract - t_read)

    top_jobs = await embed_cv_and_search_async(text, t_api_start=t_start)
    t_end = time.time()
    logger.info("Total API time: %.2fs", t_end - t_start)

    return {"top_jobs": top_jobs}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
