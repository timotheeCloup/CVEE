import asyncio
import json
import logging
import os
import re
import time

import aiohttp
from utils import search_jobs_vector_hybrid

logger = logging.getLogger(__name__)

_device = "cpu"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=_device)
    return _model


def load_french_stopwords():
    """Load French stopwords from JSON file"""
    path = os.path.join(os.path.dirname(__file__), "stopwords.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("fr", []))
    except Exception:
        return set()


FRENCH_STOPWORDS = load_french_stopwords()


def clean_text_for_fts(text):
    """Clean CV text for full-text search: remove French stopwords and short words."""
    text = re.sub(r"\s+", " ", text).lower()
    text = re.sub(r"[^\w\s\+\#\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [w for w in words if w not in FRENCH_STOPWORDS and len(w) > 1]
    return " ".join(words).strip()


async def verify_job_link(job_id: str, timeout: float = 0.2) -> dict:
    """
    Verify if a job offer link is still available on France Travail.
    Uses aggressive timeout (200ms) to fail fast on dead links.
    """
    job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job_id}"
    try:
        async with aiohttp.ClientSession() as session, session.head(
            job_url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True
        ) as resp:
            return {"job_id": job_id, "alive": resp.status == 200, "status": resp.status}
    except TimeoutError:
        return {"job_id": job_id, "alive": True, "status": "timeout"}
    except Exception:
        return {"job_id": job_id, "alive": True, "status": "error"}


async def filter_dead_jobs(top_jobs: list, max_concurrent: int = 10) -> list:
    """Filter out dead job offers using parallel HEAD requests."""
    if not top_jobs:
        return top_jobs

    t_start = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(job):
        async with semaphore:
            return await verify_job_link(job["job_id"])

    verification_results = await asyncio.gather(
        *[check_with_semaphore(job) for job in top_jobs], return_exceptions=True
    )

    alive_ids = set()
    dead_count = 0
    for result in verification_results:
        if isinstance(result, Exception):
            continue
        if result.get("alive", True):
            alive_ids.add(result["job_id"])
        else:
            dead_count += 1

    filtered_jobs = [job for job in top_jobs if job["job_id"] in alive_ids]
    t_end = time.time()
    logger.info(
        "Link verification: %.3fs | Checked: %d | Dead: %d | Alive: %d",
        t_end - t_start,
        len(top_jobs),
        dead_count,
        len(filtered_jobs),
    )
    return filtered_jobs


def embed_cv_and_search(cv_text, t_api_start=None):
    """
    Search jobs using hybrid FTS + embedding approach.

    Steps: 1. Clean CV text for FTS  2. Generate embedding  3. Hybrid search
    Uses a multilingual model, no external translation API needed.
    """
    if t_api_start is None:
        t_api_start = time.time()

    t0 = time.time()

    # Clean for FTS (French stopwords)
    cv_text_for_fts = clean_text_for_fts(cv_text)
    t1 = time.time()
    logger.info(
        "1. FTS preparation: %.2fs | original: %d chars, after stopwords: %d chars",
        t1 - t0,
        len(cv_text),
        len(cv_text_for_fts),
    )

    # Generate embedding (multilingual model handles French natively)
    embedding = _get_model().encode(cv_text).tolist()
    t2 = time.time()
    logger.info("2. Embedding: %.2fs | dim: %d", t2 - t1, len(embedding))

    # Hybrid search
    top_jobs = search_jobs_vector_hybrid(
        embedding=embedding, cv_text_fts=cv_text_for_fts, cv_text_orig=cv_text
    )
    t3 = time.time()
    logger.info("3. Hybrid search: %.2fs | results: %d jobs", t3 - t2, len(top_jobs))
    logger.info("Total time: %.2fs", t3 - t_api_start)

    return top_jobs


async def embed_cv_and_search_async(cv_text, t_api_start=None):
    """Async wrapper: blocking search + async link verification."""
    top_jobs = embed_cv_and_search(cv_text, t_api_start)
    verified_jobs = await filter_dead_jobs(top_jobs)
    return verified_jobs
