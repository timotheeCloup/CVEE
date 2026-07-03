import asyncio
import json
import os
import re
import time
from typing import Any

import aiohttp
import structlog
import torch
from utils import search_jobs_vector_hybrid

logger: Any = structlog.get_logger()

torch.set_num_threads(1)

MODEL_NAME: str = "antoinelouis/french-me5-small"
_device: str = "cpu"
_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        t0 = time.time()
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device=_device)
        logger.info("model_loaded", cold_start_duration=round(time.time() - t0, 2))
    return _model


def load_french_stopwords() -> set[str]:
    """Load French stopwords from JSON file"""
    path = os.path.join(os.path.dirname(__file__), "stopwords.json")
    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, list[str]] = json.load(f)
            return set(data.get("fr", []))
    except Exception:
        return set()


FRENCH_STOPWORDS: set[str] = load_french_stopwords()


def clean_text_for_fts(text: str) -> str:
    """Clean CV text for full-text search: remove French stopwords and short words."""
    text = text.strip()
    words = text.split()
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_len < 1.5:
        text = text.lower()
        text = re.sub(r"  +", "\n", text)
        text = re.sub(r" ", "", text)
        text = re.sub(r"\n", " ", text)
    else:
        text = re.sub(r"\s+", " ", text).lower()
    text = re.sub(r"[^\w\s\+\#\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [w for w in words if w not in FRENCH_STOPWORDS and len(w) > 1]
    return " ".join(words).strip()


async def verify_job_link(job_id: str, timeout: float = 0.2) -> dict[str, Any]:
    """
    Verify if a job offer link is still available on France Travail.
    Uses aggressive timeout (200ms) to fail fast on dead links.
    """
    job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job_id}"
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.head(
                job_url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True
            ) as resp,
        ):
            return {"job_id": job_id, "alive": resp.status == 200, "status": resp.status}
    except TimeoutError:
        return {"job_id": job_id, "alive": True, "status": "timeout"}
    except Exception:
        return {"job_id": job_id, "alive": True, "status": "error"}


async def filter_dead_jobs(
    top_jobs: list[dict[str, Any]], max_concurrent: int = 10
) -> list[dict[str, Any]]:
    """Filter out dead job offers using parallel HEAD requests."""
    if not top_jobs:
        return top_jobs

    t_start = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(job: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await verify_job_link(job["job_id"])

    verification_results: list[Any] = await asyncio.gather(
        *[check_with_semaphore(job) for job in top_jobs], return_exceptions=True
    )

    alive_ids: set[str] = set()
    dead_count = 0
    for result in verification_results:
        if isinstance(result, Exception):
            continue
        if result.get("alive", True):
            alive_ids.add(result["job_id"])
        else:
            dead_count += 1

    filtered_jobs: list[dict[str, Any]] = [job for job in top_jobs if job["job_id"] in alive_ids]
    t_end = time.time()
    logger.info(
        "link_verification",
        duration=round(t_end - t_start, 3),
        checked=len(top_jobs),
        dead=dead_count,
        alive=len(filtered_jobs),
    )
    return filtered_jobs


def embed_cv_and_search(cv_text: str, t_api_start: float | None = None) -> list[dict[str, Any]]:
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
        "fts_preparation",
        duration=round(t1 - t0, 2),
        original_chars=len(cv_text),
        after_stopwords=len(cv_text_for_fts),
    )

    # Generate embedding (multilingual model handles French natively)
    embedding: list[float] = _get_model().encode(cv_text).tolist()
    t2 = time.time()
    logger.info("embedding", duration=round(t2 - t1, 2), dim=len(embedding))

    # Hybrid search
    top_jobs: list[dict[str, Any]] = search_jobs_vector_hybrid(
        embedding=embedding, cv_text_fts=cv_text_for_fts, cv_text_orig=cv_text
    )
    t3 = time.time()
    logger.info("hybrid_search", duration=round(t3 - t2, 2), results=len(top_jobs))
    logger.info("search_complete", total_duration=round(t3 - t_api_start, 2))

    return top_jobs


async def embed_cv_and_search_async(
    cv_text: str, t_api_start: float | None = None
) -> list[dict[str, Any]]:
    """Async wrapper: blocking search in thread + async link verification."""
    top_jobs = await asyncio.to_thread(embed_cv_and_search, cv_text, t_api_start)
    verified_jobs = await filter_dead_jobs(top_jobs)
    return verified_jobs
