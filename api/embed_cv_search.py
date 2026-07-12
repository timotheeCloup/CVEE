import asyncio
import json
import os
import re
import time
from typing import Any

import aiohttp
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import structlog
from tokenizers import Tokenizer
from utils import search_jobs_vector_hybrid

logger: Any = structlog.get_logger()

# ONNX model exported from this SentenceTransformer at Docker build time.
MODEL_NAME: str = "antoinelouis/french-me5-small"
MAX_SEQ_LENGTH: int = 512
MODEL_DIR: str = os.getenv("ONNX_MODEL_DIR", os.path.join(os.path.dirname(__file__), "onnx_model"))


class OnnxEncoder:
    """SentenceTransformer.encode replacement backed by onnxruntime.

    Reproduces the model's sentence-embedding pipeline exactly:
    tokenize -> transformer -> attention-masked mean pooling -> L2 normalize.
    This yields embeddings identical to the SentenceTransformer model that
    generated the job embeddings stored in the DB (verified: cosine ~1.0),
    while dropping the torch / sentence-transformers runtime dependencies to
    slash cold-start time and image size.
    """

    def __init__(self, session: Any, tokenizer: Any) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {i.name for i in session.get_inputs()}

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        encoded = self._tokenizer.encode(text)
        ids: npt.NDArray[np.int64] = np.array([encoded.ids], dtype=np.int64)
        mask: npt.NDArray[np.int64] = np.array([encoded.attention_mask], dtype=np.int64)
        feed: dict[str, npt.NDArray[np.int64]] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        last_hidden: npt.NDArray[np.float32] = np.asarray(
            self._session.run(None, feed)[0], dtype=np.float32
        )
        mask_f = mask[:, :, None].astype(np.float32)
        summed = (last_hidden * mask_f).sum(axis=1)
        counts = np.clip(mask_f.sum(axis=1), 1e-9, None)
        mean = summed / counts
        normed = mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)
        return np.asarray(normed[0], dtype=np.float32)


_encoder: OnnxEncoder | None = None


def _get_model() -> OnnxEncoder:
    global _encoder
    if _encoder is None:
        t0 = time.time()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        _encoder = OnnxEncoder(session, tokenizer)
        logger.info("model_loaded", cold_start_duration=round(time.time() - t0, 2))
    return _encoder


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
    """Filter out dead job offers using parallel HEAD requests.

    Args:
        top_jobs: Job results from hybrid search.
        max_concurrent: Max simultaneous HEAD requests (default 10).

    Returns:
        Filtered list of jobs whose France Travail links are still alive.
    """
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


async def embed_cv_and_search(
    cv_text: str, t_api_start: float | None = None
) -> list[dict[str, Any]]:
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
    try:
        embedding: list[float] = (await asyncio.to_thread(_get_model().encode, cv_text)).tolist()
    except Exception as e:
        logger.error(
            "embedding_error",
            error=str(e),
            error_type=type(e).__name__,
            original_chars=len(cv_text),
            after_stopwords=len(cv_text_for_fts),
        )
        raise
    t2 = time.time()
    logger.info("embedding", duration=round(t2 - t1, 2), dim=len(embedding))

    # Hybrid search
    top_jobs: list[dict[str, Any]] = await search_jobs_vector_hybrid(
        embedding=embedding, cv_text_fts=cv_text_for_fts, cv_text_orig=cv_text
    )
    t3 = time.time()
    logger.info("hybrid_search", duration=round(t3 - t2, 2), results=len(top_jobs))
    logger.info("search_complete", total_duration=round(t3 - t_api_start, 2))

    return top_jobs


async def embed_cv_and_search_async(
    cv_text: str, t_api_start: float | None = None
) -> list[dict[str, Any]]:
    """Search jobs via hybrid FTS+embedding, then filter dead links.

    Args:
        cv_text: Full text extracted from the CV PDF.
        t_api_start: Optional start timestamp for total duration logging.

    Returns:
        List of verified (alive) matching job results.
    """
    top_jobs = await embed_cv_and_search(cv_text, t_api_start)
    verified_jobs = await filter_dead_jobs(top_jobs)
    return verified_jobs
