from sentence_transformers import SentenceTransformer
from utils import search_jobs_vector_hybrid
from google.cloud import translate_v2
import os
import json
import time
import asyncio
import aiohttp

device = "cpu"
# 384 dimensional embedding model
model = SentenceTransformer("BAAI/bge-small-en", device=device)
translate_client = translate_v2.Client()


def load_french_stopwords():
    """Load French stopwords from JSON file"""
    path = os.path.join(os.path.dirname(__file__), 'stopwords.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('fr', []))
    except Exception:
        return set()


FRENCH_STOPWORDS = load_french_stopwords()


def clean_text_for_fts(text):
    """
    Clean CV text for full-text search.
    Removes French stopwords and short words.
    """
    import re
    text = re.sub(r'\s+', ' ', text).lower()
    text = re.sub(r'[^\w\s\+\#\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    words = [w for w in words if w not in FRENCH_STOPWORDS and len(w) > 1]
    
    return " ".join(words).strip()


def translate_text(text):
    """
    Translate French text to English using Google Cloud Translation API.
    Does NOT remove stopwords - keeps original structure for embeddings.
    """
    if not text.strip():
        return ""
    
    try:
        result = translate_client.translate(text, target_language='en')
        return result['translatedText']
    except Exception as e:
        print(f"Translation error: {str(e)}", flush=True)
        return text


async def verify_job_link(job_id: str, timeout: float = 0.2) -> dict:
    """
    Verify if a job offer link is still available on France Travail.
    Uses aggressive timeout (100ms) to fail fast on dead links.
    France Travail returns 404 for deleted/expired offers.
    
    Args:
        job_id: Job ID from database
        timeout: Timeout in seconds (default 100ms)
    
    Returns:
        Dict with job_id and alive status
    """
    job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(job_url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                # 200 = alive, 404 = dead, timeouts = assume alive (prudent)
                is_alive = resp.status == 200
                return {"job_id": job_id, "alive": is_alive, "status": resp.status}
    except asyncio.TimeoutError:
        # Timeout = assume alive (prudent, avoid false positives)
        return {"job_id": job_id, "alive": True, "status": "timeout"}
    except Exception as e:
        # Any other error = assume alive (prudent)
        return {"job_id": job_id, "alive": True, "status": "error"}


async def filter_dead_jobs(top_jobs: list, max_concurrent: int = 10) -> list:
    """
    Filter out dead job offers using lazy checking.
    Performs parallel HEAD requests with aggressive timeout.
    
    Args:
        top_jobs: List of job offers from search
        max_concurrent: Max concurrent requests (default 10)
    
    Returns:
        List of verified alive job offers
    """
    if not top_jobs:
        return top_jobs
    
    t_start = time.time()
    
    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_with_semaphore(job):
        async with semaphore:
            return await verify_job_link(job['job_id'])
    
    # Check all jobs in parallel
    verification_results = await asyncio.gather(
        *[check_with_semaphore(job) for job in top_jobs],
        return_exceptions=True
    )
    
    # Build set of alive job IDs
    alive_ids = set()
    dead_count = 0
    
    for result in verification_results:
        if isinstance(result, Exception):
            # On exception, assume alive (prudent)
            continue
        if result.get('alive', True):
            alive_ids.add(result['job_id'])
        else:
            dead_count += 1
    
    # Filter jobs
    filtered_jobs = [job for job in top_jobs if job['job_id'] in alive_ids]
    
    t_end = time.time()
    print(f"Link verification: {t_end - t_start:.3f}s | Checked: {len(top_jobs)} | Dead: {dead_count} | Alive: {len(filtered_jobs)}", flush=True)
    
    return filtered_jobs


def embed_cv_and_search(cv_text, t_api_start=None):
    """
    Search jobs using hybrid FTS + embedding approach.
    
    Steps:
    1. Clean CV text for FTS (with French stopwords)
    2. Translate CV to English
    3. Generate embedding from translated text
    4. Search jobs with hybrid approach: 0.9*embedding + 0.1*FTS
    
    Args:
        cv_text: Raw CV text (French)
        t_api_start: API call start time
    """
    if t_api_start is None:
        t_api_start = time.time()
    
    t0 = time.time()
    
    # Clean for FTS (with French stopwords)
    cv_text_for_fts = clean_text_for_fts(cv_text)
    t1 = time.time()
    print(f"\n=== PROCESSING STEPS ===", flush=True)
    print(f"1. FTS preparation:      {t1 - t0:7.2f}s", flush=True)
    print(f"   Original length:      {len(cv_text)} chars", flush=True)
    print(f"   After stopwords:      {len(cv_text_for_fts)} chars", flush=True)
    
    # Translate for embeddings (NO stopword removal)
    cv_text_en = translate_text(cv_text)
    t2 = time.time()
    print(f"2. Translation:          {t2 - t1:7.2f}s", flush=True)
    print(f"   Translated length:    {len(cv_text_en)} chars", flush=True)
    
    # Generate embedding from English text
    embedding = model.encode(cv_text_en).tolist()
    t3 = time.time()
    print(f"3. Embedding:            {t3 - t2:7.2f}s", flush=True)
    print(f"   Embedding dim:        {len(embedding)}", flush=True)
    
    # Hybrid search: FTS (French) + Embedding (English)
    top_jobs = search_jobs_vector_hybrid(
        embedding=embedding,
        cv_text_fts=cv_text_for_fts,
        cv_text_orig=cv_text
    )
    t4 = time.time()
    print(f"4. Hybrid search:        {t4 - t3:7.2f}s", flush=True)
    print(f"   Results found:        {len(top_jobs)} jobs", flush=True)
    print(f"\n5. TOTAL TIME:           {t4 - t_api_start:7.2f}s", flush=True)
    print(f"=" * 35, flush=True)
    
    return top_jobs


async def embed_cv_and_search_async(cv_text, t_api_start=None):
    """
    Async wrapper for embed_cv_and_search with lazy link verification.
    Performs blocking search then async link verification to remove dead offers.
    
    Args:
        cv_text: Raw CV text (French)
        t_api_start: API call start time
    
    Returns:
        List of verified alive job offers
    """
    # Perform synchronous search (blocking, but necessary)
    top_jobs = embed_cv_and_search(cv_text, t_api_start)
    
    # Perform async link verification (non-blocking, removes 404 offers)
    verified_jobs = await filter_dead_jobs(top_jobs)
    
    return verified_jobs