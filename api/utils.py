import io
import re
import time
from typing import Any

import structlog
from config import settings
from psycopg_pool import AsyncConnectionPool
from pypdf import PdfReader

logger: Any = structlog.get_logger()

TOP_K: int = 100
CANDIDATE_POOL: int = 1000

# Reciprocal Rank Fusion
RRF_K: int = 60
EMBED_RRF_WEIGHT: int = 3
FTS_RRF_WEIGHT: int = 1
RRF_MAX: float = (EMBED_RRF_WEIGHT + FTS_RRF_WEIGHT) / (RRF_K + 1)

# FTS weights for tsvector levels [C, B, A] (D=0 since unused)
FTS_WEIGHTS: list[float] = [0.3, 0.6, 1.0]

_db_pool: AsyncConnectionPool | None = None


async def _get_pool() -> AsyncConnectionPool:
    global _db_pool
    if _db_pool is None and settings.db_host:
        _db_pool = AsyncConnectionPool(
            conninfo=f"host={settings.db_host} dbname={settings.db_name} user={settings.db_user} password={settings.db_password} port={settings.db_port}",
            min_size=1,
            max_size=5,
            open=True,
        )
    if _db_pool is None:
        raise RuntimeError("No DB pool available (DB_HOST not set)")
    return _db_pool


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from first page of PDF"""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:
        text = reader.pages[0].extract_text() or ""
    return text.strip()


def extract_french_keywords_from_headline(headline: Any) -> list[str]:
    """Extract French keywords from ts_headline <b>...</b> fragments"""
    if not headline:
        return []
    marked_terms = re.findall(r"<b>([^<]+)</b>", str(headline))
    keywords: list[str] = []
    seen: set[str] = set()
    for term in marked_terms:
        clean_term = re.sub(r"[^\w\s]", "", term.strip().lower())
        words = [w for w in clean_term.split() if len(w) > 2]
        if words:
            final_term = " ".join(words)
            if final_term not in seen:
                keywords.append(final_term)
                seen.add(final_term)
    return keywords[:10]


def linear_mapping(
    value: float, from_min: float, from_max: float, to_min: float, to_max: float
) -> float:
    """Linearly map a value from one range to another"""
    if from_max - from_min == 0:
        return to_min
    return to_min + (to_max - to_min) * (value - from_min) / (from_max - from_min)


def _build_fts_weights_literal() -> str:
    """Build PostgreSQL tsvector weights literal string like '{0, 0.3, 0.6, 1.0}'"""
    weights = [0] + FTS_WEIGHTS  # [D, C, B, A], D=0 unused
    return "{" + ", ".join(str(w) for w in weights) + "}"


async def search_jobs_vector_hybrid(
    embedding: list[float], cv_text_fts: str, cv_text_orig: str
) -> list[dict[str, Any]]:
    """
    Hybrid job search: embedding-first with weighted Reciprocal Rank Fusion.

    Pre-selects top candidates by embedding similarity, then scores
    candidates via weighted RRF: 3/(k + embed_rank) + 1/(k + fts_rank).

    Returns top 100 jobs sorted by weighted RRF score.
    """
    t_start = time.time()

    pool = await _get_pool()
    async with pool.connection() as conn:
        t_conn = time.time()
        logger.info("db_connection", duration=round(t_conn - t_start, 3))
        logger.info("fts_prep", fts_chars=len(cv_text_fts), embedding_dim=len(embedding))

        fts_terms = cv_text_fts.split()[:1000]
        tsquery = " | ".join(f"'{term}'" for term in fts_terms) if fts_terms else ""
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        fts_weights_literal = _build_fts_weights_literal()

        sql = """
        WITH embedding_candidates AS (
            SELECT jg.job_id,
                   (1 - (jg.embedding <-> %s))::float8 as embedding_score
            FROM jobs_gold jg
            WHERE jg.fts_tokens IS NOT NULL
            ORDER BY embedding_score DESC
            LIMIT %s
        ),
        fts_candidates AS (
            SELECT jg.job_id,
                   COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8 as fts_score
            FROM jobs_gold jg
            WHERE jg.fts_tokens IS NOT NULL
            ORDER BY fts_score DESC
            LIMIT %s
        ),
        all_candidates AS (
            SELECT job_id FROM embedding_candidates
            UNION
            SELECT job_id FROM fts_candidates
        ),
        ranked AS (
            SELECT
                c.job_id,
                (1 - (jg.embedding <-> %s))::float8 as embedding_score,
                COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8 as fts_score,
                js.intitule,
                js.entreprise->>'nom' AS entreprise,
                js.lieuTravail->>'libelle' AS lieu,
                js.typeContratLibelle,
                js.dateCreation,
                ts_headline('french',
                    js.intitule || ' ' || COALESCE(js.description, '') || ' ' ||
                    COALESCE((SELECT string_agg(elem->>'libelle', ' ')
                              FROM jsonb_array_elements(js.competences) AS elem), '') || ' ' ||
                    COALESCE((SELECT string_agg((elem->>'libelle') || ' ' || (elem->>'description'), ' ')
                              FROM jsonb_array_elements(js.qualitesprofessionnelles) AS elem), ''),
                    to_tsquery('french', %s),
                    'StartSel=<b>, StopSel=</b>, MaxWords=100, MinWords=50') as headline,
                ROW_NUMBER() OVER (ORDER BY (1 - (jg.embedding <-> %s)) DESC) as embed_rank,
                ROW_NUMBER() OVER (ORDER BY COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0) DESC) as fts_rank
            FROM all_candidates c
            JOIN jobs_gold jg ON c.job_id = jg.job_id
            JOIN jobs_silver js ON jg.job_id = js.job_id
        )
        SELECT
            job_id, embedding_score, fts_score,
            (%s::float8 / (%s + embed_rank) + %s::float8 / (%s + fts_rank))::float8 as combined_score,
            intitule, entreprise, lieu, typeContratLibelle, dateCreation, headline
        FROM ranked
        ORDER BY combined_score DESC
        LIMIT %s;
        """

        async with conn.cursor() as cur:
            await cur.execute(
                sql,
                (
                    embedding_str,        # embedding_candidates
                    CANDIDATE_POOL,       # embedding_candidates LIMIT
                    fts_weights_literal,  # fts_candidates
                    tsquery,              # fts_candidates
                    CANDIDATE_POOL,       # fts_candidates LIMIT
                    embedding_str,        # ranked embedding_score
                    fts_weights_literal,  # ranked fts_score
                    tsquery,              # ranked fts_score query
                    tsquery,              # ts_headline
                    embedding_str,        # embed_rank window
                    fts_weights_literal,  # fts_rank window weights
                    tsquery,              # fts_rank window query
                    EMBED_RRF_WEIGHT,     # outer combined_score embed weight
                    RRF_K,                # outer combined_score embed k
                    FTS_RRF_WEIGHT,       # outer combined_score fts weight
                    RRF_K,                # outer combined_score fts k
                    TOP_K,                # outer LIMIT
                ),
            )
            results = await cur.fetchall()

    t_query = time.time()
    logger.info("query_execution", duration=round(t_query - t_conn, 3), results=len(results))

    # Process results (already sorted by combined_score)
    hybrid_results = []
    for r in results:
        (
            job_id,
            embedding_score,
            fts_score,
            combined_score,
            intitule,
            entreprise,
            lieu,
            type_contrat,
            date_creation,
            headline,
        ) = r
        keywords = extract_french_keywords_from_headline(headline)
        hybrid_results.append(
            {
                "job_id": job_id,
                "embedding_score": embedding_score,
                "fts_score": fts_score,
                "combined_score": combined_score,
                "intitule": intitule,
                "entreprise": entreprise,
                "lieu": lieu,
                "type_contrat": type_contrat,
                "date_creation": date_creation,
                "keywords": keywords,
            }
        )

    t_process = time.time()
    logger.info("processing", duration=round(t_process - t_query, 3))

    # Summary stats
    fts_non_zero = sum(1 for r in hybrid_results if r["fts_score"] > 0)
    logger.info(
        "search_stats",
        rrf_k=RRF_K,
        fts_non_zero=fts_non_zero,
        total=len(hybrid_results),
    )

    if hybrid_results:
        top = hybrid_results[0]
        logger.info(
            "top_result",
            embedding_score=round(top["embedding_score"], 4),
            fts_score=round(top["fts_score"], 4),
            combined=round(top["combined_score"], 6),
            intitule=top["intitule"][:40],
        )

    # Format results for API response
    processed_results = []
    for job in hybrid_results:
        mapped = linear_mapping(job["combined_score"], 0, RRF_MAX, 0, 1)
        clamped = max(0.0, min(1.0, mapped))
        processed_results.append(
            {
                "job_id": job["job_id"],
                "similarity_score": round(clamped, 2),
                "embedding_score": round(job["embedding_score"], 4),
                "fts_score": round(job["fts_score"], 4),
                "combined_score": round(job["combined_score"], 4),
            "intitule": job["intitule"] or "",
            "entreprise": job["entreprise"] or "",
            "lieu": job["lieu"] or "",
            "type_contrat": job["type_contrat"] or "",
            "date_creation": job["date_creation"] or "",
                "matching_terms": job["keywords"],
            }
        )

    return processed_results
