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

# Reciprocal Rank Fusion constants
RRF_K: int = 60

# Title weight in RRF: title score counts TITLE_WEIGHT times more than embedding/FTS
TITLE_WEIGHT: float = 3.0

# Theoretical max RRF score: job ranked #1 in all 3 dimensions
# score = sum(weight / (RRF_K + rank)) for embed(1), fts(1), title(TITLE_WEIGHT)
RRF_SCORE_MAX: float = (1 + 1 + TITLE_WEIGHT) / (RRF_K + 1)

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
        )
        await _db_pool.open()
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
    Hybrid job search combining FTS + embedding + title via Reciprocal Rank Fusion.

    Ranks jobs independently by embedding similarity, FTS relevance, and title match,
    then fuses ranks using RRF: score(d) = 1/(k+rank_embed) + 1/(k+rank_fts) + w*1/(k+rank_title).

    Returns top 100 jobs sorted by RRF combined score.
    """
    t_start = time.time()

    pool = await _get_pool()
    async with pool.connection() as conn:
        t_conn = time.time()
        logger.info("db_connection", duration=round(t_conn - t_start, 3))
        logger.info("fts_prep", fts_chars=len(cv_text_fts), embedding_dim=len(embedding))

        fts_terms = cv_text_fts.split()[:1000]
        tsquery = " | ".join(f"'{term}'" for term in fts_terms) if fts_terms else ""
        if not tsquery:
            logger.warning(
                "empty_fts_query",
                fts_chars=len(cv_text_fts),
                original_chars=len(cv_text_orig),
            )
            tsquery = "'placeholder'"
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        fts_weights_literal = _build_fts_weights_literal()

        sql = """
        SELECT
            job_id, embedding_score, fts_score,
            (1.0 / (%s + embed_rank) + 1.0 / (%s + fts_rank) + %s * 1.0 / (%s + title_rank))::float8 as combined_score,
            intitule, entreprise, lieu, typeContratLibelle, dateCreation, headline
        FROM (
            SELECT
                jg.job_id,
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
                ROW_NUMBER() OVER (ORDER BY COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0) DESC) as fts_rank,
                ROW_NUMBER() OVER (ORDER BY COALESCE(ts_rank(to_tsvector('french', js.intitule), to_tsquery('french', %s), 2), 0) DESC) as title_rank
            FROM jobs_gold jg
            JOIN jobs_silver js ON jg.job_id = js.job_id
            WHERE jg.fts_tokens IS NOT NULL
        ) ranked
        ORDER BY combined_score DESC
        LIMIT %s;
        """

        async with conn.cursor() as cur:
            try:
                await cur.execute(
                    sql,
                    (
                        RRF_K,
                        RRF_K,
                        TITLE_WEIGHT,
                        RRF_K,
                        embedding_str,
                        fts_weights_literal,
                        tsquery,
                        tsquery,
                        embedding_str,
                        fts_weights_literal,
                        tsquery,
                        tsquery,
                        TOP_K,
                    ),
                )
                results = await cur.fetchall()
            except Exception as e:
                logger.error(
                    "db_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    tsquery=tsquery[:200],
                    fts_terms_count=len(fts_terms),
                    embedding_dim=len(embedding),
                )
                raise

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
        title_weight=TITLE_WEIGHT,
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
    # Map RRF combined_score to [0, 1] using theoretical max.
    # This preserves natural score spread: top result won't always be 100%.
    logger.info(
        "score_distribution",
        score_max_theoretical=round(RRF_SCORE_MAX, 6),
        first_score=round(hybrid_results[0]["combined_score"], 6) if hybrid_results else 0,
    )

    processed_results = []
    for job in hybrid_results:
        mapped = linear_mapping(job["combined_score"], 0, RRF_SCORE_MAX, 0, 1)
        clamped = round(max(0.0, min(1.0, mapped)), 2)
        processed_results.append(
            {
                "job_id": job["job_id"],
                "similarity_score": clamped,
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
