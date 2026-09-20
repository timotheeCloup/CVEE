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

# Absolute match score: corpus-independent (no rank-based normalization).
# ts_rank is diluted by the number of query terms (rank = matched_weight / N),
# so each keyword component is multiplied back by N to recover an absolute
# "matched weight" that does not depend on the CV length.
# score = EMBED_WEIGHT * cosine
#       + FTS_WEIGHT   * min(fts_matched   / FTS_REF, 1)
#       + TITLE_WEIGHT * min(title_matched / TITLE_REF, 1)
EMBED_WEIGHT: float = 0.3
FTS_WEIGHT: float = 0.4
TITLE_WEIGHT: float = 0.3
FTS_REF: float = 15.0
TITLE_REF: float = 0.25

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


def _build_fts_weights_literal() -> str:
    """Build PostgreSQL tsvector weights literal string like '{0, 0.3, 0.6, 1.0}'"""
    weights = [0] + FTS_WEIGHTS  # [D, C, B, A], D=0 unused
    return "{" + ", ".join(str(w) for w in weights) + "}"


async def search_jobs_vector_hybrid(
    embedding: list[float],
    cv_text_fts: str,
    cv_text_orig: str,
    departements: list[str] | None = None,
    types_contrat: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid job search combining FTS + embedding + title via Reciprocal Rank Fusion.

    Ranks jobs by an absolute match score combining embedding cosine similarity,
    full-text relevance and job-title relevance:

        score = EMBED_WEIGHT * cosine
              + FTS_WEIGHT   * min(fts_matched   / FTS_REF, 1)
              + TITLE_WEIGHT * min(title_matched / TITLE_REF, 1)

    where ``*_matched = ts_rank(...) * number_of_query_terms``. ts_rank is diluted
    by the query length (rank = matched_weight / N), so multiplying back yields an
    absolute matched weight. Every term is therefore corpus- and CV-length
    independent: the score is not inflated when the filtered corpus is small.

    Optional filters restrict the corpus before scoring:
      - departements: department codes (e.g. ["69", "75"]), matched against the
        prefix of lieuTravail->>'libelle' ("69 - Lyon" -> "69").
      - types_contrat: contract codes (e.g. ["CDI", "CDD"]), matched against typeContrat.

    Returns top 100 jobs sorted by match score.
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
        departements_filter = departements or None
        types_contrat_filter = types_contrat or None

        # Two-stage query: (1) score all jobs with the absolute match score and
        # keep the top-K, then (2) compute the expensive ts_headline snippet ONLY
        # on those K rows. ts_headline does not influence scoring (it only feeds
        # keyword highlighting), so restricting it to the final top-K is
        # result-preserving while avoiding highlighting the full corpus.
        sql = """
        WITH scored AS (
            SELECT
                jg.job_id,
                (1 - (jg.embedding <=> %s))::float8 as cosine_score,
                COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8 as fts_score,
                COALESCE(ts_rank(js.title_tsv, to_tsquery('french', %s)), 0)::float8 as title_score,
                js.intitule,
                js.entreprise->>'nom' AS entreprise,
                js.lieuTravail->>'libelle' AS lieu,
                js.typeContratLibelle,
                js.dateCreation
            FROM jobs_gold jg
            JOIN jobs_silver js ON jg.job_id = js.job_id
            WHERE jg.fts_tokens IS NOT NULL
              AND (%s::text[] IS NULL OR split_part(js.lieuTravail->>'libelle', ' - ', 1) = ANY(%s::text[]))
              AND (%s::text[] IS NULL OR js.typeContrat = ANY(%s::text[]))
        ),
        top_scored AS (
            SELECT
                job_id, cosine_score, fts_score, title_score,
                LEAST(1.0, GREATEST(0.0,
                    %s * cosine_score
                  + %s * LEAST(1.0, (fts_score * %s) / %s)
                  + %s * LEAST(1.0, (title_score * %s) / %s)
                ))::float8 as match_score,
                intitule, entreprise, lieu, typeContratLibelle, dateCreation
            FROM scored
            ORDER BY match_score DESC
            LIMIT %s
        )
        SELECT
            t.job_id, t.cosine_score, t.fts_score, t.title_score, t.match_score,
            t.intitule, t.entreprise, t.lieu, t.typeContratLibelle, t.dateCreation,
            ts_headline('french',
                js.intitule || ' ' || COALESCE(js.description, '') || ' ' ||
                COALESCE((SELECT string_agg(elem->>'libelle', ' ')
                          FROM jsonb_array_elements(js.competences) AS elem), '') || ' ' ||
                COALESCE((SELECT string_agg((elem->>'libelle') || ' ' || (elem->>'description'), ' ')
                          FROM jsonb_array_elements(js.qualitesprofessionnelles) AS elem), ''),
                to_tsquery('french', %s),
                'StartSel=<b>, StopSel=</b>, MaxWords=100, MinWords=50') as headline
        FROM top_scored t
        JOIN jobs_silver js ON js.job_id = t.job_id
        ORDER BY t.match_score DESC;
        """

        async with conn.cursor() as cur:
            try:
                # Keep the scoring sort over the full corpus in memory instead of
                # spilling to disk. Scoped to this transaction via SET LOCAL, so
                # it never leaks to pooled sessions.
                await cur.execute("SET LOCAL work_mem = '64MB'")
                await cur.execute(
                    sql,
                    (
                        embedding_str,
                        fts_weights_literal,
                        tsquery,
                        tsquery,
                        departements_filter,
                        departements_filter,
                        types_contrat_filter,
                        types_contrat_filter,
                        EMBED_WEIGHT,
                        FTS_WEIGHT,
                        len(fts_terms),
                        FTS_REF,
                        TITLE_WEIGHT,
                        len(fts_terms),
                        TITLE_REF,
                        TOP_K,
                        tsquery,
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

    # Process results (already sorted by match_score)
    hybrid_results = []
    for r in results:
        (
            job_id,
            cosine_score,
            fts_score,
            title_score,
            match_score,
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
                "embedding_score": cosine_score,
                "fts_score": fts_score,
                "title_score": title_score,
                "combined_score": match_score,
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
        embed_weight=EMBED_WEIGHT,
        fts_weight=FTS_WEIGHT,
        title_weight=TITLE_WEIGHT,
        fts_non_zero=fts_non_zero,
        total=len(hybrid_results),
    )

    if hybrid_results:
        top = hybrid_results[0]
        n_terms = len(fts_terms)
        logger.info(
            "top_result",
            cosine=round(top["embedding_score"], 4),
            fts_matched=round(top["fts_score"] * n_terms, 3),
            title_matched=round(top["title_score"] * n_terms, 4),
            match=round(top["combined_score"], 4),
        )

    processed_results = []
    for job in hybrid_results:
        processed_results.append(
            {
                "job_id": job["job_id"],
                "similarity_score": round(max(0.0, min(1.0, job["combined_score"])), 2),
                "embedding_score": round(job["embedding_score"], 4),
                "fts_score": round(job["fts_score"], 4),
                "title_score": round(job["title_score"], 4),
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
