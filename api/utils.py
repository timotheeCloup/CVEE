import io
import time
from typing import Any

import structlog
from config import settings
from psycopg_pool import AsyncConnectionPool
from pypdf import PdfReader

logger: Any = structlog.get_logger()

TOP_K: int = 100

# Absolute match score: corpus-independent (no rank-based normalization).
# score = EMBED_WEIGHT * cosine
#       + FTS_WEIGHT   * min(matched_idf / cv_idf, 1)
#       + TITLE_WEIGHT * min(title_matched / TITLE_REF, 1)
#
# The keyword component is weighted by IDF. Only the FTS_MAX_TERMS rarest CV
# terms (highest IDF against the job corpus) enter the query, and a job scores
# on the terms it actually shares with the CV. Discriminative skills ("python")
# therefore dominate generic words ("equipe", "realisation") that match almost
# every offer.
EMBED_WEIGHT: float = 0.3
FTS_WEIGHT: float = 0.4
TITLE_WEIGHT: float = 0.3
# Title saturation reference. One matched title term contributes ~0.11 of
# "matched weight", so TITLE_REF=1.0 keeps a single term a partial boost and
# requires several matching terms to saturate the title component.
TITLE_REF: float = 1.0
FTS_REF: float = 15.0
# Number of rarest CV terms kept for the keyword query and IDF weighting.
FTS_MAX_TERMS: int = 50
# Jobs re-scored with the exact IDF overlap after a cheap pre-selection. Bounds
# the per-row tsvector unnest so it never runs on the full corpus.
CANDIDATE_K: int = 300

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
    Hybrid job search combining FTS + embedding + title.

    Ranks jobs by an absolute match score:

        score = EMBED_WEIGHT * cosine
              + FTS_WEIGHT   * min(matched_idf / cv_idf, 1)
              + TITLE_WEIGHT * min(title_matched / TITLE_REF, 1)

    The keyword component is weighted by IDF. The CV is reduced to its
    FTS_MAX_TERMS rarest terms (highest IDF against the corpus stored in
    job_term_stats), and a job scores on the terms it actually shares with the
    CV. Discriminative skills ("python") therefore stay decisive while generic
    words ("equipe", "realisation") that match almost every offer are neutralized.

    Candidate selection is bounded: the CANDIDATE_K best jobs by a cheap
    ts_rank/cosine/title score are re-scored with the exact IDF overlap, so the
    per-row tsvector unnest never runs on the full corpus.

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

        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        fts_weights_literal = _build_fts_weights_literal()

        # Three-stage query, fully server-side:
        #  1. candidates: cheap cosine + ts_rank + title score over the corpus,
        #     using a keyword query restricted to the rarest CV terms.
        #  2. top_candidates: keep the CANDIDATE_K best to bound the next step.
        #  3. rescored: exact IDF overlap (tsvector unnest) on those rows only.
        # matched_terms returned for display are the shared CV/offer terms,
        # ordered by corpus rarity (rarest first).
        sql = """
        WITH corpus AS (
            SELECT count(*)::float8 AS n_docs FROM jobs_gold
        ),
        cv_terms AS (
            SELECT DISTINCT unnest(
                tsvector_to_array(to_tsvector('french', unaccent(%(cv_text)s)))
            ) AS term
        ),
        cv_terms_clean AS (
            -- Drop single characters and pure numbers (PDF extraction artifacts
            -- such as "H/F" or "bac+5"): they are noise, not skills.
            SELECT term FROM cv_terms WHERE length(term) >= 2 AND term ~ '[a-z]'
        ),
        scored_cv AS (
            SELECT c.term, s.df, ln(corpus.n_docs / GREATEST(s.df, 1))::float8 AS idf
            FROM cv_terms_clean c
            JOIN job_term_stats s ON s.term = c.term
            CROSS JOIN corpus
        ),
        ranked AS (
            SELECT term, df, idf FROM scored_cv ORDER BY df ASC LIMIT %(max_terms)s
        ),
        fts_query AS (
            SELECT COALESCE(string_agg(term, ' | '), 'placeholder') AS q FROM ranked
        ),
        idf_norm AS (
            SELECT GREATEST(COALESCE(sum(idf), 1e-9), 1e-9)::float8 AS total FROM ranked
        ),
        candidates AS (
            SELECT
                jg.job_id,
                (1 - (jg.embedding <=> %(embedding)s))::float8 AS cosine_score,
                COALESCE(ts_rank(%(fts_weights)s::float4[], jg.fts_tokens,
                                 to_tsquery('french', (SELECT q FROM fts_query))), 0)::float8 AS fts_rank_score,
                COALESCE(ts_rank(js.title_tsv,
                                 to_tsquery('french', (SELECT q FROM fts_query))), 0)::float8 AS title_score,
                js.intitule,
                js.entreprise->>'nom' AS entreprise,
                js.lieuTravail->>'libelle' AS lieu,
                js.typeContratLibelle,
                js.dateCreation
            FROM jobs_gold jg
            JOIN jobs_silver js ON jg.job_id = js.job_id
            WHERE jg.fts_tokens IS NOT NULL
              AND (%(departements)s::text[] IS NULL OR split_part(js.lieuTravail->>'libelle', ' - ', 1) = ANY(%(departements)s::text[]))
              AND (%(types_contrat)s::text[] IS NULL OR js.typeContrat = ANY(%(types_contrat)s::text[]))
        ),
        top_candidates AS (
            SELECT c.*,
                (%(embed_w)s * c.cosine_score
                 + %(fts_w)s * LEAST(1.0, (c.fts_rank_score * (SELECT count(*) FROM ranked)) / %(fts_ref)s)
                 + %(title_w)s * LEAST(1.0, (c.title_score * (SELECT count(*) FROM ranked)) / %(title_ref)s)
                )::float8 AS provisional_score
            FROM candidates c
            ORDER BY provisional_score DESC
            LIMIT %(candidate_k)s
        ),
        rescored AS (
            SELECT tc.*,
                COALESCE(m.idf_sum, 0)::float8 AS idf_sum,
                COALESCE(m.matched_terms, ARRAY[]::text[]) AS matched_terms
            FROM top_candidates tc
            JOIN jobs_gold jg ON jg.job_id = tc.job_id
            LEFT JOIN LATERAL (
                SELECT sum(r.idf) AS idf_sum,
                       array_agg(r.term ORDER BY r.df ASC) AS matched_terms
                FROM unnest(tsvector_to_array(jg.fts_tokens)) AS t(lex)
                JOIN ranked r ON r.term = t.lex
            ) m ON true
        )
        SELECT
            job_id, cosine_score,
            LEAST(1.0, idf_sum / (SELECT total FROM idf_norm))::float8 AS fts_ratio,
            title_score,
            LEAST(1.0, GREATEST(0.0,
                %(embed_w)s * cosine_score
              + %(fts_w)s * LEAST(1.0, idf_sum / (SELECT total FROM idf_norm))
              + %(title_w)s * LEAST(1.0, (title_score * (SELECT count(*) FROM ranked)) / %(title_ref)s)
            ))::float8 AS match_score,
            intitule, entreprise, lieu, typeContratLibelle, dateCreation, matched_terms
        FROM rescored
        ORDER BY match_score DESC
        LIMIT %(top_k)s;
        """

        params = {
            "cv_text": cv_text_fts,
            "embedding": embedding_str,
            "fts_weights": fts_weights_literal,
            "max_terms": FTS_MAX_TERMS,
            "departements": departements or None,
            "types_contrat": types_contrat or None,
            "embed_w": EMBED_WEIGHT,
            "fts_w": FTS_WEIGHT,
            "title_w": TITLE_WEIGHT,
            "fts_ref": FTS_REF,
            "title_ref": TITLE_REF,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
        }

        async with conn.cursor() as cur:
            try:
                # Keep the scoring sort over the full corpus in memory instead of
                # spilling to disk. Scoped to this transaction via SET LOCAL, so
                # it never leaks to pooled sessions.
                await cur.execute("SET LOCAL work_mem = '64MB'")
                await cur.execute(sql, params)
                results = await cur.fetchall()
            except Exception as e:
                logger.error(
                    "db_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    fts_chars=len(cv_text_fts),
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
            matched_terms,
        ) = r
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
                "matching_terms": matched_terms or [],
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
        logger.info(
            "top_result",
            cosine=round(top["embedding_score"], 4),
            fts=round(top["fts_score"], 4),
            title=round(top["title_score"], 4),
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
                "matching_terms": job["matching_terms"],
            }
        )

    return processed_results
