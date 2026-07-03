import io
import logging
import os
import re
import time

import psycopg2
from dotenv import load_dotenv
from psycopg2 import pool
from pypdf import PdfReader

logger = logging.getLogger(__name__)

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

TOP_K = 100

# Hybrid search weights
EMBEDDING_WEIGHT = 1
FTS_WEIGHT = 4

MAP_FROM_MIN = EMBEDDING_WEIGHT * 0.30
MAP_FROM_MAX = EMBEDDING_WEIGHT * 0.60 + FTS_WEIGHT * 0.10

# FTS weights for tsvector levels [C, B, A] (D=0 since unused)
FTS_WEIGHTS = [0.3, 0.6, 1.0]

_db_pool = None


def _get_pool():
    global _db_pool
    if _db_pool is None and DB_HOST:
        _db_pool = pool.ThreadedConnectionPool(
            1, 5, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
    return _db_pool


def db_connection():
    """Get a connection from the pool (or create one ad-hoc if no pool)"""
    p = _get_pool()
    if p:
        return p.getconn()
    return psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )


def db_release(conn):
    p = _get_pool()
    if p:
        p.putconn(conn)
    else:
        conn.close()


def extract_text_from_pdf(file_bytes):
    """Extract text from first page of PDF"""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:
        text = reader.pages[0].extract_text() or ""
    return text.strip()


def extract_french_keywords_from_headline(headline):
    """Extract French keywords from ts_headline <b>...</b> fragments"""
    if not headline:
        return []
    marked_terms = re.findall(r"<b>([^<]+)</b>", str(headline))
    keywords = []
    seen = set()
    for term in marked_terms:
        clean_term = re.sub(r"[^\w\s]", "", term.strip().lower())
        words = [w for w in clean_term.split() if len(w) > 2]
        if words:
            final_term = " ".join(words)
            if final_term not in seen:
                keywords.append(final_term)
                seen.add(final_term)
    return keywords[:10]


def linear_mapping(value, from_min, from_max, to_min, to_max):
    """Linearly map a value from one range to another"""
    if from_max - from_min == 0:
        return to_min
    return to_min + (to_max - to_min) * (value - from_min) / (from_max - from_min)


def _build_fts_weights_literal():
    """Build PostgreSQL tsvector weights literal string like '{0, 0.3, 0.6, 1.0}'"""
    weights = [0] + FTS_WEIGHTS  # [D, C, B, A], D=0 unused
    return "{" + ", ".join(str(w) for w in weights) + "}"


def search_jobs_vector_hybrid(embedding, cv_text_fts, cv_text_orig):
    """
    Hybrid job search combining FTS (French) and semantic embedding (English).

    Returns top 100 jobs sorted by combined score:
        EMBEDDING_WEIGHT * embedding_score + FTS_WEIGHT * fts_score
    """
    t_start = time.time()

    conn = db_connection()
    try:
        cur = conn.cursor()
        t_conn = time.time()
        logger.info("DB connection: %.3fs", t_conn - t_start)
        logger.info(
            "CV text for FTS: %d chars, Embedding dim: %d", len(cv_text_fts), len(embedding)
        )

        # Build French FTS query
        fts_terms = cv_text_fts.split()[:1000]
        tsquery = " | ".join(f"'{term}'" for term in fts_terms) if fts_terms else ""
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        fts_weights_literal = _build_fts_weights_literal()

        sql = """
        SELECT
            jg.job_id,
            (1 - (jg.embedding <-> %s))::float8 as embedding_score,
            COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8 as fts_score,
            (%s * (1 - (jg.embedding <-> %s))::float8 +
             %s * COALESCE(ts_rank(%s::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8)::float8 as combined_score,
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
                'StartSel=<b>, StopSel=</b>, MaxWords=100, MinWords=50') as headline
        FROM jobs_gold jg
        JOIN jobs_silver js ON jg.job_id = js.job_id
        WHERE jg.fts_tokens IS NOT NULL
        ORDER BY combined_score DESC
        LIMIT %s;
        """

        cur.execute(
            sql,
            (
                embedding_str,
                fts_weights_literal,
                tsquery,
                EMBEDDING_WEIGHT,
                embedding_str,
                FTS_WEIGHT,
                fts_weights_literal,
                tsquery,
                tsquery,
                TOP_K,
            ),
        )
        results = cur.fetchall()
    finally:
        cur.close()
        db_release(conn)

    t_query = time.time()
    logger.info("Query execution: %.3fs (retrieved %d jobs)", t_query - t_conn, len(results))

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
    logger.info("Processing: %.3fs", t_process - t_query)

    # Summary stats
    fts_non_zero = sum(1 for r in hybrid_results if r["fts_score"] > 0)
    logger.info(
        "Weights: %.0f%% embedding + %.0f%% FTS | FTS levels(C,B,A): %s | Results with FTS>0: %d/%d",
        EMBEDDING_WEIGHT * 100,
        FTS_WEIGHT * 100,
        FTS_WEIGHTS,
        fts_non_zero,
        len(hybrid_results),
    )

    if hybrid_results:
        logger.info(
            "Top result: Emb=%.4f FTS=%.4f Combined=%.4f %s",
            hybrid_results[0]["embedding_score"],
            hybrid_results[0]["fts_score"],
            hybrid_results[0]["combined_score"],
            hybrid_results[0]["intitule"][:40],
        )

    # Format results for API response
    processed_results = []
    for r in hybrid_results:
        mapped = linear_mapping(r["combined_score"], MAP_FROM_MIN, MAP_FROM_MAX, 0.15, 0.85)
        clamped = max(0.0, min(1.0, mapped))
        processed_results.append(
            {
                "job_id": r["job_id"],
                "similarity_score": round(clamped, 2),
                "embedding_score": round(r["embedding_score"], 4),
                "fts_score": round(r["fts_score"], 4),
                "combined_score": round(r["combined_score"], 4),
                "intitule": r["intitule"],
                "entreprise": r["entreprise"],
                "lieu": r["lieu"],
                "type_contrat": r["type_contrat"],
                "date_creation": r["date_creation"],
                "matching_terms": r["keywords"],
            }
        )

    return processed_results
