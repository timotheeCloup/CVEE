import io
import json
import os
import re
import time
from typing import Any

import structlog
from config import settings
from psycopg_pool import AsyncConnectionPool
from pypdf import PdfReader

logger: Any = structlog.get_logger()

# Pagination: each page holds PAGE_SIZE offers, plus PAGE_MARGIN extra scored
# rows so the live dead-link check (which runs after ranking) can drop a few and
# still fill the page. MAX_PAGE bounds how deep the UI can go.
PAGE_SIZE: int = 100
PAGE_MARGIN: int = 10
MAX_PAGE: int = 5

# Match score = EMBED_WEIGHT * cos_norm
#              + FTS_WEIGHT   * fts_norm
#              + TITLE_WEIGHT * title_score
#
# cos_norm and fts_norm are min-max normalized over the candidate set before
# weighting: raw cosine sits in a very narrow band (~0.85-0.92 for every offer,
# the embedding space is anisotropic) while fts_ratio spans 0-0.1, so applying
# fixed weights to the raw values let the embedding dominate no matter what the
# weights said. Normalizing first makes the weights mean what they say.
EMBED_WEIGHT: float = 0.4
FTS_WEIGHT: float = 0.5
TITLE_WEIGHT: float = 0.1
# Title saturation reference. One matched title term contributes ~0.11 of
# "matched weight", so TITLE_REF=2.0 keeps a single term a small boost and
# requires several matching terms to saturate the title component. Kept low on
# purpose: a lone title match is often a place/entity name, not a skill.
TITLE_REF: float = 2.0
FTS_REF: float = 15.0
# Number of rarest CV terms kept for the keyword query and IDF weighting.
FTS_MAX_TERMS: int = 50
# Minimum corpus document frequency for a CV term to enter the keyword query.
# Terms unique to the corpus (df=1) are proper nouns, emails or typos: they
# carry a maximal IDF that inflates the normalization total and adds noise, so
# they are dropped and only terms shared by at least two offers are kept.
FTS_MIN_DF: int = 2
# Jobs re-scored with the exact IDF overlap after a cheap pre-selection. Bounds
# the per-row tsvector unnest so it never runs on the full corpus. Covers every
# page (MAX_PAGE * PAGE_SIZE) with headroom for the pre-selection to differ from
# the exact rescore.
CANDIDATE_K: int = 1000

# FTS weights for tsvector levels [C, B, A] (D=0 since unused)
FTS_WEIGHTS: list[float] = [0.3, 0.6, 1.0]

_db_pool: AsyncConnectionPool | None = None

# Corpus mean embedding, used to center the cosine (see search_jobs_vector_hybrid).
# Cached for the process lifetime: the corpus mean drifts slowly and a Cloud Run
# instance is short-lived. None until the first search computes it.
_centroid: str | None = None


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


def repair_inter_char_spacing(text: str) -> str:
    """Undo a pypdf extraction artifact that space-separates every glyph.

    Depending on the PDF layout (and the pypdf version), ``extract_text`` can
    return "I n g é n i e u r  e n" instead of "Ingénieur en": one space between
    glyphs and two between words. Detected via the mean token length, then
    repaired by rebuilding word boundaries. No word is removed and phrases stay
    intact, so the text is safe to embed (unlike ``clean_text_for_fts``, which
    strips stopwords and is only meant for full-text search).
    """
    words = text.split()
    if not words:
        return text
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len >= 1.5:
        return text
    text = re.sub(r"  +", "\x00", text)
    text = text.replace(" ", "")
    return text.replace("\x00", " ")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from first page of PDF"""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:
        text = reader.pages[0].extract_text() or ""
    return repair_inter_char_spacing(text).strip()


COMMUNES_COORDS_FILE: str = os.path.join(os.path.dirname(__file__), "communes_coords.json")

# Lazy-loaded commune coordinate lookup (INSEE/postal/department -> [lat, lon]).
# ~1 MB, ~17 ms to parse, so it is only read when an offer actually lacks
# coordinates and never blocks startup/health checks.
_communes_coords: dict[str, dict[str, list[float]]] | None = None


def _load_communes_coords() -> dict[str, dict[str, list[float]]]:
    """Load the commune coordinate lookup once, on first use."""
    global _communes_coords
    if _communes_coords is not None:
        return _communes_coords
    try:
        with open(COMMUNES_COORDS_FILE, encoding="utf-8") as f:
            data: dict[str, dict[str, list[float]]] = json.load(f)
    except Exception as e:
        logger.warning("communes_coords_load_failed", error=str(e))
        data = {"insee": {}, "cp": {}, "dept": {}}
    _communes_coords = data
    return data


def _to_float(value: Any) -> float | None:
    """Convert a raw latitude/longitude value to float, or None if unusable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_department(libelle: str) -> str | None:
    """Extract the department code from a location label ("69 - Lyon" -> "69")."""
    prefix = libelle.split(" - ", 1)[0].strip()
    return prefix if re.fullmatch(r"\d{2,3}|2[AB]", prefix) else None


def resolve_coordinates(
    insee: str | None, code_postal: str | None, libelle: str | None
) -> tuple[float | None, float | None]:
    """Resolve map coordinates for an offer whose location carries no lat/lon.

    Tries the INSEE commune code, then the postal code, then the department
    centroid. Returns (None, None) for country/region-level offers ("France")
    that cannot be placed on a map.
    """
    coords = _load_communes_coords()
    for key, table in ((insee, coords["insee"]), (code_postal, coords["cp"])):
        if key and key in table:
            lat, lon = table[key]
            return lat, lon
    if libelle:
        department = _split_department(libelle)
        if department and department in coords["dept"]:
            lat, lon = coords["dept"][department]
            return lat, lon
    return None, None


def _build_fts_weights_literal() -> str:
    """Build PostgreSQL tsvector weights literal string like '{0, 0.3, 0.6, 1.0}'"""
    weights = [0] + FTS_WEIGHTS  # [D, C, B, A], D=0 unused
    return "{" + ", ".join(str(w) for w in weights) + "}"


async def _get_centroid(conn: Any, dim: int) -> str:
    """Return the corpus mean embedding as a pgvector literal, cached per process.

    Used to center the cosine: the embedding space is strongly anisotropic (every
    offer sits at ~0.87 cosine from every other), so subtracting the shared mean
    direction makes the cosine reflect how an offer differs from the average
    instead of how close it is to the common direction. Falls back to a zero
    vector (no centering) when the corpus is empty.
    """
    global _centroid
    if _centroid is None:
        async with conn.cursor() as cur:
            await cur.execute("SELECT avg(embedding)::text FROM jobs_gold")
            row = await cur.fetchone()
        _centroid = row[0] if row and row[0] else "[" + ",".join(["0"] * dim) + "]"
    return _centroid


def _normalize(values: list[float]) -> list[float]:
    """Min-max scale values to [0, 1]; all-equal values map to 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


async def search_jobs_vector_hybrid(
    embedding: list[float],
    cv_text_fts: str,
    cv_text_orig: str,
    departements: list[str] | None = None,
    types_contrat: list[str] | None = None,
    page: int = 1,
) -> list[dict[str, Any]]:
    """
    Hybrid job search combining FTS + embedding + title, one page at a time.

    Ranks jobs by a normalized match score:

        score = EMBED_WEIGHT * cos_norm
              + FTS_WEIGHT   * fts_norm
              + TITLE_WEIGHT * title_score

    where cos_norm and fts_norm are min-max scaled over the candidate set. Raw
    cosine barely varies (anisotropic space), so weighting the raw values let the
    embedding dominate whatever the weights; normalizing first restores the
    intended balance and keeps the keyword signal alive deeper in the ranking.

    The cosine is centered on the corpus mean (see ``_get_centroid``) to spread
    out those near-identical values.

    The keyword component is weighted by IDF. The CV is reduced to its
    FTS_MAX_TERMS rarest terms shared by at least FTS_MIN_DF offers (unique
    terms are proper nouns/typos, not skills), and a job scores on the terms it
    actually shares with the CV.

    Candidate selection is bounded: the CANDIDATE_K best jobs by a cheap
    ts_rank/cosine/title score are re-scored with the exact IDF overlap, so the
    per-row tsvector unnest never runs on the full corpus.

    Optional filters restrict the corpus before scoring:
      - departements: department codes (e.g. ["69", "75"]), matched against the
        prefix of lieuTravail->>'libelle' ("69 - Lyon" -> "69").
      - types_contrat: contract codes (e.g. ["CDI", "CDD"]), matched against typeContrat.

    Returns PAGE_SIZE + PAGE_MARGIN jobs for the requested page (1-indexed),
    sorted by match score; the extra PAGE_MARGIN covers the live dead-link check.
    """
    t_start = time.time()
    page = max(1, min(page, MAX_PAGE))
    offset = (page - 1) * PAGE_SIZE
    limit = PAGE_SIZE + PAGE_MARGIN

    pool = await _get_pool()
    async with pool.connection() as conn:
        t_conn = time.time()
        logger.info("db_connection", duration=round(t_conn - t_start, 3))
        logger.info("fts_prep", fts_chars=len(cv_text_fts), embedding_dim=len(embedding))

        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        fts_weights_literal = _build_fts_weights_literal()
        centroid = await _get_centroid(conn, len(embedding))

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
        cv_token_map AS (
            -- Stem -> original CV word. Frequencies and IDF are computed on the
            -- stem (french dictionary), but the display keeps the exact word so
            -- the UI shows "databricks" instead of the truncated "databrick".
            SELECT
                lower(t) AS token,
                (tsvector_to_array(to_tsvector('french', unaccent(lower(t)))))[1] AS stem
            FROM regexp_split_to_table(%(cv_text)s, '[^[:alnum:]-]+') AS t
            WHERE length(t) >= 2
        ),
        cv_display AS (
            SELECT stem, min(token) AS display
            FROM cv_token_map
            WHERE stem IS NOT NULL
            GROUP BY stem
        ),
        scored_cv AS (
            SELECT c.term, s.df, ln(corpus.n_docs / GREATEST(s.df, 1))::float8 AS idf
            FROM cv_terms_clean c
            JOIN job_term_stats s ON s.term = c.term
            CROSS JOIN corpus
        ),
        ranked AS (
            SELECT term, df, idf FROM scored_cv
            WHERE df >= %(min_df)s
            ORDER BY df ASC LIMIT %(max_terms)s
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
                (1 - ((jg.embedding - %(centroid)s::vector)
                       <=> (%(embedding)s::vector - %(centroid)s::vector)))::float8 AS cosine_score,
                COALESCE(ts_rank(%(fts_weights)s::float4[], jg.fts_tokens,
                                 to_tsquery('french', (SELECT q FROM fts_query))), 0)::float8 AS fts_rank_score,
                COALESCE(ts_rank(js.title_tsv,
                                 to_tsquery('french', (SELECT q FROM fts_query))), 0)::float8 AS title_score,
                js.intitule,
                js.entreprise->>'nom' AS entreprise,
                js.lieuTravail->>'libelle' AS lieu,
                js.typeContratLibelle,
                js.dateCreation,
                js.lieuTravail->>'latitude' AS latitude,
                js.lieuTravail->>'longitude' AS longitude,
                js.lieuTravail->>'commune' AS commune,
                js.lieuTravail->>'codePostal' AS code_postal
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
                       array_agg(COALESCE(d.display, r.term) ORDER BY r.df ASC) AS matched_terms
                FROM unnest(tsvector_to_array(jg.fts_tokens)) AS t(lex)
                JOIN ranked r ON r.term = t.lex
                LEFT JOIN cv_display d ON d.stem = r.term
            ) m ON true
        )
        SELECT
            job_id, cosine_score,
            LEAST(1.0, idf_sum / (SELECT total FROM idf_norm))::float8 AS fts_ratio,
            title_score, matched_terms,
            intitule, entreprise, lieu, typeContratLibelle, dateCreation,
            latitude, longitude, commune, code_postal
        FROM rescored
        ORDER BY provisional_score DESC;
        """

        params = {
            "cv_text": cv_text_fts,
            "embedding": embedding_str,
            "centroid": centroid,
            "fts_weights": fts_weights_literal,
            "max_terms": FTS_MAX_TERMS,
            "min_df": FTS_MIN_DF,
            "departements": departements or None,
            "types_contrat": types_contrat or None,
            "embed_w": EMBED_WEIGHT,
            "fts_w": FTS_WEIGHT,
            "title_w": TITLE_WEIGHT,
            "fts_ref": FTS_REF,
            "title_ref": TITLE_REF,
            "candidate_k": CANDIDATE_K,
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

    # Normalize each signal over the candidate set, then fuse. Doing this in
    # Python (rather than SQL) keeps the SQL free of window functions and lets
    # the raw cosine/fts values stay available for logging.
    cos_norm = _normalize([r[1] for r in results])
    fts_norm = _normalize([r[2] for r in results])
    scored = [
        (
            r,
            EMBED_WEIGHT * cos_norm[i] + FTS_WEIGHT * fts_norm[i] + TITLE_WEIGHT * r[3],
        )
        for i, r in enumerate(results)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # The page slice; the PAGE_MARGIN extra rows absorb the dead-link check that
    # runs after this function.
    page_rows = scored[offset : offset + limit]

    t_process = time.time()
    logger.info("processing", duration=round(t_process - t_query, 3))

    hybrid_results = []
    for r, match_score in page_rows:
        (
            job_id,
            cosine_score,
            fts_score,
            title_score,
            matched_terms,
            intitule,
            entreprise,
            lieu,
            type_contrat,
            date_creation,
            raw_latitude,
            raw_longitude,
            commune,
            code_postal,
        ) = r

        # Most offers already carry coordinates; the rest are placed via the
        # commune lookup (INSEE/postal/department), or left off the map.
        latitude = _to_float(raw_latitude)
        longitude = _to_float(raw_longitude)
        if latitude is None or longitude is None:
            latitude, longitude = resolve_coordinates(commune, code_postal, lieu)

        hybrid_results.append(
            {
                "job_id": job_id,
                "similarity_score": round(max(0.0, min(1.0, match_score)), 2),
                "embedding_score": round(cosine_score, 4),
                "fts_score": round(fts_score, 4),
                "title_score": round(title_score, 4),
                "combined_score": round(match_score, 4),
                "intitule": intitule or "",
                "entreprise": entreprise or "",
                "lieu": lieu or "",
                "type_contrat": type_contrat or "",
                "date_creation": date_creation or "",
                "matching_terms": matched_terms or [],
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    # Summary stats
    fts_non_zero = sum(1 for r in hybrid_results if r["fts_score"] > 0)
    logger.info(
        "search_stats",
        embed_weight=EMBED_WEIGHT,
        fts_weight=FTS_WEIGHT,
        title_weight=TITLE_WEIGHT,
        fts_non_zero=fts_non_zero,
        total=len(hybrid_results),
        page=page,
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

    return hybrid_results
