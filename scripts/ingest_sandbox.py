"""Local ingest sandbox — exercise gcs_sync on a few chosen jobs.

Runs the real ``functions/ingest-db/gcs_sync.main`` insert path against a local
pgvector database (``docker compose up -d postgres``), using a small, chosen
subset of the latest GCS silver/gold batch. Nothing is written to Supabase.

The local schema mirrors prod (including the ``fts_tokens`` column and the
``tr_update_gold_fts`` trigger, which are not covered by the Alembic
migrations yet), so the batching change is exercised exactly as in prod.

Usage:
    docker compose up -d postgres
    uv run python scripts/ingest_sandbox.py --limit 5
    uv run python scripts/ingest_sandbox.py --job-ids 214CVBS,214CTYX,214CTMT
"""

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import gcsfs
import pandas as pd
import psycopg2

REPO_ROOT = Path(__file__).resolve().parent.parent
CF_DIR = REPO_ROOT / "functions" / "ingest-db"
sys.path.insert(0, str(CF_DIR))

import gcs_sync  # noqa: E402

BUCKET = "cvee-20260208"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/cvee_db"

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE jobs_silver (
    job_id TEXT PRIMARY KEY,
    intitule TEXT,
    description TEXT,
    vector_text_input TEXT,
    dateCreation TEXT,
    dateActualisation TEXT,
    lieuTravail JSONB,
    entreprise JSONB,
    contact JSONB,
    agence JSONB,
    origineOffre JSONB,
    contexteTravail JSONB,
    salaire JSONB,
    competences JSONB,
    formations JSONB,
    langues JSONB,
    qualitesProfessionnelles JSONB,
    permis JSONB,
    romeCode TEXT,
    romeLibelle TEXT,
    appellationlibelle TEXT,
    typeContrat TEXT,
    typeContratLibelle TEXT,
    natureContrat TEXT,
    experienceExige TEXT,
    experienceLibelle TEXT,
    dureeTravailLibelle TEXT,
    dureeTravailLibelleConverti TEXT,
    alternance BOOLEAN,
    nombrePostes INTEGER,
    accessibleTH BOOLEAN,
    qualificationCode TEXT,
    qualificationLibelle TEXT,
    codeNAF TEXT,
    secteurActivite TEXT,
    secteurActiviteLibelle TEXT,
    trancheEffectifEtab TEXT,
    offresManqueCandidats BOOLEAN,
    entrepriseAdaptee BOOLEAN,
    employeurHandiEngage BOOLEAN,
    deplacementCode TEXT,
    deplacementLibelle TEXT,
    experienceCommentaire TEXT,
    complementExercice TEXT,
    ingestion_date DATE
);

CREATE TABLE jobs_gold (
    job_id TEXT PRIMARY KEY,
    embedding vector(384),
    fts_tokens tsvector,
    CONSTRAINT fk_job FOREIGN KEY (job_id)
        REFERENCES jobs_silver(job_id) ON DELETE CASCADE
);

CREATE TABLE job_term_stats (
    term TEXT PRIMARY KEY,
    df INTEGER NOT NULL
);

CREATE OR REPLACE FUNCTION fn_fill_fts_from_silver() RETURNS trigger AS $$
DECLARE
    source_record RECORD;
    clean_competences TEXT;
    clean_qualites TEXT;
BEGIN
    SELECT intitule, description, competences, qualitesprofessionnelles
    INTO source_record
    FROM jobs_silver
    WHERE job_id = NEW.job_id;

    SELECT string_agg(elem->>'libelle', ' ')
    INTO clean_competences
    FROM jsonb_array_elements(source_record.competences) AS elem;

    SELECT string_agg((elem->>'libelle') || ' ' || (elem->>'description'), ' ')
    INTO clean_qualites
    FROM jsonb_array_elements(source_record.qualitesprofessionnelles) AS elem;

    UPDATE jobs_gold
    SET fts_tokens =
        setweight(to_tsvector('french', unaccent(COALESCE(source_record.intitule, ''))), 'A') ||
        setweight(to_tsvector('french', unaccent(COALESCE(source_record.description, ''))), 'B') ||
        setweight(to_tsvector('french', unaccent(COALESCE(clean_competences, ''))), 'C') ||
        setweight(to_tsvector('french', unaccent(COALESCE(clean_qualites, ''))), 'C')
    WHERE job_id = NEW.job_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_gold_fts AFTER INSERT ON jobs_gold
FOR EACH ROW EXECUTE FUNCTION fn_fill_fts_from_silver();

CREATE OR REPLACE FUNCTION refresh_job_term_stats() RETURNS void AS $$
BEGIN
    DELETE FROM job_term_stats;
    INSERT INTO job_term_stats (term, df)
    SELECT word, ndoc
    FROM ts_stat('SELECT fts_tokens FROM jobs_gold WHERE fts_tokens IS NOT NULL');
END;
$$ LANGUAGE plpgsql;
"""


def parse_dsn(dsn):
    u = urlparse(dsn)
    return {
        "host": u.hostname or "localhost",
        "port": u.port or 5432,
        "user": u.username or "postgres",
        "password": u.password or "postgres",
        "name": (u.path or "/cvee_db").lstrip("/"),
    }


def reset_schema(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS jobs_gold CASCADE;")
        cur.execute("DROP TABLE IF EXISTS jobs_silver CASCADE;")
        cur.execute("DROP TABLE IF EXISTS job_term_stats CASCADE;")
        cur.execute(SCHEMA_SQL)
    conn.commit()


def build_subset(limit, job_ids):
    fs = gcsfs.GCSFileSystem()
    silver_uri = gcs_sync.get_latest_batch_parquet_files(BUCKET, "jobs_silver/")[0]
    gold_uri = gcs_sync.get_latest_batch_parquet_files(BUCKET, "jobs_gold/")[0]
    print(f"source silver: {silver_uri}")
    print(f"source gold:   {gold_uri}")

    df_silver = pd.read_parquet(silver_uri, filesystem=fs)
    df_gold = pd.read_parquet(gold_uri, filesystem=fs)

    if job_ids:
        chosen = [j for j in job_ids if j in set(df_silver["job_id"])]
        missing = sorted(set(job_ids) - set(chosen))
        if missing:
            print(f"ignored ids not in batch: {missing}")
    else:
        chosen = df_silver["job_id"].head(limit).tolist()

    if not chosen:
        raise SystemExit("no matching job_id in the latest batch")

    df_silver = df_silver[df_silver["job_id"].isin(chosen)].reset_index(drop=True)
    df_gold = df_gold[df_gold["job_id"].isin(chosen)].reset_index(drop=True)

    tmp = Path(tempfile.mkdtemp(prefix="cvee-sandbox-"))
    silver_path = tmp / "silver.parquet"
    gold_path = tmp / "gold.parquet"
    df_silver.to_parquet(silver_path)
    df_gold.to_parquet(gold_path)

    print(f"subset: {len(df_silver)} silver / {len(df_gold)} gold -> {tmp}")
    return str(silver_path), str(gold_path), chosen


def patch_gcs(silver_path, gold_path):
    def fake_latest(bucket, prefix):
        return [silver_path] if prefix.startswith("jobs_silver") else [gold_path]

    gcs_sync.get_latest_batch_parquet_files = fake_latest
    gcs_sync.read_parquet_from_gcs = lambda path: pd.read_parquet(path)


def report(conn, chosen):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM jobs_silver;")
        silver = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM jobs_gold;")
        gold = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM jobs_gold WHERE fts_tokens IS NOT NULL;")
        fts = cur.fetchone()[0]
        cur.execute("SELECT vector_dims(embedding) FROM jobs_gold LIMIT 1;")
        dims = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM job_term_stats;")
        terms = cur.fetchone()[0]
    print(f"DB: silver={silver} gold={gold} fts_filled={fts} vector_dims={dims} terms={terms}")
    assert silver == len(chosen), "silver row count mismatch"
    assert gold == len(chosen), "gold row count mismatch"
    assert fts == len(chosen), "fts_tokens not filled for every gold row"
    assert dims == 384, "embedding dimension mismatch"
    assert terms > 0, "job_term_stats not refreshed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="local pgvector DSN")
    parser.add_argument("--limit", type=int, default=5, help="number of jobs to ingest")
    parser.add_argument("--job-ids", default="", help="comma-separated job ids to ingest")
    args = parser.parse_args()

    job_ids = [j.strip() for j in args.job_ids.split(",") if j.strip()]
    creds = parse_dsn(args.dsn)

    silver_path, gold_path, chosen = build_subset(args.limit, job_ids)
    patch_gcs(silver_path, gold_path)

    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        dbname=creds["name"],
    )
    reset_schema(conn)

    print("--- run 1 ---")
    gcs_sync.main(
        bucket_name=BUCKET,
        sb_host=creds["host"],
        sb_port=creds["port"],
        sb_user=creds["user"],
        sb_password=creds["password"],
        sb_name=creds["name"],
    )
    report(conn, chosen)

    print("--- run 2 (idempotency) ---")
    gcs_sync.main(
        bucket_name=BUCKET,
        sb_host=creds["host"],
        sb_port=creds["port"],
        sb_user=creds["user"],
        sb_password=creds["password"],
        sb_name=creds["name"],
    )
    report(conn, chosen)

    conn.close()
    print(f"OK — {len(chosen)} jobs ingested and re-ingested without duplicates: {chosen}")


if __name__ == "__main__":
    main()
