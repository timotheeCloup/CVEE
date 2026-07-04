"""DuckDB on-the-fly analytics over GCS Parquet files.

Reads job offer data directly from GCS with zero in-memory loading,
enabling SQL queries on the data lake without Spark or BigQuery.

Usage:
    uv run python scripts/analytics.py gs://cvee-20260208
"""

import sys

import duckdb


def main(gcs_bucket: str) -> None:
    duckdb.sql("INSTALL httpfs; LOAD httpfs;")

    silver = f"{gcs_bucket}/jobs_silver/*.parquet"
    gold = f"{gcs_bucket}/jobs_gold/*.parquet"

    print("=" * 50)
    print("Total job offers")
    duckdb.sql(f"SELECT count(*) AS total FROM '{silver}'").show()

    print("=" * 50)
    print("Top 10 contract types")
    duckdb.sql(f"""
        SELECT typeContratLibelle, count(*) AS cnt
        FROM '{silver}'
        GROUP BY typeContratLibelle
        ORDER BY cnt DESC
        LIMIT 10
    """).show()

    print("=" * 50)
    print("Top 10 sectors")
    duckdb.sql(f"""
        SELECT secteurActiviteLibelle, count(*) AS cnt
        FROM '{silver}'
        GROUP BY secteurActiviteLibelle
        ORDER BY cnt DESC
        LIMIT 10
    """).show()

    print("=" * 50)
    print("Jobs per day (last 14 days)")
    duckdb.sql(f"""
        SELECT ingestion_date, count(*) AS cnt
        FROM '{silver}'
        WHERE ingestion_date >= CURRENT_DATE - INTERVAL 14 DAYS
        GROUP BY ingestion_date
        ORDER BY ingestion_date DESC
    """).show()

    print("=" * 50)
    print("Gold layer — distinct embeddings")
    duckdb.sql(f"SELECT count(DISTINCT job_id) AS jobs_with_embeddings FROM '{gold}'").show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: uv run python {__file__} gs://your-bucket")
        sys.exit(1)
    main(sys.argv[1])
