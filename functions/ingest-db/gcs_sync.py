import json
from datetime import datetime

import gcsfs
import numpy as np
import pandas as pd
import psycopg2

DAYS_BEFORE_PURGE = 30


def _parse_gcs_time(t):
    if isinstance(t, datetime):
        return t
    return datetime.fromisoformat(str(t).replace("Z", "+00:00"))


def get_latest_batch_parquet_files(bucket, prefix):
    """Return list of parquet GCS URIs from the most recent day"""
    fs = gcsfs.GCSFileSystem()
    base = f"gs://{bucket}/{prefix}"
    try:
        all_files = fs.ls(base, detail=True)
    except FileNotFoundError:
        return []

    parquet_files = [
        f for f in all_files if f["name"].endswith(".parquet") and "_SUCCESS" not in f["name"]
    ]

    if not parquet_files:
        return []

    latest_day = max(_parse_gcs_time(f["updated"]).date() for f in parquet_files)
    return [f["name"] for f in parquet_files if _parse_gcs_time(f["updated"]).date() == latest_day]


def read_parquet_from_gcs(gcs_path):
    """Read parquet file from GCS into DataFrame"""
    fs = gcsfs.GCSFileSystem()
    return pd.read_parquet(gcs_path, filesystem=fs)


def delete_old_records(cursor, days=DAYS_BEFORE_PURGE):
    """Delete old records from jobs_silver table"""
    sql = "DELETE FROM jobs_silver WHERE ingestion_date < CURRENT_DATE - INTERVAL '%s days';"
    cursor.execute(sql, (days,))
    print(f"{cursor.rowcount} old offers deleted.")


def main(bucket_name, sb_host, sb_port, sb_user, sb_password, sb_name):
    """Ingest jobs from GCS (silver + gold) → Supabase"""

    PREFIX_SILVER = "jobs_silver/"
    PREFIX_GOLD = "jobs_gold/"

    json_cols = [
        "lieuTravail",
        "entreprise",
        "contact",
        "agence",
        "origineOffre",
        "contexteTravail",
        "salaire",
        "competences",
        "formations",
        "langues",
        "permis",
        "qualitesProfessionnelles",
    ]

    # Get latest batch files from GCS
    silver_keys = get_latest_batch_parquet_files(bucket_name, PREFIX_SILVER)
    gold_keys = get_latest_batch_parquet_files(bucket_name, PREFIX_GOLD)

    # Connect to Supabase
    conn = psycopg2.connect(
        host=sb_host, database=sb_name, user=sb_user, password=sb_password, port=sb_port
    )
    cur = conn.cursor()

    # --- Processing Silver Table ---
    if not silver_keys:
        print("No Silver files found.")
    else:
        for gcs_path in silver_keys:
            print(f"Processing Silver file: {gcs_path}")
            df_silver = read_parquet_from_gcs(gcs_path)

            for col in json_cols:
                if col in df_silver.columns:
                    df_silver[col] = df_silver[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and pd.notnull(x) else x
                    )

            for _, row in df_silver.iterrows():
                cols = row.index.tolist()
                values = []
                for c in cols:
                    val = row[c]
                    if isinstance(val, (list, np.ndarray)):
                        pass  # collection, not NaN
                    elif pd.isna(val):
                        val = None

                    if c in json_cols and val is not None:
                        values.append(json.dumps(val))
                    else:
                        values.append(val)

                placeholders = ", ".join(["%s"] * len(cols))
                sql = f"INSERT INTO jobs_silver ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT (job_id) DO NOTHING;"
                cur.execute(sql, values)

            conn.commit()
            print(f"File {gcs_path} inserted successfully.")

    # --- Processing Gold Table ---
    if not gold_keys:
        print("No Gold files found.")
    else:
        for gcs_path in gold_keys:
            print(f"Processing Gold file: {gcs_path}")
            df_gold = read_parquet_from_gcs(gcs_path)
            df_gold = df_gold[["job_id", "embedding"]]

            df_gold["embedding"] = df_gold["embedding"].apply(
                lambda x: (
                    x.tolist()
                    if hasattr(x, "tolist")
                    else (json.loads(x) if isinstance(x, str) else x)
                )
            )

            for _, row in df_gold.iterrows():
                job_id = row["job_id"]
                embedding = row["embedding"]

                if embedding is not None and any(pd.isna(i) for i in embedding):
                    embedding = [None if pd.isna(i) else i for i in embedding]

                sql = "INSERT INTO jobs_gold (job_id, embedding) VALUES (%s, %s) ON CONFLICT (job_id) DO NOTHING;"
                cur.execute(sql, (job_id, embedding))

            conn.commit()
            print(f"File {gcs_path} inserted successfully.")

    delete_old_records(cur, days=30)
    conn.commit()
    conn.close()
    print("End of import to Supabase")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(
        bucket_name=os.getenv("GCS_BUCKET_NAME"),
        sb_host=os.getenv("SB_HOST"),
        sb_port=int(os.getenv("SB_PORT", "5432")),
        sb_user=os.getenv("SB_USER"),
        sb_password=os.getenv("SB_PASSWORD"),
        sb_name=os.getenv("SB_NAME"),
    )
