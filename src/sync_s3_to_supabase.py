import os
import io
import json
import psycopg2
import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DAYS_BEFORE_PURGE = 40

# Supabase configuration
SB_HOST = os.getenv("SB_HOST")
SB_USER = os.getenv("SB_USER")
SB_PASSWORD = os.getenv("SB_PASSWORD")
SB_NAME = os.getenv("SB_NAME")
SB_PORT = os.getenv("SB_PORT")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
PREFIX_SILVER = "jobs_silver/"
PREFIX_GOLD = "jobs_gold/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def get_latest_batch_parquet_files(bucket, prefix):
    """Return list of parquet keys with the most recent LastModified timestamp"""
    all_parquet_objs = []
    
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and "_SUCCESS" not in key:
                all_parquet_objs.append({
                    "Key": key,
                    "LastModified": obj["LastModified"]
                })

    if not all_parquet_objs:
        return []

    latest_mtime = max(obj["LastModified"] for obj in all_parquet_objs)
    latest_keys = [
        obj["Key"] for obj in all_parquet_objs 
        if obj["LastModified"] == latest_mtime
    ]

    return latest_keys

def read_parquet_from_s3(bucket, key):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

def delete_old_records(cursor, days=DAYS_BEFORE_PURGE):
    """Delete old records from jobs_silver table on Supabase"""
    sql = "DELETE FROM jobs_silver WHERE ingestion_date < CURRENT_DATE - INTERVAL '%s days';"
    cursor.execute(sql, (days,))
    print(f"{cursor.rowcount} old offers deleted.")


silver_keys = get_latest_batch_parquet_files(AWS_S3_BUCKET_NAME, PREFIX_SILVER)
gold_keys = get_latest_batch_parquet_files(AWS_S3_BUCKET_NAME, PREFIX_GOLD)

json_cols = ["lieuTravail", "entreprise", "contact", "agence",
             "origineOffre", "contexteTravail",
             "salaire", "competences", "formations", "langues",
             "permis", "qualitesProfessionnelles"]

conn = psycopg2.connect(
    host=SB_HOST,
    database=SB_NAME,
    user=SB_USER,
    password=SB_PASSWORD,
    port=SB_PORT
)
cur = conn.cursor()

# --- Processing Silver Table ---
if not silver_keys:
    print("No Silver files found.")
else:
    for key in silver_keys:
        print(f"Processing Silver file: {key}")
        df_silver = read_parquet_from_s3(AWS_S3_BUCKET_NAME, key)

        for col in json_cols:
            if col in df_silver.columns:
                df_silver[col] = df_silver[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and pd.notnull(x) else x
                )

        # --- Processing Silver Table ---
        for _, row in df_silver.iterrows():
            cols = row.index.tolist()
            values = []
            for c in cols:
                val = row[c]
                
                is_null = False
                if isinstance(val, (list, np.ndarray)):
                    is_null = False # It's a collection, not Nan scalar
                elif pd.isna(val):
                    is_null = True
                    
                if is_null:
                    val = None
                    
                if c in json_cols and val is not None:
                    values.append(json.dumps(val))
                else:
                    values.append(val)

            placeholders = ", ".join(["%s"] * len(cols))
            sql = f"INSERT INTO jobs_silver ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT (job_id) DO NOTHING;"
            cur.execute(sql, values)
        
        conn.commit()
        print(f"File {key} inserted successfully.")

# --- Processing Gold Table ---
if not gold_keys:
    print("No Gold files found.")
else:
    for key in gold_keys:
        print(f"Processing Gold file: {key}")
        df_gold = read_parquet_from_s3(AWS_S3_BUCKET_NAME, key)
        df_gold = df_gold[["job_id", "embedding"]]

        df_gold["embedding"] = df_gold["embedding"].apply(
            lambda x: x.tolist() if hasattr(x, "tolist") else (json.loads(x) if isinstance(x, str) else x)
        )

        for _, row in df_gold.iterrows():
            job_id = row["job_id"]
            embedding = row["embedding"]
            
            # Final safety check for NaN in embedding vectors
            if embedding is not None and any(pd.isna(i) for i in embedding):
                embedding = [None if pd.isna(i) else i for i in embedding]

            sql = "INSERT INTO jobs_gold (job_id, embedding) VALUES (%s, %s) ON CONFLICT (job_id) DO NOTHING;"
            cur.execute(sql, (job_id, embedding))

        conn.commit()
        print(f"File {key} inserted successfully.")
        
delete_old_records(cur, days=30)
conn.commit()

conn.close()
print("End of import to Supabase")
