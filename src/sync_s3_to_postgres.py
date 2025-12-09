import os
import io
import json
import psycopg2
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

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


def get_latest_parquet_file(bucket, prefix):
    """Return the most recently modified parquet file key in the given S3 bucket and prefix"""
    latest_key = None
    latest_mtime = None

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                mtime = obj["LastModified"]
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_key = key

    return latest_key



silver_key = get_latest_parquet_file(AWS_S3_BUCKET_NAME, PREFIX_SILVER)
gold_key = get_latest_parquet_file(AWS_S3_BUCKET_NAME, PREFIX_GOLD)



def read_parquet_from_s3(bucket, key):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))



json_cols = ["lieuTravail", "entreprise", "contact", "agence",
             "origineOffre", "contexteTravail",
             "salaire", "competences", "formations", "langues",
             "permis", "qualitesProfessionnelles"]


conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)
cur = conn.cursor()

# Processing Silver Table
df_silver = read_parquet_from_s3(AWS_S3_BUCKET_NAME, silver_key)

# JSON string to JSON object
for col in json_cols:
    if col in df_silver.columns:
        df_silver[col] = df_silver[col].apply(lambda x: json.loads(x) if pd.notnull(x) else None)

#ingestion into Postgres
for _, row in df_silver.iterrows():
    cols = row.index.tolist()
    values = [json.dumps(row[c]) if c in json_cols else row[c] for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO jobs_silver ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT (job_id) DO NOTHING;"
    cur.execute(sql, values)
conn.commit()
print(f"Inserted {silver_key} into jobs_silver")
    


# Processing Gold Table
df_gold = read_parquet_from_s3(AWS_S3_BUCKET_NAME, gold_key)
df_gold= df_gold[["job_id", "embedding"]]

# JSON string to vector list
df_gold["embedding"] = df_gold["embedding"].apply(lambda x: list(x) if isinstance(x, list) else list(json.loads(x)))

#ingestion into Postgres
for _, row in df_gold.iterrows():
    sql = """
        INSERT INTO jobs_gold (job_id, embedding) VALUES (%s, %s) ON CONFLICT (job_id) DO NOTHING;
    """
    cur.execute(sql, (row["job_id"], row["embedding"]))

conn.commit()
print(f"Inserted {gold_key} into jobs_gold")


conn.close()
print("End of import")


