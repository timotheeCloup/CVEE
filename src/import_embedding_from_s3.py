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
PREFIX_SILVER = "jobs_metadata_silver/"
PREFIX_GOLD = "jobs_vectors/"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def list_parquet_files(bucket, prefix):
    """Return a list of parquet file keys in the given S3 bucket and prefix"""
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


metadata_keys = list_parquet_files(AWS_S3_BUCKET_NAME, PREFIX_SILVER)
embeddings_keys = list_parquet_files(AWS_S3_BUCKET_NAME, PREFIX_GOLD)



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


for key in metadata_keys:
    df_meta = read_parquet_from_s3(AWS_S3_BUCKET_NAME, key)
    df_meta=df_meta.drop(columns=['description_clean','required_skills','required_formation','required_qualities','job_location','vector_text_input'])
    
    # JSON string to JSON object
    for col in json_cols:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
    
    #ingestion into Postgres
    for _, row in df_meta.iterrows():
        cols = row.index.tolist()
        values = [json.dumps(row[c]) if c in json_cols else row[c] for c in cols]
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO jobs_metadata ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT (job_id) DO NOTHING;"
        cur.execute(sql, values)
    conn.commit()
    print(f"Inserted {key} into jobs_metadata")
    
for key in embeddings_keys:
    df_vec = read_parquet_from_s3(AWS_S3_BUCKET_NAME, key)

    # JSON string to JSON object
    df_vec["embedding"] = df_vec["embedding"].apply(lambda x: json.dumps(x))

    #ingestion into Postgres
    for _, row in df_vec.iterrows():
        sql = """
            INSERT INTO jobs_vectors (job_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (job_id) DO NOTHING;
        """
        cur.execute(sql, (row["job_id"], row["embedding"]))

    conn.commit()
    print(f"Inserted {key} into jobs_vectors")


conn.close()
print("End of import")


