import io
import json
import psycopg2
import boto3
import pandas as pd
import numpy as np

DAYS_BEFORE_PURGE = 40


def get_s3_client(aws_access_key_id, aws_secret_access_key):
    """Create S3 client with provided credentials"""
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


def get_latest_batch_parquet_files(s3_client, bucket, prefix):
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

def read_parquet_from_s3(s3_client, bucket, key):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

def delete_old_records(cursor, days=DAYS_BEFORE_PURGE):
    """Delete old records from jobs_silver table on Supabase"""
    sql = "DELETE FROM jobs_silver WHERE ingestion_date < CURRENT_DATE - INTERVAL '%s days';"
    cursor.execute(sql, (days,))
    print(f"{cursor.rowcount} old offers deleted.")


def main(aws_access_key_id, aws_secret_access_key, aws_s3_bucket_name, 
         sb_host, sb_port, sb_user, sb_password, sb_name):
    """Main function to orchestrate S3 to Supabase ingestion"""
    
    s3_client = get_s3_client(aws_access_key_id, aws_secret_access_key)
    
    PREFIX_SILVER = "jobs_silver/"
    PREFIX_GOLD = "jobs_gold/"
    
    json_cols = ["lieuTravail", "entreprise", "contact", "agence",
                 "origineOffre", "contexteTravail",
                 "salaire", "competences", "formations", "langues",
                 "permis", "qualitesProfessionnelles"]
    
    # Get latest batch files
    silver_keys = get_latest_batch_parquet_files(s3_client, aws_s3_bucket_name, PREFIX_SILVER)
    gold_keys = get_latest_batch_parquet_files(s3_client, aws_s3_bucket_name, PREFIX_GOLD)
    
    # Connect to Supabase
    conn = psycopg2.connect(
        host=sb_host,
        database=sb_name,
        user=sb_user,
        password=sb_password,
        port=sb_port
    )
    cur = conn.cursor()
    
    # --- Processing Silver Table ---
    if not silver_keys:
        print("No Silver files found.")
    else:
        for key in silver_keys:
            print(f"Processing Silver file: {key}")
            df_silver = read_parquet_from_s3(s3_client, aws_s3_bucket_name, key)

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
            df_gold = read_parquet_from_s3(s3_client, aws_s3_bucket_name, key)
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


if __name__ == "__main__":
    # For local testing only
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    main(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_s3_bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
        sb_host=os.getenv("SB_HOST"),
        sb_port=int(os.getenv("SB_PORT")),
        sb_user=os.getenv("SB_USER"),
        sb_password=os.getenv("SB_PASSWORD"),
        sb_name=os.getenv("SB_NAME"),
    )
