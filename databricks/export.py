# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Export to GCS
# Reads Delta via Spark, uploads Parquet to GCS via google-cloud-storage.
# (Spark GCS connector blocked by Spark Connect — Python client as workaround.)
# ---------------------------------------------------------------------------
# COMMAND ----------
# MAGIC %pip install google-cloud-storage --quiet

# COMMAND ----------
# Load GCS credentials from _secrets.py (gitignored, never committed).
# Copy _secrets_template.py → _secrets.py and fill in the values.
# MAGIC %run ./_secrets

# COMMAND ----------

from datetime import datetime
import json as _json, io as _io

import pandas as _pd
import pyspark.sql.functions as F
from google.cloud import storage
from google.oauth2 import service_account

from common import JSON_COLS

# COMMAND ----------

GCS_BUCKET = "cvee-20260208"
SILVER_TABLE = "cvee.jobs_silver"
GOLD_TABLE = "cvee.jobs_gold"

_creds = service_account.Credentials.from_service_account_info(
    {
        "type": "service_account",
        "project_id": GCS_BUCKET,
        "private_key_id": _gcs_key_id,
        "private_key": _gcs_key,
        "client_email": _gcs_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)
_sclient = storage.Client(project=GCS_BUCKET, credentials=_creds)
_bucket = _sclient.bucket(GCS_BUCKET)
print("GCS client ready.")

# COMMAND ----------

print("Reading silver Delta table ...")
df_silver = spark.table(SILVER_TABLE).drop("vector_text_input")

for col_name in JSON_COLS:
    if col_name in df_silver.columns:
        df_silver = df_silver.withColumn(col_name, F.to_json(F.col(col_name)))

max_date = df_silver.agg(F.max("ingestion_date")).collect()[0][0]
df_silver = df_silver.filter(F.col("ingestion_date") == max_date)

_pdf_silver = df_silver.toPandas()
_pdf_silver.attrs = {}
print(f"  {len(_pdf_silver)} rows in silver")

# COMMAND ----------

print("Reading gold Delta table ...")
df_gold = spark.table(GOLD_TABLE)
df_gold = df_gold.filter(F.col("ingestion_date") == max_date)

_pdf_gold = df_gold.toPandas()
_pdf_gold.attrs = {}
_pdf_gold["embedding"] = _pdf_gold["embedding"].apply(
    lambda x: _json.dumps(x.tolist() if hasattr(x, "tolist") else x)
)
print(f"  {len(_pdf_gold)} rows in gold")

# COMMAND ----------

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Uploading silver -> jobs_silver/jobs_silver_{ts}.parquet ...")
_buf = _io.BytesIO()
_pdf_silver.to_parquet(_buf, engine="pyarrow", index=False)
_buf.seek(0)
_bucket.blob(f"jobs_silver/jobs_silver_{ts}.parquet").upload_from_file(_buf)
print(f"  {len(_pdf_silver)} jobs uploaded")

print(f"Uploading gold -> jobs_gold/jobs_gold_{ts}.parquet ...")
_buf2 = _io.BytesIO()
_pdf_gold.to_parquet(_buf2, engine="pyarrow", index=False)
_buf2.seek(0)
_bucket.blob(f"jobs_gold/jobs_gold_{ts}.parquet").upload_from_file(_buf2)
print(f"  {len(_pdf_gold)} jobs uploaded")

print("Export — DONE")
