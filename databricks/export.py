# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Export to GCS
# Reads Delta via Spark, uploads Parquet to GCS via google-cloud-storage.
# (Spark GCS connector blocked by Spark Connect — Python client as workaround.)
#
# Exports the same silver/gold schema as functions/pipeline/core.py (the
# reference Cloud Function pipeline) so the ingest treats both identically.
# ---------------------------------------------------------------------------
# COMMAND ----------
# MAGIC %pip install google-cloud-storage --quiet

# COMMAND ----------
# GCS credentials come from a Databricks secret scope (the service-account
# JSON), so no key lives in the repo or in a workspace notebook.

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
GCS_CREDENTIALS_SCOPE = "cvee"
GCS_CREDENTIALS_KEY = "gcs-service-account-json"

_sa_info = _json.loads(dbutils.secrets.get(scope=GCS_CREDENTIALS_SCOPE, key=GCS_CREDENTIALS_KEY))
_creds = service_account.Credentials.from_service_account_info(_sa_info)
_sclient = storage.Client(project=GCS_BUCKET, credentials=_creds)
_bucket = _sclient.bucket(GCS_BUCKET)
print("GCS client ready.")

# COMMAND ----------

print("Reading silver Delta table ...")
# The Delta table can carry columns from earlier API schemas that the current
# raw feed no longer returns (kept alive by the merge's schema alignment).
# Drop them so the export matches the reference pipeline's silver schema.
STALE_SILVER_COLUMNS = ["complementExercice"]
df_silver = spark.table(SILVER_TABLE).drop(*STALE_SILVER_COLUMNS)

for col_name in JSON_COLS:
    if col_name in df_silver.columns:
        df_silver = df_silver.withColumn(col_name, F.to_json(F.col(col_name)))

max_date = df_silver.agg(F.max("ingestion_date")).collect()[0][0]
df_silver = df_silver.filter(F.col("ingestion_date") == max_date)
# core.py stores ingestion_date as a "YYYY-MM-DD" string.
df_silver = df_silver.withColumn("ingestion_date", F.date_format("ingestion_date", "yyyy-MM-dd"))

_pdf_silver = df_silver.toPandas()
_pdf_silver.attrs = {}
print(f"  {len(_pdf_silver)} rows in silver")

# COMMAND ----------

print("Reading gold Delta table ...")
# core.py exports gold as [job_id, embedding] only.
df_gold = (
    spark.table(GOLD_TABLE)
    .filter(F.col("ingestion_date") == max_date)
    .select("job_id", "embedding")
)

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
