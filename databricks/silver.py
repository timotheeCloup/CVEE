# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Silver Layer
# Reads raw jobs from GCS, cleans HTML, aggregates JSON fields,
# merges into Delta table cvee.jobs_silver.
# ---------------------------------------------------------------------------
# COMMAND ----------

import re
from datetime import datetime

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.functions import col, concat_ws, current_date, udf
from pyspark.sql.types import StringType

from common import clean_html

# COMMAND ----------

GCS_RAW_PATH = "gs://cvee-20260208/jobs_raw/"
SILVER_TABLE = "cvee.jobs_silver"

# File Arrival trigger passes the path via a widget — use it if available
raw_file = None
try:
    raw_file = dbutils.widgets.get("raw_file")
    print(f"Triggered by file arrival: {raw_file}")
except Exception:
    pass

if not raw_file:
    # Manual run: list latest file via public GCS API
    print(f"Listing latest raw file from {GCS_RAW_PATH} ...")
    import urllib.request, json as _json

    _api_url = "https://storage.googleapis.com/storage/v1/b/cvee-20260208/o?prefix=jobs_raw/"
    _resp = urllib.request.urlopen(_api_url)
    _items = _json.loads(_resp.read()).get("items", [])
    parquet_files = [
        "gs://cvee-20260208/" + item["name"] for item in _items if item["name"].endswith(".parquet")
    ]

    def _extract_date_from_name(path):
        match = re.search(r"(\d{8}_\d{6})", path)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return datetime.min

    raw_file = max(parquet_files, key=_extract_date_from_name)

print(f"  → {raw_file}")

# Download raw parquet via HTTP into memory (no local file, no GCS auth)
import io

import pandas as _pd

_http_url = raw_file.replace("gs://cvee-20260208/", "https://storage.googleapis.com/cvee-20260208/")
_resp = urllib.request.urlopen(_http_url)
_pdf = _pd.read_parquet(io.BytesIO(_resp.read()), engine="pyarrow")
df_raw = spark.createDataFrame(_pdf)
print(f"  {df_raw.count()} raw jobs loaded")

# COMMAND ----------

print("Cleaning HTML and aggregating JSON fields ...")

clean_html_udf = udf(clean_html, StringType())

df_cleaned = (
    df_raw.withColumn("description_clean", clean_html_udf(col("description")))
    .withColumn(
        "competences_aggregated",
        F.expr("concat_ws(' ', transform(competences, x -> x.libelle))"),
    )
    .withColumn(
        "formations_aggregated",
        F.expr("concat_ws(' ', transform(formations, x -> x.domaineLibelle))"),
    )
    .withColumn(
        "qualites_aggregated",
        F.expr("concat_ws(' ', transform(qualitesProfessionnelles, x -> x.libelle))"),
    )
)

df_final = df_cleaned.withColumn(
    "vector_text_input",
    concat_ws(
        " ",
        col("intitule"),
        col("description_clean"),
        col("competences_aggregated"),
        col("formations_aggregated"),
        col("qualites_aggregated"),
    ),
)

df_final = df_final.dropDuplicates(["id"])
df_final = df_final.drop(
    "description",
    "competences_aggregated",
    "formations_aggregated",
    "qualites_aggregated",
)
df_final = (
    df_final.withColumnRenamed("id", "job_id")
    .withColumnRenamed("description_clean", "description")
    .withColumn("ingestion_date", current_date())
)

print(f"  {df_final.count()} jobs after dedup")

# COMMAND ----------

print("No translation needed — multilingual embedding model handles French natively.")

# COMMAND ----------

print(f"Merging into Delta table {SILVER_TABLE} ...")

try:
    target_schema = spark.table(SILVER_TABLE).schema
    df_source_aligned = df_final.select(
        F.from_json(F.to_json(F.struct("*")), target_schema).alias("data")
    ).select("data.*")

    delta_table = DeltaTable.forName(spark, SILVER_TABLE)
    old_count = delta_table.toDF().count()

    delta_table.alias("target").merge(
        df_source_aligned.alias("source"), "target.job_id = source.job_id"
    ).whenNotMatchedInsertAll().execute()

    new_count = delta_table.toDF().count()
    added = new_count - old_count
    print(f"  {added} new rows inserted ({new_count} total)")

except Exception:
    print("  Table does not exist yet — creating ...")
    df_final.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)
    final_count = spark.table(SILVER_TABLE).count()
    print(f"  Table created with {final_count} rows")

print("Silver layer — DONE")
