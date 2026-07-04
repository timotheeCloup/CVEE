# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Gold Layer
# Collects unprocessed jobs from Silver, generates embeddings on driver
# (no Pandas UDF to avoid OOM on Community Edition), merges into Delta.
# ---------------------------------------------------------------------------
# COMMAND ----------
# MAGIC %pip install sentence-transformers==2.2.2 torch huggingface_hub==0.24.0 --quiet

# COMMAND ----------

import os

from delta.tables import DeltaTable
from pyspark.sql import functions as F

os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

# COMMAND ----------

SILVER_TABLE = "cvee.jobs_silver"
GOLD_TABLE = "cvee.jobs_gold"
MAX_JOBS = 10000  # Covers a full daily raw batch (~3k jobs max)

# COMMAND ----------

print(f"Reading {SILVER_TABLE} and filtering already processed jobs ...")

df_silver = spark.table(SILVER_TABLE).select("job_id", "vector_text_input", "ingestion_date")
df_gold = spark.table(GOLD_TABLE).select("job_id")
df_to_process = df_silver.join(df_gold, on="job_id", how="left_anti")

total_new = df_to_process.count()
print(f"  {total_new} new offers to embed")

if total_new > MAX_JOBS:
    print(f"  Limiting to {MAX_JOBS} jobs")
    df_to_process = df_to_process.limit(MAX_JOBS)

rows = df_to_process.select(
    "job_id",
    "ingestion_date",
    F.coalesce(F.col("vector_text_input"), F.lit("")).alias("text_input"),
).collect()

job_ids = [r.job_id for r in rows]
dates = [r.ingestion_date for r in rows]
texts = [r.text_input[:5000] for r in rows]

print(f"  Processing {len(texts)} offers")

if len(texts) == 0:
    print("  No new jobs — skipping embeddings generation.")
else:
    print("Generating embeddings (driver-side, no UDF) ...")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    model = SentenceTransformer("antoinelouis/french-me5-small", device="cpu")

    with torch.no_grad():
        embeddings = model.encode(
            texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True
        )

    print(f"  {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")

    import pandas as pd

    _pdf = pd.DataFrame(
        {
            "job_id": job_ids,
            "ingestion_date": [str(d) for d in dates],
            "embedding": [emb.tolist() for emb in embeddings],
        }
    )

    df_final = spark.createDataFrame(_pdf)

    print(f"Merging into {GOLD_TABLE} ...")

    try:
        gold_table = DeltaTable.forName(spark, GOLD_TABLE)
        gold_table.alias("target").merge(
            df_final.alias("source"), "target.job_id = source.job_id"
        ).whenNotMatchedInsertAll().execute()
        print("  Merge completed successfully.")
    except Exception:
        print("  Table does not exist yet — creating ...")
        df_final.write.format("delta").mode("overwrite").saveAsTable(GOLD_TABLE)
        final_count = spark.table(GOLD_TABLE).count()
        print(f"  Table created with {final_count} rows")

print("Gold layer — DONE")
