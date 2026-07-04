# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Full Pipeline
# Orchestrates the 3-step ETL: silver → gold → GCS export.
# Run this notebook to execute the complete Databricks pipeline.
# ---------------------------------------------------------------------------
# COMMAND ----------

print("=" * 60)
print("CVEE Databricks Pipeline — Bronze → Silver → Gold → GCS")
print("=" * 60)

# COMMAND ----------

print("Step 1/4 — Cleanup (60-day retention) ...")

# COMMAND ----------
# MAGIC %run ./cleanup

# COMMAND ----------

print("Step 2/4 — Silver layer ...")

# COMMAND ----------
# MAGIC %run ./silver

# COMMAND ----------

print("Step 3/4 — Gold layer (embeddings) ...")

# COMMAND ----------
# MAGIC %run ./gold

# COMMAND ----------

print("Step 4/4 — Export to GCS ...")

# COMMAND ----------
# MAGIC %run ./export

# COMMAND ----------

print("\n" + "=" * 60)
print("Pipeline completed successfully.")
print("=" * 60)
