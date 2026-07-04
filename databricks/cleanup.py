# Databricks notebook source
# COMMAND ----------
# ---------------------------------------------------------------------------
# CVEE Databricks — Cleanup
# Deletes Delta rows older than 60 days to keep tables lean.
# ---------------------------------------------------------------------------
# COMMAND ----------

from datetime import datetime, timedelta

from delta.tables import DeltaTable

# COMMAND ----------

RETENTION_DAYS = 60
SILVER_TABLE = "cvee.jobs_silver"
GOLD_TABLE = "cvee.jobs_gold"

cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
print(f"Deleting rows older than {cutoff} ({RETENTION_DAYS} days) ...")

# COMMAND ----------

print(f"Silver table ({SILVER_TABLE}) ...")
try:
    silver_before = spark.table(SILVER_TABLE).count()
    delta_silver = DeltaTable.forName(spark, SILVER_TABLE)
    delta_silver.delete(f"ingestion_date < '{cutoff}'")
    silver_after = spark.table(SILVER_TABLE).count()
    print(f"  {silver_before} → {silver_after} ({silver_before - silver_after} deleted)")
except Exception as e:
    print(f"  Skipped: {e}")

# COMMAND ----------

print(f"Gold table ({GOLD_TABLE}) ...")
try:
    gold_before = spark.table(GOLD_TABLE).count()
    delta_gold = DeltaTable.forName(spark, GOLD_TABLE)
    delta_gold.delete(f"ingestion_date < '{cutoff}'")
    gold_after = spark.table(GOLD_TABLE).count()
    print(f"  {gold_before} → {gold_after} ({gold_before - gold_after} deleted)")
except Exception as e:
    print(f"  Skipped: {e}")

print("Cleanup — DONE")
