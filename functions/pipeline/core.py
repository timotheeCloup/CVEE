import json
import math
import os
import re
from datetime import datetime, timedelta

import gcsfs
import numpy as np
import polars as pl
import structlog
import torch
from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.makedirs("/tmp/huggingface", exist_ok=True)

torch.set_num_threads(1)

logger = structlog.get_logger()

MODEL_NAME = "antoinelouis/french-me5-small"
BATCH_SIZE = 32

PREFIX_RAW = "jobs_raw"
PREFIX_SILVER = "jobs_silver"
PREFIX_GOLD = "jobs_gold"

JSON_COLS = [
    "lieuTravail",
    "entreprise",
    "contact",
    "agence",
    "origineOffre",
    "contexteTravail",
    "salaire",
    "competences",
    "formations",
    "langues",
    "permis",
    "qualitesProfessionnelles",
]


def _is_na(val):
    """Check if a value is None or NaN (polars uses None for null)."""
    return val is None or (isinstance(val, float) and math.isnan(val))


def clean_html(text):
    """Strip HTML tags and entities, collapse whitespace.

    Args:
        text: Raw text potentially containing HTML.

    Returns:
        Cleaned plain text string, or ``""`` if input is None/NaN.
    """
    if _is_na(text):
        return ""
    text = str(text)
    clean_re = re.compile(r"<.*?>|&([a-zA-Z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});")
    text = re.sub(clean_re, "", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text).strip()
    return text


def _extract_field(val, field="libelle"):
    if _is_na(val):
        return ""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return str(val)
    if isinstance(val, (list, np.ndarray)):
        parts = []
        for item in val:
            if isinstance(item, dict):
                v = item.get(field, "")
                if v:
                    parts.append(str(v))
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(val, dict):
        return str(val.get(field, ""))
    return str(val)


def serialize_json_col(val):
    """Convert nested dict/list/ndarray to JSON string for JSONB compatibility.

    Args:
        val: Value to serialize (dict, list, ndarray, str, None).

    Returns:
        JSON string if val is a complex type, the same string if already str,
        or None if val is None/NaN.
    """
    if _is_na(val):
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False, default=_numpy_to_python)


def _numpy_to_python(obj):
    """Convert numpy/polars types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict)):
        return list(obj)
    return str(obj)


def _databricks_already_produced(bucket_name):
    """Check if Databricks already produced today's silver+gold output in GCS.

    Both silver AND gold must exist to confirm a complete Databricks run.
    Incomplete runs (silver only) are treated as NOT done → GCP fallback.
    """
    fs = gcsfs.GCSFileSystem()
    today = datetime.now().strftime("%Y%m%d")
    try:
        silver_exists = (
            len(fs.glob(f"gs://{bucket_name}/{PREFIX_SILVER}/jobs_silver_{today}*.parquet")) > 0
        )
        gold_exists = (
            len(fs.glob(f"gs://{bucket_name}/{PREFIX_GOLD}/jobs_gold_{today}*.parquet")) > 0
        )
    except FileNotFoundError:
        return False
    return silver_exists and gold_exists


def _list_raw_files(bucket_name, days=None):
    """Return sorted list of raw parquet file URIs from GCS.

    days=None → latest file only
    days=N   → all files from last N days
    """
    fs = gcsfs.GCSFileSystem()
    base = f"gs://{bucket_name}/{PREFIX_RAW}/"
    try:
        all_files = fs.glob(f"{base}*.parquet")
    except FileNotFoundError:
        return []

    if not all_files:
        return []

    if days is None:
        return [max(all_files)]

    cutoff = datetime.now() - timedelta(days=days)
    recent = []
    for f in all_files:
        try:
            info = fs.info(f)
            updated = info.get("updated", datetime.min)
            if isinstance(updated, str):
                updated = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            if updated.replace(tzinfo=None) >= cutoff:
                recent.append(f)
        except Exception:
            continue
    return sorted(recent) if recent else [max(all_files)]


def _deduplicate(df):
    """Deduplicate by id, keeping first occurrence."""
    if "id" not in df.columns:
        return df
    return df.unique(subset=["id"], keep="first", maintain_order=True)


def run_pipeline(bucket_name, days=None, max_jobs=None, force=False):
    """Bronze → Silver → Gold.

    days=None → latest raw file only (daily mode)
    days=N   → last N days of raw files, deduplicated (manual backfill)
    max_jobs → limit number of jobs processed (for fast tests)
    force    → skip Databricks check and run unconditionally
    """
    if not force and _databricks_already_produced(bucket_name):
        logger.info("databricks_skip")
        return None, None

    raw_files = _list_raw_files(bucket_name, days=days)
    if not raw_files:
        logger.info("no_raw_files")
        return None, None

    mode = f"last {days} days" if days else "daily (latest file)"
    logger.info("pipeline_mode", mode=mode, file_count=len(raw_files))

    dfs = []
    for rf in raw_files:
        logger.info("loading_raw", file=rf)
        dfs.append(pl.read_parquet(rf, use_pyarrow=True))
    df = pl.concat(dfs, how="vertical")
    total_before = df.height
    df = _deduplicate(df)
    total_after = df.height
    if total_before != total_after:
        logger.info(
            "deduplication",
            removed=total_before - total_after,
            kept=total_after,
        )
    logger.info("jobs_loaded", count=total_after)

    if max_jobs and max_jobs < total_after:
        logger.info("limiting_jobs", limit=max_jobs, total=total_after)
        df = df.head(max_jobs)

    logger.info("step_clean_html")
    if "description" in df.columns:
        df = df.with_columns(
            pl.col("description")
            .map_elements(clean_html, return_dtype=pl.String)
            .alias("description_clean")
        )
    else:
        df = df.with_columns(pl.lit("").alias("description_clean"))

    logger.info("step_aggregate")
    competences_text = df["competences"].map_elements(
        lambda x: _extract_field(x, "libelle"), return_dtype=pl.String
    )
    formations_text = df["formations"].map_elements(
        lambda x: _extract_field(x, "domaineLibelle"), return_dtype=pl.String
    )
    qualites_text = pl.Series("qualites", [""] * df.height)
    if "qualitesProfessionnelles" in df.columns:
        qualites_text = df["qualitesProfessionnelles"].map_elements(
            lambda x: _extract_field(x, "libelle"), return_dtype=pl.String
        )

    df = df.with_columns(
        (
            pl.col("intitule").fill_null("")
            + pl.lit(" ")
            + pl.col("description_clean").fill_null("")
            + pl.lit(" ")
            + competences_text.fill_null("")
            + pl.lit(" ")
            + formations_text.fill_null("")
            + pl.lit(" ")
            + qualites_text.fill_null("")
        )
        .str.slice(0, 5000)
        .alias("vector_text_input")
    )

    logger.info("step_embeddings", model=MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = df["vector_text_input"].fill_null("").to_list()
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    )
    logger.info("embeddings_generated", count=len(embeddings), dims=embeddings.shape[1])

    logger.info("step_silver")
    df_silver = df.clone()
    df_silver = df_silver.with_columns(pl.col("id").cast(pl.String).alias("job_id"))
    df_silver = df_silver.drop(["id", "description"])
    df_silver = df_silver.rename({"description_clean": "description"})
    df_silver = df_silver.with_columns(
        pl.lit(datetime.now().strftime("%Y-%m-%d")).alias("ingestion_date")
    )

    for col in JSON_COLS:
        if col in df_silver.columns:
            df_silver = df_silver.with_columns(
                pl.col(col).map_elements(serialize_json_col, return_dtype=pl.String).alias(col)
            )

    logger.info("step_gold")
    df_gold = pl.DataFrame(
        {
            "job_id": df_silver["job_id"].to_list(),
            "embedding": [emb.tolist() for emb in embeddings],
        }
    )

    logger.info("step_write")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    silver_path = f"gs://{bucket_name}/{PREFIX_SILVER}/jobs_silver_{ts}.parquet"
    gold_path = f"gs://{bucket_name}/{PREFIX_GOLD}/jobs_gold_{ts}.parquet"

    df_silver.write_parquet(silver_path)
    df_gold.write_parquet(gold_path)

    logger.info("silver_written", path=silver_path, count=df_silver.height)
    logger.info("gold_written", path=gold_path, count=df_gold.height)
    logger.info("pipeline_success")

    return silver_path, gold_path
