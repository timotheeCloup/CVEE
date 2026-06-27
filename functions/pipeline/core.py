import json
import os
import re
from datetime import datetime, timedelta

import gcsfs
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Cloud Functions: /tmp is the only writable path
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.makedirs("/tmp/huggingface", exist_ok=True)

torch.set_num_threads(1)

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


def clean_html(text):
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text)
    clean_re = re.compile(r"<.*?>|&([a-zA-Z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});")
    text = re.sub(clean_re, "", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text).strip()
    return text


def _extract_field(val, field="libelle"):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
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


def _deep_to_list(obj):
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_deep_to_list(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _deep_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def serialize_json_col(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str):
        return val
    val = _deep_to_list(val)
    return json.dumps(val, ensure_ascii=False)


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
            if info.get("updated", datetime.min).replace(tzinfo=None) >= cutoff:
                recent.append(f)
        except Exception:
            continue
    return sorted(recent) if recent else [max(all_files)]


def _deduplicate(df):
    """Deduplicate by id, keeping first occurrence."""
    if "id" not in df.columns:
        return df
    return df.drop_duplicates(subset=["id"], keep="first")


def run_pipeline(bucket_name, days=None):
    """Bronze → Silver → Gold.

    days=None → latest raw file only (daily mode)
    days=N   → last N days of raw files, deduplicated (manual backfill)
    """
    fs = gcsfs.GCSFileSystem()
    raw_files = _list_raw_files(bucket_name, days=days)
    if not raw_files:
        print("No raw files found. Aborting.")
        return None, None

    mode = f"last {days} days" if days else "daily (latest file)"
    print(f"Pipeline mode: {mode} — {len(raw_files)} raw file(s)")

    # --- 1. Load + deduplicate raw ---
    dfs = []
    for rf in raw_files:
        print(f"  Loading {rf}")
        dfs.append(pd.read_parquet(rf, filesystem=fs))
    df = pd.concat(dfs, ignore_index=True)
    total_before = len(df)
    df = _deduplicate(df)
    total_after = len(df)
    if total_before != total_after:
        print(f"  Deduplication: {total_before - total_after} duplicates removed ({total_after} kept)")
    print(f"  {total_after} unique jobs loaded")

    # --- 2. Clean HTML ---
    print("2. Cleaning HTML...")
    if "description" in df.columns:
        df["description_clean"] = df["description"].apply(clean_html)
    else:
        df["description_clean"] = ""

    # --- 3. Aggregate JSON arrays ---
    print("3. Aggregating competences/formations/qualites...")
    competences_text = df["competences"].apply(lambda x: _extract_field(x, "libelle"))
    formations_text = df["formations"].apply(lambda x: _extract_field(x, "domaineLibelle"))
    qualites_col = df.get("qualitesProfessionnelles", pd.Series([None] * len(df)))
    qualites_text = qualites_col.apply(lambda x: _extract_field(x, "libelle"))

    df["vector_text_input"] = (
        df["intitule"].fillna("")
        + " "
        + df["description_clean"].fillna("")
        + " "
        + competences_text.fillna("")
        + " "
        + formations_text.fillna("")
        + " "
        + qualites_text.fillna("")
    ).str.slice(0, 5000)

    # --- 4. Generate embeddings ---
    print(f"4. Generating embeddings with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = df["vector_text_input"].fillna("").tolist()
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"   {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")

    # --- 5. Build silver ---
    print("5. Building silver layer...")
    df_silver = df.copy()
    df_silver["job_id"] = df_silver["id"].astype(str)
    df_silver.drop(columns=["id", "description_clean"], inplace=True, errors="ignore")
    df_silver["description"] = df["description_clean"]
    df_silver["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")

    for col in JSON_COLS:
        if col in df_silver.columns:
            df_silver[col] = df_silver[col].apply(serialize_json_col)

    # --- 6. Build gold ---
    print("6. Building gold layer...")
    df_gold = pd.DataFrame(
        {"job_id": df_silver["job_id"], "embedding": [emb.tolist() for emb in embeddings]}
    )

    # --- 7. Write to GCS ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    silver_path = f"gs://{bucket_name}/{PREFIX_SILVER}/jobs_silver_{ts}.parquet"
    gold_path = f"gs://{bucket_name}/{PREFIX_GOLD}/jobs_gold_{ts}.parquet"

    print("7. Writing to GCS...")
    df_silver.to_parquet(silver_path, engine="pyarrow", index=False, filesystem=fs)
    df_gold.to_parquet(gold_path, engine="pyarrow", index=False, filesystem=fs)

    print(f"   Silver: {silver_path}  ({len(df_silver)} jobs)")
    print(f"   Gold:   {gold_path}  ({len(df_gold)} jobs)")
    print("Pipeline completed successfully.")

    return silver_path, gold_path
