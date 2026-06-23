"""
CVEE Data Pipeline: Bronze → Silver → Gold
Replaces Databricks notebooks: reads raw jobs from GCS, cleans,
generates embeddings (multilingual model), writes silver + gold Parquet back to GCS.

Usage:
    python pipeline.py                          # reads latest raw batch
    python pipeline.py --raw gs://bucket/jobs_raw/file.parquet  # specific file
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import gcsfs
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Single-thread torch to avoid conflicts with ThreadPoolExecutor
torch.set_num_threads(1)

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "cvee-20260208")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"
BATCH_SIZE = 32


def clean_html(text):
    """Strip HTML tags and entities, collapse whitespace"""
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


def extract_field_from_array(val, field="libelle"):
    """Extract a field from each element of a JSON-like array, join with space"""
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
    """Recursively convert numpy types to Python native types"""
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
    """Convert nested dict/list/ndarray to JSON string for Supabase JSONB compatibility"""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str):
        return val
    val = _deep_to_list(val)
    return json.dumps(val, ensure_ascii=False)


def main(raw_path=None):
    t_total = time.time()

    # --- 1. Read raw data ---
    fs = gcsfs.GCSFileSystem()
    if raw_path and raw_path.startswith("gs://"):
        raw_file = raw_path
    else:
        raw_files = sorted(fs.glob(f"gs://{GCS_BUCKET}/jobs_raw/*.parquet"))
        if not raw_files:
            print("No raw files found in GCS.")
            return
        raw_file = raw_files[-1]

    print(f"1. Loading {raw_file} ...")
    df = pd.read_parquet(raw_file, filesystem=fs)
    print(f"   {len(df)} jobs loaded")
    t_load = time.time()

    # --- 2. Clean HTML ---
    print("2. Cleaning HTML...")
    if "description" in df.columns:
        df["description_clean"] = df["description"].apply(clean_html)
    else:
        df["description_clean"] = ""
    t_clean = time.time()
    print(f"   Done in {t_clean - t_load:.1f}s")

    # --- 3. Aggregate JSON arrays ---
    print("3. Aggregating competences/formations/qualites...")
    competences_text = df["competences"].apply(lambda x: extract_field_from_array(x, "libelle"))
    formations_text = df["formations"].apply(
        lambda x: extract_field_from_array(x, "domaineLibelle")
    )
    qualites_text = df.get("qualitesProfessionnelles", pd.Series([None] * len(df)))
    qualites_text = qualites_text.apply(lambda x: extract_field_from_array(x, "libelle"))

    # Build concatenated text (for embedding via multilingual model)
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
    t_agg = time.time()
    print(f"   Done in {t_agg - t_clean:.1f}s")

    # --- 4. Generate embeddings (multilingual model, no translation needed) ---
    print("4. Generating embeddings...")
    model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)
    texts = df["vector_text_input"].fillna("").tolist()
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
    )
    t_embed = time.time()
    print(f"   {len(embeddings)} embeddings ({embeddings.shape[1]} dims) in {t_embed - t_agg:.1f}s")

    # --- 5. Build silver DataFrame ---
    print("5. Building silver layer...")
    df_silver = df.copy()

    # Rename id → job_id, use cleaned description
    df_silver["job_id"] = df_silver["id"].astype(str)
    df_silver.drop(columns=["id", "description_clean"], inplace=True)
    df_silver["description"] = df["description_clean"]
    df_silver["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")

    # Serialize JSON struct columns
    json_cols = [
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
    for col in json_cols:
        if col in df_silver.columns:
            df_silver[col] = df_silver[col].apply(serialize_json_col)

    # --- 6. Build gold DataFrame ---
    print("6. Building gold layer...")
    df_gold = pd.DataFrame(
        {"job_id": df_silver["job_id"], "embedding": [emb.tolist() for emb in embeddings]}
    )

    # --- 7. Write to GCS ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    silver_path = f"gs://{GCS_BUCKET}/jobs_silver/jobs_silver_{ts}.parquet"
    gold_path = f"gs://{GCS_BUCKET}/jobs_gold/jobs_gold_{ts}.parquet"

    print("7. Writing to GCS...")
    df_silver.to_parquet(silver_path, engine="pyarrow", index=False, filesystem=fs)
    df_gold.to_parquet(gold_path, engine="pyarrow", index=False, filesystem=fs)

    t_end = time.time()
    print(f"\n✅ Pipeline completed in {t_end - t_total:.1f}s")
    print(f"   Silver: {silver_path}  ({len(df_silver)} jobs)")
    print(f"   Gold:   {gold_path}  ({len(df_gold)} jobs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVEE Data Pipeline")
    parser.add_argument("--raw", type=str, default=None, help="Specific raw parquet file")
    args = parser.parse_args()
    main(raw_path=args.raw)
