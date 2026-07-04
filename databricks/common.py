"""Shared utilities for Databricks pipeline notebooks."""

import json
import re
from datetime import datetime


def clean_html(text):
    """Strip HTML tags and entities, collapse whitespace.

    Args:
        text: Raw text potentially containing HTML markup.

    Returns:
        Cleaned plain text string, or ``""`` if input is None.
    """
    if text is None:
        return ""
    clean_re = re.compile(r"<.*?>|&([a-zA-Z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});")
    text = re.sub(clean_re, "", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text).strip()
    return text


def serialize_json_col(val):
    """Convert nested dict/list to JSON string for JSONB compatibility.

    Args:
        val: Value to serialize (dict, list, str, None).

    Returns:
        JSON string if val is a complex type, the same string if already str,
        or None if val is None.
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False)


def find_latest_raw_file(gcs_raw_path):
    """Find the most recent parquet file in the raw GCS prefix.

    Args:
        gcs_raw_path: GCS prefix to search, e.g. ``"gs://bucket/jobs_raw/"``.

    Returns:
        The GCS URI of the most recent parquet file.

    Raises:
        FileNotFoundError: If no parquet files are found in the prefix.
    """
    try:
        jvm = spark._jvm
        uri = jvm.java.net.URI(gcs_raw_path)
        fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, spark._jsc.hadoopConfiguration())
        path = jvm.org.apache.hadoop.fs.Path(gcs_raw_path)
        file_statuses = fs.listStatus(path)
        parquet_files = [
            f.getPath().toUri().toString()
            for f in file_statuses
            if f.getPath().getName().endswith(".parquet")
        ]
    except Exception:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        parquet_files = sorted(fs.glob(f"{gcs_raw_path.rstrip('/')}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {gcs_raw_path}")
    return max(parquet_files, key=_extract_date_from_name)


def _extract_date_from_name(path):
    match = re.search(r"(\d{8}_\d{6})", path)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return datetime.min


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
