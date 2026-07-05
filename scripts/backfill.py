#!/usr/bin/env python3
"""
Historical backfill script for CVEE.

Fetches France Travail jobs for a given date range (month by month or day by day)
and exports to GCS. Can optionally run the pipeline (bronze→silver→gold) after each chunk.

Usage:
    # Backfill January + February 2026, month by month
    uv run python scripts/backfill.py --date-min 2026-01-01 --date-max 2026-02-28

    # Backfill with pipeline run after each month
    uv run python scripts/backfill.py --date-min 2026-01-01 --date-max 2026-06-01 --pipeline

    # Backfill a single specific month
    uv run python scripts/backfill.py --date-min 2026-05-01 --date-max 2026-05-31

    # Backfill last N months (from today)
    uv run python scripts/backfill.py --months 6

    # Call deployed Cloud Function via curl:
    curl -X POST "https://europe-west1-cvee-20260208.cloudfunctions.net/api-to-gcs-cf?date_min=2026-01-01&date_max=2026-01-31"
"""

import argparse
import calendar
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "functions" / "api-to-gcs"))
from ft_client import export_to_gcs, fetch_jobs_data, get_ft_token  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "functions" / "pipeline"))
from core import run_pipeline  # noqa: E402


def generate_month_ranges(date_min_str, date_max_str):
    """Yield (start, end) pairs for each month in [date_min, date_max]"""
    start = datetime.strptime(date_min_str, "%Y-%m-%d")
    end = datetime.strptime(date_max_str, "%Y-%m-%d")

    current = start
    while current <= end:
        month_start = current.strftime("%Y-%m-%d")
        month_end_date = min(
            end,
            datetime(
                current.year, current.month, calendar.monthrange(current.year, current.month)[1]
            ),
        )
        month_end = month_end_date.strftime("%Y-%m-%d")
        yield month_start, month_end

        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def run_backfill(
    date_min_str,
    date_max_str,
    bucket_name,
    ft_client_id,
    ft_client_secret,
    with_pipeline=False,
    max_index=3000,
):
    print(f"\n{'=' * 60}")
    print(f"Backfill: {date_min_str} → {date_max_str}")
    print(f"{'=' * 60}")

    token = get_ft_token(ft_client_id, ft_client_secret)
    if not token:
        raise RuntimeError("Failed to obtain FT API token")

    jobs = fetch_jobs_data(
        token,
        date_min=date_min_str,
        date_max=date_max_str,
        page_size=150,
        max_index=max_index,
    )

    if not jobs:
        print(f"No jobs found for {date_min_str} → {date_max_str}")
        return

    export_to_gcs(jobs, bucket_name, date_min=date_min_str, date_max=date_max_str)

    if with_pipeline:
        print("\nRunning pipeline (Polars)...")
        silver_path, gold_path = run_pipeline(bucket_name, force=True)
        if silver_path and gold_path:
            print(f"Pipeline completed — Silver: {silver_path}, Gold: {gold_path}")
        else:
            print("Pipeline returned no output (no raw files or skipped)")


def main():
    parser = argparse.ArgumentParser(description="CVEE Historical Backfill")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date-min", type=str, help="Start date (YYYY-MM-DD)")
    group.add_argument("--months", type=int, help="Backfill last N months from today")
    parser.add_argument(
        "--date-max", type=str, help="End date (YYYY-MM-DD), required if --date-min is set"
    )
    parser.add_argument("--pipeline", action="store_true", help="Run pipeline after each chunk")
    parser.add_argument(
        "--max-index", type=int, default=3000, help="Max results per chunk (default: 3000)"
    )

    args = parser.parse_args()

    ft_client_id = os.getenv("FT_CLIENT_ID")
    ft_client_secret = os.getenv("FT_CLIENT_SECRET")
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not all([ft_client_id, ft_client_secret, bucket_name]):
        print(
            "ERROR: Missing environment variables. Set FT_CLIENT_ID, FT_CLIENT_SECRET, GCS_BUCKET_NAME"
        )
        print("You can create a .env file at the project root.")
        sys.exit(1)

    if args.months:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.months * 30)
        date_min = start_date.strftime("%Y-%m-%d")
        date_max = end_date.strftime("%Y-%m-%d")
    else:
        if not args.date_max:
            parser.error("--date-max is required when using --date-min")
        date_min = args.date_min
        date_max = args.date_max

    try:
        datetime.strptime(date_min, "%Y-%m-%d")
        datetime.strptime(date_max, "%Y-%m-%d")
    except ValueError:
        print("ERROR: Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    chunks = list(generate_month_ranges(date_min, date_max))
    print(f"Backfill will run in {len(chunks)} monthly chunk(s):")
    for start, end in chunks:
        print(f"  {start} → {end}")

    for start, end in chunks:
        try:
            run_backfill(
                start,
                end,
                bucket_name,
                ft_client_id,
                ft_client_secret,
                with_pipeline=args.pipeline,
                max_index=args.max_index,
            )
        except Exception as e:
            print(f"ERROR on chunk {start}→{end}: {e}")
            print("Continuing with next chunk...")

    print("\n✅ Backfill complete.")


if __name__ == "__main__":
    main()
