"""
Raw data loader and schema validator for Customer Support on Twitter dataset.

Loads twcs.csv, validates the required schema, computes baseline statistics,
and logs row counts to pipeline_stats.json.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("load_raw")

REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

DEFAULT_RAW_PATH = "data/raw/twcs.csv"
DEFAULT_STATS_PATH = "reports/pipeline_stats.json"


class SchemaValidationError(ValueError):
    """Raised when twcs.csv does not conform to the expected schema."""
    pass


def update_pipeline_stats(stats_path: str, step_name: str, data: Dict[str, Any]) -> None:
    """Safely append or update a step's stats in pipeline_stats.json."""
    stats_file = Path(stats_path)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {}
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing stats file {stats_path}: {e}")

    stats[step_name] = {
        **data,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Updated pipeline stats at {stats_path} for step '{step_name}'")


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame has the exact required columns and non-empty content.
    Fails loudly with SchemaValidationError if invalid.
    """
    existing_cols = list(df.columns)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in existing_cols]

    if missing_cols:
        msg = (
            f"Schema Validation Failed! twcs.csv is missing required column(s): {missing_cols}.\n"
            f"Expected: {REQUIRED_COLUMNS}\n"
            f"Found: {existing_cols}"
        )
        logger.error(msg)
        raise SchemaValidationError(msg)

    if df.empty:
        msg = "Schema Validation Failed! twcs.csv is completely empty (0 rows)."
        logger.error(msg)
        raise SchemaValidationError(msg)

    logger.info("Schema validation passed successfully. All 7 required columns are present.")


def load_and_validate_raw(
    csv_path: str = DEFAULT_RAW_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw twcs.csv, validate schema, compute summary stats, and persist stats.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (df, stats_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {csv_path}")

    logger.info(f"Loading raw dataset from {csv_path}...")
    start_time = datetime.utcnow()

    # Load with optimal dtypes for memory & speed
    # Note: in_response_to_tweet_id can be NaN, response_tweet_id can be comma-separated list
    dtype_spec = {
        "author_id": "str",
        "inbound": "bool",
        "text": "str",
        "response_tweet_id": "str",
    }

    # Reading CSV
    df = pd.read_csv(
        csv_path,
        dtype=dtype_spec,
        low_memory=False,
    )

    load_duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Loaded {len(df):,} rows in {load_duration:.2f}s.")

    # Validate Schema
    validate_schema(df)

    # Compute Statistics
    total_rows = len(df)
    inbound_count = int(df["inbound"].sum())
    outbound_count = total_rows - inbound_count

    # Null counts per column
    null_counts = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}

    # Parse created_at timestamps
    logger.info("Parsing timestamps to determine date range...")
    # twitter format: Tue Oct 31 22:10:47 +0000 2017
    try:
        created_at_dt = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce")
    except Exception:
        created_at_dt = pd.to_datetime(df["created_at"], errors="coerce")
    malformed_dates = int(created_at_dt.isna().sum())
    min_date = str(created_at_dt.min()) if not created_at_dt.empty else None
    max_date = str(created_at_dt.max()) if not created_at_dt.empty else None

    # Check for malformed tweet_id
    invalid_tweet_ids = int(pd.to_numeric(df["tweet_id"], errors="coerce").isna().sum())

    stats = {
        "raw_file_path": str(csv_path),
        "total_rows": total_rows,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "inbound_pct": round(100.0 * inbound_count / total_rows, 2) if total_rows > 0 else 0,
        "outbound_pct": round(100.0 * outbound_count / total_rows, 2) if total_rows > 0 else 0,
        "date_range_min": min_date,
        "date_range_max": max_date,
        "null_counts": null_counts,
        "malformed_dates_count": malformed_dates,
        "malformed_tweet_ids_count": invalid_tweet_ids,
        "load_duration_seconds": round(load_duration, 2),
    }

    # Print summary report
    print("=" * 60)
    print("STAGE 1: RAW INGESTION & SCHEMA VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Rows Ingested:        {total_rows:,}")
    print(f"Date Range:                 {min_date} --> {max_date}")
    print(f"Inbound (Customer) Tweets:  {inbound_count:,} ({stats['inbound_pct']}%)")
    print(f"Outbound (Brand) Tweets:    {outbound_count:,} ({stats['outbound_pct']}%)")
    print("Null Counts:")
    for col, count in null_counts.items():
        print(f"  - {col:25s}: {count:,} ({count / total_rows * 100:.2f}%)")
    print(f"Malformed Dates:            {malformed_dates}")
    print(f"Malformed Tweet IDs:        {invalid_tweet_ids}")
    print("=" * 60)

    # Persist stats
    update_pipeline_stats(stats_path, "stage1_raw_ingestion", stats)

    return df, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Ingest and validate twcs.csv raw dataset.")
    parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Path to twcs.csv")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH, help="Path to pipeline_stats.json")
    args = parser.parse_args()

    load_and_validate_raw(csv_path=args.input, stats_path=args.stats)
