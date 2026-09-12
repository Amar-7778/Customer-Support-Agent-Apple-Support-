"""
Golden evaluation holdout extraction module for Stage 4 (Step 1).

Selects exactly 300 customer threads from the 6,000 verified Groq-classified sample,
stratified proportionally across all 8 intents, using a fixed random seed.
Permanently isolates these threads into data/processed/golden_eval_candidates.parquet
to guarantee zero data leakage into the retrieval vector index.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("holdout_split")

DEFAULT_CLASSIFIED_PATH = "data/processed/AppleSupport_classified_sample.parquet"
DEFAULT_HOLDOUT_PATH = "data/processed/golden_eval_candidates.parquet"
DEFAULT_INDEXABLE_PATH = "data/processed/indexable_precedents_input.parquet"
DEFAULT_HOLDOUT_SIZE = 300
DEFAULT_SEED = 42


def split_golden_holdout(
    classified_path: str = DEFAULT_CLASSIFIED_PATH,
    holdout_output_path: str = DEFAULT_HOLDOUT_PATH,
    indexable_output_path: str = DEFAULT_INDEXABLE_PATH,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the 6,000 classified messages into:
    1. A golden evaluation holdout set (~300 threads, stratified across 8 intents).
    2. An indexable pool (remaining 5,700 threads) for Stage 4 precedent indexing.

    Strict assertion guards guarantee zero leakage between holdout and indexable pools.
    """
    if not os.path.exists(classified_path):
        raise FileNotFoundError(f"Classified corpus not found at {classified_path}")

    logger.info(f"Loading classified sample from {classified_path}...")
    df = pd.read_parquet(classified_path)
    total_records = len(df)
    logger.info(f"Loaded {total_records:,} verified customer inquiries.")

    if holdout_size >= total_records:
        raise ValueError(f"Holdout size ({holdout_size}) must be less than total records ({total_records})")

    # Proportional stratified sampling across the 8 intents
    intent_counts = df["predicted_intent"].value_counts()
    holdout_dfs = []
    
    # Calculate exact allocation per stratum
    allocated_counts = {}
    for intent, count in intent_counts.items():
        n_strat = int(round(holdout_size * (count / total_records)))
        allocated_counts[intent] = max(1, n_strat)

    # Adjust rounding discrepancy if any to hit exact holdout_size
    diff = holdout_size - sum(allocated_counts.values())
    if diff != 0:
        # Allocate difference to the most populous category
        top_intent = intent_counts.index[0]
        allocated_counts[top_intent] += diff

    for intent, n_target in allocated_counts.items():
        intent_slice = df[df["predicted_intent"] == intent]
        sampled_slice = intent_slice.sample(n=n_target, random_state=seed)
        holdout_dfs.append(sampled_slice)

    holdout_df = pd.concat(holdout_dfs, ignore_index=True).sort_values("thread_id").reset_index(drop=True)
    
    # Isolate non-holdout rows
    holdout_threads = set(holdout_df["thread_id"])
    indexable_df = df[~df["thread_id"].isin(holdout_threads)].sort_values("thread_id").reset_index(drop=True)

    # PERMANENT ZERO-LEAKAGE CONSERVATION ASSERTIONS
    assert len(holdout_df) == holdout_size, f"Holdout size mismatch: {len(holdout_df)} != {holdout_size}"
    expected_indexable = total_records - holdout_size
    assert len(indexable_df) == expected_indexable, (
        f"Indexable size mismatch: {len(indexable_df)} != {expected_indexable}"
    )
    assert set(holdout_df["thread_id"]).isdisjoint(set(indexable_df["thread_id"])), (
        "CRITICAL LEAKAGE ERROR: Overlapping thread_ids detected between holdout and indexable pool!"
    )
    assert len(set(holdout_df["thread_id"]).union(set(indexable_df["thread_id"]))) == total_records, (
        "Conservation error: Total union does not match 6,000 input messages!"
    )

    # Save outputs
    Path(holdout_output_path).parent.mkdir(parents=True, exist_ok=True)
    holdout_df.to_parquet(holdout_output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(holdout_df):,} golden eval holdout candidates to {holdout_output_path}")

    Path(indexable_output_path).parent.mkdir(parents=True, exist_ok=True)
    indexable_df.to_parquet(indexable_output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(indexable_df):,} indexable candidates to {indexable_output_path}")

    # Print stratification comparison
    print("\n" + "=" * 80)
    print(f"STAGE 4 HOLDOUT PARTITION REPORT (Seed={seed})")
    print("=" * 80)
    format_row = "{:<36} | {:<12} | {:<12} | {:<10}"
    print(format_row.format("Intent Name", "Total (6k)", "Holdout (300)", "Indexable (5.7k)"))
    print("-" * 80)
    for intent, count in intent_counts.items():
        h_cnt = (holdout_df["predicted_intent"] == intent).sum()
        i_cnt = (indexable_df["predicted_intent"] == intent).sum()
        print(format_row.format(intent, f"{count:,}", f"{h_cnt:>4} ({h_cnt/holdout_size*100:.1f}%)", f"{i_cnt:,}"))
    print("-" * 80)
    print(format_row.format("TOTAL", f"{total_records:,}", f"{len(holdout_df):,} (100.0%)", f"{len(indexable_df):,}"))
    print("=" * 80 + "\n")

    return holdout_df, indexable_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split classified dataset into golden holdout and indexable pool.")
    parser.add_argument("--classified", default=DEFAULT_CLASSIFIED_PATH)
    parser.add_argument("--holdout-output", default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument("--indexable-output", default=DEFAULT_INDEXABLE_PATH)
    parser.add_argument("--holdout-size", type=int, default=DEFAULT_HOLDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    split_golden_holdout(
        classified_path=args.classified,
        holdout_output_path=args.holdout_output,
        indexable_output_path=args.indexable_output,
        holdout_size=args.holdout_size,
        seed=args.seed,
    )
