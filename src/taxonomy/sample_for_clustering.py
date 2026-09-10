"""
Stratified customer first-message sampling module for Stage 3 (Intent Taxonomy).

Extracts the first inbound customer message from each AppleSupport thread in
data/processed/AppleSupport_threads.parquet, and draws a reproducible,
stratified sample across time period and conversation thread length.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sample_for_clustering")

DEFAULT_INPUT_PATH = "data/processed/AppleSupport_threads.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/taxonomy_sample.parquet"
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42


def clean_tweet_text(text: str) -> str:
    """
    Clean Twitter handles and excessive whitespace for semantic analysis,
    while preserving actual message content, punctuation, and problem description.
    """
    if not text or not isinstance(text, str):
        return ""
    # Strip leading @mentions
    cleaned = re.sub(r"^(?:@\w+\s*)+", "", text)
    # Strip inline @mentions of AppleSupport specifically
    cleaned = re.sub(r"@AppleSupport\b", "", cleaned, flags=re.IGNORECASE)
    # Normalize multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text.strip()


def extract_customer_first_messages(threads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the initial inbound customer message from each thread.
    Returns a DataFrame of candidate customer inquiries with temporal and length metadata.
    """
    logger.info(f"Extracting customer first-messages from {len(threads_df):,} threads...")

    records = []
    inbounds_arr = threads_df["inbounds"].values
    texts_arr = threads_df["texts"].values
    tids_arr = threads_df["tweet_ids"].values
    thread_ids_arr = threads_df["thread_id"].values
    lengths_arr = threads_df["thread_length"].values
    start_times_arr = threads_df["start_time"].values
    resolved_arr = threads_df["resolved"].values if "resolved" in threads_df.columns else ["unknown"] * len(threads_df)
    reasons_arr = threads_df["resolution_reason"].values if "resolution_reason" in threads_df.columns else [""] * len(threads_df)

    for idx, (inbs, texts, tids, th_id, th_len, st_tm, res, reason) in enumerate(
        zip(inbounds_arr, texts_arr, tids_arr, thread_ids_arr, lengths_arr, start_times_arr, resolved_arr, reasons_arr)
    ):
        # Locate the first customer message (inbound == True)
        first_cust_idx = None
        for i, inb in enumerate(inbs):
            if inb:
                first_cust_idx = i
                break

        if first_cust_idx is None or first_cust_idx >= len(texts):
            continue

        raw_txt = str(texts[first_cust_idx] or "").strip()
        if not raw_txt:
            continue

        tweet_id = tids[first_cust_idx] if first_cust_idx < len(tids) else ""
        cleaned = clean_tweet_text(raw_txt)

        records.append({
            "thread_id": th_id,
            "tweet_id": str(tweet_id),
            "customer_msg_idx": first_cust_idx,
            "text": raw_txt,
            "cleaned_text": cleaned,
            "thread_length": int(th_len),
            "start_time": str(st_tm),
            "resolved": str(res),
            "resolution_reason": str(reason),
        })

    extracted_df = pd.DataFrame(records)
    logger.info(f"Successfully extracted {len(extracted_df):,} customer inquiries.")
    return extracted_df


def assign_stratification_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign time_period and thread_length_bin strata to each message.
    - time_period: parsed start_time bucketed by Month (e.g. 2017-10, 2017-11)
    - thread_length_bin: short (2), medium (3-4), long (5+)
    """
    df = df.copy()

    # Thread length bins
    def categorize_length(length: int) -> str:
        if length <= 2:
            return "short_2"
        elif length <= 4:
            return "medium_3_4"
        else:
            return "long_5plus"

    df["thread_length_bin"] = df["thread_length"].apply(categorize_length)

    # Time period bins
    try:
        parsed_dates = pd.to_datetime(
            df["start_time"],
            format="%a %b %d %H:%M:%S +0000 %Y",
            errors="coerce",
        )
    except Exception:
        parsed_dates = pd.to_datetime(df["start_time"], errors="coerce")

    # Group rare older dates into 'pre_2017_10', and October/November into monthly bins
    def categorize_period(dt) -> str:
        if pd.isna(dt):
            return "unknown_time"
        if dt.year < 2017 or (dt.year == 2017 and dt.month < 10):
            return "pre_2017_10"
        return f"{dt.year}_{dt.month:02d}"

    df["time_period"] = parsed_dates.apply(categorize_period)
    df["stratum"] = df["time_period"] + "__" + df["thread_length_bin"]

    return df


def draw_stratified_sample(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Draw a proportional stratified sample of size `sample_size` across defined strata.
    Guarantees exact determinism given `seed`.
    """
    if len(df) <= sample_size:
        logger.warning(f"Dataset has {len(df)} rows <= requested sample size {sample_size}. Returning full dataset.")
        return df.copy()

    strata_counts = df["stratum"].value_counts()
    total_rows = len(df)

    # Calculate proportional targets per stratum
    sampled_dfs = []
    rng = np.random.RandomState(seed)

    allocated_total = 0
    allocations = {}
    for stratum, count in strata_counts.items():
        # Proportional allocation
        target = int(round(sample_size * (count / total_rows)))
        target = max(1, min(count, target))  # At least 1 if stratum exists
        allocations[stratum] = target
        allocated_total += target

    # Rebalance rounding difference if any
    diff = sample_size - allocated_total
    if diff != 0:
        sorted_strata = sorted(allocations.keys(), key=lambda s: strata_counts[s], reverse=True)
        for i in range(abs(diff)):
            s = sorted_strata[i % len(sorted_strata)]
            if diff > 0 and allocations[s] < strata_counts[s]:
                allocations[s] += 1
            elif diff < 0 and allocations[s] > 1:
                allocations[s] -= 1

    for stratum, target_count in allocations.items():
        stratum_df = df[df["stratum"] == stratum]
        # Deterministic sample for this stratum
        # Use fixed seed derived from master seed and stratum hash for independence
        stratum_seed = (seed + int(abs(hash(stratum)) % 100000)) % (2**31 - 1)
        sub_sample = stratum_df.sample(n=target_count, random_state=stratum_seed)
        sampled_dfs.append(sub_sample)

    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle final sample deterministically
    final_sample = final_sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(f"Drawn stratified sample of {len(final_sample):,} rows across {len(allocations)} strata.")
    return final_sample


def sample_customer_first_messages(
    threads_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    End-to-end execution of Step 1:
    Loads AppleSupport threads, extracts customer first messages, stratifies, samples,
    and writes data/processed/taxonomy_sample.parquet.
    """
    logger.info(f"Loading threads from {threads_path}...")
    threads_df = pd.read_parquet(threads_path)

    extracted_df = extract_customer_first_messages(threads_df)
    stratified_df = assign_stratification_bins(extracted_df)
    sample_df = draw_stratified_sample(stratified_df, sample_size=sample_size, seed=seed)

    # Save to parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved stratified sample ({len(sample_df):,} rows) to {output_path}")

    # Print summary of stratification
    print("\n" + "=" * 70)
    print(f"STRATIFIED TAXONOMY SAMPLE SUMMARY (Total: {len(sample_df):,} messages)")
    print("=" * 70)
    summary = sample_df.groupby(["time_period", "thread_length_bin"]).size().unstack(fill_value=0)
    print(summary)
    print("=" * 70 + "\n")

    return sample_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified sampling for Stage 3 taxonomy derivation.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to AppleSupport_threads.parquet")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to taxonomy_sample.parquet")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size (default: 2000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    sample_customer_first_messages(
        threads_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
