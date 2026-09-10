"""
Brand filter and candidate evaluation module for Customer Support on Twitter dataset.

Filters reconstructed conversation threads for a specific target brand (e.g. AppleSupport),
evaluates top candidate brands by volume to support data-driven brand selection,
applies the configurable resolution heuristic to tag threads with resolved status and reason,
and writes the stage 2 processed output artifact.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .heuristics import ResolutionConfig, evaluate_thread_resolution
from .load_raw import DEFAULT_STATS_PATH, update_pipeline_stats
from .reconstruct_threads import DEFAULT_THREADS_OUTPUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("filter_brand")

DEFAULT_CANDIDATES_PATH = "reports/brand_candidates.csv"


def generate_brand_candidate_report(
    threads_df: pd.DataFrame,
    top_n: int = 20,
    output_csv: str = DEFAULT_CANDIDATES_PATH,
    config: Optional[ResolutionConfig] = None,
    dataset_max_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Evaluate top N candidate brands by volume across reconstructed threads.
    Computes total tweets, total threads, average thread length, and % resolved.
    Saves results to reports/brand_candidates.csv.
    """
    logger.info(f"Analyzing top {top_n} candidate brands across {len(threads_df):,} threads...")
    t0 = time.time()
    if config is None:
        config = ResolutionConfig()

    # Determine global max time if not provided
    if dataset_max_time is None and "end_time" in threads_df.columns:
        try:
            dataset_max_time = pd.to_datetime(threads_df["end_time"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce").max()
        except Exception:
            dataset_max_time = pd.to_datetime(threads_df["end_time"], errors="coerce").max()

    # Explode brand_handles_involved to count brand occurrences
    brand_exploded = threads_df[["thread_id", "brand_handles_involved", "thread_length"]].explode("brand_handles_involved")
    brand_exploded = brand_exploded.dropna(subset=["brand_handles_involved"])
    brand_counts = brand_exploded["brand_handles_involved"].value_counts()

    top_brands = brand_counts.head(top_n).index.tolist()
    logger.info(f"Top {len(top_brands)} brands by thread volume: {top_brands}")

    records = []
    for brand in top_brands:
        # Fast index slice for threads containing this brand
        indices = brand_exploded.index[brand_exploded["brand_handles_involved"] == brand].unique()
        b_threads = threads_df.iloc[indices]
        num_threads = len(b_threads)
        if num_threads == 0:
            continue

        # Count total tweets in threads and total brand tweets
        avg_len = round(float(b_threads["thread_length"].mean()), 2)
        total_thread_tweets = int(b_threads["thread_length"].sum())

        # Count brand's own tweets
        def count_brand_msgs(authors):
            return sum(1 for a in authors if str(a).lower() == brand.lower())

        total_brand_tweets = int(b_threads["author_ids"].apply(count_brand_msgs).sum())

        # Fast batch resolution evaluation
        resolved_count = 0
        unresolved_count = 0
        unknown_count = 0

        inbounds_list = b_threads["inbounds"].values
        authors_list = b_threads["author_ids"].values
        texts_list = b_threads["texts"].values
        end_times_list = b_threads["end_time"].values
        has_created_ats = "created_ats" in b_threads.columns
        created_ats_list = b_threads["created_ats"].values if has_created_ats else None

        for idx, (inbounds, authors, texts, end_tm) in enumerate(zip(inbounds_list, authors_list, texts_list, end_times_list)):
            if has_created_ats:
                cats = created_ats_list[idx]
                tw_list = [
                    {"inbound": inb, "author_id": auth, "text": txt, "created_at": cat}
                    for inb, auth, txt, cat in zip(inbounds, authors, texts, cats)
                ]
            else:
                tw_list = [
                    {"inbound": inb, "author_id": auth, "text": txt, "created_at": None}
                    for inb, auth, txt in zip(inbounds, authors, texts)
                ]
                if tw_list:
                    tw_list[-1]["created_at"] = end_tm
            status, _ = evaluate_thread_resolution(
                tweets=tw_list,
                brand_handle=brand,
                dataset_max_time=dataset_max_time,
                config=config,
            )
            if status == "true":
                resolved_count += 1
            elif status == "false":
                unresolved_count += 1
            else:
                unknown_count += 1

        pct_resolved = round(100.0 * resolved_count / num_threads, 2) if num_threads > 0 else 0.0

        records.append({
            "brand_handle": brand,
            "total_brand_tweets": total_brand_tweets,
            "total_thread_tweets": total_thread_tweets,
            "total_threads": num_threads,
            "avg_thread_length": avg_len,
            "resolved_threads": resolved_count,
            "unresolved_threads": unresolved_count,
            "unknown_threads": unknown_count,
            "pct_resolved": pct_resolved,
        })

    cand_df = pd.DataFrame(records)
    cand_df.sort_values(by="total_threads", ascending=False, inplace=True)

    # Save to CSV
    cand_path = Path(output_csv)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(cand_path, index=False)
    logger.info(f"Saved candidate brand report to {output_csv} in {time.time() - t0:.2f}s.")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"TOP {len(cand_df)} BRAND CANDIDATES BENCHMARK (reports/brand_candidates.csv)")
    print("=" * 80)
    format_row = "{:<16} | {:<12} | {:<12} | {:<10} | {:<12}"
    print(format_row.format("Brand Handle", "Brand Tweets", "Total Threads", "Avg Length", "% Resolved"))
    print("-" * 80)
    for _, row in cand_df.iterrows():
        print(format_row.format(
            str(row["brand_handle"]),
            f"{row['total_brand_tweets']:,}",
            f"{row['total_threads']:,}",
            f"{row['avg_thread_length']:.2f}",
            f"{row['pct_resolved']:.1f}%",
        ))
    print("=" * 80 + "\n")

    return cand_df


def filter_brand_threads(
    threads_df: pd.DataFrame,
    brand_handle: str = "AppleSupport",
    output_path: Optional[str] = None,
    stats_path: str = DEFAULT_STATS_PATH,
    config: Optional[ResolutionConfig] = None,
    dataset_max_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Filter conversation threads for a specific brand, evaluate the resolution heuristic,
    and persist data/processed/{brand}_threads.parquet.
    """
    if config is None:
        config = ResolutionConfig()

    norm_brand = brand_handle.strip()
    if not output_path:
        output_path = f"data/processed/{norm_brand}_threads.parquet"

    logger.info(f"Filtering threads for target brand: '{norm_brand}'...")
    t0 = time.time()

    # Global dataset max time
    if dataset_max_time is None and "end_time" in threads_df.columns:
        try:
            dataset_max_time = pd.to_datetime(threads_df["end_time"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce").max()
        except Exception:
            dataset_max_time = pd.to_datetime(threads_df["end_time"], errors="coerce").max()

    # Fast index slice for threads containing this brand
    brand_exploded = threads_df[["brand_handles_involved"]].explode("brand_handles_involved").dropna()
    target_lower = norm_brand.lower()
    match_indices = brand_exploded.index[brand_exploded["brand_handles_involved"].astype(str).str.lower() == target_lower].unique()
    filtered_df = threads_df.iloc[match_indices].copy()
    num_brand_threads = len(filtered_df)

    if num_brand_threads == 0:
        logger.warning(f"No threads found containing brand '{norm_brand}'. Check spelling against candidates report.")
        # We still continue with empty dataframe to preserve schema
        filtered_df["resolved"] = pd.Series(dtype=str)
        filtered_df["resolution_reason"] = pd.Series(dtype=str)
        filtered_df.to_parquet(output_path, index=False, engine="pyarrow")
        return filtered_df

    logger.info(f"Found {num_brand_threads:,} threads containing brand '{norm_brand}'. Tagging resolution heuristic...")

    # Fast evaluate resolution heuristic for each thread
    resolved_tags = []
    resolution_reasons = []

    f_inbounds = filtered_df["inbounds"].values
    f_authors = filtered_df["author_ids"].values
    f_texts = filtered_df["texts"].values
    f_end_times = filtered_df["end_time"].values
    has_f_created_ats = "created_ats" in filtered_df.columns
    f_created_ats = filtered_df["created_ats"].values if has_f_created_ats else None

    for idx, (inbounds, authors, texts, end_tm) in enumerate(zip(f_inbounds, f_authors, f_texts, f_end_times)):
        if has_f_created_ats:
            cats = f_created_ats[idx]
            tw_list = [
                {"inbound": inb, "author_id": auth, "text": txt, "created_at": cat}
                for inb, auth, txt, cat in zip(inbounds, authors, texts, cats)
            ]
        else:
            tw_list = [
                {"inbound": inb, "author_id": auth, "text": txt, "created_at": None}
                for inb, auth, txt in zip(inbounds, authors, texts)
            ]
            if tw_list:
                tw_list[-1]["created_at"] = end_tm

        status, reason = evaluate_thread_resolution(
            tweets=tw_list,
            brand_handle=norm_brand,
            dataset_max_time=dataset_max_time,
            config=config,
        )
        resolved_tags.append(status)
        resolution_reasons.append(reason)

    filtered_df["resolved"] = resolved_tags
    filtered_df["resolution_reason"] = resolution_reasons

    # Write output artifact
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {num_brand_threads:,} tagged threads to {output_path}...")
    filtered_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Stage 2 output artifact successfully saved: {output_path}")

    # Compute brand-level statistics
    total_tweets = int(filtered_df["thread_length"].sum())
    resolved_count = int((filtered_df["resolved"] == "true").sum())
    unresolved_count = int((filtered_df["resolved"] == "false").sum())
    unknown_count = int((filtered_df["resolved"] == "unknown").sum())
    pct_resolved = round(100.0 * resolved_count / num_brand_threads, 2) if num_brand_threads > 0 else 0.0

    reason_distribution = filtered_df["resolution_reason"].value_counts().to_dict()

    stats = {
        "brand_handle": norm_brand,
        "output_path": str(output_path),
        "total_threads_reconstructed_input": len(threads_df),
        "brand_threads_count": num_brand_threads,
        "brand_total_tweets_in_threads": total_tweets,
        "avg_thread_length": round(float(filtered_df["thread_length"].mean()), 2),
        "resolved_threads_count": resolved_count,
        "resolved_pct": pct_resolved,
        "unresolved_threads_count": unresolved_count,
        "unknown_threads_count": unknown_count,
        "resolution_reason_breakdown": reason_distribution,
        "inactivity_hours_threshold": config.inactivity_hours,
        "duration_seconds": round(time.time() - t0, 2),
    }

    print("=" * 60)
    print(f"STAGE 2: BRAND FILTER REPORT FOR '{norm_brand}'")
    print("=" * 60)
    print(f"Total Threads Found:         {num_brand_threads:,}")
    print(f"Total Tweets Across Threads: {total_tweets:,}")
    print(f"Average Conversation Length: {stats['avg_thread_length']} tweets")
    print(f"Resolved Threads:            {resolved_count:,} ({pct_resolved}%)")
    print(f"Unresolved Threads:          {unresolved_count:,} ({round(100*unresolved_count/num_brand_threads, 1)}%)")
    print(f"Unknown / Indeterminate:     {unknown_count:,} ({round(100*unknown_count/num_brand_threads, 1)}%)")
    print("\nResolution Reason Distribution:")
    for reason, count in reason_distribution.items():
        print(f"  - {reason:40s}: {count:,} ({count / num_brand_threads * 100:.1f}%)")
    print(f"\nArtifact Saved: {output_path}")
    print("=" * 60)

    # Persist stats
    update_pipeline_stats(stats_path, f"stage2_brand_filter_{norm_brand}", stats)

    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Filter conversation threads by brand and evaluate resolution.")
    parser.add_argument("--brand", default="AppleSupport", help="Brand handle to filter (e.g. AppleSupport)")
    parser.add_argument("--threads", default=DEFAULT_THREADS_OUTPUT, help="Path to threads.parquet from Stage 1")
    parser.add_argument("--output", default=None, help="Path to output {brand}_threads.parquet")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES_PATH, help="Path to brand_candidates.csv")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH, help="Path to pipeline_stats.json")
    parser.add_argument("--top_n", type=int, default=20, help="Number of candidate brands to analyze")
    parser.add_argument("--inactivity_hours", type=float, default=24.0, help="Inactivity hours threshold for closure heuristic")
    args = parser.parse_args()

    if not os.path.exists(args.threads):
        logger.error(f"Threads file not found at {args.threads}. Please run Stage 1 first.")
        sys.exit(1)

    logger.info(f"Loading Stage 1 threads from {args.threads}...")
    threads = pd.read_parquet(args.threads)

    cfg = ResolutionConfig(inactivity_hours=args.inactivity_hours)

    # 1. Generate candidate brands comparison
    generate_brand_candidate_report(
        threads_df=threads,
        top_n=args.top_n,
        output_csv=args.candidates,
        config=cfg,
    )

    # 2. Filter target brand and output artifact
    filter_brand_threads(
        threads_df=threads,
        brand_handle=args.brand,
        output_path=args.output,
        stats_path=args.stats,
        config=cfg,
    )
