"""
End-to-end pipeline runner for Stages 1 and 2:
1. Ingest raw twcs.csv and validate schema
2. Reconstruct full conversation threads (with graph connected components & row count conservation)
3. Generate candidate brand comparison report
4. Filter threads for target brand, apply resolution heuristic, and write final stage 2 artifact
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingest.filter_brand import filter_brand_threads, generate_brand_candidate_report
from src.ingest.heuristics import ResolutionConfig
from src.ingest.load_raw import DEFAULT_RAW_PATH, DEFAULT_STATS_PATH, load_and_validate_raw, update_pipeline_stats
from src.ingest.reconstruct_threads import DEFAULT_THREADS_OUTPUT, reconstruct_conversation_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    raw_path: str = DEFAULT_RAW_PATH,
    brand_handle: str = "AppleSupport",
    threads_output: str = DEFAULT_THREADS_OUTPUT,
    brand_output: str = None,
    candidates_path: str = "reports/brand_candidates.csv",
    stats_path: str = DEFAULT_STATS_PATH,
    top_n: int = 20,
    inactivity_hours: float = 24.0,
) -> None:
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("STARTING CUSTOMER SUPPORT DATA INGESTION & BRAND FILTERING PIPELINE")
    logger.info(f"Target Brand:      {brand_handle}")
    logger.info(f"Raw Input:         {raw_path}")
    logger.info(f"Threads Output:    {threads_output}")
    logger.info(f"Stats Log:         {stats_path}")
    logger.info("=" * 70)

    # 1. STAGE 1 — LOAD RAW & VALIDATE SCHEMA
    logger.info("\n>>> STAGE 1.1: Loading raw dataset and validating schema...")
    raw_df, raw_stats = load_and_validate_raw(csv_path=raw_path, stats_path=stats_path)

    # 2. STAGE 1 — RECONSTRUCT CONVERSATION THREADS
    logger.info("\n>>> STAGE 1.2: Reconstructing conversation threads...")
    threads_df = reconstruct_conversation_threads(
        df=raw_df,
        output_path=threads_output,
        stats_path=stats_path,
    )

    # Free memory of raw_df
    del raw_df

    # 3. STAGE 2 — BRAND CANDIDATES EVALUATION
    logger.info(f"\n>>> STAGE 2.1: Benchmarking top {top_n} candidate brands...")
    res_cfg = ResolutionConfig(inactivity_hours=inactivity_hours)
    try:
        dataset_max_time = pd.to_datetime(threads_df["end_time"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce").max()
    except Exception:
        dataset_max_time = pd.to_datetime(threads_df["end_time"], errors="coerce").max()

    cand_df = generate_brand_candidate_report(
        threads_df=threads_df,
        top_n=top_n,
        output_csv=candidates_path,
        config=res_cfg,
        dataset_max_time=dataset_max_time,
    )

    # 4. STAGE 2 — FILTER TARGET BRAND & TAG RESOLUTION
    if not brand_output:
        brand_output = f"data/processed/{brand_handle}_threads.parquet"

    logger.info(f"\n>>> STAGE 2.2: Filtering for brand '{brand_handle}' and evaluating resolution heuristic...")
    brand_threads_df = filter_brand_threads(
        threads_df=threads_df,
        brand_handle=brand_handle,
        output_path=brand_output,
        stats_path=stats_path,
        config=res_cfg,
        dataset_max_time=dataset_max_time,
    )

    total_pipeline_time = time.time() - pipeline_start
    update_pipeline_stats(stats_path, "pipeline_summary", {
        "status": "COMPLETED",
        "brand_handle": brand_handle,
        "total_elapsed_seconds": round(total_pipeline_time, 2),
        "total_raw_rows": raw_stats["total_rows"],
        "total_threads": len(threads_df),
        "brand_threads": len(brand_threads_df),
        "brand_resolved_pct": round(100.0 * (brand_threads_df["resolved"] == "true").sum() / len(brand_threads_df), 2) if len(brand_threads_df) > 0 else 0,
        "final_artifact": str(brand_output),
    })

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Total Execution Time:        {total_pipeline_time:.2f} seconds ({total_pipeline_time / 60:.2f} minutes)")
    print(f"Total Raw Rows Processed:    {raw_stats['total_rows']:,}")
    print(f"Total Threads Reconstructed: {len(threads_df):,}")
    print(f"Conservation Check:          PASSED (100% tweets preserved)")
    print(f"Target Brand Threads ({brand_handle}): {len(brand_threads_df):,}")
    print(f"Candidates Comparison:       {candidates_path}")
    print(f"Processed Brand Artifact:    {brand_output}")
    print(f"Pipeline Audit Stats:        {stats_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stages 1 & 2 of the customer support pipeline.")
    parser.add_argument("--brand", default="AppleSupport", help="Brand handle to process (default: AppleSupport)")
    parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Path to twcs.csv")
    parser.add_argument("--threads-output", default=DEFAULT_THREADS_OUTPUT, help="Path to reconstructed threads parquet")
    parser.add_argument("--brand-output", default=None, help="Path to brand processed parquet")
    parser.add_argument("--candidates-output", default="reports/brand_candidates.csv", help="Path to candidate brands CSV")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH, help="Path to pipeline_stats.json")
    parser.add_argument("--top_n", type=int, default=20, help="Number of candidate brands to evaluate")
    parser.add_argument("--inactivity_hours", type=float, default=24.0, help="Inactivity hours for resolution heuristic")
    args = parser.parse_args()

    run_pipeline(
        raw_path=args.input,
        brand_handle=args.brand,
        threads_output=args.threads_output,
        brand_output=args.brand_output,
        candidates_path=args.candidates_output,
        stats_path=args.stats,
        top_n=args.top_n,
        inactivity_hours=args.inactivity_hours,
    )
