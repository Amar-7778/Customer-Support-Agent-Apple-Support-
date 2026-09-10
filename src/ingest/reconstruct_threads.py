"""
Thread reconstruction module for Customer Support on Twitter dataset.

Constructs full conversation threads by walking in_response_to_tweet_id and response_tweet_id
links as an undirected reply graph and computing connected components using SciPy.

Handles:
- Broken links: References to missing tweet IDs are omitted, making the earliest observed tweet the thread root.
- Branching: Multiple responses to the same tweet are kept in the conversation tree.
- Single-sided threads: Isolated tweets without replies or parents form valid 1-message threads.
- Exact conservation: Zero tweets are dropped; sum(thread_lengths) == total raw tweets.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .load_raw import DEFAULT_RAW_PATH, DEFAULT_STATS_PATH, load_and_validate_raw, update_pipeline_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("reconstruct_threads")

DEFAULT_THREADS_OUTPUT = "data/processed/threads.parquet"


def build_reply_graph(
    df: pd.DataFrame,
) -> Tuple[csr_matrix, Dict[Any, int], np.ndarray]:
    """
    Build a sparse adjacency matrix of the tweet reply graph.
    Only creates edges between tweets that BOTH exist in the dataset.
    Missing parents/children (broken links) are cleanly ignored.
    """
    n = len(df)
    logger.info(f"Building reply graph for {n:,} tweets...")
    t0 = time.time()

    # Map tweet_id -> index in dataframe
    tweet_id_series = df["tweet_id"].values
    tweet_to_idx = {tid: i for i, tid in enumerate(tweet_id_series)}

    # Collect edges from in_response_to_tweet_id
    parent_series = df["in_response_to_tweet_id"].values
    src_list: List[int] = []
    dst_list: List[int] = []

    # Vectorized / fast array scan for parent edges
    for child_idx, parent_id in enumerate(parent_series):
        if pd.notna(parent_id):
            # Normalise parent_id to int or str depending on tweet_id type
            try:
                norm_pid = int(float(parent_id))
            except (ValueError, TypeError):
                norm_pid = str(parent_id)

            parent_idx = tweet_to_idx.get(norm_pid)
            if parent_idx is None:
                # Try str lookup if int failed
                parent_idx = tweet_to_idx.get(str(norm_pid))

            if parent_idx is not None:
                src_list.append(child_idx)
                dst_list.append(parent_idx)

    # Collect any additional edges from response_tweet_id (comma-separated children)
    if "response_tweet_id" in df.columns:
        resp_series = df["response_tweet_id"].values
        for parent_idx, resp_val in enumerate(resp_series):
            if pd.notna(resp_val):
                resp_str = str(resp_val).strip()
                if resp_str:
                    for child_str in resp_str.split(","):
                        c_str = child_str.strip()
                        if not c_str:
                            continue
                        try:
                            norm_cid = int(float(c_str))
                        except (ValueError, TypeError):
                            norm_cid = c_str

                        child_idx = tweet_to_idx.get(norm_cid)
                        if child_idx is None:
                            child_idx = tweet_to_idx.get(str(norm_cid))

                        if child_idx is not None:
                            src_list.append(parent_idx)
                            dst_list.append(child_idx)

    num_directed_edges = len(src_list)
    logger.info(f"Extracted {num_directed_edges:,} valid reply edges in {time.time() - t0:.2f}s.")

    # Symmetrize edges for undirected connected components
    u_arr = np.concatenate([src_list, dst_list])
    v_arr = np.concatenate([dst_list, src_list])
    data = np.ones(len(u_arr), dtype=np.int8)

    graph = csr_matrix((data, (u_arr, v_arr)), shape=(n, n))
    return graph, tweet_to_idx, tweet_id_series


def reconstruct_conversation_threads(
    df: pd.DataFrame,
    output_path: str = DEFAULT_THREADS_OUTPUT,
    stats_path: str = DEFAULT_STATS_PATH,
) -> pd.DataFrame:
    """
    Reconstruct full conversation threads from raw tweets dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Validated raw tweets dataframe.
    output_path : str
        Parquet destination path.
    stats_path : str
        JSON path to persist pipeline statistics.

    Returns
    -------
    pd.DataFrame
        Reconstructed threads table.
    """
    start_time = time.time()
    total_raw_rows = len(df)
    logger.info(f"Starting thread reconstruction for {total_raw_rows:,} tweets...")

    # Ensure created_at is converted to datetime for accurate chronological ordering
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        logger.info("Converting created_at to datetime...")
        try:
            df["created_at_dt"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce")
        except Exception:
            df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df = df.copy()
        df["created_at_dt"] = df["created_at"]

    # Build graph and run connected components
    graph, tweet_to_idx, _ = build_reply_graph(df)
    logger.info("Computing connected components with SciPy csgraph...")
    t_cc = time.time()
    n_components, labels = connected_components(graph, directed=False)
    logger.info(f"Found {n_components:,} connected components (threads) in {time.time() - t_cc:.2f}s.")

    df["component_id"] = labels

    # Check for branching: count number of incoming replies per tweet
    logger.info("Detecting branching in conversations...")
    valid_parents = df["in_response_to_tweet_id"].dropna()
    try:
        valid_parents = valid_parents.astype(float).astype(np.int64)
    except Exception:
        pass
    parent_counts = valid_parents.value_counts()
    branching_parent_ids = set(parent_counts[parent_counts > 1].index)
    df["is_branching_child"] = df["in_response_to_tweet_id"].isin(branching_parent_ids)

    # Sort entire dataframe by component_id and created_at_dt, then tweet_id
    logger.info("Sorting tweets chronologically within components...")
    t_sort = time.time()
    df.sort_values(by=["component_id", "created_at_dt", "tweet_id"], ascending=[True, True, True], inplace=True)
    logger.info(f"Sorted in {time.time() - t_sort:.2f}s.")

    # High performance aggregation using Polars if installed, else contiguous group slices
    logger.info("Aggregating components into structured threads table...")
    t_agg = time.time()

    # Fast group aggregation
    try:
        import polars as pl
        logger.info("Using Polars for accelerated thread aggregation...")
        pldf = pl.from_pandas(df[[
            "component_id", "tweet_id", "author_id", "inbound",
            "created_at", "text", "in_response_to_tweet_id", "is_branching_child"
        ]])

        # Aggregate per component
        agg_exprs = [
            pl.col("tweet_id").alias("tweet_ids"),
            pl.col("author_id").alias("author_ids"),
            pl.col("inbound").alias("inbounds"),
            pl.col("created_at").alias("created_ats"),
            pl.col("text").alias("texts"),
            pl.col("in_response_to_tweet_id").alias("in_response_to_tweet_ids"),
            pl.col("is_branching_child").any().alias("has_branching"),
            pl.len().alias("thread_length"),
            pl.col("created_at").first().alias("start_time"),
            pl.col("created_at").last().alias("end_time"),
        ]
        pl_threads = pldf.group_by("component_id", maintain_order=True).agg(agg_exprs)
        threads_df = pl_threads.to_pandas()

    except Exception as e:
        logger.info(f"Falling back to pandas aggregation ({e})...")
        grouped = df.groupby("component_id", sort=False)
        threads_df = grouped.agg(
            tweet_ids=("tweet_id", list),
            author_ids=("author_id", list),
            inbounds=("inbound", list),
            created_ats=("created_at", list),
            texts=("text", list),
            in_response_to_tweet_ids=("in_response_to_tweet_id", list),
            has_branching=("is_branching_child", "any"),
            thread_length=("tweet_id", "count"),
            start_time=("created_at", "first"),
            end_time=("created_at", "last"),
        ).reset_index()

    logger.info(f"Aggregation complete in {time.time() - t_agg:.2f}s.")

    # Generate deterministic thread_id: "T_{first_tweet_id}"
    threads_df["thread_id"] = threads_df["tweet_ids"].apply(lambda tids: f"T_{tids[0]}")

    # Compute brand_handles_involved (outbound author_ids in each thread)
    def extract_brands(authors: List[Any], inbounds: List[bool]) -> List[str]:
        brands = set()
        for auth, inb in zip(authors, inbounds):
            if not inb and pd.notna(auth):
                brands.add(str(auth))
        return sorted(list(brands))

    threads_df["brand_handles_involved"] = [
        extract_brands(auths, inbs)
        for auths, inbs in zip(threads_df["author_ids"], threads_df["inbounds"])
    ]

    # Reorder columns as requested:
    # thread_id, tweet_ids, author_ids, brand_handles_involved, thread_length, start_time, end_time, ...
    cols_order = [
        "thread_id",
        "tweet_ids",
        "author_ids",
        "brand_handles_involved",
        "thread_length",
        "start_time",
        "end_time",
        "has_branching",
        "inbounds",
        "texts",
        "in_response_to_tweet_ids",
    ]
    threads_df = threads_df[cols_order]

    # Conservation check
    total_tweets_conserved = int(threads_df["thread_length"].sum())
    conservation_passed = (total_tweets_conserved == total_raw_rows)
    logger.info(f"CONSERVATION CHECK: Total tweets in threads = {total_tweets_conserved:,} / Raw input = {total_raw_rows:,}. Passed: {conservation_passed}")
    if not conservation_passed:
        raise RuntimeError(
            f"Row count conservation failed! Ingested {total_raw_rows} tweets, but threads contain {total_tweets_conserved} tweets."
        )

    # Save to Parquet
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing threads table to {output_path}...")
    threads_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(threads_df):,} threads to {output_path}.")

    # Gather statistics
    single_message_threads = int((threads_df["thread_length"] == 1).sum())
    multi_message_threads = int((threads_df["thread_length"] > 1).sum())
    branching_threads = int(threads_df["has_branching"].sum())
    threads_with_brand = int(threads_df["brand_handles_involved"].apply(lambda b: len(b) > 0).sum())

    total_duration = time.time() - start_time
    stats = {
        "output_path": str(output_path),
        "total_raw_rows_input": total_raw_rows,
        "total_threads": len(threads_df),
        "total_tweets_conserved": total_tweets_conserved,
        "conservation_check_passed": conservation_passed,
        "single_message_threads": single_message_threads,
        "multi_message_threads": multi_message_threads,
        "branching_threads": branching_threads,
        "threads_with_brand_involved": threads_with_brand,
        "min_thread_length": int(threads_df["thread_length"].min()),
        "max_thread_length": int(threads_df["thread_length"].max()),
        "avg_thread_length": round(float(threads_df["thread_length"].mean()), 2),
        "reconstruction_duration_seconds": round(total_duration, 2),
    }

    print("=" * 60)
    print("STAGE 1: THREAD RECONSTRUCTION REPORT")
    print("=" * 60)
    print(f"Total Threads Reconstructed: {len(threads_df):,}")
    print(f"Total Tweets Conserved:      {total_tweets_conserved:,} (100.0%)")
    print(f"Single-Tweet Threads:        {single_message_threads:,} ({single_message_threads / len(threads_df) * 100:.1f}%)")
    print(f"Multi-Tweet Conversations:   {multi_message_threads:,} ({multi_message_threads / len(threads_df) * 100:.1f}%)")
    print(f"Threads with Branching:      {branching_threads:,} ({branching_threads / len(threads_df) * 100:.1f}%)")
    print(f"Threads with Brand Reply:    {threads_with_brand:,} ({threads_with_brand / len(threads_df) * 100:.1f}%)")
    print(f"Avg Conversation Length:     {stats['avg_thread_length']} tweets")
    print(f"Max Conversation Length:     {stats['max_thread_length']} tweets")
    print(f"Elapsed Time:                {total_duration:.2f}s")
    print("=" * 60)

    # Persist stats
    update_pipeline_stats(stats_path, "stage1_thread_reconstruction", stats)

    return threads_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Reconstruct conversation threads from twcs.csv.")
    parser.add_argument("--input", default=DEFAULT_RAW_PATH, help="Path to twcs.csv")
    parser.add_argument("--output", default=DEFAULT_THREADS_OUTPUT, help="Path to output threads.parquet")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH, help="Path to pipeline_stats.json")
    args = parser.parse_args()

    raw_df, _ = load_and_validate_raw(csv_path=args.input, stats_path=args.stats)
    reconstruct_conversation_threads(raw_df, output_path=args.output, stats_path=args.stats)
