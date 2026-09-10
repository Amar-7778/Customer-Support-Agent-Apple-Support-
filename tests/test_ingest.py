"""
Pytest test suite for Stages 1 and 2 of the customer support pipeline:
1. Schema validation (loud failure on missing columns)
2. Thread reconstruction row conservation check (zero tweets dropped)
3. Branching and broken links handling
4. Heuristic determinism and rule validation
5. Brand filtering produces non-empty output for real brand in dataset
"""

import os
from datetime import datetime, timezone
import pandas as pd
import pytest

from src.ingest.heuristics import ResolutionConfig, evaluate_thread_resolution
from src.ingest.load_raw import SchemaValidationError, validate_schema
from src.ingest.reconstruct_threads import reconstruct_conversation_threads
from src.ingest.filter_brand import filter_brand_threads


def test_schema_validation_success():
    """Verify that a valid DataFrame passes schema validation without errors."""
    valid_df = pd.DataFrame({
        "tweet_id": [1, 2],
        "author_id": ["user1", "AppleSupport"],
        "inbound": [True, False],
        "created_at": ["2017-10-31 22:00:00", "2017-10-31 22:05:00"],
        "text": ["Need help with iPhone", "We are here to help!"],
        "response_tweet_id": ["2", None],
        "in_response_to_tweet_id": [None, 1],
    })
    validate_schema(valid_df)  # Should not raise


def test_schema_validation_missing_column():
    """Verify that a DataFrame missing a required column fails loudly with SchemaValidationError."""
    invalid_df = pd.DataFrame({
        "tweet_id": [1],
        "author_id": ["user1"],
        # "inbound" is missing!
        "created_at": ["2017-10-31 22:00:00"],
        "text": ["Help"],
        "response_tweet_id": [None],
        "in_response_to_tweet_id": [None],
    })
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_schema(invalid_df)
    assert "inbound" in str(exc_info.value)


def test_thread_reconstruction_conservation_and_branching(tmp_path):
    """
    Test that thread reconstruction:
    1. Never drops tweets (exact conservation check).
    2. Correctly groups branched conversations into the same thread.
    3. Handles broken links (pointing to non-existent tweet ID).
    4. Handles single-sided isolated tweets.
    """
    sample_data = pd.DataFrame({
        "tweet_id": [101, 102, 103, 104, 105, 106, 107],
        "author_id": ["cust1", "AppleSupport", "AppleSupport", "cust1", "cust2", "UberSupport", "cust3"],
        "inbound": [True, False, False, True, True, False, True],
        "created_at": [
            "2017-10-31 10:00:00",  # 101 (root)
            "2017-10-31 10:05:00",  # 102 (reply to 101 - branch A)
            "2017-10-31 10:06:00",  # 103 (reply to 101 - branch B)
            "2017-10-31 10:10:00",  # 104 (reply to 102)
            "2017-10-31 11:00:00",  # 105 (broken link: in_response_to 9999 which is not in dataset)
            "2017-10-31 11:05:00",  # 106 (reply to 105)
            "2017-10-31 12:00:00",  # 107 (isolated tweet, single-sided, no replies, no parent)
        ],
        "text": [
            "Help with iPad",
            "What iOS version? (branch A)",
            "Try restarting? (branch B)",
            "Thanks, restarting worked!",
            "Can I get a ride?",
            "We are on it!",
            "Just tweeting something general",
        ],
        "response_tweet_id": ["102,103", "104", None, None, "106", None, None],
        "in_response_to_tweet_id": [None, 101, 101, 102, 9999, 105, None],
    })

    out_file = str(tmp_path / "test_threads.parquet")
    stats_file = str(tmp_path / "test_stats.json")

    threads_df = reconstruct_conversation_threads(
        df=sample_data,
        output_path=out_file,
        stats_path=stats_file,
    )

    # 1. Exact Row Conservation: All 7 tweets must be present in threads
    total_tweets_in_threads = threads_df["thread_length"].sum()
    assert total_tweets_in_threads == len(sample_data), "Row conservation failed: tweets were lost or duplicated!"

    # Flatten all tweet_ids across threads and ensure it matches original tweet_ids exactly
    all_reconstructed_tids = [tid for tids in threads_df["tweet_ids"] for tid in tids]
    assert set(all_reconstructed_tids) == set(sample_data["tweet_id"]), "Tweet IDs set mismatch!"

    # 2. Branching conversation: 101, 102, 103, 104 must be in the same thread
    branch_thread = threads_df[threads_df["tweet_ids"].apply(lambda tids: 101 in tids)].iloc[0]
    assert set(branch_thread["tweet_ids"]) == {101, 102, 103, 104}
    assert branch_thread["thread_length"] == 4
    assert branch_thread["has_branching"] == True

    # 3. Broken link: 105 referenced missing 9999, but 105 and 106 must form their own thread
    broken_link_thread = threads_df[threads_df["tweet_ids"].apply(lambda tids: 105 in tids)].iloc[0]
    assert set(broken_link_thread["tweet_ids"]) == {105, 106}
    assert broken_link_thread["thread_length"] == 2

    # 4. Isolated tweet: 107 must form its own thread of length 1
    isolated_thread = threads_df[threads_df["tweet_ids"].apply(lambda tids: 107 in tids)].iloc[0]
    assert isolated_thread["tweet_ids"] == [107]
    assert isolated_thread["thread_length"] == 1


def test_resolved_heuristic_determinism():
    """
    Test that the resolved-thread heuristic runs deterministically:
    1. Produces identical output on repeated runs.
    2. Correctly identifies customer gratitude closure.
    3. Correctly identifies brand final reply dormancy.
    4. Correctly identifies pending brand reply.
    """
    config = ResolutionConfig(inactivity_hours=24.0)

    # Case A: Customer thanks brand -> resolved = true, reason = customer_gratitude_closure
    thread_gratitude = [
        {"inbound": True, "author_id": "cust1", "created_at": "2017-10-01 10:00:00", "text": "My phone broke"},
        {"inbound": False, "author_id": "AppleSupport", "created_at": "2017-10-01 10:05:00", "text": "Try resetting"},
        {"inbound": True, "author_id": "cust1", "created_at": "2017-10-01 10:10:00", "text": "Thank you, that resolved it!"},
    ]
    res1, reason1 = evaluate_thread_resolution(thread_gratitude, brand_handle="AppleSupport", config=config)
    res2, reason2 = evaluate_thread_resolution(thread_gratitude, brand_handle="AppleSupport", config=config)
    assert res1 == "true"
    assert res1 == res2, "Heuristic is not deterministic!"
    assert reason1 == reason2
    assert "gratitude" in reason1

    # Case B: Brand replied last, dormant > 24 hours
    thread_dormant = [
        {"inbound": True, "author_id": "cust1", "created_at": "2017-10-01 10:00:00", "text": "Need assistance"},
        {"inbound": False, "author_id": "AppleSupport", "created_at": "2017-10-01 10:15:00", "text": "We sent you a DM to assist."},
    ]
    ds_max = datetime(2017, 10, 10, 0, 0, 0, tzinfo=timezone.utc)
    res_b, reason_b = evaluate_thread_resolution(
        thread_dormant,
        brand_handle="AppleSupport",
        dataset_max_time=ds_max,
        config=config,
    )
    assert res_b == "true"
    assert "dormant" in reason_b

    # Case C: Customer replied last without gratitude -> pending brand reply
    thread_pending = [
        {"inbound": True, "author_id": "cust1", "created_at": "2017-10-01 10:00:00", "text": "Need assistance"},
        {"inbound": False, "author_id": "AppleSupport", "created_at": "2017-10-01 10:15:00", "text": "What model?"},
        {"inbound": True, "author_id": "cust1", "created_at": "2017-10-01 10:20:00", "text": "iPhone 7 Plus, still not working"},
    ]
    res_c, reason_c = evaluate_thread_resolution(
        thread_pending,
        brand_handle="AppleSupport",
        dataset_max_time=ds_max,
        config=config,
    )
    assert res_c == "false"
    assert reason_c == "pending_brand_reply"


def test_brand_filtering_non_empty_output(tmp_path):
    """
    Test brand filtering produces non-empty output and valid resolved tags
    when filtering for a brand present in the reconstructed threads.
    """
    threads_df = pd.DataFrame({
        "thread_id": ["T_1", "T_2", "T_3"],
        "tweet_ids": [[1, 2], [3, 4], [5]],
        "author_ids": [["c1", "AppleSupport"], ["c2", "UberSupport"], ["c3"]],
        "brand_handles_involved": [["AppleSupport"], ["UberSupport"], []],
        "thread_length": [2, 2, 1],
        "start_time": ["2017-10-01 10:00:00", "2017-10-01 11:00:00", "2017-10-01 12:00:00"],
        "end_time": ["2017-10-01 10:05:00", "2017-10-01 11:05:00", "2017-10-01 12:00:00"],
        "has_branching": [False, False, False],
        "inbounds": [[True, False], [True, False], [True]],
        "texts": [["Help", "Fixed!"], ["Ride issue", "On it"], ["Hello"]],
        "created_ats": [
            ["2017-10-01 10:00:00", "2017-10-01 10:05:00"],
            ["2017-10-01 11:00:00", "2017-10-01 11:05:00"],
            ["2017-10-01 12:00:00"],
        ],
        "in_response_to_tweet_ids": [[None, 1], [None, 3], [None]],
    })

    out_parquet = str(tmp_path / "AppleSupport_threads.parquet")
    filtered = filter_brand_threads(
        threads_df=threads_df,
        brand_handle="AppleSupport",
        output_path=out_parquet,
        stats_path=str(tmp_path / "stats.json"),
    )

    assert not filtered.empty, "Brand filtering returned empty output for an existing brand!"
    assert len(filtered) == 1
    assert "resolved" in filtered.columns
    assert "resolution_reason" in filtered.columns
    assert os.path.exists(out_parquet)


def test_real_dataset_output_artifact():
    """
    Validate the real processed output artifact for AppleSupport if present.
    Verifies that the deliverable is derived from actual raw data with valid schemas and tags.
    """
    artifact_path = "data/processed/AppleSupport_threads.parquet"
    if not os.path.exists(artifact_path):
        pytest.skip(f"Artifact {artifact_path} not yet generated; skipping real data integration check.")

    df = pd.read_parquet(artifact_path)
    assert not df.empty, "AppleSupport artifact is empty!"
    assert len(df) > 10000, f"Expected substantial threads count, got {len(df):,}"

    # Required columns
    expected_cols = [
        "thread_id", "tweet_ids", "author_ids", "brand_handles_involved",
        "thread_length", "start_time", "end_time", "resolved", "resolution_reason"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in AppleSupport artifact!"

    # Validate resolved tags
    valid_statuses = {"true", "false", "unknown"}
    actual_statuses = set(df["resolved"].unique())
    assert actual_statuses.issubset(valid_statuses), f"Unexpected resolved statuses: {actual_statuses}"

    # Validate non-empty reasons
    assert df["resolution_reason"].isna().sum() == 0, "Found null resolution_reason!"
    assert (df["resolution_reason"] == "").sum() == 0, "Found empty resolution_reason strings!"

    # Verify that AppleSupport actually appears in every filtered thread
    for brands in df["brand_handles_involved"]:
        assert any("applesupport" in str(b).lower() for b in brands), f"AppleSupport not in {brands}"

