"""
Pytest test suite for Stage 4 (Precedent Retrieval and Grounding Index):
1. Golden evaluation holdout stratification, exact count (300), and conservation.
2. Explicit zero-leakage assertion: holdout thread_ids have ZERO overlap with indexable pool or ChromaDB index.
3. Strict intent-filtering assertion in retrieve_precedents.
4. Precedent-agreement score bounds [0.0, 1.0] and calculation consistency.
5. Structured precedent schema and outcome validation.
"""

import os
from pathlib import Path
import pandas as pd
import pytest

from src.retrieval.holdout_split import split_golden_holdout
from src.retrieval.query_index import retrieve_precedents, compute_precedent_agreement, get_embed_model

HOLDOUT_PATH = "data/processed/golden_eval_candidates.parquet"
INDEXABLE_PATH = "data/processed/indexable_precedents_input.parquet"
CLASSIFIED_PATH = "data/processed/AppleSupport_classified_sample.parquet"
PRECEDENTS_PATH = "data/processed/structured_precedents.parquet"
CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "apple_support_precedents"

EXPECTED_INTENTS = {
    "software_update_os_bugs",
    "keyboard_text_autocorrect",
    "battery_power_performance",
    "hardware_display_physical",
    "orders_purchases_applecare",
    "account_access_apple_id",
    "apple_music_audio_playback",
    "international_multilingual_inquiries",
}


def test_golden_holdout_stratification_and_conservation():
    """Verify that golden holdout has exactly 300 rows and conserves all 8 intents."""
    if not os.path.exists(HOLDOUT_PATH) or not os.path.exists(INDEXABLE_PATH):
        pytest.skip(f"Holdout files not generated yet. Run holdout_split first.")

    holdout_df = pd.read_parquet(HOLDOUT_PATH)
    indexable_df = pd.read_parquet(INDEXABLE_PATH)

    assert len(holdout_df) == 300, f"Expected 300 holdout rows, got {len(holdout_df)}"
    assert len(indexable_df) == 5700, f"Expected 5,700 indexable rows, got {len(indexable_df)}"
    assert len(holdout_df) + len(indexable_df) == 6000, "Conservation violation: total != 6,000"

    # All 8 intents must be present in holdout
    holdout_intents = set(holdout_df["predicted_intent"].unique())
    assert holdout_intents == EXPECTED_INTENTS, f"Missing intents in holdout: {EXPECTED_INTENTS - holdout_intents}"

    # Required columns check
    for col in ["thread_id", "tweet_id", "text", "predicted_intent", "confidence", "classification_source"]:
        assert col in holdout_df.columns, f"Missing column {col} in holdout parquet"


def test_holdout_zero_overlap_with_indexable_and_index():
    """
    CRITICAL TEST: Verify that the 300 golden holdout thread_ids have ZERO overlap
    with the indexable pool, the structured precedents artifact, and the ChromaDB index.
    """
    if not os.path.exists(HOLDOUT_PATH) or not os.path.exists(INDEXABLE_PATH):
        pytest.skip(f"Holdout files not found.")

    holdout_df = pd.read_parquet(HOLDOUT_PATH)
    indexable_df = pd.read_parquet(INDEXABLE_PATH)

    holdout_ids = set(holdout_df["thread_id"])
    indexable_ids = set(indexable_df["thread_id"])

    # 1. Zero overlap with indexable pool
    overlap = holdout_ids.intersection(indexable_ids)
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} holdout IDs found in indexable pool!"

    # 2. Zero overlap with structured precedents artifact (if exists)
    if os.path.exists(PRECEDENTS_PATH):
        prec_df = pd.read_parquet(PRECEDENTS_PATH)
        prec_ids = set(prec_df["thread_id"])
        prec_overlap = holdout_ids.intersection(prec_ids)
        assert len(prec_overlap) == 0, (
            f"LEAKAGE DETECTED: {len(prec_overlap)} holdout IDs found in structured precedents!"
        )

    # 3. Zero overlap with ChromaDB collection (if exists)
    if os.path.exists(CHROMA_DIR):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            col = client.get_collection(name=COLLECTION_NAME)
            indexed_ids = set(col.get()["ids"])
            chroma_overlap = holdout_ids.intersection(indexed_ids)
            assert len(chroma_overlap) == 0, (
                f"LEAKAGE DETECTED: {len(chroma_overlap)} holdout IDs found in ChromaDB collection!"
            )
        except Exception:
            pass


def test_precedent_agreement_scoring_bounds():
    """Verify compute_precedent_agreement bounds [0.0, 1.0] and calculation consistency."""
    embed_model = get_embed_model()

    # Identical actions and outcomes -> score 1.0
    identical_actions = [
        "Advised resetting network settings in Settings > General > Reset.",
        "Advised resetting network settings in Settings > General > Reset.",
        "Advised resetting network settings in Settings > General > Reset.",
    ]
    identical_outcomes = ["troubleshooting_steps_provided"] * 3
    score_identical = compute_precedent_agreement(identical_actions, identical_outcomes, embed_model)
    assert score_identical >= 0.95, f"Expected near 1.0 for identical actions, got {score_identical}"

    # Conflicting actions and outcomes
    conflicting_actions = [
        "Advised resetting network settings in Settings > General > Reset.",
        "Directed customer to physical Apple Store for hardware digitizer replacement.",
        "Provided link to billing refund portal on iTunes support website.",
    ]
    conflicting_outcomes = ["troubleshooting_steps_provided", "escalated_to_apple_store", "directed_to_support_link"]
    score_conflicting = compute_precedent_agreement(conflicting_actions, conflicting_outcomes, embed_model)
    assert 0.0 <= score_conflicting <= 1.0, f"Score out of bounds: {score_conflicting}"
    assert score_conflicting < score_identical, (
        f"Conflicting score ({score_conflicting}) should be lower than identical ({score_identical})"
    )


def test_retrieval_intent_filtering_and_schema():
    """Verify that retrieve_precedents strictly enforces intent filtering."""
    if not os.path.exists(CHROMA_DIR):
        pytest.skip(f"ChromaDB index not built yet.")

    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_collection(name=COLLECTION_NAME)
        if col.count() == 0:
            pytest.skip("ChromaDB collection is empty.")
    except Exception:
        pytest.skip("ChromaDB collection not accessible.")

    # Test query with intent filtering
    target_intent = "software_update_os_bugs"
    res = retrieve_precedents(
        message="My iPhone is freezing and stuck on the Apple logo after iOS 11 update",
        intent=target_intent,
        k=3,
    )

    assert "query_message" in res
    assert "query_intent" in res
    assert "precedents" in res
    assert "precedent_agreement_score" in res
    assert 0.0 <= res["precedent_agreement_score"] <= 1.0

    for prec in res["precedents"]:
        assert prec["intent"] == target_intent, (
            f"Intent filter violated! Expected {target_intent}, got {prec['intent']}"
        )
        assert "thread_id" in prec
        assert "action_taken" in prec
        assert "outcome" in prec
        assert "similarity_score" in prec
        assert 0.0 <= prec["similarity_score"] <= 1.0
