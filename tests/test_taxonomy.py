"""
Pytest test suite for Stage 3 (Intent Taxonomy Derivation and Classification):
1. Determinism of stratified sampling given a fixed seed.
2. Determinism of K-Means clustering and silhouette scoring.
3. Schema and business rule validation of taxonomy.yaml (6-10 intents, escalation defaults, 3-5 examples).
4. 100% classification coverage on real customer messages (no nulls/unhandled classifications).
5. Exact conservation of class distribution count (sum of class counts == total input customer messages).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.taxonomy.sample_for_clustering import (
    clean_tweet_text,
    assign_stratification_bins,
    draw_stratified_sample,
)
from src.taxonomy.cluster_messages import evaluate_k_sweep
from src.taxonomy.review_taxonomy import load_draft_taxonomy, consolidate_draft_taxonomy


def test_clean_tweet_text():
    """Verify handle stripping and text normalization while preserving grievance."""
    raw1 = "@AppleSupport @115858 My iPhone 7 battery is draining fast!"
    cleaned1 = clean_tweet_text(raw1)
    assert "battery is draining fast!" in cleaned1
    assert "@AppleSupport" not in cleaned1

    raw2 = "Why is this happening? @AppleSupport"
    cleaned2 = clean_tweet_text(raw2)
    assert "Why is this happening?" in cleaned2
    assert "@AppleSupport" not in cleaned2


def test_stratified_sampling_determinism():
    """Verify that stratified sampling is 100% deterministic given the random seed."""
    sample_path = "data/processed/taxonomy_sample.parquet"
    if not os.path.exists(sample_path):
        pytest.skip(f"{sample_path} not found. Run sample_for_clustering first.")

    df = pd.read_parquet(sample_path)
    assert len(df) == 2000, f"Expected 2000 sampled rows, got {len(df)}"

    required_cols = [
        "thread_id", "tweet_id", "text", "cleaned_text",
        "thread_length", "start_time", "time_period", "thread_length_bin"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in sample parquet"

    # Test that draw_stratified_sample yields identical results twice
    sample1 = draw_stratified_sample(df, sample_size=100, seed=42)
    sample2 = draw_stratified_sample(df, sample_size=100, seed=42)

    assert sample1["tweet_id"].tolist() == sample2["tweet_id"].tolist(), (
        "Sampling failed determinism check: different rows selected!"
    )


def test_clustering_determinism_and_silhouette():
    """Verify clustering determinism and silhouette score computation on synthetic embeddings."""
    np.random.seed(42)
    # 50 points in 2D space, 3 clusters
    c1 = np.random.randn(20, 4) + 5
    c2 = np.random.randn(20, 4) - 5
    c3 = np.random.randn(10, 4)
    data = np.vstack([c1, c2, c3]).astype(np.float32)

    scores1, best_k1, models1 = evaluate_k_sweep(data, k_min=2, k_max=4, seed=42)
    scores2, best_k2, models2 = evaluate_k_sweep(data, k_min=2, k_max=4, seed=42)

    assert scores1 == scores2, "Silhouette scores were not deterministic across runs!"
    assert best_k1 == best_k2, "Optimal k varied between identical runs!"
    for k, score in scores1.items():
        assert -1.0 <= score <= 1.0, f"Silhouette score {score} out of valid range [-1, 1]"

    assert list(models1[best_k1].labels_) == list(models2[best_k2].labels_), (
        "Cluster assignments differed between identical runs!"
    )


def test_taxonomy_yaml_structure():
    """
    Validate the finalized taxonomy.yaml:
    - 6 to 10 intents
    - Required fields: intent_name, description, escalation_default, representative_examples
    - 3 to 5 real examples per intent
    - account_access_apple_id and orders_purchases_applecare default to escalation
    """
    taxonomy_path = "taxonomy.yaml"
    if not os.path.exists(taxonomy_path):
        pytest.skip(f"{taxonomy_path} not found. Run review_taxonomy first.")

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)

    intents = tax.get("intents", [])
    assert 6 <= len(intents) <= 10, f"Expected 6-10 intents, got {len(intents)}"

    always_escalate_found = set()

    for item in intents:
        assert "intent_name" in item and len(item["intent_name"]) > 0
        assert "description" in item and len(item["description"]) > 0
        assert "escalation_default" in item and isinstance(item["escalation_default"], bool)
        assert "representative_examples" in item
        examples = item["representative_examples"]
        assert 3 <= len(examples) <= 5, (
            f"Intent {item['intent_name']} must have 3-5 examples, got {len(examples)}"
        )

        for ex in examples:
            assert "tweet_id" in ex and len(str(ex["tweet_id"])) > 0
            assert "text" in ex and len(ex["text"]) > 0

        if item["escalation_default"]:
            always_escalate_found.add(item["intent_name"])

    assert "account_access_apple_id" in always_escalate_found, (
        "account_access_apple_id must default to escalation_default: true"
    )
    assert "orders_purchases_applecare" in always_escalate_found, (
        "orders_purchases_applecare must default to escalation_default: true"
    )


@pytest.mark.slow
def test_full_corpus_classification_coverage_and_conservation():
    """
    Verify classified output artifact (stratified sample):
    - Exactly 6,000 real customer messages classified (rescoped from 74k corpus).
    - 100% of messages receive a valid intent label (no nulls or empty strings).
    - Class distribution sum matches exactly 6,000 messages.
    - Escalation default flag properly aligned with taxonomy.yaml.
    - Provenance flag verifies 100% Groq LLM classification (zero fallback).
    - Confidence scores strictly within [0.0, 1.0].
    """
    sample_path = "data/processed/AppleSupport_classified_sample.parquet"
    if not os.path.exists(sample_path):
        pytest.skip(f"{sample_path} does not exist yet.")

    df = pd.read_parquet(sample_path)
    assert not df.empty, "Classified dataset is empty!"
    assert len(df) == 6000, f"Expected exactly 6,000 classified messages, got {len(df):,}"

    # Required columns
    expected_cols = [
        "thread_id",
        "tweet_id",
        "text",
        "predicted_intent",
        "confidence",
        "classification_source",
        "escalation_default",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in classified dataset"

    # Coverage: zero nulls, zero empty strings
    assert df["predicted_intent"].isna().sum() == 0, "Found null values in predicted_intent!"
    assert (df["predicted_intent"] == "").sum() == 0, "Found empty string in predicted_intent!"

    # Allowed categories matching taxonomy.yaml
    with open("taxonomy.yaml", "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    valid_intents = {item["intent_name"] for item in tax["intents"]}
    actual_intents = set(df["predicted_intent"].unique())
    assert actual_intents.issubset(valid_intents), f"Found unexpected intents: {actual_intents - valid_intents}"

    # Conservation: sum of class distribution == 6,000
    class_counts = df["predicted_intent"].value_counts()
    assert int(class_counts.sum()) == 6000, "Sum of class distribution != 6,000!"

    # Confidence scores within [0.0, 1.0]
    assert (df["confidence"] >= 0.0).all() and (df["confidence"] <= 1.0).all(), (
        "Confidence scores out of bounds [0.0, 1.0]!"
    )

    # Provenance verification: 100% groq_llm
    assert (df["classification_source"] == "groq_llm").all(), "Found non-Groq classification sources!"

