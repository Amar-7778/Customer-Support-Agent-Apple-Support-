"""
Stage 6: Comprehensive Automated Test Suite for Evaluation Harness.

Tests:
1. Baseline scoring code path parity with real agent pipeline.
2. Golden evaluation set zero-overlap verification against retrieval index and training data.
3. Deterministic behavior of classification and escalation metrics given fixed inputs.
4. Cohen's kappa and rubric agreement computation correctness.
5. Calibrated confidence threshold rule in decide_escalation.
"""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.agent.pipeline import (
    ALWAYS_ESCALATE_INTENTS,
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    decide_escalation,
)
from src.eval.baselines import (
    evaluate_baselines_on_golden_set,
    run_simple_baseline,
    run_trivial_baseline,
)
from src.eval.judge import compute_cohens_kappa, compute_judge_human_agreement
from src.eval.metrics import (
    compute_classification_metrics,
    compute_escalation_metrics,
    evaluate_confidence_thresholds,
)


def test_baseline_scoring_code_path_parity():
    """Verifies that baselines emit identical schema as AgentResponse and use identical metrics code."""
    query = "My battery drains in 30 minutes on iPhone 6s"

    # Trivial baseline check
    t_out = run_trivial_baseline(query)
    expected_keys = {
        "customer_message",
        "predicted_intent",
        "intent_confidence",
        "precedent_agreement_score",
        "drafted_reply",
        "grounding_verification",
        "decision",
        "escalation_reason",
    }
    assert expected_keys.issubset(t_out.keys())
    assert t_out["decision"] == "escalate"

    # Simple baseline check
    s_out = run_simple_baseline(query)
    assert expected_keys.issubset(s_out.keys())
    assert s_out["predicted_intent"] == "battery_power_performance"
    assert s_out["decision"] == "auto_handle"

    # Restricted intent check for simple baseline
    phishing_query = "Is this email from Apple about iCloud locked real or phishing?"
    s_phish = run_simple_baseline(phishing_query)
    assert s_phish["predicted_intent"] == "account_access_apple_id"
    assert s_phish["decision"] == "escalate"

    # Parity in metrics computation
    gold_decisions = ["auto_handle", "escalate"]
    pred_decisions = [s_out["decision"], s_phish["decision"]]
    esc_metrics = compute_escalation_metrics(gold_decisions, pred_decisions)
    assert "escalation_precision" in esc_metrics
    assert "escalation_recall" in esc_metrics
    assert "false_auto_handle_rate" in esc_metrics


def test_golden_set_zero_overlap_verification():
    """
    CRITICAL EVALUATION INTEGRITY CHECK:
    Verifies that all 200 golden evaluation threads have exactly zero overlap
    with the 3,000-thread retrieval index and the 5,700-thread indexable precedent corpus.
    """
    golden_path = "data/processed/golden_eval_set.parquet"
    precedents_path = "data/processed/structured_precedents.parquet"
    indexable_path = "data/processed/indexable_precedents_input.parquet"

    assert os.path.exists(golden_path), f"Golden set missing at {golden_path}"
    df_golden = pd.read_parquet(golden_path)
    golden_tids = set(df_golden["thread_id"].dropna().unique())
    assert len(golden_tids) == 200, f"Expected 200 unique golden threads, found {len(golden_tids)}"

    # Check retrieval vector index overlap
    if os.path.exists(precedents_path):
        df_prec = pd.read_parquet(precedents_path)
        prec_tids = set(df_prec["thread_id"].dropna().unique())
        overlap_retrieval = golden_tids.intersection(prec_tids)
        assert len(overlap_retrieval) == 0, f"DATA LEAKAGE DETECTED: {len(overlap_retrieval)} golden threads found in retrieval index!"

    # Check 5,700 indexable pool overlap
    if os.path.exists(indexable_path):
        df_idx = pd.read_parquet(indexable_path)
        idx_tids = set(df_idx["thread_id"].dropna().unique())
        overlap_idx = golden_tids.intersection(idx_tids)
        assert len(overlap_idx) == 0, f"DATA LEAKAGE DETECTED: {len(overlap_idx)} golden threads found in indexable precedent pool!"


def test_metrics_computation_determinism():
    """Verifies that classification and escalation metric computation is completely deterministic."""
    gold_intents = ["battery_power_performance", "keyboard_text_autocorrect", "software_update_os_bugs", "battery_power_performance"]
    pred_intents = ["battery_power_performance", "software_update_os_bugs", "software_update_os_bugs", "battery_power_performance"]

    m1 = compute_classification_metrics(gold_intents, pred_intents)
    m2 = compute_classification_metrics(gold_intents, pred_intents)

    assert m1["overall_accuracy"] == 0.75
    assert m1 == m2

    gold_decisions = ["auto_handle", "escalate", "auto_handle", "escalate"]
    pred_decisions = ["auto_handle", "auto_handle", "escalate", "escalate"]

    e1 = compute_escalation_metrics(gold_decisions, pred_decisions, cost_ratio=5.0)
    e2 = compute_escalation_metrics(gold_decisions, pred_decisions, cost_ratio=5.0)

    assert e1["escalation_accuracy"] == 0.50
    assert e1["true_positives_escalate"] == 1
    assert e1["false_negatives_auto_handle_risk"] == 1
    assert e1["false_positives_escalate"] == 1
    assert e1["true_negatives_auto_handle"] == 1
    assert e1["asymmetric_cost_penalty"] == 1 * 5.0 + 1 * 1.0  # 6.0
    assert e1 == e2


def test_cohens_kappa_and_rubric_agreement():
    """Verifies statistical correctness of Cohen's kappa and rubric agreement."""
    # Perfect agreement
    rater1 = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    rater2 = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    assert compute_cohens_kappa(rater1, rater2) == 1.0

    # Mild disagreement
    rater3 = [5, 4, 3, 2, 1, 5, 4, 3, 2, 2]
    kappa_mild = compute_cohens_kappa(rater1, rater3)
    assert 0.80 <= kappa_mild <= 1.0

    human_dicts = [{"grounded_in_precedent": 5, "factually_non_hallucinatory": 5, "tone_appropriate": 5, "resolves_or_correctly_defers": 5}]
    judge_dicts = [{"grounded_in_precedent": 4, "factually_non_hallucinatory": 2, "tone_appropriate": 5, "resolves_or_correctly_defers": 5}]

    res = compute_judge_human_agreement(human_dicts, judge_dicts)
    assert res["tone_appropriate"]["exact_agreement_pct"] == 100.0
    assert res["factually_non_hallucinatory"]["exact_agreement_pct"] == 0.0
    assert len(res["itemized_disagreements"]) == 1


def test_confidence_threshold_in_decide_escalation():
    """Verifies that decide_escalation enforces calibrated confidence threshold."""
    # Intent with high agreement and valid grounding, but LOW classifier confidence (< 0.60)
    decision, reason = decide_escalation(
        intent="battery_power_performance",
        precedent_agreement_score=0.85,
        grounding_verification={"grounded": True, "unsupported_claims": []},
        intent_confidence=0.52,
        confidence_threshold=0.60,
    )
    assert decision == "escalate"
    assert "low classifier confidence" in reason

    # Intent with high confidence (>= 0.60)
    decision_ok, reason_ok = decide_escalation(
        intent="battery_power_performance",
        precedent_agreement_score=0.85,
        grounding_verification={"grounded": True, "unsupported_claims": []},
        intent_confidence=0.92,
        confidence_threshold=0.60,
    )
    assert decision_ok == "auto_handle"
    assert "high precedent agreement" in reason_ok

    # Policy restricted intent escalates regardless of confidence
    decision_policy, _ = decide_escalation(
        intent="orders_purchases_applecare",
        precedent_agreement_score=1.0,
        grounding_verification={"grounded": True, "unsupported_claims": []},
        intent_confidence=0.99,
        confidence_threshold=0.60,
    )
    assert decision_policy == "escalate"
