"""
Stage 6: Evaluation Metrics Module.

Implements:
1. Per-intent and aggregate classification metrics (Accuracy, Precision, Recall, F1).
2. Escalation decision metrics modeling cost asymmetry (False Auto-Handle vs. False Escalate).
3. Confidence threshold calibration curves for data-backed threshold selection.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_classification_metrics(
    gold_intents: List[str],
    pred_intents: List[str],
) -> Dict[str, Any]:
    """Computes per-intent and aggregate classification performance."""
    labels = sorted(list(set(gold_intents).union(set(pred_intents))))
    n = len(gold_intents)

    per_intent = {}
    f1_list = []
    weights = []

    correct_total = sum(1 for g, p in zip(gold_intents, pred_intents) if g == p)
    overall_accuracy = round(correct_total / n, 4)

    for label in labels:
        tp = sum(1 for g, p in zip(gold_intents, pred_intents) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_intents, pred_intents) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_intents, pred_intents) if g == label and p != label)
        support = sum(1 for g in gold_intents if g == label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_intent[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)
        weights.append(support)

    macro_f1 = round(float(np.mean(f1_list)), 4)
    weighted_f1 = round(float(np.average(f1_list, weights=weights)), 4)

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_eval_samples": n,
        "per_intent": per_intent,
    }


def compute_escalation_metrics(
    gold_decisions: List[str],
    pred_decisions: List[str],
    cost_ratio: float = 5.0,
) -> Dict[str, Any]:
    """
    Computes binary escalation metrics emphasizing safety-critical cost asymmetry:
    - Positive class: 'escalate'
    - Negative class: 'auto_handle'
    - False Auto-Handle (False Negative): Safety hazard (Cost = cost_ratio * 1.0)
    - False Escalate (False Positive): Operational friction (Cost = 1.0)
    """
    n = len(gold_decisions)
    tp = sum(1 for g, p in zip(gold_decisions, pred_decisions) if g == "escalate" and p == "escalate")
    fp = sum(1 for g, p in zip(gold_decisions, pred_decisions) if g == "auto_handle" and p == "escalate")
    fn = sum(1 for g, p in zip(gold_decisions, pred_decisions) if g == "escalate" and p == "auto_handle")
    tn = sum(1 for g, p in zip(gold_decisions, pred_decisions) if g == "auto_handle" and p == "auto_handle")

    actual_escalate = tp + fn
    actual_auto = tn + fp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Safety Risk: False Auto-Handle Rate
    false_auto_handle_rate = round(fn / actual_escalate, 4) if actual_escalate > 0 else 0.0
    # Human Labor Friction: False Escalation Rate
    false_escalate_rate = round(fp / actual_auto, 4) if actual_auto > 0 else 0.0

    # Total Asymmetric Penalty
    total_penalty = round(fn * cost_ratio + fp * 1.0, 2)
    normalized_cost = round(total_penalty / n, 4)

    return {
        "escalation_accuracy": round((tp + tn) / n, 4),
        "escalation_precision": round(prec, 4),
        "escalation_recall": round(rec, 4),
        "escalation_f1": round(f1, 4),
        "true_positives_escalate": tp,
        "false_positives_escalate": fp,
        "false_negatives_auto_handle_risk": fn,
        "true_negatives_auto_handle": tn,
        "false_auto_handle_rate": false_auto_handle_rate,
        "false_escalate_rate": false_escalate_rate,
        "asymmetric_cost_penalty": total_penalty,
        "normalized_risk_cost": normalized_cost,
    }


def evaluate_confidence_thresholds(
    gold_decisions: List[str],
    predicted_intents: List[str],
    intent_confidences: List[float],
    precedent_agreement_scores: List[float],
    grounding_results: List[Dict[str, Any]],
    candidate_thresholds: List[float] = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    agreement_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Evaluates escalation performance across candidate classification confidence thresholds:
    If intent_confidence < tau -> escalate.
    """
    from src.agent.pipeline import ALWAYS_ESCALATE_INTENTS

    results = []

    for tau in candidate_thresholds:
        simulated_decisions = []
        for it, conf, agr, gv in zip(predicted_intents, intent_confidences, precedent_agreement_scores, grounding_results):
            # Base policy rules
            if it in ALWAYS_ESCALATE_INTENTS:
                dec = "escalate"
            elif agr < agreement_threshold:
                dec = "escalate"
            elif not gv.get("grounded", True) or len(gv.get("unsupported_claims", [])) > 0:
                dec = "escalate"
            elif conf < tau:
                dec = "escalate"
            else:
                dec = "auto_handle"
            simulated_decisions.append(dec)

        metrics = compute_escalation_metrics(gold_decisions, simulated_decisions)
        results.append({
            "confidence_threshold": tau,
            "escalation_precision": metrics["escalation_precision"],
            "escalation_recall": metrics["escalation_recall"],
            "false_auto_handle_rate": metrics["false_auto_handle_rate"],
            "false_escalate_rate": metrics["false_escalate_rate"],
            "normalized_risk_cost": metrics["normalized_risk_cost"],
            "escalated_count": sum(1 for d in simulated_decisions if d == "escalate"),
            "auto_handled_count": sum(1 for d in simulated_decisions if d == "auto_handle"),
        })

    return results
