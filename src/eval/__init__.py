"""
Stage 6: Evaluation Harness, Golden Set, Baselines, and Metrics.
"""

# Lazy / optional imports to allow module initialization
try:
    from .baselines import run_trivial_baseline, run_simple_baseline
except ImportError:
    pass

try:
    from .judge import evaluate_reply_quality, compute_judge_human_agreement
except ImportError:
    pass

try:
    from .metrics import compute_classification_metrics, compute_escalation_metrics
except ImportError:
    pass
