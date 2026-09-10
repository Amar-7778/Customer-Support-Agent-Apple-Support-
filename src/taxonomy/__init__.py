"""
Intent Taxonomy Derivation and Classification Module (Stage 3).

Provides:
- Stratified sampling of real customer first inquiries across time and thread depth.
- Dense semantic message embedding and parameter-swept K-Means clustering.
- Human-in-the-loop taxonomy review, merging, and escalation configuration.
- Full-corpus few-shot intent classification and class distribution auditing.
"""

from .sample_for_clustering import sample_customer_first_messages
from .cluster_messages import cluster_messages, evaluate_k_sweep
from .review_taxonomy import load_draft_taxonomy, save_final_taxonomy
from .classify_full_corpus import classify_corpus

__all__ = [
    "sample_customer_first_messages",
    "cluster_messages",
    "evaluate_k_sweep",
    "load_draft_taxonomy",
    "save_final_taxonomy",
    "classify_corpus",
]
