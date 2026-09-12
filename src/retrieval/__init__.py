"""
Retrieval and grounding package for Stage 4.
Provides structured precedent extraction, ChromaDB vector indexing,
intent-filtered precedent retrieval, and precedent-agreement scoring.
"""

from .holdout_split import split_golden_holdout
from .extract_precedents import extract_structured_precedents
from .build_index import build_precedent_index
from .query_index import retrieve_precedents

__all__ = [
    "split_golden_holdout",
    "extract_structured_precedents",
    "build_precedent_index",
    "retrieve_precedents",
]
