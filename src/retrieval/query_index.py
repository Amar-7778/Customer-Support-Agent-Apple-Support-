"""
Query and precedent retrieval module for Stage 4 (Step 4).

Implements retrieve_precedents(message, intent, k=5):
- Filters candidate precedents strictly by intent metadata.
- Ranks candidates by dense semantic cosine similarity.
- Computes precedent-agreement score (fraction of top-k sharing a consistent resolution action)
  to feed Stage 5's escalation decision logic.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from fastembed import TextEmbedding

from src.taxonomy.sample_for_clustering import clean_tweet_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("query_index")

DEFAULT_CHROMA_DIR = "data/processed/chroma_db"
DEFAULT_COLLECTION_NAME = "apple_support_precedents"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons for low-latency queries
_EMBED_MODEL: Optional[TextEmbedding] = None
_CHROMA_COLLECTION: Optional[Any] = None


def get_embed_model(model_name: str = DEFAULT_MODEL_NAME) -> TextEmbedding:
    """Lazy loader for FastEmbed model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        threads = max(1, os.cpu_count() or 4)
        _EMBED_MODEL = TextEmbedding(model_name=model_name, threads=threads)
    return _EMBED_MODEL


def get_collection(
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """Lazy loader for persistent ChromaDB collection."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(f"ChromaDB directory not found at {chroma_dir}. Run build_index first.")
        client = chromadb.PersistentClient(path=chroma_dir)
        _CHROMA_COLLECTION = client.get_collection(name=collection_name)
    return _CHROMA_COLLECTION


def compute_precedent_agreement(
    retrieved_actions: List[str],
    retrieved_outcomes: List[str],
    embed_model: TextEmbedding,
    similarity_threshold: float = 0.65,
) -> float:
    """
    Computes the precedent-agreement score in [0.0, 1.0]:
    Determines what fraction of the top-k retrieved precedents share a consistent resolution strategy.

    Combines:
    1. Action Semantic Agreement: fraction of retrieved actions whose dense embedding
       cosine similarity to the primary (rank 1) action is >= similarity_threshold.
    2. Outcome Category Agreement: fraction of precedents sharing the modal outcome category.
    """
    if not retrieved_actions or len(retrieved_actions) <= 1:
        return 1.0

    k = len(retrieved_actions)

    # 1. Action Semantic Similarity against primary action
    action_embeds = list(embed_model.embed(retrieved_actions))
    action_mat = np.array([e / (np.linalg.norm(e) + 1e-9) for e in action_embeds])
    primary_vec = action_mat[0]
    
    # Cosine similarities to rank 1 action
    cos_sims = np.dot(action_mat, primary_vec)
    action_match_count = sum(1 for s in cos_sims if s >= similarity_threshold)
    action_agreement_ratio = action_match_count / k

    # 2. Outcome category consistency
    from collections import Counter
    outcome_counts = Counter(retrieved_outcomes)
    modal_outcome_count = outcome_counts.most_common(1)[0][1]
    outcome_agreement_ratio = modal_outcome_count / k

    # Blended agreement score (60% action semantic agreement + 40% outcome agreement)
    blended_score = round(0.60 * action_agreement_ratio + 0.40 * outcome_agreement_ratio, 4)
    return float(np.clip(blended_score, 0.0, 1.0))


def retrieve_precedents(
    message: str,
    intent: str,
    k: int = 5,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    """
    Query interface for grounding candidate replies (Stage 4 Step 4):
    1. Filters candidate precedents strictly by `intent` metadata.
    2. Embeds incoming query message and retrieves top-k nearest precedent neighbors.
    3. Calculates precedent-agreement score across top-k actions.
    4. Returns structured precedents and metadata to feed Stage 5 agent pipeline.
    """
    collection = get_collection(chroma_dir=chroma_dir, collection_name=collection_name)
    embed_model = get_embed_model(model_name=model_name)

    cleaned_msg = clean_tweet_text(message)
    query_text = f"Customer: {cleaned_msg}"
    query_emb = list(embed_model.embed([query_text]))[0].tolist()

    # Query ChromaDB with intent pre-filter
    chroma_results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where={"intent": intent},
        include=["metadatas", "documents", "distances"],
    )

    precedents = []
    actions = []
    outcomes = []

    if chroma_results["ids"] and len(chroma_results["ids"][0]) > 0:
        ids = chroma_results["ids"][0]
        metas = chroma_results["metadatas"][0]
        docs = chroma_results["documents"][0]
        dists = chroma_results["distances"][0]

        for p_id, meta, doc, dist in zip(ids, metas, docs, dists):
            sim = round(max(0.0, 1.0 - float(dist)), 4)
            act = meta.get("action_taken", "")
            out = meta.get("outcome", "")
            actions.append(act)
            outcomes.append(out)

            precedents.append({
                "thread_id": p_id,
                "tweet_id": meta.get("tweet_id"),
                "intent": meta.get("intent"),
                "action_taken": act,
                "outcome": out,
                "brand_reply_text": meta.get("brand_reply_text", ""),
                "similarity_score": sim,
                "cosine_distance": round(float(dist), 4),
                "document": doc,
            })

    # Precedent-agreement score across top-k actions
    agreement_score = compute_precedent_agreement(
        retrieved_actions=actions,
        retrieved_outcomes=outcomes,
        embed_model=embed_model,
    )

    return {
        "query_message": message,
        "query_intent": intent,
        "precedents_retrieved_count": len(precedents),
        "precedent_agreement_score": agreement_score,
        "precedents": precedents,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query precedent vector index with intent filter.")
    parser.add_argument("--message", required=True, help="Customer inquiry text to match.")
    parser.add_argument("--intent", required=True, help="Target intent category.")
    parser.add_argument("-k", type=int, default=5, help="Number of precedents to retrieve.")
    args = parser.parse_args()

    res = retrieve_precedents(message=args.message, intent=args.intent, k=args.k)
    print(json.dumps(res, indent=2))
