"""
Semantic clustering and draft taxonomy generation module for Stage 3.

Embeds sampled customer first-messages, evaluates K-Means clustering across a range of k (5-15)
using silhouette scores, extracts real exemplar messages for each cluster, and uses an LLM
(or heuristic keyphrase extractor fallback) to propose descriptive intent labels.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cluster_messages")

DEFAULT_SAMPLE_PATH = "data/processed/taxonomy_sample.parquet"
DEFAULT_DRAFT_OUTPUT = "data/processed/draft_taxonomy.json"
DEFAULT_EMBEDDING_CACHE = "data/processed/taxonomy_embeddings.npy"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEED = 42


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Initialize fastembed TextEmbedding model."""
    from fastembed import TextEmbedding
    threads = os.cpu_count() or 4
    logger.info(f"Loading embedding model '{model_name}' via ONNX with {threads} CPU threads...")
    return TextEmbedding(model_name=model_name, threads=threads)


def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    cache_path: Optional[str] = DEFAULT_EMBEDDING_CACHE,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Generate dense L2-normalized embeddings for a list of text strings.
    Utilizes local ONNX fastembed runtime for speed and zero API cost.
    Supports disk caching to speed up iterative analysis.
    """
    if not force_recompute and cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        if len(embeddings) == len(texts):
            return embeddings
        logger.warning(f"Cached embeddings count ({len(embeddings)}) != texts count ({len(texts)}). Recomputing...")

    t0 = time.time()
    model = get_embedding_model(model_name)
    logger.info(f"Generating embeddings for {len(texts):,} texts...")

    # fastembed returns generator of numpy arrays
    raw_embeddings = list(model.embed(texts))
    embeddings = np.array(raw_embeddings, dtype=np.float32)

    # L2 normalize so Euclidean distance matches Cosine distance
    embeddings = normalize(embeddings, norm="l2", axis=1)
    logger.info(f"Embeddings generated in {time.time() - t0:.2f}s, shape: {embeddings.shape}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings cache to {cache_path}")

    return embeddings


def evaluate_k_sweep(
    embeddings: np.ndarray,
    k_min: int = 5,
    k_max: int = 15,
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[int, float], int, Dict[int, KMeans]]:
    """
    Perform a sweep over k in [k_min, k_max] and compute silhouette scores.
    Returns:
    - scores: Dict mapping k -> silhouette_score
    - best_k: k with the maximum silhouette score
    - models: Dict mapping k -> fitted KMeans model
    """
    logger.info(f"Evaluating clustering sweep for k in [{k_min}, {k_max}] (seed={seed})...")
    scores: Dict[int, float] = {}
    models: Dict[int, KMeans] = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels, metric="euclidean"))
        scores[k] = round(score, 4)
        models[k] = kmeans
        logger.info(f"  k = {k:2d} | Silhouette Score = {score:.4f}")

    best_k = max(scores.keys(), key=lambda k: scores[k])
    logger.info(f"Optimal k from silhouette sweep: k = {best_k} (score = {scores[best_k]:.4f})")
    return scores, best_k, models


def extract_cluster_exemplars(
    df: pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    embeddings: np.ndarray,
    n_exemplars: int = 15,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    For each cluster, find the n_exemplars messages closest to the cluster centroid.
    """
    exemplars_by_cluster = {}
    k = len(centroids)

    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            exemplars_by_cluster[cluster_id] = []
            continue

        cluster_embeddings = embeddings[indices]
        centroid = centroids[cluster_id]

        # Compute Euclidean distances to centroid (since L2-normalized, lower distance = higher cosine similarity)
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        ranked_order = np.argsort(dists)

        top_indices = indices[ranked_order[:n_exemplars]]

        cluster_items = []
        for idx in top_indices:
            row = df.iloc[idx]
            cluster_items.append({
                "tweet_id": str(row["tweet_id"]),
                "thread_id": str(row["thread_id"]),
                "text": str(row["text"]),
                "cleaned_text": str(row.get("cleaned_text", row["text"])),
            })
        exemplars_by_cluster[cluster_id] = cluster_items

    return exemplars_by_cluster


def _extract_heuristic_label(exemplar_texts: List[str]) -> Tuple[str, str]:
    """
    Fallback high-precision heuristic intent extraction using term frequencies
    and domain-specific regex when external LLM API is unavailable.
    """
    combined = " ".join(exemplar_texts).lower()

    # Rule-based intent matching on top keywords
    rules = [
        (r"\b(update|ios\s*\d+|11\.|install|download|updating)\b",
         "software_update_installation",
         "Issues updating iOS/macOS, download failures, or post-update regressions."),
        (r"\b(battery|drain|dying|percentage|charge|charging|charger|shutting\s*off)\b",
         "battery_charging_power",
         "Abnormal battery drain, battery health degradation, or failure to charge."),
        (r"\b(apple\s*id|password|locked|disabled|passcode|icloud|login|security)\b",
         "account_access_apple_id_security",
         "Apple ID lockouts, two-factor authentication, password resets, and account access."),
        (r"\b(music|itunes|app\s*store|subscription|bill|charge|refund|purchased?)\b",
         "billing_purchases_subscriptions",
         "App Store purchases, unrecognized charges, Apple Music and iCloud subscriptions."),
        (r"\b(screen|cracked|display|black\s*screen|frozen|unresponsive|touch|broken)\b",
         "hardware_display_touch_defect",
         "Physical display damage, unresponsive touch screen, or frozen black screen."),
        (r"\b(wifi|wi-fi|bluetooth|connection|cellular|carrier|service|lte|sim|network)\b",
         "connectivity_wifi_cellular_bluetooth",
         "Connectivity drops, Wi-Fi authentication issues, Bluetooth pairing, or cellular service loss."),
        (r"\b(message|imessage|text|sms|send|receive|call|calling|contact)\b",
         "messaging_calling_communication",
         "iMessage activation errors, failed SMS delivery, or phone call dropped audio."),
        (r"\b(camera|photo|pictures?|video|flash|focus)\b",
         "camera_photos_multimedia",
         "Camera blur, shutter lag, photo syncing errors, or flash malfunction."),
        (r"\b(sound|speaker|microphone|audio|volume|earpiece|headphones|airpods)\b",
         "audio_speaker_microphone",
         "Distorted speaker audio, microphone not picking up voice, or AirPods connectivity."),
    ]

    for pattern, label, desc in rules:
        matches = len(re.findall(pattern, combined))
        if matches >= 3:
            return label, desc

    # General fallback based on frequent words
    words = [w for w in re.findall(r"\b[a-z]{4,}\b", combined) if w not in {"have", "with", "this", "that", "from", "when", "your", "what", "apple", "applesupport", "phone", "iphone"}]
    top_words = [w for w, _ in Counter(words).most_common(2)]
    label_str = "_".join(top_words) if top_words else "general_technical_support"
    return f"{label_str}_inquiry", f"Customer assistance inquiries regarding {' and '.join(top_words) if top_words else 'Apple devices'}."


def propose_cluster_label(
    exemplar_texts: List[str],
    cluster_id: int,
    cluster_size: int,
) -> Tuple[str, str]:
    """
    Propose a short, human-readable intent label and one-line description.
    Attempts LLM API call if GEMINI_API_KEY/GOOGLE_API_KEY or OPENAI_API_KEY is present;
    otherwise falls back gracefully to domain heuristic keyphrase extraction.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    prompt = (
        "You are an expert customer support taxonomist. Propose a short, snake_case intent label "
        "(e.g. battery_charging_power, software_update_ios, account_security_apple_id) and a one-line description "
        "for the following customer messages sent to Apple Support on Twitter.\n\n"
        "Messages:\n"
        + "\n".join(f"- {txt}" for txt in exemplar_texts[:10])
        + "\n\nFormat your response as strict JSON: {\"label\": \"<snake_case_label>\", \"description\": \"<one_line_description>\"}"
    )

    if gemini_key:
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            req_data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text_resp = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text_resp)
                return parsed["label"], parsed["description"]
        except Exception as e:
            logger.warning(f"Gemini API call failed for cluster {cluster_id}: {e}. Using heuristic fallback.")

    elif openai_key:
        try:
            import urllib.request
            url = "https://api.openai.com/v1/chat/completions"
            req_data = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text_resp = data["choices"][0]["message"]["content"]
                parsed = json.loads(text_resp)
                return parsed["label"], parsed["description"]
        except Exception as e:
            logger.warning(f"OpenAI API call failed for cluster {cluster_id}: {e}. Using heuristic fallback.")

    return _extract_heuristic_label(exemplar_texts)


def cluster_messages(
    sample_path: str = DEFAULT_SAMPLE_PATH,
    output_path: str = DEFAULT_DRAFT_OUTPUT,
    k_min: int = 5,
    k_max: int = 15,
    chosen_k: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Main orchestration for Step 2:
    - Loads sampled customer first-messages
    - Computes real semantic embeddings
    - Evaluates k across [k_min, k_max] with silhouette scores
    - Fits clustering model for chosen_k (or best_k)
    - Extracts 10-15 real exemplars per cluster
    - Proposes labels and descriptions
    - Persists draft_taxonomy.json
    """
    logger.info(f"Loading sample from {sample_path}...")
    sample_df = pd.read_parquet(sample_path)
    logger.info(f"Loaded {len(sample_df):,} customer inquiries.")

    # Use cleaned_text for embedding if available
    texts_to_embed = sample_df["cleaned_text"].tolist() if "cleaned_text" in sample_df.columns else sample_df["text"].tolist()
    embeddings = generate_embeddings(texts_to_embed)

    # 1. Silhouette score sweep
    scores, best_k, models = evaluate_k_sweep(embeddings, k_min=k_min, k_max=k_max, seed=seed)

    if chosen_k is None:
        chosen_k = best_k

    logger.info(f"Selected k = {chosen_k} for draft taxonomy clustering.")
    chosen_model = models[chosen_k]
    labels = chosen_model.labels_
    centroids = chosen_model.cluster_centers_

    # 2. Extract real exemplars per cluster
    exemplars = extract_cluster_exemplars(
        df=sample_df,
        labels=labels,
        centroids=centroids,
        embeddings=embeddings,
        n_exemplars=15,
    )

    # 3. Propose labels and descriptions
    clusters_data = []
    cluster_counts = pd.Series(labels).value_counts().to_dict()

    print("\n" + "=" * 80)
    print(f"DRAFT TAXONOMY CLUSTERING RESULTS (k = {chosen_k})")
    print("=" * 80)
    format_row = "{:<4} | {:<32} | {:<8} | {:<50}"
    print(format_row.format("ID", "Proposed Label", "Size", "Sample First Message"))
    print("-" * 80)

    for cluster_id in range(chosen_k):
        size = cluster_counts.get(cluster_id, 0)
        cluster_ex = exemplars.get(cluster_id, [])
        ex_texts = [item["text"] for item in cluster_ex]

        label, desc = propose_cluster_label(ex_texts, cluster_id, size)

        first_ex_snippet = ex_texts[0][:47] + "..." if ex_texts else "N/A"
        print(format_row.format(str(cluster_id), label, str(size), first_ex_snippet))

        clusters_data.append({
            "cluster_id": cluster_id,
            "proposed_label": label,
            "description": desc,
            "cluster_size": size,
            "pct_of_sample": round(100.0 * size / len(sample_df), 2),
            "exemplars": cluster_ex,
        })

    print("=" * 80 + "\n")

    draft_taxonomy = {
        "sample_size": len(sample_df),
        "chosen_k": chosen_k,
        "silhouette_scores": {str(k): score for k, score in scores.items()},
        "clusters": clusters_data,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(draft_taxonomy, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved draft taxonomy to {output_path}")
    return draft_taxonomy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering and draft intent taxonomy derivation.")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE_PATH, help="Path to taxonomy_sample.parquet")
    parser.add_argument("--output", default=DEFAULT_DRAFT_OUTPUT, help="Path to draft_taxonomy.json")
    parser.add_argument("--k-min", type=int, default=5, help="Min k for silhouette sweep")
    parser.add_argument("--k-max", type=int, default=15, help="Max k for silhouette sweep")
    parser.add_argument("--k", type=int, default=None, help="Explicit k to select")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    cluster_messages(
        sample_path=args.sample,
        output_path=args.output,
        k_min=args.k_min,
        k_max=args.k_max,
        chosen_k=args.k,
        seed=args.seed,
    )
