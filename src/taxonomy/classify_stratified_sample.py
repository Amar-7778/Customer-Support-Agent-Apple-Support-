"""
Stratified sample customer inquiry intent classification module for Stage 3.

Rescoped from exhaustive full-corpus classification (74k msgs, ~23.5hrs at free-tier Groq TPM)
to a representative, stratified sample of 6,000 messages sized for practical runtime.
Uses exclusively the official Groq Python SDK (qwen/qwen3.8-27b) with few-shot prompting
grounded in taxonomy.yaml. Zero fallback to TF-IDF or centroids; all rows classified by Groq.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv, find_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIError
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .sample_for_clustering import (
    clean_tweet_text,
    extract_customer_first_messages,
    assign_stratification_bins,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("classify_stratified_sample")

DEFAULT_THREADS_PATH = "data/processed/AppleSupport_threads.parquet"
DEFAULT_TAXONOMY_PATH = "taxonomy.yaml"
DEFAULT_SAMPLE_2K_PATH = "data/processed/taxonomy_sample.parquet"
DEFAULT_EMBEDDINGS_2K_PATH = "data/processed/taxonomy_embeddings.npy"
DEFAULT_STRATIFIED_SAMPLE_PATH = "data/processed/AppleSupport_sample_6000.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/AppleSupport_classified_sample.parquet"
DEFAULT_CHECKPOINT_PATH = "data/processed/AppleSupport_classified_sample_checkpoint.parquet"
DEFAULT_STATS_PATH = "reports/pipeline_stats.json"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_SEED = 42


def initialize_groq_client() -> Tuple[Groq, str]:
    """
    Validates GROQ_API_KEY at startup and initializes the official Groq client.
    Fails loudly and clearly if the key is unset or empty.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "\n" + "=" * 80 + "\n"
            "CRITICAL CONFIGURATION ERROR: GROQ_API_KEY is unset or empty in .env.\n"
            "Per pipeline architecture constraints, Groq is the EXCLUSIVE provider for all LLM inference.\n"
            "No other provider (Gemini, OpenAI, Anthropic) is permitted.\n"
            f"Please open {env_path} and set:\n"
            "  GROQ_API_KEY=gsk_your_actual_groq_key_here\n"
            + "=" * 80
        )

    logger.info("GROQ_API_KEY validated successfully at startup.")
    client = Groq(api_key=api_key.strip())
    return client, api_key.strip()


def load_taxonomy(taxonomy_path: str = DEFAULT_TAXONOMY_PATH) -> Dict[str, Any]:
    """Load finalized taxonomy.yaml configuration."""
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy configuration not found at {taxonomy_path}.")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_groq_system_prompt(taxonomy: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, bool]]:
    """
    Constructs the few-shot system prompt from taxonomy.yaml.
    Includes every intent name, official description, and representative customer tweet exemplars.
    """
    intents = taxonomy.get("intents", [])
    valid_intents = []
    escalation_flags = {}
    intent_blocks = []

    for item in intents:
        name = item["intent_name"]
        desc = item.get("description", "")
        examples = item.get("representative_examples", [])
        escalate = bool(item.get("escalation_default", False))

        valid_intents.append(name)
        escalation_flags[name] = escalate

        ex_lines = "\n".join([f'    * "{clean_tweet_text(ex.get("text", ""))}"' for ex in examples[:3]])
        intent_blocks.append(
            f"- Intent: `{name}`\n"
            f"  Description: {desc}\n"
            f"  Exemplars:\n{ex_lines}"
        )

    intents_text = "\n\n".join(intent_blocks)
    prompt = (
        "You are an expert customer support intent classifier for Apple Support inquiries.\n"
        "Here is the finalized intent taxonomy grounded strictly in real customer interactions:\n\n"
        f"{intents_text}\n\n"
        f"Allowed intent categories (choose exactly one per message):\n{json.dumps(valid_intents)}\n\n"
        "Instructions:\n"
        "1. For each customer message provided, determine the customer's primary intent.\n"
        "2. Respond ONLY with a valid JSON object with the key 'results', containing a list of objects:\n"
        '   {"results": [{"idx": <int>, "intent": "<intent_name>", "confidence": <float 0.0-1.0>}]}\n'
        "3. Every intent MUST be one of the allowed categories. Do not invent new intents."
    )

    return prompt, valid_intents, escalation_flags


def draw_stratified_6k_sample(
    threads_path: str = DEFAULT_THREADS_PATH,
    sample_2k_path: str = DEFAULT_SAMPLE_2K_PATH,
    embeddings_2k_path: str = DEFAULT_EMBEDDINGS_2K_PATH,
    output_sample_path: str = DEFAULT_STRATIFIED_SAMPLE_PATH,
    sample_size: int = 6000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Draws a 6,000-message stratified sample from the full 73,997 resolved customer corpus,
    stratified proportionally across the 8 draft clusters from stage 3's k-means output
    AND across time period, with a fixed random seed.
    Documents overlap with the earlier 2,000-message clustering sample.
    """
    if os.path.exists(output_sample_path):
        logger.info(f"Loading existing stratified sample from {output_sample_path}...")
        df = pd.read_parquet(output_sample_path)
        if len(df) == sample_size:
            return df

    logger.info(f"Generating stratified sample of {sample_size:,} messages from {threads_path}...")

    # 1. Fit KMeans on Stage 3 2k sample embeddings to establish 8 cluster labels
    sample_2k = pd.read_parquet(sample_2k_path)
    embeds = np.load(embeddings_2k_path)
    kmeans = KMeans(n_clusters=8, random_state=seed, n_init=10)
    sample_2k["draft_cluster"] = kmeans.fit_predict(embeds)

    # 2. Train fast classifier to project draft cluster labels across full 74k corpus
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train = vec.fit_transform(sample_2k["cleaned_text"])
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_train, sample_2k["draft_cluster"])

    # 3. Load full resolved corpus
    threads_df = pd.read_parquet(threads_path)
    resolved_df = threads_df[threads_df["resolved"] == "true"].copy()
    full_inquiries = extract_customer_first_messages(resolved_df).reset_index(drop=True)
    full_inquiries["cleaned_text"] = full_inquiries["text"].apply(clean_tweet_text)
    full_inquiries["draft_cluster"] = clf.predict(vec.transform(full_inquiries["cleaned_text"]))

    # 4. Bin by time period and formulate composite strata
    full_binned = assign_stratification_bins(full_inquiries)
    full_binned["composite_stratum"] = (
        full_binned["time_period"] + "__cluster_" + full_binned["draft_cluster"].astype(str)
    )

    # 5. Draw proportional sample across composite strata
    strata_counts = full_binned["composite_stratum"].value_counts()
    sampled_dfs = []
    for st, count in strata_counts.items():
        n_sub = max(1, int(round(sample_size * (count / len(full_binned)))))
        sub_slice = full_binned[full_binned["composite_stratum"] == st]
        if len(sub_slice) <= n_sub:
            sampled_dfs.append(sub_slice)
        else:
            sampled_dfs.append(sub_slice.sample(n=n_sub, random_state=seed))

    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    elif len(sample_df) < sample_size:
        diff = sample_size - len(sample_df)
        remaining = full_binned[~full_binned["tweet_id"].isin(sample_df["tweet_id"])]
        add_df = remaining.sample(n=diff, random_state=seed)
        sample_df = pd.concat([sample_df, add_df], ignore_index=True)

    # 6. Document overlap with 2,000 clustering sample
    overlap = set(sample_df["tweet_id"]).intersection(set(sample_2k["tweet_id"]))
    logger.info(
        f"Drawn {len(sample_df):,} stratified messages across {len(strata_counts)} strata. "
        f"Overlap with 2,000 clustering sample: {len(overlap)} / {len(sample_2k)} "
        f"({len(overlap) / len(sample_df) * 100:.2f}% of 6,000 sample)."
    )

    Path(output_sample_path).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(output_sample_path, index=False, engine="pyarrow")
    logger.info(f"Saved stratified sample to {output_sample_path}")
    return sample_df


def call_groq_batch(
    client: Groq,
    model: str,
    system_prompt: str,
    batch_payload: List[Dict[str, Any]],
    valid_intents: List[str],
    max_retries: int = 6,
) -> Tuple[List[Dict[str, Any]], int, int, float]:
    """
    Submits a batch of customer inquiries to Groq via chat completions.
    Handles rate limiting (HTTP 429) with exponential backoff and returns:
    - parsed valid items
    - prompt tokens
    - completion tokens
    - latency seconds
    """
    t0 = time.time()
    valid_set = set(valid_intents)
    user_prompt = f"Messages to classify:\n{json.dumps(batch_payload)}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            dur = time.time() - t0
            usage = response.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            content = response.choices[0].message.content
            parsed = json.loads(content)
            results = parsed.get("results", []) if isinstance(parsed, dict) else parsed

            clean_results = []
            for item in results:
                idx = item.get("idx")
                intent = item.get("intent")
                conf = float(item.get("confidence", 0.90))
                if idx is not None and intent in valid_set:
                    clean_results.append({"idx": int(idx), "intent": intent, "confidence": conf})

            return clean_results, p_tok, c_tok, dur

        except RateLimitError as rle:
            wait_time = (2.0 ** attempt) * 5.0
            m = re.search(r"try again in ([\d\.]+)s", str(rle), re.IGNORECASE)
            if m:
                wait_time = max(wait_time, float(m.group(1)) + 1.0)
            logger.warning(
                f"Groq Rate Limit (attempt {attempt+1}/{max_retries}): {rle}. Backing off {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
        except (APIConnectionError, APIError) as api_err:
            wait_time = (2.0 ** attempt) * 3.0
            logger.warning(
                f"Groq API error (attempt {attempt+1}/{max_retries}): {api_err}. Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
        except Exception as e:
            logger.warning(f"Unexpected error in Groq call (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3.0)

    logger.error(f"Batch failed after {max_retries} attempts.")
    return [], 0, 0, time.time() - t0


def classify_stratified_sample(
    threads_path: str = DEFAULT_THREADS_PATH,
    taxonomy_path: str = DEFAULT_TAXONOMY_PATH,
    sample_path: str = DEFAULT_STRATIFIED_SAMPLE_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    model: str = DEFAULT_GROQ_MODEL,
    sample_size: int = 6000,
    batch_size: int = 25,
    max_messages: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Main orchestration for Groq-exclusive stratified sample classification:
    - Validates GROQ_API_KEY at startup
    - Draws or loads the 6,000 stratified customer inquiries
    - Classifies via Groq LLM API with structured batching and exponential backoff
    - Records row-level classification provenance ('groq_llm') for every row
    - Checkpoints after every batch for crash resilience and resume capability
    - Recomputes exact total tokens and API calls from response objects
    - Saves classified sample to parquet and updates pipeline_stats.json
    """
    t_start = time.time()
    logger.info("=" * 80)
    logger.info("STAGE 3 STRATIFIED SAMPLE CLASSIFICATION — EXCLUSIVELY VIA GROQ")
    logger.info(f"Target Sample Size: {sample_size:,} messages")
    logger.info(f"Threads Input:      {threads_path}")
    logger.info(f"Taxonomy Config:    {taxonomy_path}")
    logger.info(f"Stratified Sample:  {sample_path}")
    logger.info(f"Output Parquet:     {output_path}")
    logger.info(f"Groq Model:         {model}")
    logger.info(f"Batch Size:         {batch_size} msgs/call")
    logger.info("=" * 80)

    # 1. Validate Groq client
    client, _ = initialize_groq_client()

    # 2. Load Taxonomy
    taxonomy = load_taxonomy(taxonomy_path)
    system_prompt, valid_intents, escalation_map = build_groq_system_prompt(taxonomy)

    # 3. Draw or Load 6k Stratified Sample
    sample_df = draw_stratified_6k_sample(
        threads_path=threads_path,
        sample_2k_path=DEFAULT_SAMPLE_2K_PATH,
        embeddings_2k_path=DEFAULT_EMBEDDINGS_2K_PATH,
        output_sample_path=sample_path,
        sample_size=sample_size,
        seed=seed,
    )

    if max_messages is not None and max_messages < len(sample_df):
        logger.info(f"Capping evaluation to first {max_messages:,} messages per --max-messages...")
        sample_df = sample_df.iloc[:max_messages].copy().reset_index(drop=True)

    num_records = len(sample_df)
    predicted_labels = [""] * num_records
    confidence_scores = [0.0] * num_records
    source_flags = [""] * num_records

    # Check for existing checkpoint to support resuming
    api_calls_made = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if os.path.exists(checkpoint_path):
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            if len(ckpt_df) == num_records and "predicted_intent" in ckpt_df.columns:
                valid_mask = ckpt_df["predicted_intent"].str.len() > 0
                for i in range(num_records):
                    if valid_mask.iloc[i]:
                        predicted_labels[i] = ckpt_df["predicted_intent"].iloc[i]
                        confidence_scores[i] = float(ckpt_df["confidence"].iloc[i])
                        source_flags[i] = "groq_llm"
                resumed_count = sum(1 for s in source_flags if s == "groq_llm")
                logger.info(f"Resumed {resumed_count:,}/{num_records} previously classified rows from checkpoint.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    # 4. Formulate Batches for Unclassified Rows
    unclassified_indices = [i for i in range(num_records) if not predicted_labels[i]]
    logger.info(f"Remaining rows to classify via Groq: {len(unclassified_indices):,} of {num_records:,}")

    batches = []
    for s_idx in range(0, len(unclassified_indices), batch_size):
        sub_indices = unclassified_indices[s_idx : s_idx + batch_size]
        payload = [
            {"idx": int(idx), "text": clean_tweet_text(sample_df.iloc[idx]["text"])}
            for idx in sub_indices
        ]
        batches.append((sub_indices, payload))

    logger.info(f"Dispatching {len(batches):,} Groq API batches ({batch_size} msgs/batch)...")

    # 5. Execute Groq Batches
    for b_idx, (sub_indices, payload) in enumerate(batches):
        results, p_tok, c_tok, dur = call_groq_batch(
            client=client,
            model=model,
            system_prompt=system_prompt,
            batch_payload=payload,
            valid_intents=valid_intents,
        )

        api_calls_made += 1
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        # Map predictions back to rows
        for item in results:
            idx = item["idx"]
            if 0 <= idx < num_records:
                predicted_labels[idx] = item["intent"]
                confidence_scores[idx] = item["confidence"]
                source_flags[idx] = "groq_llm"
                # Row-by-row monotonic logging
                logger.info(
                    f"[Row {idx+1}/{num_records}] [groq_llm] tweet_id: {sample_df.iloc[idx]['tweet_id']} "
                    f"-> {item['intent']} ({item['confidence']:.2f})"
                )

        completed_rows = sum(1 for s in source_flags if s == "groq_llm")
        # Save checkpoint after every batch for 100% crash resilience
        sample_df["predicted_intent"] = predicted_labels
        sample_df["confidence"] = confidence_scores
        sample_df["classification_source"] = source_flags
        sample_df.to_parquet(checkpoint_path, index=False)

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == len(batches):
            logger.info(
                f"Groq Batch {b_idx+1}/{len(batches)} Complete "
                f"({completed_rows}/{num_records} rows, {api_calls_made} calls, "
                f"{total_prompt_tokens + total_completion_tokens:,} tokens, latency: {dur:.2f}s)"
            )

        # Smooth pacing to adhere to Groq 1,000 OTPM limit (~20-22s per batch of 25)
        sleep_dur = max(2.0, 22.0 - dur)
        time.sleep(sleep_dur)

    # 6. Retry any missing rows
    missing = [i for i in range(num_records) if not predicted_labels[i]]
    if missing:
        logger.warning(f"Retrying {len(missing)} unassigned rows with individual calls...")
        for m_idx in missing:
            row_payload = [{"idx": m_idx, "text": clean_tweet_text(sample_df.iloc[m_idx]["text"])}]
            results, p_tok, c_tok, _ = call_groq_batch(
                client=client,
                model=model,
                system_prompt=system_prompt,
                batch_payload=row_payload,
                valid_intents=valid_intents,
            )
            api_calls_made += 1
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            if results:
                predicted_labels[m_idx] = results[0]["intent"]
                confidence_scores[m_idx] = results[0]["confidence"]
                source_flags[m_idx] = "groq_llm"
            time.sleep(2.0)

    total_tokens = total_prompt_tokens + total_completion_tokens
    total_wall_clock = time.time() - t_start

    # Assign columns
    sample_df["predicted_intent"] = predicted_labels
    sample_df["confidence"] = confidence_scores
    sample_df["classification_source"] = source_flags
    sample_df["escalation_default"] = [escalation_map.get(lbl, False) for lbl in predicted_labels]

    # Verify coverage: 100% of rows must have valid labels from Groq
    assert sample_df["predicted_intent"].isna().sum() == 0, "Found null predictions!"
    assert (sample_df["predicted_intent"] == "").sum() == 0, "Found empty predictions!"
    assert set(sample_df["predicted_intent"].unique()).issubset(set(valid_intents))
    assert (sample_df["classification_source"] == "groq_llm").all(), "Non-Groq classification source found!"

    # 7. Save Classified Sample Parquet Output
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(out_file, index=False, engine="pyarrow")
    logger.info(f"Saved {len(sample_df):,} Groq-classified records to {output_path}")

    # Remove temporary checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    # 8. Compute Distribution and Cost
    # Pricing for qwen/qwen3.8-27b: $0.15 / 1M prompt, $0.60 / 1M completion
    estimated_cost_usd = round(
        (total_prompt_tokens * 0.00000015) + (total_completion_tokens * 0.00000060), 4
    )
    dist_counts = sample_df["predicted_intent"].value_counts().to_dict()
    dist_pct = {k: round(100.0 * v / num_records, 2) for k, v in dist_counts.items()}
    escalated_count = int(sample_df["escalation_default"].sum())
    escalated_pct = round(100.0 * escalated_count / num_records, 2)

    # 9. Print Verified Summary Table
    print("\n" + "=" * 85)
    print(f"STAGE 3 GROQ-CLASSIFIED STRATIFIED SAMPLE REPORT ({num_records:,} Messages)")
    print(f"Provider: Groq (Official SDK) | Model: {model}")
    print(f"API Calls Made: {api_calls_made:,} | Prompt Tokens: {total_prompt_tokens:,} | Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,} ({total_tokens / num_records:.1f} tokens/msg) | Cost: ${estimated_cost_usd:.4f}")
    print(f"Wall-Clock Duration: {total_wall_clock:.2f}s ({num_records / total_wall_clock:.2f} msgs/sec)")
    print("=" * 85)
    format_row = "{:<36} | {:<10} | {:<8} | {:<12}"
    print(format_row.format("Intent Name", "Count", "Pct (%)", "Escalation"))
    print("-" * 85)
    for intent, count in dist_counts.items():
        print(
            format_row.format(
                intent,
                f"{count:,}",
                f"{dist_pct[intent]:.2f}%",
                "ALWAYS" if escalation_map.get(intent, False) else "standard",
            )
        )
    print("-" * 85)
    print(format_row.format("TOTAL", f"{num_records:,}", "100.00%", f"{escalated_pct:.1f}% Escalate"))
    print("=" * 85 + "\n")

    # 10. Update reports/pipeline_stats.json
    existing_stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                existing_stats = json.load(f)
        except Exception:
            existing_stats = {}

    existing_stats["stage3_classified_stratified_sample"] = {
        "status": "COMPLETED",
        "provider": "Groq",
        "model": model,
        "sample_size": num_records,
        "input_threads_path": str(threads_path),
        "stratified_sample_path": str(sample_path),
        "classified_artifact_path": str(output_path),
        "classification_method": "few_shot_llm_groq_sdk",
        "api_calls_made": api_calls_made,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_message": round(total_tokens / num_records, 2) if num_records else 0,
        "estimated_cost_usd": estimated_cost_usd,
        "duration_seconds": round(total_wall_clock, 2),
        "class_distribution_counts": dist_counts,
        "class_distribution_percentages": dist_pct,
        "always_escalate_count": escalated_count,
        "always_escalate_pct": escalated_pct,
        "scoping_decision": (
            "Rescoped from exhaustive full-corpus classification (73,997 msgs, ~23.5hrs at 8,000 TPM) "
            "to a representative 6,000-message sample stratified across the 8 draft clusters and time period."
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(existing_stats, f, indent=2)

    logger.info(f"Updated pipeline stats in {stats_path}")
    return sample_df


# For backwards compatibility with classify_full_corpus imports
classify_corpus = classify_stratified_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groq-exclusive few-shot intent classification on stratified sample (Stage 3).")
    parser.add_argument("--threads", default=DEFAULT_THREADS_PATH)
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--sample-output", default=DEFAULT_STRATIFIED_SAMPLE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH)
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    classify_stratified_sample(
        threads_path=args.threads,
        taxonomy_path=args.taxonomy,
        sample_path=args.sample_output,
        output_path=args.output,
        stats_path=args.stats,
        model=args.model,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_messages=args.max_messages,
        seed=args.seed,
    )
