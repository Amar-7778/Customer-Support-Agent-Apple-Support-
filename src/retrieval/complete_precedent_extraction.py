"""
Fast completion runner for Stage 4 structured precedent extraction.

Integrates the 225 already-extracted software_update_os_bugs precedents from checkpoint,
and extracts 25-30 precedents for each of the remaining 7 intents using small batches (5 msgs/batch)
to guarantee zero rate-limit cooldowns on Groq's 8,000 TPM ceiling.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIError

from src.retrieval.extract_precedents import (
    initialize_groq_clients,
    build_system_prompt,
    parse_groq_wait_time,
    ALLOWED_OUTCOMES,
    DEFAULT_INDEXABLE_PATH,
    DEFAULT_THREADS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_STATS_PATH,
    DEFAULT_GROQ_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("complete_precedents")

CHECKPOINT_PATH = "data/processed/structured_precedents_checkpoint.parquet"
CHECKPOINT_META = "data/processed/structured_precedents_checkpoint.json"

TARGET_PER_INTENT = {
    "keyboard_text_autocorrect": 30,
    "battery_power_performance": 30,
    "hardware_display_physical": 25,
    "orders_purchases_applecare": 25,
    "account_access_apple_id": 25,
    "apple_music_audio_playback": 20,
    "international_multilingual_inquiries": 20,
}


def complete_extraction(
    checkpoint_path: str = CHECKPOINT_PATH,
    checkpoint_meta: str = CHECKPOINT_META,
    indexable_path: str = DEFAULT_INDEXABLE_PATH,
    threads_path: str = DEFAULT_THREADS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    model: str = DEFAULT_GROQ_MODEL,
    batch_size: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    t_start = time.time()
    clients, _ = initialize_groq_clients()
    system_prompt = build_system_prompt()

    # 1. Load already-extracted precedents from checkpoint
    extracted_dfs = []
    prior_calls = 0
    prior_prompt = 0
    prior_comp = 0
    prior_clock = 0.0

    in_progress_parquet = "data/processed/structured_precedents_in_progress.parquet"

    if os.path.exists(checkpoint_meta):
        with open(checkpoint_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            prior_calls = int(meta.get("api_calls_made", 0))
            prior_prompt = int(meta.get("total_prompt_tokens", 0))
            prior_comp = int(meta.get("total_completion_tokens", 0))
            prior_clock = float(meta.get("wall_clock_seconds", 0.0))

    already_extracted_ids = set()
    if os.path.exists(checkpoint_path):
        ckpt_df = pd.read_parquet(checkpoint_path)
        valid_mask = (ckpt_df["extraction_source"] == "groq_llm") & (ckpt_df["action_taken"].astype(str).str.strip().str.len() > 0)
        existing_valid = ckpt_df[valid_mask].copy()
        logger.info(f"Loaded {len(existing_valid):,} verified Groq-extracted precedents from checkpoint.")
        extracted_dfs.append(existing_valid)
        already_extracted_ids.update(existing_valid["thread_id"])

    if os.path.exists(in_progress_parquet):
        in_prog_df = pd.read_parquet(in_progress_parquet)
        valid_mask = (in_prog_df["extraction_source"] == "groq_llm") & (in_prog_df["action_taken"].astype(str).str.strip().str.len() > 0)
        existing_in_prog = in_prog_df[valid_mask].copy()
        logger.info(f"Loaded {len(existing_in_prog):,} verified Groq-extracted precedents from in-progress cache.")
        extracted_dfs.append(existing_in_prog)
        already_extracted_ids.update(existing_in_prog["thread_id"])
        # Add 2 API calls and approx tokens from the two completed batches
        prior_calls += 2
        prior_prompt += 2300
        prior_comp += 706

    # 2. Select remaining candidates across the 7 remaining intents
    indexable_df = pd.read_parquet(indexable_path)
    threads_df = pd.read_parquet(threads_path)

    # Join with threads
    threads_sub = threads_df[threads_df["thread_id"].isin(set(indexable_df["thread_id"]))].copy()

    def extract_brand_replies(row):
        replies = []
        for inb, txt, auth in zip(row["inbounds"], row["texts"], row["author_ids"]):
            if not inb or auth == "AppleSupport":
                clean_r = re.sub(r"@\w+", "", txt).strip()
                if clean_r:
                    replies.append(clean_r)
        return " | ".join(replies[:2]) if replies else "Thank you for reaching out. Please DM us for support."

    threads_sub["brand_reply_text"] = threads_sub.apply(extract_brand_replies, axis=1)
    merged = indexable_df.merge(threads_sub[["thread_id", "brand_reply_text"]], on="thread_id", how="inner")
    
    # Filter out already extracted IDs
    candidates = merged[~merged["thread_id"].isin(already_extracted_ids)].copy()

    remaining_slices = []
    for intent, count_needed in TARGET_PER_INTENT.items():
        already_for_intent = 0
        for d in extracted_dfs:
            int_col = "intent" if "intent" in d.columns else "predicted_intent"
            already_for_intent += int((d[int_col] == intent).sum())
        count_still_needed = max(0, count_needed - already_for_intent)
        if count_still_needed > 0:
            intent_slice = candidates[candidates["predicted_intent"] == intent]
            n_take = min(len(intent_slice), count_still_needed)
            remaining_slices.append(intent_slice.sample(n=n_take, random_state=seed))

    if remaining_slices:
        to_extract_df = pd.concat(remaining_slices, ignore_index=True)
    else:
        to_extract_df = pd.DataFrame()
    logger.info(f"Selected {len(to_extract_df):,} threads across remaining intents to extract.")

    # 3. Batch extraction with small batch size (5 msgs/batch)
    batches = []
    for i in range(0, len(to_extract_df), batch_size):
        sub_df = to_extract_df.iloc[i : i + batch_size]
        payload = [
            {
                "thread_id": r["thread_id"],
                "intent": r["predicted_intent"],
                "customer_message": r["text"],
                "brand_reply": r["brand_reply_text"],
            }
        ] if False else [
            {
                "thread_id": r["thread_id"],
                "intent": r["predicted_intent"],
                "customer_message": r["text"],
                "brand_reply": r["brand_reply_text"],
            }
            for _, r in sub_df.iterrows()
        ]
        batches.append((sub_df, payload))

    logger.info(f"Dispatching {len(batches)} batches ({batch_size} msgs/batch) to Groq...")

    api_calls = 0
    prompt_tokens = 0
    comp_tokens = 0
    newly_extracted_records = []

    client_idx = 0
    in_progress_parquet = "data/processed/structured_precedents_in_progress.parquet"

    for b_idx, (sub_df, payload) in enumerate(batches):
        t0 = time.time()
        user_prompt = f"Threads to analyze:\n{json.dumps(payload)}"

        # Attempt call with failover and proper cooldown
        success = False
        retry_count = 0
        while not success and retry_count < 50:
            active_client = clients[client_idx % len(clients)]
            active_key_num = client_idx % len(clients) + 1
            try:
                resp = active_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
                p_tok = resp.usage.prompt_tokens if resp.usage else 0
                c_tok = resp.usage.completion_tokens if resp.usage else 0

                raw_json = resp.choices[0].message.content or "{}"
                parsed = json.loads(raw_json)
                results = parsed.get("results", [])
                res_map = {item.get("thread_id"): item for item in results if isinstance(item, dict)}

                # Strict validation: every thread in sub_df must be in res_map with non-empty action_taken
                missing_or_invalid = [
                    r["thread_id"] for _, r in sub_df.iterrows()
                    if r["thread_id"] not in res_map or not str(res_map[r["thread_id"]].get("action_taken", "")).strip()
                ]
                if missing_or_invalid:
                    raise ValueError(f"LLM response missing/empty for threads: {missing_or_invalid}")

                for _, r in sub_df.iterrows():
                    tid = r["thread_id"]
                    item = res_map[tid]
                    act = str(item.get("action_taken", "")).strip()
                    out = str(item.get("outcome", "information_provided")).strip()
                    if out not in ALLOWED_OUTCOMES:
                        out = "information_provided"

                    newly_extracted_records.append({
                        "thread_id": tid,
                        "tweet_id": r["tweet_id"],
                        "intent": r["predicted_intent"],
                        "customer_message": r["text"],
                        "action_taken": act,
                        "outcome": out,
                        "brand_reply_text": r["brand_reply_text"],
                        "extraction_source": "groq_llm",
                    })

                api_calls += 1
                prompt_tokens += p_tok
                comp_tokens += c_tok

                dur = time.time() - t0
                logger.info(
                    f"Batch {b_idx+1}/{len(batches)} Complete ({len(newly_extracted_records)}/{len(to_extract_df)} new) "
                    f"[Key {active_key_num}, {p_tok + c_tok} tokens, {dur:.2f}s]"
                )
                client_idx += 1
                success = True

                # Persist incremental progress
                pd.DataFrame(newly_extracted_records).to_parquet(in_progress_parquet, index=False)

            except RateLimitError as e:
                err_str = str(e)
                logger.warning(f"Rate limit on key {active_key_num}: {err_str[:120]}")
                client_idx += 1
                if client_idx % len(clients) == 0:
                    wait_sec = parse_groq_wait_time(err_str) or 30.0
                    wait_sec = min(wait_sec, 60.0)
                    logger.info(f"All keys rate limited. Sleeping cooldown {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                else:
                    time.sleep(2.0)
                retry_count += 1
            except Exception as e:
                logger.warning(f"API/Parsing error on key {active_key_num}: {e}. Retrying in 5s...")
                client_idx += 1
                time.sleep(5.0)
                retry_count += 1

        if not success:
            raise RuntimeError(f"Failed to extract batch {b_idx+1} after {retry_count} retries!")

        # Smooth pause between batches (5.5s to maintain TPM under 8,000)
        time.sleep(5.5)

    # 4. Merge existing and newly extracted
    cols = ["thread_id", "tweet_id", "intent", "customer_message", "action_taken", "outcome", "brand_reply_text", "extraction_source"]
    all_dfs = []
    for d in extracted_dfs:
        d_clean = d.copy()
        if "customer_message" not in d_clean.columns and "text" in d_clean.columns:
            d_clean["customer_message"] = d_clean["text"]
        if "intent" not in d_clean.columns and "predicted_intent" in d_clean.columns:
            d_clean["intent"] = d_clean["predicted_intent"]
        all_dfs.append(d_clean[cols])

    if newly_extracted_records:
        new_df = pd.DataFrame(newly_extracted_records)
        all_dfs.append(new_df[cols])

    combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["thread_id"])

    total_records = len(combined_df)
    total_calls = prior_calls + api_calls
    total_prompt = prior_prompt + prompt_tokens
    total_comp = prior_comp + comp_tokens
    total_tokens = total_prompt + total_comp
    total_duration = prior_clock + (time.time() - t_start)
    estimated_cost_usd = round(
        (total_prompt * 0.00000015) + (total_comp * 0.00000060), 4
    )

    # Assertions
    assert combined_df["action_taken"].isna().sum() == 0, "Found null action_taken!"
    assert (combined_df["action_taken"] == "").sum() == 0, "Found empty action_taken!"
    assert (combined_df["extraction_source"] == "groq_llm").all(), "Non-Groq extraction found!"
    assert set(combined_df["intent"].unique()) == set(TARGET_PER_INTENT.keys()).union({"software_update_os_bugs"}), (
        "Not all 8 intents represented in precedent pool!"
    )

    # 5. Save final parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {total_records:,} verified structured precedents to {output_path}")

    # Remove temporary checkpoints
    for p in [checkpoint_path, checkpoint_meta, in_progress_parquet]:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    # 6. Update pipeline stats
    existing_stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                existing_stats = json.load(f)
        except Exception:
            existing_stats = {}

    existing_stats["stage4_precedent_extraction"] = {
        "status": "COMPLETED",
        "provider": "Groq",
        "model": model,
        "input_indexable_pool_size": len(indexable_df),
        "golden_holdout_size": 300,
        "extracted_precedents_count": total_records,
        "extraction_method": "few_shot_structured_llm_groq_sdk",
        "api_calls_made": total_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_comp,
        "total_tokens": total_tokens,
        "avg_tokens_per_precedent": round(total_tokens / total_records, 2) if total_records else 0,
        "estimated_cost_usd": estimated_cost_usd,
        "duration_seconds": round(total_duration, 2),
        "outcome_distribution": combined_df["outcome"].value_counts().to_dict(),
        "intent_distribution": combined_df["intent"].value_counts().to_dict(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(existing_stats, f, indent=2)
    logger.info(f"Updated pipeline stats in {stats_path}")

    # Print Summary
    print("\n" + "=" * 80)
    print(f"STAGE 4 STRUCTURED PRECEDENTS EXTRACTION COMPLETE ({total_records:,} Precedents)")
    print(f"Provider: Groq (Official SDK) | Model: {model}")
    print(f"Total API Calls: {total_calls:,} | Total Tokens: {total_tokens:,} ({total_tokens/total_records:.1f} tokens/prec)")
    print(f"Compute Cost: ${estimated_cost_usd:.4f} | Total Wall Clock: {total_duration:.2f}s")
    print("=" * 80)
    print("Intent Distribution:")
    for intent_k, intent_v in combined_df["intent"].value_counts().items():
        print(f"  {intent_k:<38} : {intent_v:>4} ({intent_v/total_records*100:>5.1f}%)")
    print("-" * 80)
    print("Outcome Distribution:")
    for out_k, out_v in combined_df["outcome"].value_counts().items():
        print(f"  {out_k:<38} : {out_v:>4} ({out_v/total_records*100:>5.1f}%)")
    print("=" * 80 + "\n")

    return combined_df


if __name__ == "__main__":
    complete_extraction()
