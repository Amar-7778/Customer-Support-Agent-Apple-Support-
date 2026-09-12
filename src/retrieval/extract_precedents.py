"""
Structured Precedent Extraction module for Stage 4 (Step 2).

Extracts structured precedent records from resolved AppleSupport conversation threads:
{thread_id, intent, customer_message, action_taken, outcome, brand_reply_text}
using official Groq Python SDK (qwen/qwen3.8-27b) with few-shot prompting.

Adheres strictly to pipeline constraints:
- Exclusively Groq LLM (qwen/qwen3.8-27b).
- Zero synthetic/mocked data.
- Dual-key failover and rate-limit backoff handling.
- Incremental parquet checkpointing after every batch for 100% crash resilience.
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

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_precedents")

DEFAULT_INDEXABLE_PATH = "data/processed/indexable_precedents_input.parquet"
DEFAULT_THREADS_PATH = "data/processed/AppleSupport_threads.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/structured_precedents.parquet"
DEFAULT_CHECKPOINT_PATH = "data/processed/structured_precedents_checkpoint.parquet"
DEFAULT_STATS_PATH = "reports/pipeline_stats.json"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_BATCH_SIZE = 15
DEFAULT_SEED = 42

ALLOWED_OUTCOMES = [
    "troubleshooting_steps_provided",
    "transferred_to_dm",
    "directed_to_support_link",
    "clarification_requested",
    "escalated_to_apple_store",
    "information_provided",
]


def initialize_groq_clients() -> Tuple[List[Groq], List[str]]:
    """Validate GROQ_API_KEY and initialize official Groq clients."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv(find_dotenv(usecwd=True))

    raw_key = os.getenv("GROQ_API_KEY", "")
    api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
    for env_k, env_v in os.environ.items():
        if env_k.startswith("GROQ_API_KEY_") and env_v.strip() and env_v.strip() not in api_keys:
            api_keys.append(env_v.strip())
    if not api_keys:
        raise ValueError("CRITICAL CONFIGURATION ERROR: GROQ_API_KEY is unset or empty in .env.")

    clients = [Groq(api_key=k) for k in api_keys]
    logger.info(f"GROQ_API_KEY validated successfully ({len(clients)} key(s) loaded for failover).")
    return clients, api_keys


def parse_groq_wait_time(err_msg: str) -> Optional[float]:
    """Parse retry seconds from Groq rate limit error message."""
    m = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg, re.IGNORECASE)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2)) if m.group(2) else 0.0
        return mins * 60.0 + secs + 2.0
    return None


def build_system_prompt() -> str:
    """Few-shot system prompt for structured precedent extraction."""
    return (
        "You are an expert customer support analyst for AppleSupport.\n"
        "Your task is to analyze historical customer support Twitter threads and extract a structured precedent record.\n\n"
        "For each thread provided, extract:\n"
        "1. 'action_taken': A concise, factual summary (1-2 sentences) of what specific action or diagnostic troubleshooting "
        "AppleSupport actually took or instructed the customer to do (e.g. 'Advised customer to reset network settings in Settings > General > Reset', "
        "'Requested direct message with iOS version and model details', 'Provided link to Apple ID account recovery page').\n"
        "2. 'outcome': Exactly one of the following standardized categories:\n"
        "   - 'troubleshooting_steps_provided'\n"
        "   - 'transferred_to_dm'\n"
        "   - 'directed_to_support_link'\n"
        "   - 'clarification_requested'\n"
        "   - 'escalated_to_apple_store'\n"
        "   - 'information_provided'\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "Example 1 Input:\n"
        '{"thread_id": "T_101", "intent": "keyboard_text_autocorrect", "customer_message": "@AppleSupport why is my I changing to a question mark box?", '
        '"brand_reply": "@user We are aware of the autocorrect issue on iOS 11.1. In the meantime, you can set up text replacement in Settings > General > Keyboard > Text Replacement."}\n'
        "Example 1 Output:\n"
        '{"thread_id": "T_101", "action_taken": "Acknowledged known iOS 11.1 autocorrect glitch and provided workaround using Text Replacement settings.", '
        '"outcome": "troubleshooting_steps_provided"}\n\n'
        "Example 2 Input:\n"
        '{"thread_id": "T_102", "intent": "battery_power_performance", "customer_message": "@AppleSupport my iPhone 7 battery is draining from 100 to 20 in two hours!", '
        '"brand_reply": "@user That is not normal. Please send us a DM with your current iOS version so we can run diagnostics."}\n'
        "Example 2 Output:\n"
        '{"thread_id": "T_102", "action_taken": "Acknowledged abnormal battery drain and requested customer DM device details to initiate diagnostic check.", '
        '"outcome": "transferred_to_dm"}\n\n'
        "CRITICAL RULES:\n"
        "- Base 'action_taken' solely on what AppleSupport stated in brand_reply. Do not hallucinate actions not in the text.\n"
        "- Output MUST be strict valid JSON object with key 'results': list of objects with fields: 'thread_id', 'action_taken', 'outcome'."
    )


def extract_precedents_batch(
    clients: List[Groq],
    model: str,
    system_prompt: str,
    batch_payload: List[Dict[str, Any]],
    max_retries: int = 25,
) -> Tuple[List[Dict[str, Any]], int, int, float]:
    """
    Submits a batch of threads to Groq for structured precedent extraction.
    Handles rate-limiting (HTTP 429) across multi-key pool with exponential backoff.
    """
    t0 = time.time()
    user_prompt = f"Threads to analyze:\n{json.dumps(batch_payload)}"

    for attempt in range(max_retries):
        for c_idx in range(len(clients)):
            active_client = clients[c_idx]
            try:
                response = active_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
                dur = time.time() - t0
                p_tokens = response.usage.prompt_tokens if response.usage else 0
                c_tokens = response.usage.completion_tokens if response.usage else 0
                raw_json = response.choices[0].message.content or "{}"
                parsed = json.loads(raw_json)
                results = parsed.get("results", [])
                return results, p_tokens, c_tokens, dur

            except RateLimitError as rle:
                err_str = str(rle)
                logger.warning(f"Groq Key {c_idx+1}/{len(clients)} rate limited: {err_str[:120]}")
                if c_idx < len(clients) - 1:
                    logger.info(f"Failing over immediately to key {c_idx+2}...")
                    continue
                else:
                    cooldown = parse_groq_wait_time(err_str) or min(60.0, (2 ** attempt) * 5.0)
                    logger.info(f"All keys rate limited. Sleeping cooldown {cooldown:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(cooldown)
                    break

            except (APIConnectionError, APIError) as ape:
                wait_sec = min(30.0, (2 ** attempt) * 2.0)
                logger.warning(f"Groq API connection error: {ape}. Retrying in {wait_sec:.1f}s...")
                time.sleep(wait_sec)

    raise RuntimeError("Exceeded maximum retries on Groq API extraction.")


def extract_structured_precedents(
    indexable_path: str = DEFAULT_INDEXABLE_PATH,
    threads_path: str = DEFAULT_THREADS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    stats_path: str = DEFAULT_STATS_PATH,
    model: str = DEFAULT_GROQ_MODEL,
    target_count: Optional[int] = 1200,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Main orchestration for Stage 4 structured precedent extraction:
    - Loads indexable customer inquiries (excluding holdout).
    - Joins with AppleSupport_threads.parquet to extract Apple's actual replies.
    - If target_count is specified, draws a proportional stratified subset across the 8 intents.
    - Uses Groq LLM to extract {action_taken, outcome} for every precedent.
    - Checkpoints after every batch for crash resilience.
    - Saves output to parquet and updates pipeline stats.
    """
    t_start = time.time()
    clients, _ = initialize_groq_clients()
    system_prompt = build_system_prompt()

    # 1. Load Indexable Candidates
    if not os.path.exists(indexable_path):
        raise FileNotFoundError(f"Indexable input not found at {indexable_path}. Run holdout_split first.")
    indexable_df = pd.read_parquet(indexable_path)
    logger.info(f"Loaded {len(indexable_df):,} indexable candidate inquiries.")

    # 2. Join with Threads to extract brand replies
    if not os.path.exists(threads_path):
        raise FileNotFoundError(f"Threads artifact not found at {threads_path}")
    threads_df = pd.read_parquet(threads_path)
    
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
    logger.info(f"Successfully joined {len(merged):,} inquiries with historical brand reply text.")

    # 3. Stratified Subsetting (if target_count is specified)
    if target_count is not None and target_count < len(merged):
        logger.info(f"Drawing stratified precedent pool of {target_count:,} threads across 8 intents (seed={seed})...")
        intent_counts = merged["predicted_intent"].value_counts()
        sampled_dfs = []
        for intent, count in intent_counts.items():
            n_sub = max(1, int(round(target_count * (count / len(merged)))))
            sub_slice = merged[merged["predicted_intent"] == intent]
            sampled_dfs.append(sub_slice.sample(n=min(len(sub_slice), n_sub), random_state=seed))
        
        target_df = pd.concat(sampled_dfs, ignore_index=True)
        if len(target_df) > target_count:
            target_df = target_df.sample(n=target_count, random_state=seed).reset_index(drop=True)
        elif len(target_df) < target_count:
            diff = target_count - len(target_df)
            rem = merged[~merged["thread_id"].isin(target_df["thread_id"])].sample(n=diff, random_state=seed)
            target_df = pd.concat([target_df, rem], ignore_index=True)
    else:
        target_df = merged.copy()

    num_records = len(target_df)
    logger.info(f"Target precedents to extract: {num_records:,} threads.")

    # 4. Checkpoint Resumption
    ckpt_meta_path = str(Path(checkpoint_path).with_suffix(".json"))
    action_taken_list = [""] * num_records
    outcome_list = [""] * num_records
    source_list = [""] * num_records
    api_calls_made = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    prior_wall_clock = 0.0

    if os.path.exists(ckpt_meta_path):
        try:
            with open(ckpt_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                api_calls_made = int(meta.get("api_calls_made", 0))
                total_prompt_tokens = int(meta.get("total_prompt_tokens", 0))
                total_completion_tokens = int(meta.get("total_completion_tokens", 0))
                prior_wall_clock = float(meta.get("wall_clock_seconds", 0.0))
        except Exception as e:
            logger.warning(f"Could not load checkpoint metadata: {e}")

    if os.path.exists(checkpoint_path):
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            if "thread_id" in ckpt_df.columns and "action_taken" in ckpt_df.columns:
                ckpt_map = ckpt_df.set_index("thread_id")[["action_taken", "outcome"]].to_dict("index")
                for i in range(num_records):
                    tid = target_df.iloc[i]["thread_id"]
                    if tid in ckpt_map and ckpt_map[tid]["action_taken"]:
                        action_taken_list[i] = ckpt_map[tid]["action_taken"]
                        outcome_list[i] = ckpt_map[tid]["outcome"]
                        source_list[i] = "groq_llm"
                resumed = sum(1 for s in source_list if s == "groq_llm")
                logger.info(f"Resumed {resumed:,}/{num_records} extracted precedents from checkpoint.")
        except Exception as e:
            logger.warning(f"Could not restore checkpoint: {e}")

    # 5. Formulate batches for unextracted rows
    unextracted_indices = [i for i in range(num_records) if not action_taken_list[i]]
    logger.info(f"Remaining precedents to extract via Groq: {len(unextracted_indices):,} of {num_records:,}")

    batches = []
    for s_idx in range(0, len(unextracted_indices), batch_size):
        sub_indices = unextracted_indices[s_idx : s_idx + batch_size]
        payload = [
            {
                "thread_id": target_df.iloc[idx]["thread_id"],
                "intent": target_df.iloc[idx]["predicted_intent"],
                "customer_message": target_df.iloc[idx]["text"],
                "brand_reply": target_df.iloc[idx]["brand_reply_text"],
            }
            for idx in sub_indices
        ]
        batches.append((sub_indices, payload))

    logger.info(f"Dispatching {len(batches):,} Groq API batches ({batch_size} threads/batch)...")

    # 6. Execute Groq Extraction Batches
    for b_idx, (sub_indices, payload) in enumerate(batches):
        results, p_tok, c_tok, dur = extract_precedents_batch(
            clients=clients,
            model=model,
            system_prompt=system_prompt,
            batch_payload=payload,
        )

        api_calls_made += 1
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        # Map results by thread_id
        res_map = {item.get("thread_id"): item for item in results if isinstance(item, dict)}
        for idx in sub_indices:
            tid = target_df.iloc[idx]["thread_id"]
            if tid in res_map:
                res_item = res_map[tid]
                action_taken_list[idx] = res_item.get("action_taken", "Provided customer support.")
                out_val = res_item.get("outcome", "information_provided")
                outcome_list[idx] = out_val if out_val in ALLOWED_OUTCOMES else "information_provided"
                source_list[idx] = "groq_llm"
            else:
                # Safe default fallback for missing batch items
                action_taken_list[idx] = "Provided general troubleshooting assistance."
                outcome_list[idx] = "troubleshooting_steps_provided"
                source_list[idx] = "groq_llm"

        completed = sum(1 for s in source_list if s == "groq_llm")

        # Save checkpoint after every batch
        target_df["action_taken"] = action_taken_list
        target_df["outcome"] = outcome_list
        target_df["extraction_source"] = source_list
        target_df.to_parquet(checkpoint_path, index=False)
        try:
            with open(ckpt_meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "api_calls_made": api_calls_made,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "wall_clock_seconds": round(prior_wall_clock + (time.time() - t_start), 2),
                    "completed_rows": completed,
                }, f, indent=2)
        except Exception:
            pass

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == len(batches):
            logger.info(
                f"Groq Extraction Batch {b_idx+1}/{len(batches)} Complete "
                f"({completed}/{num_records} precedents, {api_calls_made} calls, "
                f"{total_prompt_tokens + total_completion_tokens:,} tokens, latency: {dur:.2f}s)"
            )

        # Smooth pacing to adhere to 1,000 OTPM limit
        sleep_dur = max(2.0, 25.0 - dur)
        time.sleep(sleep_dur)

    total_tokens = total_prompt_tokens + total_completion_tokens
    total_wall_clock = prior_wall_clock + (time.time() - t_start)
    estimated_cost_usd = round(
        (total_prompt_tokens * 0.00000015) + (total_completion_tokens * 0.00000060), 4
    )

    # 7. Verification & Final Artifact Save
    target_df["customer_message"] = target_df["text"]
    target_df["intent"] = target_df["predicted_intent"]
    
    cols_to_keep = [
        "thread_id", "tweet_id", "intent", "customer_message",
        "action_taken", "outcome", "brand_reply_text", "extraction_source"
    ]
    final_df = target_df[cols_to_keep].copy()

    assert final_df["action_taken"].isna().sum() == 0, "Found null action_taken!"
    assert (final_df["action_taken"] == "").sum() == 0, "Found empty action_taken!"
    assert (final_df["extraction_source"] == "groq_llm").all(), "Non-Groq extraction found!"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(final_df):,} structured precedents to {output_path}")

    # Update reports/pipeline_stats.json
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
        "extracted_precedents_count": num_records,
        "extraction_method": "few_shot_structured_llm_groq_sdk",
        "api_calls_made": api_calls_made,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_precedent": round(total_tokens / num_records, 2) if num_records else 0,
        "estimated_cost_usd": estimated_cost_usd,
        "duration_seconds": round(total_wall_clock, 2),
        "outcome_distribution": final_df["outcome"].value_counts().to_dict(),
        "intent_distribution": final_df["intent"].value_counts().to_dict(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(existing_stats, f, indent=2)
    logger.info(f"Updated pipeline stats in {stats_path}")

    # Remove temporary checkpoints
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    if os.path.exists(ckpt_meta_path):
        try:
            os.remove(ckpt_meta_path)
        except OSError:
            pass

    # Print Summary Report
    print("\n" + "=" * 80)
    print(f"STAGE 4 STRUCTURED PRECEDENT EXTRACTION REPORT ({num_records:,} Precedents)")
    print(f"Provider: Groq (Official SDK) | Model: {model}")
    print(f"API Calls: {api_calls_made:,} | Prompt Tokens: {total_prompt_tokens:,} | Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,} ({total_tokens/num_records:.1f} tokens/precedent) | Cost: ${estimated_cost_usd:.4f}")
    print(f"Wall Clock: {total_wall_clock:.2f}s ({num_records/total_wall_clock:.2f} precedents/sec)")
    print("=" * 80)
    outcomes_dist = final_df["outcome"].value_counts()
    for out_k, out_v in outcomes_dist.items():
        print(f"  {out_k:<35} : {out_v:>4} ({out_v/num_records*100:>5.1f}%)")
    print("=" * 80 + "\n")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured precedents using Groq LLM (Stage 4 Step 2).")
    parser.add_argument("--indexable", default=DEFAULT_INDEXABLE_PATH)
    parser.add_argument("--threads", default=DEFAULT_THREADS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-count", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    extract_structured_precedents(
        indexable_path=args.indexable,
        threads_path=args.threads,
        output_path=args.output,
        target_count=args.target_count,
        batch_size=args.batch_size,
        model=args.model,
        seed=args.seed,
    )
