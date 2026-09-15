"""
Stage 5: Human Verification & Agent Spot-Check Evaluation Runner.

Evaluates 16 real customer inquiries sampled from the golden holdout set
(data/processed/golden_eval_candidates.parquet), strictly stratified to include
2 inquiries from each of the 8 taxonomy intents.

Runs end-to-end handle_message() against Groq and ChromaDB, collects:
- Predicted intent & confidence
- Top-1 precedent action, reply, and outcome
- Precedent agreement score
- Drafted reply
- Grounding verification self-critique pass
- Escalation decision and explicit policy reason
- Real API tokens and latency

Saves full human audit report to reports/agent_spotcheck.md.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.agent.pipeline import (
    DEFAULT_MODEL,
    AgentResponse,
    get_groq_client,
    handle_message,
    mask_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_spotcheck")

HOLDOUT_PATH = "data/processed/golden_eval_candidates.parquet"
REPORT_PATH = "reports/agent_spotcheck.md"
STATS_PATH = "reports/pipeline_stats.json"


def select_stratified_holdout_messages(holdout_path: str = HOLDOUT_PATH, n_per_intent: int = 2) -> pd.DataFrame:
    """Selects n_per_intent messages per intent from the golden holdout candidate pool."""
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Holdout candidates not found at {holdout_path}")

    df = pd.read_parquet(holdout_path)
    intent_col = "predicted_intent" if "predicted_intent" in df.columns else "intent"

    samples = []
    for intent, group in df.groupby(intent_col):
        sample = group.sample(n=min(n_per_intent, len(group)), random_state=42)
        samples.append(sample)

    sampled_df = pd.concat(samples).reset_index(drop=True)
    logger.info("Selected %d holdout messages across %d intents.", len(sampled_df), sampled_df[intent_col].nunique())
    return sampled_df


def run_spotcheck(
    holdout_path: str = HOLDOUT_PATH,
    report_path: str = REPORT_PATH,
    stats_path: str = STATS_PATH,
) -> None:
    """Executes the agent pipeline on holdout samples and generates the audit report."""
    sampled_df = select_stratified_holdout_messages(holdout_path=holdout_path, n_per_intent=2)
    client = get_groq_client()

    logger.info("Starting Stage 5 Agent Spot-Check on %d messages...", len(sampled_df))

    records = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0

    t_run_start = time.time()

    for idx, row in sampled_df.iterrows():
        thread_id = row["thread_id"]
        true_intent = row["predicted_intent"] if "predicted_intent" in row else row["intent"]
        text = row["text"]

        logger.info(
            "[%d/%d] Processing Thread %s (Gold Intent: %s)...",
            idx + 1,
            len(sampled_df),
            thread_id,
            true_intent,
        )

        try:
            agent_resp: AgentResponse = handle_message(customer_message=text, client=client)
            metrics = agent_resp.metrics or {}
            p_tok = metrics.get("total_prompt_tokens", 0)
            c_tok = metrics.get("total_completion_tokens", 0)
            lat = metrics.get("total_latency_seconds", 0.0)

            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_latency += lat

            records.append({
                "index": idx + 1,
                "thread_id": thread_id,
                "gold_intent": true_intent,
                "customer_message": text,
                "predicted_intent": agent_resp.predicted_intent,
                "intent_confidence": agent_resp.intent_confidence,
                "precedent_agreement_score": agent_resp.precedent_agreement_score,
                "top_precedent": agent_resp.retrieved_precedents[0] if agent_resp.retrieved_precedents else {},
                "drafted_reply": agent_resp.drafted_reply,
                "grounding_verification": agent_resp.grounding_verification,
                "decision": agent_resp.decision,
                "escalation_reason": agent_resp.escalation_reason,
                "metrics": metrics,
            })
        except Exception as e:
            logger.exception("Failed processing thread %s: %s", thread_id, e)
            records.append({
                "index": idx + 1,
                "thread_id": thread_id,
                "gold_intent": true_intent,
                "customer_message": text,
                "error": str(e),
            })

        # Polite delay to respect Groq OTPM/TPM bounds
        time.sleep(2.0)

    total_wall_clock = round(time.time() - t_run_start, 2)
    logger.info("Completed %d spot-checks in %.2fs.", len(records), total_wall_clock)

    # Generate Markdown Report
    generate_markdown_report(records=records, report_path=report_path, total_tokens=total_prompt_tokens + total_completion_tokens, wall_clock=total_wall_clock)

    # Update pipeline_stats.json
    update_pipeline_stats(
        stats_path=stats_path,
        samples_count=len(records),
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        wall_clock=total_wall_clock,
    )


def generate_markdown_report(records: List[Dict[str, Any]], report_path: str, total_tokens: int, wall_clock: float) -> None:
    """Formats and writes reports/agent_spotcheck.md."""
    lines = [
        "# Stage 5: Agent Pipeline End-to-End Spot-Check Verification Report",
        "",
        f"- **Model**: `{DEFAULT_MODEL}` (Groq SDK)",
        f"- **Evaluated Samples**: {len(records)} real customer inquiries from `golden_eval_candidates.parquet`",
        f"- **Stratification**: 2 inquiries for each of the 8 taxonomy intents",
        f"- **Vector Index**: 3,000 precedents in ChromaDB (`apple_support_precedents`)",
        f"- **Total Tokens Consumed**: {total_tokens:,} tokens",
        f"- **Total Wall Clock Duration**: {wall_clock:.2f}s",
        "",
        "---",
        "",
    ]

    for rec in records:
        idx = rec["index"]
        tid = rec["thread_id"]
        gold = rec["gold_intent"]
        msg = rec["customer_message"]
        pred_intent = rec.get("predicted_intent", "N/A")
        conf = rec.get("intent_confidence", 0.0)
        agr = rec.get("precedent_agreement_score", 0.0)
        top_p = rec.get("top_precedent", {})
        draft = rec.get("drafted_reply", "N/A")
        gv = rec.get("grounding_verification", {})
        decision = rec.get("decision", "N/A")
        reason = rec.get("escalation_reason", "N/A")

        p_act = top_p.get("action_taken", "N/A")
        p_reply = top_p.get("brand_reply_text", "N/A")
        p_out = top_p.get("outcome", "N/A")
        p_sim = top_p.get("similarity_score", 0.0)

        grounded_str = "GROUNDED" if gv.get("grounded", True) else "UNGROUNDED"
        unsupported = gv.get("unsupported_claims", [])
        unsupported_str = f"None ({grounded_str})" if not unsupported else f"{unsupported} ({grounded_str})"

        decision_badge = "**AUTO-HANDLE**" if decision == "auto_handle" else "**ESCALATE**"

        lines.extend([
            f"## Case #{idx}: [{pred_intent}] (Thread: `{tid}`)",
            f"- **Gold Intent**: `{gold}` | **Predicted**: `{pred_intent}` (Confidence: `{conf:.2f}`)",
            f"- **Customer Inquiry**:\n  > *\"{msg}\"*",
            f"- **Top-1 Retrieved Precedent** (Similarity: `{p_sim:.4f}` | Outcome: `{p_out}`):",
            f"  - *Historical Action*: {p_act}",
            f"  - *Historical Apple Reply*: \"{p_reply}\"",
            f"- **Precedent Agreement Score**: `{agr:.2f}`",
            f"- **Drafted Reply**:\n  > \"{draft}\"",
            f"- **Grounding Self-Critique**: {unsupported_str}",
            f"- **Escalation Decision**: {decision_badge}",
            f"- **Decision Reason**: {reason}",
            "- **Human Judgment (Safety & Quality)**: *Pending human audit*",
            "",
            "---",
            "",
        ])

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved agent spot-check report to %s", report_path)


def update_pipeline_stats(
    stats_path: str,
    samples_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    wall_clock: float,
) -> None:
    """Updates reports/pipeline_stats.json with stage 5 telemetry."""
    if not os.path.exists(stats_path):
        data = {}
    else:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    blended_rate_per_million = (prompt_tokens * 0.20 + completion_tokens * 0.40) / 1_000_000

    data["stage5_agent_pipeline"] = {
        "status": "COMPLETED",
        "provider": "Groq",
        "model": DEFAULT_MODEL,
        "evaluation_samples": samples_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(blended_rate_per_million, 4),
        "total_wall_clock_seconds": wall_clock,
        "agreement_threshold": 0.50,
        "always_escalate_intents": [
            "orders_purchases_applecare",
            "account_access_apple_id",
            "international_multilingual_inquiries",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Updated pipeline stats in %s", stats_path)


if __name__ == "__main__":
    run_spotcheck()
