"""
Stage 6: Interactive Human Labeling & Blind Calibration Tool.

Provides both an interactive Terminal CLI and a Local Web Interface for:
1. Golden Set Labeling (200 holdout threads):
   - Shows full real multi-turn thread context from AppleSupport_threads.parquet.
   - Prompts for:
     * gold_intent (from the 8 canonical taxonomy options)
     * gold_decision (auto_handle or escalate)
     * gold_escalation_reason (one-line justification)
     * gold_reply_guidance (what a good reply should contain)
   - Appends immediately to disk checkpoint (data/processed/golden_eval_set_human.json)
     so nothing is lost if interrupted.
   - Shows live progress (e.g. "47/200 labeled") and resumes automatically.
   - Logs verifiable timestamp and elapsed seconds per item in
     data/processed/golden_labeling_session_log.jsonl.
2. Blind Judge-Calibration Scoring (40 candidate replies):
   - Shows customer message, retrieved precedents, and drafted reply.
   - Strictly blind: LLM judge scores are hidden.
   - Prompts for 1-5 score on each of the 4 axes + auditor notes.
   - Saves immediately to data/processed/judge_calibration_human.json.

Usage:
  python src/eval/labeling_tool.py --cli          # Interactive Terminal CLI (Recommended for fast keyboard entry)
  python src/eval/labeling_tool.py                # Local Web Interface on http://localhost:8000
  python src/eval/labeling_tool.py --calib --cli  # Terminal CLI for 40 Blind Calibration samples
  python src/eval/labeling_tool.py --port 8080    # Custom web port
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("labeling_tool")

BASE_DIR = Path(__file__).resolve().parents[2]
GOLDEN_PARQUET = BASE_DIR / "data" / "processed" / "golden_eval_set.parquet"
THREADS_PARQUET = BASE_DIR / "data" / "processed" / "AppleSupport_threads.parquet"
HUMAN_LABELS_JSON = BASE_DIR / "data" / "processed" / "golden_eval_set_human.json"
CALIB_LABELS_JSON = BASE_DIR / "data" / "processed" / "judge_calibration_human.json"
SESSION_LOG = BASE_DIR / "data" / "processed" / "golden_labeling_session_log.jsonl"
CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / "golden_agent_eval_checkpoint.json"

CANONICAL_INTENTS = [
    {
        "id": "battery_power_performance",
        "name": "Battery & Power Performance",
        "desc": "Abnormal battery drain, device overheating, sudden shutdowns, charging failures",
    },
    {
        "id": "software_update_os_bugs",
        "name": "Software Update & OS Bugs",
        "desc": "iOS/macOS update errors, boot loops, system freezing, app crashes",
    },
    {
        "id": "keyboard_text_autocorrect",
        "name": "Keyboard & Autocorrect",
        "desc": "Autocorrect glitches, iOS 11 'I' rendering bug, typing lag, box glyphs",
    },
    {
        "id": "apple_music_audio_playback",
        "name": "Apple Music & Audio Playback",
        "desc": "Music app crashes, missing offline library, playlist errors, car audio",
    },
    {
        "id": "orders_purchases_applecare",
        "name": "Orders, Purchases & AppleCare",
        "desc": "iPhone pre-orders, reservation errors, AppleCare warranty, billing, refunds",
    },
    {
        "id": "account_access_apple_id",
        "name": "Account Access & Apple ID",
        "desc": "Password resets, 2FA locks, activation lock, security questions, phishing",
    },
    {
        "id": "hardware_display_physical",
        "name": "Hardware, Display & Physical Damage",
        "desc": "Cracked screens, broken buttons, camera, water damage, vibrating haptics",
    },
    {
        "id": "international_multilingual_inquiries",
        "name": "International & Multilingual Support",
        "desc": "Inquiries written in Spanish, Portuguese, German, French, etc.",
    },
]

INTENT_CHOICES = {str(i + 1): item["id"] for i, item in enumerate(CANONICAL_INTENTS)}


def get_df_golden() -> pd.DataFrame:
    """Loads the 200 golden evaluation candidates."""
    if not GOLDEN_PARQUET.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_PARQUET}")
    return pd.read_parquet(GOLDEN_PARQUET)


def load_human_labels() -> Dict[str, Any]:
    """Loads existing human labels from disk."""
    if HUMAN_LABELS_JSON.exists():
        try:
            with open(HUMAN_LABELS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_human_label(thread_id: str, payload: Dict[str, Any]) -> None:
    """Saves a single human label entry immediately to JSON and session log."""
    labels = load_human_labels()
    payload["labeled_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    labels[thread_id] = payload

    # Write human JSON checkpoint immediately
    with open(HUMAN_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Append to verifiable session log with exact timestamp and duration
    with open(SESSION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "human_label_saved",
            "timestamp": payload["labeled_at"],
            "thread_id": thread_id,
            "gold_intent": payload.get("gold_intent"),
            "gold_decision": payload.get("gold_decision"),
            "elapsed_seconds": payload.get("elapsed_seconds", 0),
        }) + "\n")


def load_calibration_data() -> List[Dict[str, Any]]:
    """Loads 40 candidate evaluation items for blind scoring."""
    df_golden = get_df_golden()

    # Load agent drafted replies from checkpoint if available
    agent_replies = {}
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                chk = json.load(f)
                for r in chk:
                    agent_replies[r["thread_id"]] = r
        except Exception:
            pass

    records = []
    # Pick 40 candidates
    for idx in range(min(40, len(df_golden))):
        row = df_golden.iloc[idx]
        tid = row["thread_id"]
        chk_item = agent_replies.get(tid, {})
        draft = chk_item.get("drafted_reply", "We'd like to look into this. Please send us a DM with your device details: https://t.co/GDrqU22YpT")
        precedents = chk_item.get("retrieved_precedents", [
            {"action_taken": "Invite to DM", "brand_reply_text": "Send us a DM and we'll help get this sorted: https://t.co/GDrqU22YpT", "outcome": "DM initiated"}
        ])
        records.append({
            "idx": idx,
            "thread_id": tid,
            "customer_message": row["text"],
            "drafted_reply": draft,
            "retrieved_precedents": precedents,
        })
    return records


def load_calib_scores() -> Dict[str, Any]:
    if CALIB_LABELS_JSON.exists():
        try:
            with open(CALIB_LABELS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_calib_score(thread_id: str, payload: Dict[str, Any]) -> None:
    scores = load_calib_scores()
    payload["scored_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    scores[thread_id] = payload
    with open(CALIB_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


# ==============================================================================
# CLI Labeling Loops (Fast, Interactive Terminal Mode)
# ==============================================================================

def run_cli_golden_labeling():
    """Runs interactive terminal prompt loop for 200 golden examples."""
    df = get_df_golden()
    total = len(df)

    print("\n" + "=" * 80)
    print("  APPLE SUPPORT GOLDEN EVALUATION SET: HUMAN HAND-LABELING CLI")
    print(f"  Total Holdout Threads: {total}")
    print("  - Real multi-turn thread transcripts from AppleSupport_threads.parquet")
    print("  - Saves immediately after each example to: data/processed/golden_eval_set_human.json")
    print("  - Enter 'q' at any prompt to save and exit; resume anytime.")
    print("=" * 80 + "\n")

    while True:
        human_labels = load_human_labels()
        labeled_count = len(human_labels)

        # Find first unlabeled index
        next_idx = None
        for i in range(total):
            tid = df.iloc[i]["thread_id"]
            if tid not in human_labels:
                next_idx = i
                break

        if next_idx is None:
            print("\n🎉 ALL 200 GOLDEN EXAMPLES HAVE BEEN LABELED BY HUMAN!")
            print(f"Human labels stored at: {HUMAN_LABELS_JSON}")
            break

        row = df.iloc[next_idx]
        tid = row["thread_id"]
        tweet_id = str(row.get("tweet_id", ""))
        context = str(row.get("full_thread_context", f"[Customer]: {row['text']}"))

        print("\n" + "-" * 80)
        print(f"  ITEM [{labeled_count + 1}/{total}] | Thread ID: {tid} | Tweet: {tweet_id}")
        print(f"  Progress: {labeled_count}/{total} labeled ({round(labeled_count/total*100, 1)}%)")
        print("-" * 80)
        print("\nFULL CONVERSATION THREAD TRANSCRIPT:")
        for line in context.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Customer]:"):
                print(f"  \033[94m{line}\033[0m")
            elif "AppleSupport" in line:
                print(f"  \033[92m{line}\033[0m")
            else:
                print(f"  {line}")
        print()

        t0 = time.time()

        # 1. Intent selection
        print("SELECT CANONICAL INTENT:")
        for k, it_id in INTENT_CHOICES.items():
            it_name = next(x["name"] for x in CANONICAL_INTENTS if x["id"] == it_id)
            print(f"  [{k}] {it_name}")

        chosen_intent = None
        while chosen_intent is None:
            ans = input("Choice [1-8] (or 'q' to quit): ").strip()
            if ans.lower() == "q":
                print("\nExiting. All completed labels are safely persisted.")
                return
            if ans in INTENT_CHOICES:
                chosen_intent = INTENT_CHOICES[ans]
            else:
                print("Invalid selection. Please enter a number 1-8.")

        # 2. Decision selection
        print("\nSELECT ESCALATION DECISION:")
        print("  [a] auto_handle  (Safe for autonomous precedent/template reply)")
        print("  [e] escalate     (Requires human specialist, security/billing/repair channel)")
        chosen_decision = None
        while chosen_decision is None:
            ans = input("Decision [a/e] (or 'q' to quit): ").strip().lower()
            if ans == "q":
                print("\nExiting. All completed labels are safely persisted.")
                return
            if ans in ["a", "auto", "auto_handle", "1"]:
                chosen_decision = "auto_handle"
            elif ans in ["e", "esc", "escalate", "2"]:
                chosen_decision = "escalate"
            else:
                print("Invalid selection. Enter 'a' for auto_handle or 'e' for escalate.")

        # 3. Escalation Reason
        print("\nENTER ESCALATION REASON:")
        print("  (One-line rationale explaining why this should auto-handle or escalate)")
        reason = input("Reason: ").strip()
        if reason.lower() == "q":
            return
        if not reason:
            reason = f"Standard {chosen_decision} decision for {chosen_intent}"

        # 4. Good Reply Guidance
        print("\nENTER GOOD REPLY GUIDANCE:")
        print("  (Brief note on what a compliant, accurate, helpful reply must contain)")
        guidance = input("Guidance: ").strip()
        if guidance.lower() == "q":
            return
        if not guidance:
            guidance = f"Provide standard diagnostic troubleshooting for {chosen_intent}"

        elapsed = round(time.time() - t0, 1)

        payload = {
            "thread_id": tid,
            "gold_intent": chosen_intent,
            "gold_decision": chosen_decision,
            "gold_escalation_reason": reason,
            "gold_reply_guidance": guidance,
            "elapsed_seconds": elapsed,
            "labeler": "human",
        }

        save_human_label(tid, payload)
        print(f"\n✓ Saved Thread {tid} in {elapsed}s. ({labeled_count + 1}/{total} completed)\n")


def run_cli_blind_calibration():
    """Runs interactive terminal prompt loop for 40 blind judge-calibration samples."""
    items = load_calibration_data()
    total = len(items)

    print("\n" + "=" * 80)
    print("  BLIND JUDGE-CALIBRATION AUDIT: 40 SAMPLES")
    print("  - LLM judge scores are strictly hidden for unbiased blind scoring")
    print("  - Score each rubric axis on a 1-5 integer scale")
    print("  - Enter 'q' at any prompt to quit; resume anytime.")
    print("=" * 80 + "\n")

    while True:
        scores = load_calib_scores()
        scored_count = len(scores)

        next_idx = None
        for i, item in enumerate(items):
            if item["thread_id"] not in scores:
                next_idx = i
                break

        if next_idx is None:
            print("\n🎉 ALL 40 BLIND CALIBRATION ITEMS HAVE BEEN SCORED BY HUMAN!")
            print(f"Scores saved to: {CALIB_LABELS_JSON}")
            break

        item = items[next_idx]
        tid = item["thread_id"]

        print("\n" + "-" * 80)
        print(f"  CALIBRATION SAMPLE [{scored_count + 1}/{total}] | Thread ID: {tid}")
        print("-" * 80)
        print(f"\nCUSTOMER MESSAGE:\n  \"{item['customer_message']}\"\n")

        print("RETRIEVED PRECEDENTS:")
        for j, p in enumerate(item["retrieved_precedents"], 1):
            act = p.get("action_taken", "N/A")
            rep = p.get("brand_reply_text", "N/A")
            print(f"  Precedent #{j}: Action='{act}' | Reply='{rep}'")
        print()

        print("DRAFTED REPLY TO EVALUATE (BLIND):")
        print(f"  \033[93m\"{item['drafted_reply']}\"\033[0m\n")

        t0 = time.time()

        def prompt_score(name: str, desc: str) -> int:
            while True:
                val = input(f"{name} (1-5) [{desc}]: ").strip()
                if val.lower() == "q":
                    sys.exit(0)
                if val in ["1", "2", "3", "4", "5"]:
                    return int(val)
                print("Enter an integer from 1 to 5.")

        s_g = prompt_score("1. Grounded in Precedent", "1=Contradicts precedent, 5=Faithful to precedent")
        s_f = prompt_score("2. Factually Non-Hallucinatory", "Strict: 1=Fabricated specifics/URLs/versions, 5=Non-hallucinatory")
        s_t = prompt_score("3. Tone Appropriate", "1=Unprofessional, 5=Empathetic, < 280 chars, Apple brand voice")
        s_r = prompt_score("4. Resolves or Correctly Defers", "1=Unsafe auto-handle, 5=Proper resolution or safe escalation")

        notes = input("Auditor Notes / Disagreement details: ").strip()
        elapsed = round(time.time() - t0, 1)

        payload = {
            "thread_id": tid,
            "grounded_in_precedent": s_g,
            "factually_non_hallucinatory": s_f,
            "tone_appropriate": s_t,
            "resolves_or_correctly_defers": s_r,
            "notes": notes,
            "elapsed_seconds": elapsed,
            "labeler": "human",
        }

        save_calib_score(tid, payload)
        print(f"\n✓ Saved Calibration Score for {tid} ({scored_count + 1}/{total} completed)\n")


# ==============================================================================
# Web Interface (FastAPI + Modern HTML/CSS/JS)
# ==============================================================================

def create_web_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Apple Support Golden Labeling Tool")

    @app.get("/api/golden_stats")
    def api_golden_stats():
        df = get_df_golden()
        human = load_human_labels()
        total = len(df)
        labeled_count = len(human)
        return {
            "total": total,
            "labeled": labeled_count,
            "percent": round(labeled_count / total * 100, 1) if total else 0,
            "labeled_ids": list(human.keys()),
        }

    @app.get("/api/golden_item/{index}")
    def api_golden_item(index: int):
        df = get_df_golden()
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=404, detail="Index out of bounds")
        row = df.iloc[index]
        tid = row["thread_id"]
        human_labels = load_human_labels()
        existing_human = human_labels.get(tid, None)

        context_raw = str(row.get("full_thread_context", f"[Customer]: {row['text']}"))
        turns = []
        for line in context_raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Customer]:"):
                turns.append({"speaker": "Customer", "is_customer": True, "text": line.replace("[Customer]:", "").strip()})
            elif "AppleSupport" in line:
                turns.append({"speaker": "AppleSupport", "is_customer": False, "text": line[line.find("]:") + 2:].strip()})
            else:
                turns.append({"speaker": "Note", "is_customer": False, "text": line})

        return {
            "index": index,
            "total": len(df),
            "thread_id": tid,
            "tweet_id": str(row.get("tweet_id", "")),
            "start_time": str(row.get("start_time", "")),
            "customer_message": row["text"],
            "turns": turns,
            "historical_reply": str(row.get("historical_brand_reply", "")),
            "intents_list": CANONICAL_INTENTS,
            "human_label": existing_human,  # None if not labeled yet!
        }

    @app.post("/api/save_golden_item/{index}")
    async def api_save_golden(index: int, request: Request):
        df = get_df_golden()
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=404, detail="Index out of bounds")
        data = await request.json()
        tid = df.iloc[index]["thread_id"]
        save_human_label(tid, data)
        return {"status": "success", "thread_id": tid}

    @app.get("/api/calib_items")
    def api_calib_items():
        items = load_calibration_data()
        scores = load_calib_scores()
        for item in items:
            tid = item["thread_id"]
            item["human_score"] = scores.get(tid, None)
        return {
            "total": len(items),
            "scored_count": len(scores),
            "items": items,
        }

    @app.post("/api/save_calib_item/{index}")
    async def api_save_calib(index: int, request: Request):
        items = load_calibration_data()
        if index < 0 or index >= len(items):
            raise HTTPException(status_code=404, detail="Index out of bounds")
        data = await request.json()
        tid = items[index]["thread_id"]
        save_calib_score(tid, data)
        return {"status": "success", "thread_id": tid}

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        # Clean local HTML web UI
        return HTMLResponse(content=WEB_HTML)

    return app


WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Apple Support Golden Set Human Labeling Tool</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #0c0f14;
      --bg-card: #151a23;
      --bg-card-hover: #1c222e;
      --border-color: #262e3d;
      --text-main: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-dim: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --customer-bubble: #1e293b;
      --agent-bubble: #064e3b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background-color: var(--bg-main);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background-color: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand-title {
      font-weight: 700;
      font-size: 1.1rem;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 9999px;
      background: var(--accent-dim);
      color: var(--accent);
      border: 1px solid rgba(56,189,248,0.3);
      font-weight: 600;
    }
    .mode-switch {
      display: flex;
      background: var(--bg-main);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 3px;
      gap: 4px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }
    .mode-btn.active {
      background: var(--accent);
      color: #000;
    }
    .progress-box {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .progress-bar-container {
      width: 140px;
      height: 8px;
      background: var(--border-color);
      border-radius: 4px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: var(--success);
      width: 0%;
      transition: width 0.3s ease;
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      overflow: hidden;
    }
    .pane {
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .pane-left {
      border-right: 1px solid var(--border-color);
      background-color: #0e1218;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
    }
    .card-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .transcript-container {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .turn {
      padding: 12px 16px;
      border-radius: 10px;
      font-size: 0.92rem;
      line-height: 1.5;
      max-width: 92%;
    }
    .turn.customer {
      align-self: flex-start;
      background: var(--customer-bubble);
      border: 1px solid #334155;
    }
    .turn.agent {
      align-self: flex-end;
      background: var(--agent-bubble);
      border: 1px solid #047857;
    }
    .speaker-name {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
      color: #94a3b8;
    }
    .turn.agent .speaker-name { color: #6ee7b7; }
    .turn.customer .speaker-name { color: #38bdf8; }

    /* Form Styles */
    .intent-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .intent-card {
      background: #111620;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .intent-card:hover {
      border-color: var(--accent);
      background: var(--bg-card-hover);
    }
    .intent-card.selected {
      border-color: var(--accent);
      background: var(--accent-dim);
    }
    .intent-header {
      font-size: 0.85rem;
      font-weight: 600;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .intent-desc {
      font-size: 0.72rem;
      color: var(--text-muted);
      line-height: 1.3;
    }
    .decision-row {
      display: flex;
      gap: 16px;
      margin-top: 6px;
    }
    .decision-btn {
      flex: 1;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      background: #111620;
      color: var(--text-muted);
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }
    .decision-btn.selected.auto_handle {
      background: #064e3b;
      color: #6ee7b7;
      border-color: #10b981;
    }
    .decision-btn.selected.escalate {
      background: #7f1d1d;
      color: #fca5a5;
      border-color: #ef4444;
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .input-label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .text-input, .text-area {
      background: #0e1218;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      color: #fff;
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
      transition: border 0.15s;
    }
    .text-input:focus, .text-area:focus {
      border-color: var(--accent);
    }
    .text-area { resize: vertical; min-height: 70px; }
    footer {
      background-color: var(--bg-card);
      border-top: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nav-group {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .action-btn {
      padding: 10px 20px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border-color);
      background: #1e293b;
      color: #fff;
      transition: all 0.15s ease;
    }
    .action-btn:hover { background: #334155; }
    .action-btn.primary {
      background: var(--accent);
      color: #000;
      border-color: var(--accent);
    }
    .action-btn.primary:hover { background: #7dd3fc; }
    .status-badge {
      font-size: 0.8rem;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
    }
    .status-badge.unlabeled {
      background: rgba(245, 158, 11, 0.15);
      color: var(--warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-badge.labeled {
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand-title">
      <span> Apple Support Holdout Evaluation</span>
      <span class="badge">Human Labeling</span>
    </div>
    <div class="mode-switch">
      <button class="mode-btn active" id="btnModeGolden" onclick="switchMode('golden')">1. Golden Set (200)</button>
      <button class="mode-btn" id="btnModeCalib" onclick="switchMode('calib')">2. Blind Calibration (40)</button>
    </div>
    <div class="progress-box">
      <span id="progressText">0 / 200 (0.0%)</span>
      <div class="progress-bar-container">
        <div class="progress-fill" id="progressFill"></div>
      </div>
    </div>
  </header>

  <main id="mainContainer"></main>

  <footer>
    <div class="nav-group">
      <button class="action-btn" onclick="prevItem()">← Previous</button>
      <button class="action-btn" onclick="nextItem()">Next →</button>
      <button class="action-btn" onclick="jumpNextUnlabeled()">Next Unlabeled ⏭</button>
      <input type="number" id="jumpIndex" style="width: 70px; padding: 8px; background: #0c0f14; border: 1px solid var(--border-color); color: #fff; border-radius: 6px; text-align: center;" min="1" max="200" onchange="jumpTo(this.value - 1)">
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
      <span id="currentStatus" class="status-badge unlabeled">Unlabeled</span>
      <button class="action-btn primary" id="btnSave" onclick="saveCurrentItem()">Save & Next (Enter)</button>
    </div>
  </footer>

  <script>
    let currentMode = 'golden';
    let currentIndex = 0;
    let totalItems = 200;
    let currentItem = null;
    let itemStartTime = Date.now();
    let selectedIntent = null;
    let selectedDecision = null;

    async function init() {
      await updateStats();
      await loadItem(currentIndex);
    }

    async function updateStats() {
      if (currentMode === 'golden') {
        const res = await fetch('/api/golden_stats');
        const data = await res.json();
        totalItems = data.total;
        document.getElementById('progressText').innerText = `${data.labeled} / ${data.total} (${data.percent}%)`;
        document.getElementById('progressFill').style.width = `${data.percent}%`;
      } else {
        const res = await fetch('/api/calib_items');
        const data = await res.json();
        totalItems = data.total;
        const pct = Math.round(data.scored_count / data.total * 100);
        document.getElementById('progressText').innerText = `${data.scored_count} / ${data.total} (${pct}%)`;
        document.getElementById('progressFill').style.width = `${pct}%`;
      }
    }

    async function loadItem(index) {
      currentIndex = index;
      document.getElementById('jumpIndex').value = index + 1;
      itemStartTime = Date.now();

      if (currentMode === 'golden') {
        const res = await fetch(`/api/golden_item/${index}`);
        currentItem = await res.json();
        renderGoldenItem(currentItem);
      } else {
        const res = await fetch('/api/calib_items');
        const data = await res.json();
        currentItem = data.items[index];
        renderCalibItem(currentItem, index, data.total);
      }
    }

    function renderGoldenItem(item) {
      const isLabeled = item.human_label !== null;
      const statusBadge = document.getElementById('currentStatus');
      statusBadge.className = isLabeled ? 'status-badge labeled' : 'status-badge unlabeled';
      statusBadge.innerText = isLabeled ? '✓ Labeled by Human' : '⚠ Unlabeled (Awaiting Human)';

      selectedIntent = isLabeled ? item.human_label.gold_intent : null;
      selectedDecision = isLabeled ? item.human_label.gold_decision : null;

      const turnsHtml = item.turns.map(t => `
        <div class="turn ${t.is_customer ? 'customer' : 'agent'}">
          <div class="speaker-name">${t.speaker}</div>
          <div>${escapeHtml(t.text)}</div>
        </div>
      `).join('');

      const intentCardsHtml = item.intents_list.map((it, i) => `
        <div class="intent-card ${selectedIntent === it.id ? 'selected' : ''}" id="intent_${it.id}" onclick="selectIntent('${it.id}')">
          <div class="intent-header">
            <span style="opacity: 0.6;">[${i+1}]</span> ${it.name}
          </div>
          <div class="intent-desc">${it.desc}</div>
        </div>
      `).join('');

      const existingReason = isLabeled ? (item.human_label.gold_escalation_reason || '') : '';
      const existingGuidance = isLabeled ? (item.human_label.gold_reply_guidance || '') : '';

      document.getElementById('mainContainer').innerHTML = `
        <div class="pane pane-left">
          <div class="card">
            <div class="card-title">
              <span>Thread Metadata</span>
              <span class="badge">#${item.index + 1} of ${item.total}</span>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.6;">
              <div><strong>Thread ID:</strong> <code>${item.thread_id}</code></div>
              <div><strong>Initial Tweet ID:</strong> <code>${item.tweet_id}</code></div>
              <div><strong>Start Time:</strong> ${item.start_time || 'N/A'}</div>
            </div>
          </div>

          <div class="card" style="flex: 1;">
            <div class="card-title">
              <span>Full Real Conversation Context</span>
              <span style="font-size: 0.75rem; color: var(--text-muted);">${item.turns.length} Turn(s)</span>
            </div>
            <div class="transcript-container">
              ${turnsHtml}
            </div>
          </div>
        </div>

        <div class="pane pane-right">
          <div class="card">
            <div class="card-title">1. Canonical Intent Category (Choose 1)</div>
            <div class="intent-grid">
              ${intentCardsHtml}
            </div>
          </div>

          <div class="card">
            <div class="card-title">2. Escalation vs. Autonomous Handle Decision</div>
            <div class="decision-row">
              <div class="decision-btn auto_handle ${selectedDecision === 'auto_handle' ? 'selected' : ''}" id="dec_auto" onclick="selectDecision('auto_handle')">
                ✓ Auto-Handle (Autonomous Reply Safe)
              </div>
              <div class="decision-btn escalate ${selectedDecision === 'escalate' ? 'selected' : ''}" id="dec_esc" onclick="selectDecision('escalate')">
                ⚠ Escalate to Human / DM
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">3. Escalation Reason (One-line Justification)</div>
            <div class="input-group">
              <input type="text" id="escalationReason" class="text-input" placeholder="e.g. Mandatory policy escalation for billing or sensitive security inquiry" value="${escapeHtml(existingReason)}">
            </div>
          </div>

          <div class="card">
            <div class="card-title">4. Good Reply Guidance (What a good reply should contain)</div>
            <div class="input-group">
              <textarea id="replyGuidance" class="text-area" placeholder="Brief note on what a compliant, helpful reply must contain...">${escapeHtml(existingGuidance)}</textarea>
            </div>
          </div>
        </div>
      `;
    }

    function renderCalibItem(item, index, total) {
      const isScored = item.human_score !== null;
      const statusBadge = document.getElementById('currentStatus');
      statusBadge.className = isScored ? 'status-badge labeled' : 'status-badge unlabeled';
      statusBadge.innerText = isScored ? '✓ Blind Scored by Human' : '⚠ Unscored (Awaiting Human)';

      const existing = item.human_score || {};
      const g_val = existing.grounded_in_precedent || 3;
      const f_val = existing.factually_non_hallucinatory || 3;
      const t_val = existing.tone_appropriate || 4;
      const r_val = existing.resolves_or_correctly_defers || 3;
      const notes = existing.notes || '';

      const precHtml = item.retrieved_precedents.map((p, i) => `
        <div style="background: #111620; padding: 10px; border-radius: 6px; border: 1px solid var(--border-color); font-size: 0.82rem; margin-bottom: 8px;">
          <div><strong>Precedent #${i+1} Action:</strong> ${escapeHtml(p.action_taken || 'N/A')}</div>
          <div style="color: var(--text-muted); margin-top: 4px;">"${escapeHtml(p.brand_reply_text || 'N/A')}"</div>
        </div>
      `).join('');

      document.getElementById('mainContainer').innerHTML = `
        <div class="pane pane-left">
          <div class="card">
            <div class="card-title">
              <span>Candidate Context</span>
              <span class="badge">Calibration #${index + 1} of ${total}</span>
            </div>
            <div style="margin-bottom: 12px;"><strong>Customer Message:</strong></div>
            <div class="turn customer" style="max-width: 100%; margin-bottom: 16px;">
              ${escapeHtml(item.customer_message)}
            </div>
            <div style="margin-bottom: 8px;"><strong>Retrieved Precedents:</strong></div>
            ${precHtml}
          </div>

          <div class="card">
            <div class="card-title">Drafted Reply to Evaluate (Strictly Blind)</div>
            <div class="turn agent" style="max-width: 100%; font-size: 0.95rem;">
              "${escapeHtml(item.drafted_reply)}"
            </div>
            <div style="font-size: 0.75rem; color: var(--accent); margin-top: 8px;">
              🔒 Note: LLM Judge scores are hidden to ensure unbiased, blind human auditor grading.
            </div>
          </div>
        </div>

        <div class="pane pane-right">
          <div class="card">
            <div class="card-title">Rubric Evaluation (1 to 5 Scale)</div>
            <div style="display: flex; flex-direction: column; gap: 16px;">
              <div>
                <label class="input-label">1. Grounded in Precedent (1=Invented action, 5=Faithfully matches precedent)</label>
                <input type="range" id="score_g" min="1" max="5" value="${g_val}" oninput="this.nextElementSibling.value = this.value" style="width: 100%;">
                <output style="font-weight: 700; color: var(--accent);">${g_val}</output> / 5
              </div>
              <div>
                <label class="input-label">2. Factually Non-Hallucinatory (Strict: penalize invented versions or ungrounded URLs)</label>
                <input type="range" id="score_f" min="1" max="5" value="${f_val}" oninput="this.nextElementSibling.value = this.value" style="width: 100%;">
                <output style="font-weight: 700; color: var(--accent);">${f_val}</output> / 5
              </div>
              <div>
                <label class="input-label">3. Tone Appropriate (Polite, empathetic, &lt; 280 chars, Twitter brand voice)</label>
                <input type="range" id="score_t" min="1" max="5" value="${t_val}" oninput="this.nextElementSibling.value = this.value" style="width: 100%;">
                <output style="font-weight: 700; color: var(--accent);">${t_val}</output> / 5
              </div>
              <div>
                <label class="input-label">4. Resolves or Correctly Defers (Safe handling; does not auto-handle unsafe complaints)</label>
                <input type="range" id="score_r" min="1" max="5" value="${r_val}" oninput="this.nextElementSibling.value = this.value" style="width: 100%;">
                <output style="font-weight: 700; color: var(--accent);">${r_val}</output> / 5
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Auditor Notes & Hallucination Details</div>
            <textarea id="calibNotes" class="text-area" placeholder="Detail any ungrounded entities, invented version numbers, or tone discrepancies...">${escapeHtml(notes)}</textarea>
          </div>
        </div>
      `;
    }

    function selectIntent(intentId) {
      selectedIntent = intentId;
      document.querySelectorAll('.intent-card').forEach(c => c.classList.remove('selected'));
      const el = document.getElementById(`intent_${intentId}`);
      if (el) el.classList.add('selected');
    }

    function selectDecision(dec) {
      selectedDecision = dec;
      document.querySelectorAll('.decision-btn').forEach(b => b.classList.remove('selected'));
      const el = document.getElementById(dec === 'auto_handle' ? 'dec_auto' : 'dec_esc');
      if (el) el.classList.add('selected');
    }

    async function saveCurrentItem() {
      const elapsed = Math.round((Date.now() - itemStartTime) / 1000);

      if (currentMode === 'golden') {
        if (!selectedIntent) {
          alert('Please select a Gold Intent Category.');
          return;
        }
        if (!selectedDecision) {
          alert('Please select an Escalation Decision (Auto-Handle or Escalate).');
          return;
        }

        const reason = document.getElementById('escalationReason').value.trim();
        const guidance = document.getElementById('replyGuidance').value.trim();

        const payload = {
          thread_id: currentItem.thread_id,
          gold_intent: selectedIntent,
          gold_decision: selectedDecision,
          gold_escalation_reason: reason,
          gold_reply_guidance: guidance,
          elapsed_seconds: elapsed,
          labeler: 'human',
        };

        await fetch(`/api/save_golden_item/${currentIndex}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        await updateStats();
        if (currentIndex < totalItems - 1) {
          await loadItem(currentIndex + 1);
        } else {
          alert('All 200 golden examples have been labeled!');
        }
      } else {
        const payload = {
          thread_id: currentItem.thread_id,
          grounded_in_precedent: parseInt(document.getElementById('score_g').value),
          factually_non_hallucinatory: parseInt(document.getElementById('score_f').value),
          tone_appropriate: parseInt(document.getElementById('score_t').value),
          resolves_or_correctly_defers: parseInt(document.getElementById('score_r').value),
          notes: document.getElementById('calibNotes').value.trim(),
          elapsed_seconds: elapsed,
          labeler: 'human',
        };

        await fetch(`/api/save_calib_item/${currentIndex}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        await updateStats();
        if (currentIndex < totalItems - 1) {
          await loadItem(currentIndex + 1);
        } else {
          alert('All 40 blind calibration items scored!');
        }
      }
    }

    async function switchMode(mode) {
      currentMode = mode;
      document.getElementById('btnModeGolden').className = mode === 'golden' ? 'mode-btn active' : 'mode-btn';
      document.getElementById('btnModeCalib').className = mode === 'calib' ? 'mode-btn active' : 'mode-btn';
      currentIndex = 0;
      await updateStats();
      await loadItem(0);
    }

    async function prevItem() {
      if (currentIndex > 0) await loadItem(currentIndex - 1);
    }
    async function nextItem() {
      if (currentIndex < totalItems - 1) await loadItem(currentIndex + 1);
    }
    async function jumpTo(idx) {
      const i = parseInt(idx);
      if (i >= 0 && i < totalItems) await loadItem(i);
    }

    async function jumpNextUnlabeled() {
      if (currentMode === 'golden') {
        const res = await fetch('/api/golden_stats');
        const stats = await res.json();
        for (let i = 0; i < totalItems; i++) {
          const it_res = await fetch(`/api/golden_item/${i}`);
          const it = await it_res.json();
          if (!it.human_label) {
            await loadItem(i);
            return;
          }
        }
        alert('All 200 golden examples have been labeled!');
      } else {
        const res = await fetch('/api/calib_items');
        const data = await res.json();
        for (let i = 0; i < data.items.length; i++) {
          if (!data.items[i].human_score) {
            await loadItem(i);
            return;
          }
        }
        alert('All 40 calibration items have been scored!');
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
          saveCurrentItem();
        }
        return;
      }
      if (e.key === 'ArrowRight') nextItem();
      if (e.key === 'ArrowLeft') prevItem();
      if (e.key === 'Enter') saveCurrentItem();
      if (e.key >= '1' && e.key <= '8') {
        const idx = parseInt(e.key) - 1;
        if (idx < CANONICAL_INTENTS.length) selectIntent(CANONICAL_INTENTS[idx].id);
      }
      if (e.key.toLowerCase() === 'a') selectDecision('auto_handle');
      if (e.key.toLowerCase() === 'e') selectDecision('escalate');
    });

    init();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Apple Support Golden Labeling Tool")
    parser.add_argument("--cli", action="store_true", help="Launch interactive Terminal CLI mode")
    parser.add_argument("--calib", action="store_true", help="Launch 40 blind calibration mode (use with --cli)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server for web mode (default 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    args = parser.parse_args()

    if args.cli:
        if args.calib:
            run_cli_blind_calibration()
        else:
            run_cli_golden_labeling()
    else:
        import uvicorn
        print("\n" + "=" * 70)
        print("  APPLE SUPPORT GOLDEN EVALUATION LABELING WEB INTERFACE")
        print("=" * 70)
        print(f"\n  Serving interactive UI at: http://{args.host}:{args.port}")
        print("  - Mode 1: 200 Golden Set Human Labeling with full multi-turn thread context")
        print("  - Mode 2: 40 Blind Judge Calibration Scoring (Strictly blind)")
        print("  - Instant persistence to data/processed/golden_eval_set_human.json")
        print("  - Real-time audit log in data/processed/golden_labeling_session_log.jsonl")
        print("\n  Press Ctrl+C in terminal to stop server.\n" + "=" * 70 + "\n")
        app = create_web_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
