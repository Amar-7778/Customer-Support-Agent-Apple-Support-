"""
Extracts a curated set of real customer inquiries from data/processed/golden_eval_candidates.parquet
for the Apple Support Agent demo frontend.
"""

import json
from pathlib import Path
import pandas as pd


def export_demo_examples():
    workspace = Path(__file__).resolve().parents[2]
    parquet_path = workspace / "data" / "processed" / "golden_eval_candidates.parquet"
    out_dir = workspace / "frontend" / "src" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "examples.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # 8 canonical intents with human-readable labels and descriptions
    intent_metadata = {
        "battery_power_performance": {
            "label": "Battery & Power",
            "tag": "Hardware & OS",
            "expected_decision": "escalate / auto_handle",
        },
        "software_update_os_bugs": {
            "label": "Software Update & OS Bugs",
            "tag": "iOS 11",
            "expected_decision": "auto_handle",
        },
        "apple_music_audio_playback": {
            "label": "Music & Audio Playback",
            "tag": "Media & CarPlay",
            "expected_decision": "auto_handle",
        },
        "keyboard_text_autocorrect": {
            "label": "Keyboard & Autocorrect",
            "tag": "Input & Typing",
            "expected_decision": "auto_handle",
        },
        "hardware_display_physical": {
            "label": "Hardware & Display Issues",
            "tag": "Physical Repairs",
            "expected_decision": "escalate",
        },
        "account_access_apple_id": {
            "label": "Account Access & Apple ID",
            "tag": "Security Critical",
            "expected_decision": "escalate (Policy Restricted)",
        },
        "orders_purchases_applecare": {
            "label": "Orders & AppleCare Billing",
            "tag": "Financial Critical",
            "expected_decision": "escalate (Policy Restricted)",
        },
        "international_multilingual_inquiries": {
            "label": "International & Multilingual",
            "tag": "Routing Critical",
            "expected_decision": "escalate (Policy Restricted)",
        },
    }

    selected_examples = []

    for intent, meta in intent_metadata.items():
        sub = df[df["predicted_intent"] == intent].copy()
        # Filter for inquiries between 40 and 220 characters
        sub = sub[sub["text"].str.len().between(40, 240)]
        if sub.empty:
            sub = df[df["predicted_intent"] == intent].head(2)

        for _, row in sub.head(2).iterrows():
            text = str(row["text"]).strip()
            # Clean non-ascii if needed
            clean_text = text.encode("ascii", "ignore").decode("ascii").strip()
            if not clean_text:
                clean_text = text

            selected_examples.append({
                "id": str(row["thread_id"]),
                "intent": intent,
                "intent_label": meta["label"],
                "category_tag": meta["tag"],
                "expected_decision": meta["expected_decision"],
                "text": clean_text,
                "length": len(clean_text),
            })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(selected_examples, f, indent=2)

    print(f"Successfully exported {len(selected_examples)} real candidate inquiries to {out_file}")


if __name__ == "__main__":
    export_demo_examples()
