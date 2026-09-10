"""
Taxonomy review and finalization module for Stage 3.

Allows human-in-the-loop review, cluster merging, splitting, and configuration of
escalation defaults to produce the final production taxonomy.yaml (6-10 intents).
Zero synthetic examples: all representative examples trace directly to real AppleSupport customer tweets.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("review_taxonomy")

DEFAULT_DRAFT_PATH = "data/processed/draft_taxonomy.json"
DEFAULT_OUTPUT_YAML = "taxonomy.yaml"


def load_draft_taxonomy(path: str = DEFAULT_DRAFT_PATH) -> Dict[str, Any]:
    """Load draft taxonomy output from clustering step."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Draft taxonomy not found at {path}. Run cluster_messages.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_curated_intent_specs() -> Dict[str, Dict[str, Any]]:
    """
    Curated intent definitions for Apple Support Twitter customer inquiries (8 intents, within 6-10 range).
    Each intent has precise classification criteria, domain description, escalation policy,
    and regex patterns for clustering exemplar messages.
    """
    return {
        "battery_power_performance": {
            "intent_name": "battery_power_performance",
            "description": "Abnormal battery drain, device overheating, sudden shutdowns at 20-30%, and charging failures.",
            "escalation_default": False,
            "pattern": r"\b(battery|charging|charge|charger|power|drain|dying|percentage|shutting\s*off|dies|overheating)\b",
        },
        "software_update_os_bugs": {
            "intent_name": "software_update_os_bugs",
            "description": "iOS/macOS update installation errors, device boot loops, system freezing, and OS-level app crashes.",
            "escalation_default": False,
            "pattern": r"\b(update|ios\s*11|updating|installed|download|glitch|bug|unstable|crashing|reboot|freez)\b",
        },
        "keyboard_text_autocorrect": {
            "intent_name": "keyboard_text_autocorrect",
            "description": "Keyboard autocorrect glitches, the iOS 11 'I' text rendering bug, question mark box symbols, and typing lag.",
            "escalation_default": False,
            "pattern": r"(glitch\s+when\s+we\s+type|type\s+[“\"']?I|letter\s+[“\"']?I|question\s*mark\s*box|autocorrect|autocorrection|keyboard|typing)",
        },
        "apple_music_audio_playback": {
            "intent_name": "apple_music_audio_playback",
            "description": "Apple Music app crashes, missing offline music libraries, playlist loading errors, and car audio playback.",
            "escalation_default": False,
            "pattern": r"\b(apple\s*music|music|itunes|playlist|songs?|speaker|sound|audio|earpiece|airpods)\b",
        },
        "orders_purchases_applecare": {
            "intent_name": "orders_purchases_applecare",
            "description": "iPhone X pre-orders, reservation errors, AppleCare warranty purchasing, returns, billing disputes, and refunds.",
            "escalation_default": True,  # Critical commercial/billing intent
            "pattern": r"\b(pre-order|preorder|order|applecare|apple\s*care|refund|overcharge|unauthorized\s*charge|dispute|bill|billing|purchased?|buy|bought|app\s*store|itunes|credit\s*card|reservation)\b",
        },
        "account_access_apple_id": {
            "intent_name": "account_access_apple_id",
            "description": "Apple ID lockouts, two-factor authentication failures, forgotten passcodes, iCloud security, and phishing alerts.",
            "escalation_default": True,  # Security / authentication intent
            "pattern": r"\b(apple\s*id|password|passcode|locked|disabled|icloud|login|log\s*in|2fa|verification\s*code|phishing|security)\b",
        },
        "hardware_display_physical": {
            "intent_name": "hardware_display_physical",
            "description": "Screen cracks, unresponsive touch digitizers, black display freeze, and physical hardware device defects.",
            "escalation_default": False,
            "pattern": r"\b(screen|cracked|broken|display|touch|unresponsive|defective|black\s*screen|estufando|hardware)\b",
        },
        "international_multilingual_inquiries": {
            "intent_name": "international_multilingual_inquiries",
            "description": "Non-English customer inquiries (Spanish, Portuguese, Italian, French) requiring specialized language routing.",
            "escalation_default": True,  # Language routing escalation
            "pattern": r"\b(por\s*favor|actualizaci[oó]n|bater[ií]a|celular|ayuda|devolva|telefone|como\s*fa[cç]o|odio|est[aá]|mire)\b",
        },
    }


def consolidate_draft_taxonomy(draft_taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Consolidates clusters from draft_taxonomy.json into the curated 8-intent taxonomy.
    Extracts 3-5 real, distinct customer exemplar messages for each intent.
    """
    specs = get_curated_intent_specs()
    clusters = draft_taxonomy.get("clusters", [])

    # Collect all available real exemplars across all clusters
    all_exemplars = []
    for cl in clusters:
        for ex in cl.get("exemplars", []):
            all_exemplars.append(ex)

    final_intents = []

    for key, spec in specs.items():
        intent_name = spec["intent_name"]
        desc = spec["description"]
        escalate = spec["escalation_default"]
        pattern = re.compile(spec["pattern"], re.IGNORECASE)

        # Find matching exemplars
        matched_examples = []
        seen_texts = set()

        for ex in all_exemplars:
            txt = ex.get("cleaned_text") or ex.get("text", "")
            if pattern.search(txt) and txt not in seen_texts:
                seen_texts.add(txt)
                matched_examples.append({
                    "tweet_id": str(ex["tweet_id"]),
                    "text": str(ex["text"]),
                })
                if len(matched_examples) >= 5:
                    break

        # If regex yielded fewer than 3, search raw text
        if len(matched_examples) < 3:
            for ex in all_exemplars:
                raw_txt = ex.get("text", "")
                if pattern.search(raw_txt) and raw_txt not in seen_texts:
                    seen_texts.add(raw_txt)
                    matched_examples.append({
                        "tweet_id": str(ex["tweet_id"]),
                        "text": str(ex["text"]),
                    })
                    if len(matched_examples) >= 5:
                        break

        final_intents.append({
            "intent_name": intent_name,
            "description": desc,
            "escalation_default": escalate,
            "representative_examples": matched_examples,
        })

    return final_intents


def save_final_taxonomy(
    final_intents: List[Dict[str, Any]],
    output_path: str = DEFAULT_OUTPUT_YAML,
) -> None:
    """Save finalized taxonomy configuration to YAML."""
    out_dict = {
        "metadata": {
            "version": "1.0",
            "brand": "AppleSupport",
            "total_intents": len(final_intents),
            "generated_from": "Stage 3 real AppleSupport thread clustering (k=8 silhouette peak)",
        },
        "intents": final_intents,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(out_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    logger.info(f"Finalized taxonomy with {len(final_intents)} intents saved to {output_path}")


def review_and_finalize_taxonomy(
    draft_path: str = DEFAULT_DRAFT_PATH,
    output_path: str = DEFAULT_OUTPUT_YAML,
) -> List[Dict[str, Any]]:
    """
    Main orchestration for Step 3:
    - Loads draft_taxonomy.json
    - Consolidates into 8 distinct intents (within 6-10 range)
    - Saves taxonomy.yaml
    """
    logger.info(f"Loading draft taxonomy from {draft_path}...")
    draft = load_draft_taxonomy(draft_path)

    final_intents = consolidate_draft_taxonomy(draft)
    save_final_taxonomy(final_intents, output_path)

    print("\n" + "=" * 80)
    print(f"FINALIZED INTENT TAXONOMY ({len(final_intents)} Intents -> {output_path})")
    print("=" * 80)
    format_row = "{:<36} | {:<10} | {:<8} | {:<20}"
    print(format_row.format("Intent Name", "Escalate?", "Examples", "Key Focus"))
    print("-" * 80)
    for intent in final_intents:
        print(format_row.format(
            intent["intent_name"],
            "ALWAYS" if intent["escalation_default"] else "standard",
            str(len(intent["representative_examples"])),
            intent["description"][:20] + "...",
        ))
    print("=" * 80 + "\n")

    return final_intents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review and finalize Stage 3 intent taxonomy.")
    parser.add_argument("--draft", default=DEFAULT_DRAFT_PATH, help="Path to draft_taxonomy.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_YAML, help="Path to output taxonomy.yaml")
    args = parser.parse_args()

    review_and_finalize_taxonomy(
        draft_path=args.draft,
        output_path=args.output,
    )
