"""
Stage 6: Baseline Models for Customer Support Evaluation.

Provides two baseline implementations scored on identical metrics as the real agent:
1. Trivial Baseline: Escalate everything with the single most frequent historical canned reply.
2. Simple Baseline: Regex/keyword classifier + intent-specific template + policy escalation.
"""

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.agent.pipeline import ALWAYS_ESCALATE_INTENTS

TRIVIAL_CANNED_REPLY = (
    "Here’s what you can do to work around the issue until it’s fixed in a future software update: "
    "https://t.co/xXaXeeSRt9"
)

INTENT_TEMPLATES = {
    "battery_power_performance": (
        "We want to make sure you get the most out of your battery. "
        "Send us a DM with your device model and iOS version: https://t.co/GDrqU22YpT"
    ),
    "keyboard_text_autocorrect": (
        "Here’s what you can do to work around the issue until it’s fixed in a future software update: "
        "https://t.co/xXaXeeSRt9"
    ),
    "software_update_os_bugs": (
        "We'd like to look into this. Send us a DM with your device model and software version: "
        "https://t.co/GDrqU22YpT"
    ),
    "hardware_display_physical": (
        "Thanks for reaching out. Reach out to us in DM so we can take a closer look: "
        "https://t.co/GDrqU22YpT"
    ),
    "apple_music_audio_playback": (
        "Let's look into this for you. Which device and OS are you currently using? "
        "Let us know in DM: https://t.co/GDrqU22YpT"
    ),
    "orders_purchases_applecare": (
        "Please contact our iTunes Store Support team to review your purchase history: "
        "https://t.co/SDIe7UiyJN"
    ),
    "account_access_apple_id": (
        "Please send us a DM so we can gather more details and assist with your account: "
        "https://t.co/GDrqU22YpT"
    ),
    "international_multilingual_inquiries": (
        "Since our Twitter support is available in English, get help at "
        "https://t.co/IBIY3vMgPj or join https://t.co/pvaOFfPbjt"
    ),
}

KEYWORD_PATTERNS = [
    (
        "international_multilingual_inquiries",
        re.compile(r"(\b(por qué|qué|hola|ayuda|celular|actualización|batería|gracias|meu|minha|não|está|travando|bandeira)\b|[¿¡])", re.IGNORECASE),
    ),
    (
        "account_access_apple_id",
        re.compile(r"\b(apple\s*id|icloud|password|passcode|phishing|scam|legit|activation\s*lock|locked)\b", re.IGNORECASE),
    ),
    (
        "orders_purchases_applecare",
        re.compile(r"(\b(order|purchase|applecare|refund|charged|charge|bought|receipt|bill|subscription|dollars?)\b|\$)", re.IGNORECASE),
    ),
    (
        "hardware_display_physical",
        re.compile(r"\b(screen|crack|display|button|touch|broken|speaker|vibration|camera|digitizer)\b", re.IGNORECASE),
    ),
    (
        "apple_music_audio_playback",
        re.compile(r"\b(music|songs?|playlists?|audio|sound|play(ing)?|volume|itunes)\b", re.IGNORECASE),
    ),
    (
        "battery_power_performance",
        re.compile(r"\b(battery|drain(ing)?|charge|charging|charger|dying|overheat(ing)?|power)\b", re.IGNORECASE),
    ),
    (
        "keyboard_text_autocorrect",
        re.compile(r"\b(keyboard|type|typing|autocorrect|letter\s*i|question\s*mark|glitch)\b", re.IGNORECASE),
    ),
    (
        "software_update_os_bugs",
        re.compile(r"\b(update|ios|os|freeze|freezing|crash|crashes|reboot|boot\s*loop)\b", re.IGNORECASE),
    ),
]


def classify_simple_baseline(message: str) -> Tuple[str, float]:
    """Crude rule/regex keyword matching classifier."""
    for intent, pattern in KEYWORD_PATTERNS:
        if pattern.search(message):
            return intent, 0.70
    return "software_update_os_bugs", 0.30


def run_trivial_baseline(customer_message: str) -> Dict[str, Any]:
    """Trivial Baseline: Escalate 100% of messages with canned reply."""
    return {
        "customer_message": customer_message,
        "predicted_intent": "software_update_os_bugs",
        "intent_confidence": 0.0,
        "precedent_agreement_score": 0.0,
        "drafted_reply": TRIVIAL_CANNED_REPLY,
        "grounding_verification": {"grounded": True, "unsupported_claims": []},
        "decision": "escalate",
        "escalation_reason": "trivial baseline: escalate all messages",
    }


def run_simple_baseline(customer_message: str) -> Dict[str, Any]:
    """Simple Baseline: Regex intent classifier + intent template + policy escalation."""
    pred_intent, conf = classify_simple_baseline(customer_message)
    reply = INTENT_TEMPLATES.get(pred_intent, TRIVIAL_CANNED_REPLY)

    if pred_intent in ALWAYS_ESCALATE_INTENTS:
        decision = "escalate"
        reason = f"intent policy: always escalate for {pred_intent}"
    else:
        decision = "auto_handle"
        reason = "simple baseline: non-restricted intent auto_handled"

    return {
        "customer_message": customer_message,
        "predicted_intent": pred_intent,
        "intent_confidence": conf,
        "precedent_agreement_score": 0.50,
        "drafted_reply": reply,
        "grounding_verification": {"grounded": True, "unsupported_claims": []},
        "decision": decision,
        "escalation_reason": reason,
    }


def evaluate_baselines_on_golden_set(df_golden: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs both baselines across all golden eval inquiries."""
    trivial_results = []
    simple_results = []

    for _, row in df_golden.iterrows():
        msg = row["text"]
        t_res = run_trivial_baseline(msg)
        s_res = run_simple_baseline(msg)
        trivial_results.append(t_res)
        simple_results.append(s_res)

    return trivial_results, simple_results
