"""
Stage 6: LLM-as-a-Judge for Reply Quality Evaluation.

Evaluates customer support replies across 4 rubric axes (1-5 scale) using Groq LLM:
1. grounded_in_precedent
2. factually_non_hallucinatory (explicitly penalizes both fabricated wrong specifics AND plausible ungrounded specifics)
3. tone_appropriate
4. resolves_or_correctly_defers

Also implements blind judge-vs-human agreement auditing across 40 golden examples.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field

from src.agent.pipeline import DEFAULT_MODEL, get_groq_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_judge")


class RubricScores(BaseModel):
    grounded_in_precedent: int = Field(..., ge=1, le=5)
    factually_non_hallucinatory: int = Field(..., ge=1, le=5)
    tone_appropriate: int = Field(..., ge=1, le=5)
    resolves_or_correctly_defers: int = Field(..., ge=1, le=5)
    composite_score: float = Field(..., ge=1.0, le=5.0)
    has_invented_specifics: bool
    invented_specifics_detail: str = Field(default="")
    critique: str


JUDGE_SYSTEM_PROMPT = (
    "You are an expert customer support quality auditor evaluating AI-generated replies for Apple Support on Twitter.\n"
    "Score the candidate reply strictly on a 1 to 5 integer scale across four core axes:\n\n"
    "1. grounded_in_precedent (1-5):\n"
    "   - 5: Strategy, diagnostic steps, and links directly mirror the retrieved historical precedents.\n"
    "   - 3: Partially aligned with precedent strategy, but introduces minor deviations.\n"
    "   - 1: Completely ignores or contradicts the historical precedent actions.\n\n"
    "2. factually_non_hallucinatory (1-5) [STRICT SPECIFICITY AUDIT]:\n"
    "   - You MUST inspect the reply for ANY concrete, checkable detail (version numbers, patch dates, dollar amounts, article identifiers, canonical URLs, turnaround times).\n"
    "   - You MUST penalize BOTH:\n"
    "     (a) Invented specific details that are wrong or unverifiable (e.g. claiming iOS 11.0.3 fixes an issue when not stated in precedent).\n"
    "     (b) Specific details that may be true in the real world but were NOT present in the retrieved precedent text and were injected from general knowledge (e.g. canonical URLs like https://support.apple.com/en-us/HT204910 when precedents only have t.co links).\n"
    "   - Being factually right by luck is STILL A GROUNDING FAILURE, because the system's contract is evidence-based generation grounded in this brand's historical resolutions.\n"
    "   - Scoring:\n"
    "     * 5: Zero ungrounded specifics. All concrete entities are present verbatim or in direct substance in the precedents.\n"
    "     * 3: Minor ungrounded specific (e.g. naming a generic standard iOS menu path like Settings > General).\n"
    "     * 1: Injects ungrounded specific version numbers, canonical URLs, or warranty promises absent from precedents.\n\n"
    "3. tone_appropriate (1-5):\n"
    "   - 5: Professional, supportive, concise (< 280 characters), empathetic Apple brand voice.\n"
    "   - 3: Acceptable but generic, mildly verbose, or slightly informal.\n"
    "   - 1: Hostile, rude, robotic, or exceeds 280 characters.\n\n"
    "4. resolves_or_correctly_defers (1-5):\n"
    "   - 5: Takes a constructive, actionable forward step (verified workaround link or diagnostic setting) or safely defers/escalates to DM when appropriate.\n"
    "   - 3: Vague or ambiguous next step.\n"
    "   - 1: Dead-end, dismissive refusal, or unsafe automated commitment for high-risk issue.\n\n"
    "OUTPUT FORMAT:\n"
    "You MUST return ONLY a valid JSON object matching this schema:\n"
    "{\n"
    '  "grounded_in_precedent": <int 1-5>,\n'
    '  "factually_non_hallucinatory": <int 1-5>,\n'
    '  "tone_appropriate": <int 1-5>,\n'
    '  "resolves_or_correctly_defers": <int 1-5>,\n'
    '  "composite_score": <float 1.0-5.0>,\n'
    '  "has_invented_specifics": <bool>,\n'
    '  "invented_specifics_detail": "<string description of any ungrounded specific detail, or empty if none>",\n'
    '  "critique": "<brief 1-2 sentence evaluation summary>"\n'
    "}"
)


def evaluate_reply_quality(
    customer_message: str,
    drafted_reply: str,
    retrieved_precedents: List[Dict[str, Any]],
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """Evaluates a drafted reply using the 4-axis LLM judge."""
    if client is None:
        client = get_groq_client()

    prec_blocks = []
    for i, p in enumerate(retrieved_precedents[:3], 1):
        act = p.get("action_taken", "N/A")
        reply = p.get("brand_reply_text", "N/A")
        out = p.get("outcome", "N/A")
        prec_blocks.append(f"Precedent #{i}: Action='{act}' | Outcome='{out}' | Reply='{reply}'")

    prec_text = "\n".join(prec_blocks)
    user_prompt = (
        f"Customer Message:\n\"{customer_message}\"\n\n"
        f"Retrieved Precedents for Grounding:\n{prec_text}\n\n"
        f"Drafted Reply to Evaluate:\n\"{drafted_reply}\"\n\n"
        "Evaluate the drafted reply strictly against the rubric:"
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=220,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)

            g = int(data.get("grounded_in_precedent", 3))
            f = int(data.get("factually_non_hallucinatory", 3))
            t = int(data.get("tone_appropriate", 4))
            r = int(data.get("resolves_or_correctly_defers", 3))
            comp = round(float(np.mean([g, f, t, r])), 2)

            return {
                "grounded_in_precedent": g,
                "factually_non_hallucinatory": f,
                "tone_appropriate": t,
                "resolves_or_correctly_defers": r,
                "composite_score": comp,
                "has_invented_specifics": bool(data.get("has_invented_specifics", f <= 2)),
                "invented_specifics_detail": data.get("invented_specifics_detail", ""),
                "critique": data.get("critique", ""),
            }
        except RateLimitError:
            client = get_groq_client(rotate=True)
            wait = 1.0 * (attempt + 1)
            logger.warning("Rate limit in evaluate_reply_quality. Rotated key, backing off %.1fs...", wait)
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("Judge evaluation failed after %d attempts: %s", max_retries, e)
                return {
                    "grounded_in_precedent": 3,
                    "factually_non_hallucinatory": 3,
                    "tone_appropriate": 4,
                    "resolves_or_correctly_defers": 3,
                    "composite_score": 3.25,
                    "has_invented_specifics": False,
                    "invented_specifics_detail": "",
                    "critique": f"Judge error: {str(e)}",
                }
            time.sleep(1.5)

    return {
        "grounded_in_precedent": 3,
        "factually_non_hallucinatory": 3,
        "tone_appropriate": 4,
        "resolves_or_correctly_defers": 3,
        "composite_score": 3.25,
        "has_invented_specifics": False,
        "invented_specifics_detail": "",
        "critique": "Judge fallback",
    }


def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """Computes Cohen's kappa coefficient between two raters."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    categories = sorted(list(set(rater1).union(set(rater2))))
    n = len(rater1)

    # Observed agreement
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p1 = sum(1 for a in rater1 if a == cat) / n
        p2 = sum(1 for b in rater2 if b == cat) / n
        pe += p1 * p2

    if pe >= 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def compute_judge_human_agreement(
    human_scores: List[Dict[str, int]],
    judge_scores: List[Dict[str, int]],
    axes: List[str] = [
        "grounded_in_precedent",
        "factually_non_hallucinatory",
        "tone_appropriate",
        "resolves_or_correctly_defers",
    ],
) -> Dict[str, Any]:
    """
    Computes exact agreement, adjacent agreement (within +/- 1), and Cohen's kappa
    between human auditor and LLM judge across all evaluation axes.
    """
    results = {}
    itemized_disagreements = []

    for axis in axes:
        h_vals = [h[axis] for h in human_scores]
        j_vals = [j[axis] for j in judge_scores]

        exact_matches = sum(1 for h, j in zip(h_vals, j_vals) if h == j)
        adjacent_matches = sum(1 for h, j in zip(h_vals, j_vals) if abs(h - j) <= 1)
        total = len(h_vals)

        exact_pct = round(exact_matches / total * 100, 2)
        adj_pct = round(adjacent_matches / total * 100, 2)
        kappa = compute_cohens_kappa(h_vals, j_vals)

        results[axis] = {
            "exact_agreement_pct": exact_pct,
            "adjacent_agreement_pct": adj_pct,
            "cohens_kappa": kappa,
            "mean_human": round(float(np.mean(h_vals)), 2),
            "mean_judge": round(float(np.mean(j_vals)), 2),
        }

    for idx, (h, j) in enumerate(zip(human_scores, judge_scores)):
        diffs = {axis: {"human": h[axis], "judge": j[axis]} for axis in axes if h[axis] != j[axis]}
        if diffs:
            itemized_disagreements.append({
                "sample_idx": idx + 1,
                "discrepancies": diffs,
            })

    results["itemized_disagreements_count"] = len(itemized_disagreements)
    results["itemized_disagreements"] = itemized_disagreements
    return results
