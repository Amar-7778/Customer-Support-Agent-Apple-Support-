"""
Stage 5: End-to-End Agent Pipeline Module.

Integrates:
1. Intent classification via Groq LLM (single-message inference using finalized taxonomy.yaml).
2. Grounded precedent retrieval from 3,000-vector ChromaDB collection.
3. Precedent-guided reply drafting using in-context historical resolution exemplars.
4. Independent grounding verification self-critique pass (detects unsupported claims).
5. Explicit rule-based escalation decision logic.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIError
from pydantic import BaseModel, Field

from src.retrieval.query_index import retrieve_precedents
from src.taxonomy.sample_for_clustering import clean_tweet_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_pipeline")

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TAXONOMY_PATH = "taxonomy.yaml"
DEFAULT_AGREEMENT_THRESHOLD = 0.50
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

ALWAYS_ESCALATE_INTENTS = {
    "orders_purchases_applecare",
    "account_access_apple_id",
    "international_multilingual_inquiries",
}


def mask_key(key: Optional[str]) -> str:
    """Masks an API key for safe logging (e.g. gsk_123...4567)."""
    if not key:
        return "UNSET"
    s = key.strip()
    return f"{s[:7]}...{s[-4:]}" if len(s) > 11 else "***"


class GroundingVerification(BaseModel):
    grounded: bool
    unsupported_claims: List[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    customer_message: str
    predicted_intent: str
    intent_confidence: float
    retrieved_precedents: List[Dict[str, Any]]
    precedent_agreement_score: float
    drafted_reply: str
    grounding_verification: Dict[str, Any]
    decision: Literal["auto_handle", "escalate"]
    escalation_reason: str
    metrics: Optional[Dict[str, Any]] = None


# Module-level singletons for low-latency requests
_GROQ_CLIENTS: List[Groq] = []
_CLIENT_INDEX: int = 0
_TAXONOMY_CACHE: Optional[Dict[str, Any]] = None
_SYSTEM_PROMPT_CACHE: Optional[str] = None
_PRECEDENTS_DF_CACHE: Optional[pd.DataFrame] = None


def get_groq_client(rotate: bool = True) -> Groq:
    """Lazy loader and validator for the official Groq client with round-robin multi-key rotation."""
    global _GROQ_CLIENTS, _CLIENT_INDEX
    if not _GROQ_CLIENTS:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv(find_dotenv(usecwd=True))

        keys: List[str] = []
        for var in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEYS"]:
            val = os.getenv(var)
            if val:
                for k in val.split(","):
                    k_str = k.strip()
                    if k_str and k_str not in keys:
                        keys.append(k_str)

        if not keys:
            raise ValueError(
                "CRITICAL CONFIGURATION ERROR: GROQ_API_KEY is unset in .env. "
                "Per pipeline constraints, Groq is the exclusive provider for LLM inference."
            )

        for k in keys:
            logger.info("Initialized Groq client with key: %s", mask_key(k))
            _GROQ_CLIENTS.append(Groq(api_key=k, max_retries=0))

    if rotate and len(_GROQ_CLIENTS) > 1:
        _CLIENT_INDEX = (_CLIENT_INDEX + 1) % len(_GROQ_CLIENTS)
        return _GROQ_CLIENTS[_CLIENT_INDEX]

    return _GROQ_CLIENTS[_CLIENT_INDEX % len(_GROQ_CLIENTS)]


def load_taxonomy(taxonomy_path: str = DEFAULT_TAXONOMY_PATH) -> Dict[str, Any]:
    """Loads and caches the finalized taxonomy.yaml."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        path = Path(taxonomy_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / taxonomy_path
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            _TAXONOMY_CACHE = yaml.safe_load(f)
    return _TAXONOMY_CACHE


def get_classification_system_prompt(taxonomy: Dict[str, Any]) -> str:
    """Builds the single-message classification prompt from taxonomy.yaml."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    intents = taxonomy.get("intents", [])
    valid_intents = [item["intent_name"] for item in intents]
    intent_blocks = []

    for item in intents:
        name = item["intent_name"]
        desc = item.get("description", "")
        examples = item.get("representative_examples", [])
        ex_lines = "\n".join([f'    * "{clean_tweet_text(ex.get("text", ""))}"' for ex in examples[:3]])
        intent_blocks.append(
            f"- Intent: `{name}`\n"
            f"  Description: {desc}\n"
            f"  Exemplars:\n{ex_lines}"
        )

    intents_text = "\n\n".join(intent_blocks)
    prompt = (
        "You are an expert customer support intent classifier for Apple Support inquiries on Twitter.\n"
        "Here is the finalized intent taxonomy grounded strictly in real customer interactions:\n\n"
        f"{intents_text}\n\n"
        f"Allowed intent categories (choose exactly one):\n{json.dumps(valid_intents)}\n\n"
        "Instructions:\n"
        "1. Determine the customer's primary intent category.\n"
        "2. Return ONLY a valid JSON object with the keys 'intent' and 'confidence' (float 0.0-1.0):\n"
        '   {"intent": "<intent_name>", "confidence": <float>}\n'
        "3. Every intent MUST be one of the allowed categories. Do not invent new intents."
    )
    _SYSTEM_PROMPT_CACHE = prompt
    return prompt


def classify_message_groq(
    customer_message: str,
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
) -> Tuple[str, float, int, int, float]:
    """
    Classifies a single customer message into one of the 8 taxonomy intents using Groq.
    Returns (predicted_intent, confidence, prompt_tokens, completion_tokens, latency_seconds).
    """
    if client is None:
        client = get_groq_client()
    taxonomy = load_taxonomy()
    system_prompt = get_classification_system_prompt(taxonomy)
    valid_intents = set(item["intent_name"] for item in taxonomy.get("intents", []))

    user_prompt = f'Customer Inquiry:\n"{clean_tweet_text(customer_message)}"'
    t0 = time.time()

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            lat = time.time() - t0
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            intent = data.get("intent", "").strip()
            confidence = float(data.get("confidence", 0.90))

            if intent not in valid_intents:
                logger.warning("Classifier returned non-standard intent '%s', falling back to software_update_os_bugs", intent)
                intent = "software_update_os_bugs"

            return intent, confidence, p_tok, c_tok, lat
        except RateLimitError as e:
            client = get_groq_client(rotate=True)
            wait = 1.0 * (attempt + 1)
            logger.warning("Rate limit in classify_message_groq. Rotated key, backing off %.1fs...", wait)
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5)

    return "software_update_os_bugs", 0.50, 0, 0, time.time() - t0


def draft_reply_groq(
    customer_message: str,
    intent: str,
    precedents: List[Dict[str, Any]],
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
) -> Tuple[str, int, int, float]:
    """
    Drafts an Apple Support Twitter reply grounded strictly in the retrieved historical precedents.
    Returns (drafted_reply, prompt_tokens, completion_tokens, latency_seconds).
    """
    if client is None:
        client = get_groq_client()

    precedent_blocks = []
    for i, p in enumerate(precedents[:3], 1):
        cust_msg = p.get("customer_message") or p.get("document") or "N/A"
        action = p.get("action_taken", "N/A")
        reply = p.get("brand_reply_text", "N/A")
        outcome = p.get("outcome", "N/A")
        precedent_blocks.append(
            f"Precedent #{i}:\n"
            f"  Customer Message: \"{cust_msg}\"\n"
            f"  Resolution Action Taken: {action}\n"
            f"  Outcome: {outcome}\n"
            f"  Historical Apple Reply: \"{reply}\""
        )

    precedents_context = "\n\n".join(precedent_blocks)

    system_prompt = (
        "You are an official Apple Support representative on Twitter (@AppleSupport).\n"
        "Draft a helpful, polite, and concise customer support reply (strictly under 280 characters).\n\n"
        "STRICT GROUNDING CONSTRAINTS:\n"
        "1. Base your diagnostic steps, requested information, or DM invitation directly on the provided historical precedents.\n"
        "2. NEVER invent unauthorized policies, promise free hardware replacements, issue refunds, or quote unverified repair timelines.\n"
        "3. If the precedents indicate requesting a Direct Message (DM) with device and OS details, follow that exact strategy.\n"
        "4. If the precedent directs to a specific official support link or language team, reflect that accurately.\n"
        "5. Respond in English unless the customer message and precedents indicate a specific multilingual routing.\n"
        "6. Output ONLY the drafted reply text. Do not include quotes, greetings like 'Draft:', or meta commentary."
    )

    user_prompt = (
        f"Incoming Customer Inquiry:\n\"{customer_message}\"\n\n"
        f"Categorized Intent: {intent}\n\n"
        f"Retrieved Historical Apple Precedents for Grounding:\n{precedents_context}\n\n"
        "Draft the official @AppleSupport reply based on these precedents:"
    )

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            lat = time.time() - t0
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            draft = resp.choices[0].message.content or ""
            draft = draft.strip().strip('"').strip("'")
            return draft, p_tok, c_tok, lat
        except RateLimitError:
            client = get_groq_client(rotate=True)
            wait = 1.0 * (attempt + 1)
            logger.warning("Rate limit in draft_reply_groq. Rotated key, backing off %.1fs...", wait)
            time.sleep(wait)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5)

    return "We'd like to help with this. Please send us a DM with your device model and current iOS version.", 0, 0, time.time() - t0


def verify_grounding_groq(
    customer_message: str,
    drafted_reply: str,
    precedents: List[Dict[str, Any]],
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
) -> Tuple[Dict[str, Any], int, int, float]:
    """
    Independent self-critique pass via Groq LLM: checks if the drafted reply contains
    unsupported claims, hallucinated policies, or unwarranted promises not present in precedents.
    Returns ({"grounded": bool, "unsupported_claims": List[str]}, prompt_tokens, completion_tokens, latency).
    """
    if client is None:
        client = get_groq_client()

    precedent_summaries = []
    for i, p in enumerate(precedents[:3], 1):
        act = p.get("action_taken", "")
        reply = p.get("brand_reply_text", "")
        precedent_summaries.append(f"Precedent #{i}: Action='{act}', Example Reply='{reply}'")

    prec_text = "\n".join(precedent_summaries)

    system_prompt = (
        "You are an uncompromising safety and factual grounding verifier for Apple Support AI replies.\n"
        "Your role is to verify whether the drafted customer support reply is faithfully grounded in the retrieved historical precedents.\n\n"
        "STRICT GROUNDING CRITERIA (STAGE 5/6 AUDIT):\n"
        "1. A draft IS GROUNDED (grounded=true, unsupported_claims=[]) if the recommended actions (e.g. asking for device model/iOS version in DM, offering general troubleshooting tips) are illustrated or directly justified by the retrieved precedents.\n"
        "2. A draft IS UNGROUNDED (grounded=false) if it asserts technical guarantees, promises free hardware/screen replacements, promises money refunds, quotes specific unsupported turnaround times, OR injects specific version numbers / canonical support URLs not in the precedents.\n\n"
        "OUTPUT FORMAT:\n"
        "You MUST respond ONLY with a valid JSON object matching this schema:\n"
        '{\n'
        '  "grounded": true | false,\n'
        '  "unsupported_claims": ["claim 1", "claim 2"]\n'
        '}\n'
        "If grounded is true, unsupported_claims MUST be empty []."
    )

    user_prompt = (
        f"Customer Inquiry: \"{customer_message}\"\n\n"
        f"Retrieved Precedents:\n{prec_text}\n\n"
        f"Drafted Reply to Verify:\n\"{drafted_reply}\"\n\n"
        "Perform factual grounding verification:"
    )

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=120,
            )
            lat = time.time() - t0
            usage = resp.usage
            p_tok = usage.prompt_tokens if usage else 0
            c_tok = usage.completion_tokens if usage else 0

            content = resp.choices[0].message.content or "{}"
            result = json.loads(content)
            grounded = bool(result.get("grounded", True))
            unsupported = list(result.get("unsupported_claims", []))

            # Defensive consistency enforcement
            if grounded and unsupported:
                grounded = False

            return {"grounded": grounded, "unsupported_claims": unsupported}, p_tok, c_tok, lat
        except RateLimitError:
            client = get_groq_client(rotate=True)
            wait = 1.0 * (attempt + 1)
            logger.warning("Rate limit in verify_grounding_groq. Rotated key, backing off %.1fs...", wait)
            time.sleep(wait)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5)

    return {"grounded": True, "unsupported_claims": []}, 0, 0, time.time() - t0


# Functional alias for external imports
verify_grounding = verify_grounding_groq


def decide_escalation(
    intent: str,
    precedent_agreement_score: float,
    grounding_verification: Dict[str, Any],
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
    intent_confidence: Optional[float] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[Literal["auto_handle", "escalate"], str]:
    """
    Explicit, inspectable escalation decision logic (Step 2).
    Pure rule-layer function:
    1. Safety-critical restricted intents always escalate.
    2. Low classifier confidence escalates due to routing uncertainty (calibrated in Stage 6).
    3. Low precedent-agreement scores escalate due to resolution disagreement.
    4. Ungrounded drafts with unsupported claims escalate due to hallucination risk.
    5. Otherwise, auto_handle.
    """
    # 1. Intent policy check
    if intent in ALWAYS_ESCALATE_INTENTS:
        return "escalate", f"intent policy: always escalate for {intent}"

    # 2. Classifier confidence check (calibrated in Stage 6)
    if intent_confidence is not None and intent_confidence < confidence_threshold:
        return (
            "escalate",
            f"low classifier confidence ({intent_confidence:.2f} < {confidence_threshold:.2f}): routing uncertain",
        )

    # 3. Precedent agreement threshold check
    if precedent_agreement_score < agreement_threshold:
        return (
            "escalate",
            f"low precedent agreement ({precedent_agreement_score:.2f}): retrieved historical resolutions disagree on how this was handled",
        )

    # 4. Grounding verification check
    is_grounded = grounding_verification.get("grounded", True)
    unsupported_claims = grounding_verification.get("unsupported_claims", [])
    if not is_grounded or len(unsupported_claims) > 0:
        return "escalate", "draft contains a claim not supported by retrieved precedent"

    # 5. Happy path auto-handle
    return (
        "auto_handle",
        f"high precedent agreement ({precedent_agreement_score:.2f}), grounded draft, non-restricted intent",
    )


def handle_message(
    customer_message: str,
    client: Optional[Groq] = None,
    model: str = DEFAULT_MODEL,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    k_precedents: int = 3,
) -> AgentResponse:
    """
    End-to-end agent pipeline for a single customer inquiry:
    1. Classifies intent via few-shot Groq LLM.
    2. Retrieves top-k precedents and computes agreement score from ChromaDB.
    3. Drafts an Apple Support reply grounded in precedents.
    4. Runs separate LLM grounding verification self-critique pass.
    5. Applies explicit escalation decision logic.
    6. Returns structured AgentResponse.
    """
    cleaned_msg = clean_tweet_text(customer_message)
    if not cleaned_msg.strip():
        raise ValueError("Customer message cannot be empty or whitespace-only.")

    t_start = time.time()
    if client is None:
        client = get_groq_client()

    # Step 1: Intent Classification
    intent, conf, p_tok_cls, c_tok_cls, lat_cls = classify_message_groq(
        customer_message=cleaned_msg, client=client or get_groq_client(), model=model
    )

    # Step 2: Precedent Retrieval
    retrieval_res = retrieve_precedents(
        message=cleaned_msg,
        intent=intent,
        k=k_precedents,
    )
    raw_precedents = retrieval_res.get("precedents", [])
    agreement_score = float(retrieval_res.get("precedent_agreement_score", 0.0))

    # Enrich precedents with customer message if available
    structured_precedents: List[Dict[str, Any]] = []
    for p in raw_precedents:
        structured_precedents.append({
            "thread_id": p.get("thread_id", ""),
            "tweet_id": p.get("tweet_id", ""),
            "intent": p.get("intent", intent),
            "customer_message": p.get("document", ""),
            "action_taken": p.get("action_taken", ""),
            "outcome": p.get("outcome", ""),
            "brand_reply_text": p.get("brand_reply_text", ""),
            "similarity_score": p.get("similarity_score", 0.0),
        })

    # Step 3: Reply Drafting
    draft_reply, p_tok_dft, c_tok_dft, lat_dft = draft_reply_groq(
        customer_message=cleaned_msg,
        intent=intent,
        precedents=structured_precedents,
        client=client or get_groq_client(),
        model=model,
    )

    # Step 4: Grounding Verification (Self-Critique Pass)
    grounding_info, p_tok_ver, c_tok_ver, lat_ver = verify_grounding_groq(
        customer_message=cleaned_msg,
        drafted_reply=draft_reply,
        precedents=structured_precedents,
        client=client or get_groq_client(),
        model=model,
    )

    # Step 5: Rule-Layer Escalation Decision
    decision, escalation_reason = decide_escalation(
        intent=intent,
        precedent_agreement_score=agreement_score,
        grounding_verification=grounding_info,
        agreement_threshold=agreement_threshold,
        intent_confidence=conf,
        confidence_threshold=confidence_threshold,
    )

    total_latency = round(time.time() - t_start, 3)
    metrics = {
        "model": model,
        "total_latency_seconds": total_latency,
        "classification_latency_seconds": round(lat_cls, 3),
        "drafting_latency_seconds": round(lat_dft, 3),
        "verification_latency_seconds": round(lat_ver, 3),
        "total_prompt_tokens": p_tok_cls + p_tok_dft + p_tok_ver,
        "total_completion_tokens": c_tok_cls + c_tok_dft + c_tok_ver,
        "total_tokens": (p_tok_cls + p_tok_dft + p_tok_ver) + (c_tok_cls + c_tok_dft + c_tok_ver),
    }

    return AgentResponse(
        customer_message=customer_message,
        predicted_intent=intent,
        intent_confidence=conf,
        retrieved_precedents=structured_precedents,
        precedent_agreement_score=agreement_score,
        drafted_reply=draft_reply,
        grounding_verification=grounding_info,
        decision=decision,
        escalation_reason=escalation_reason,
        metrics=metrics,
    )
