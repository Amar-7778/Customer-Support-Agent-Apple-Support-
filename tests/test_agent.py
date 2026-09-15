"""
Tests for Stage 5: Multi-Agent Pipeline, Escalation Logic, Grounding Verifier, and FastAPI API.
"""

import pytest
from fastapi.testclient import TestClient

from src.agent.api import app
from src.agent.pipeline import (
    ALWAYS_ESCALATE_INTENTS,
    DEFAULT_AGREEMENT_THRESHOLD,
    AgentResponse,
    decide_escalation,
    handle_message,
)

client = TestClient(app)


# ==============================================================================
# 1. Pure Function Rule Logic Unit Tests (Zero API Calls)
# ==============================================================================


@pytest.mark.parametrize("intent", list(ALWAYS_ESCALATE_INTENTS))
def test_decide_escalation_always_escalate_intents_safety_critical(intent: str):
    """
    CRITICAL SAFETY INVARIANT:
    Restricted intents (e.g. account access, billing/orders, international multilingual)
    must ALWAYS escalate to a human agent, regardless of perfect agreement score (1.0)
    or completely verified grounding status.
    """
    perfect_agreement = 1.0
    verified_grounding = {"grounded": True, "unsupported_claims": []}

    decision, reason = decide_escalation(
        intent=intent,
        precedent_agreement_score=perfect_agreement,
        grounding_verification=verified_grounding,
        agreement_threshold=0.50,
    )

    assert decision == "escalate", f"Safety violation: Restricted intent '{intent}' was not escalated!"
    assert reason == f"intent policy: always escalate for {intent}"


def test_decide_escalation_low_precedent_agreement():
    """
    Low precedent agreement (< 0.50) indicates historical Apple Support agents
    disagreed on the resolution action (e.g., DM vs. support link vs. troubleshooting),
    which must trigger human escalation.
    """
    decision, reason = decide_escalation(
        intent="battery_power_performance",
        precedent_agreement_score=0.45,
        grounding_verification={"grounded": True, "unsupported_claims": []},
        agreement_threshold=0.50,
    )

    assert decision == "escalate"
    assert "low precedent agreement" in reason
    assert "0.45" in reason


def test_decide_escalation_ungrounded_draft_flagged():
    """
    If the grounding verification self-critique pass detects ungrounded claims
    or returns grounded=False, the agent must escalate rather than risk hallucination.
    """
    # Case 1: grounded=False
    decision1, reason1 = decide_escalation(
        intent="software_update_os_bugs",
        precedent_agreement_score=0.85,
        grounding_verification={"grounded": False, "unsupported_claims": ["promised free replacement"]},
        agreement_threshold=0.50,
    )
    assert decision1 == "escalate"
    assert reason1 == "draft contains a claim not supported by retrieved precedent"

    # Case 2: unsupported_claims non-empty even if grounded=True
    decision2, reason2 = decide_escalation(
        intent="keyboard_text_autocorrect",
        precedent_agreement_score=0.75,
        grounding_verification={"grounded": True, "unsupported_claims": ["claims iOS 11.2 fixes issue"]},
        agreement_threshold=0.50,
    )
    assert decision2 == "escalate"
    assert reason2 == "draft contains a claim not supported by retrieved precedent"


def test_decide_escalation_happy_path_auto_handle():
    """
    When the inquiry has a non-restricted intent, precedent agreement >= 0.50,
    and a fully verified grounded draft, the agent may safely auto_handle.
    """
    decision, reason = decide_escalation(
        intent="battery_power_performance",
        precedent_agreement_score=0.80,
        grounding_verification={"grounded": True, "unsupported_claims": []},
        agreement_threshold=0.50,
    )

    assert decision == "auto_handle"
    assert "high precedent agreement (0.80)" in reason
    assert "grounded draft" in reason


# ==============================================================================
# 2. FastAPI Endpoint Validation & Contract Tests
# ==============================================================================


def test_fastapi_health_endpoint():
    """Test GET /health returns 200 OK and expected metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "apple-support-agent"
    assert data["provider"] == "Groq"
    assert data["collection"] == "apple_support_precedents"
    assert data["index_size"] == 3000


@pytest.mark.parametrize(
    "invalid_message",
    [
        "",  # empty string
        "   ",  # whitespace only
        "\n\t  \n",  # whitespace with newlines
    ],
)
def test_fastapi_handle_message_rejects_empty(invalid_message: str):
    """Test POST /handle_message rejects empty or whitespace-only messages with 422."""
    response = client.post("/handle_message", json={"message": invalid_message})
    assert response.status_code == 422


def test_fastapi_handle_message_rejects_oversized():
    """Test POST /handle_message rejects messages exceeding 1,000 characters with 422."""
    oversized_text = "AppleSupport " * 90  # > 1,000 characters
    response = client.post("/handle_message", json={"message": oversized_text})
    assert response.status_code == 422


# ==============================================================================
# 3. Live Integration Test (Real Groq API + Vector DB)
# ==============================================================================


@pytest.mark.integration
def test_handle_message_live_e2e_holdout_query():
    """
    Live end-to-end integration test against real Groq SDK and ChromaDB.
    Uses a real message from the holdout dataset (T_854614).
    """
    real_holdout_query = (
        "What’s with the horrible battery life of iphone after upgrading to #ios11 @AppleSupport "
        "need to charge my phone every 2 hours"
    )

    agent_resp = handle_message(customer_message=real_holdout_query)

    assert isinstance(agent_resp, AgentResponse)
    assert agent_resp.predicted_intent in [
        "battery_power_performance",
        "software_update_os_bugs",
    ]
    assert 0.0 <= agent_resp.intent_confidence <= 1.0
    assert len(agent_resp.retrieved_precedents) > 0
    assert 0.0 <= agent_resp.precedent_agreement_score <= 1.0
    assert len(agent_resp.drafted_reply.strip()) > 0
    assert "grounded" in agent_resp.grounding_verification
    assert agent_resp.decision in ["auto_handle", "escalate"]
    assert len(agent_resp.escalation_reason) > 0
    assert agent_resp.metrics is not None
    assert agent_resp.metrics["total_tokens"] > 0
