"""
Stage 5: Autonomous Multi-Agent Support Pipeline with Intent Routing,
Precedent-Grounded Generation, Self-Critique Grounding Verification,
and Explicit Rule-Based Escalation Decision Logic.
"""

from .pipeline import (
    ALWAYS_ESCALATE_INTENTS,
    DEFAULT_AGREEMENT_THRESHOLD,
    AgentResponse,
    decide_escalation,
    handle_message,
    verify_grounding,
)

__all__ = [
    "AgentResponse",
    "handle_message",
    "decide_escalation",
    "verify_grounding",
    "ALWAYS_ESCALATE_INTENTS",
    "DEFAULT_AGREEMENT_THRESHOLD",
]
