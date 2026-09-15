"""
FastAPI Service for Stage 5: Autonomous Apple Support Agent.

Endpoints:
- POST /handle_message: Accepts {"message": str}, executes pipeline, returns AgentResponse.
- GET /health: Health check reporting service readiness and vector store status.
"""

import logging
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.agent.pipeline import (
    DEFAULT_MODEL,
    AgentResponse,
    get_groq_client,
    handle_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent_api")

app = FastAPI(
    title="Apple Support Agent Service",
    version="1.0.0",
    description="End-to-end agentic customer support service powered by Groq and ChromaDB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        description="Customer inquiry text (e.g. from Twitter).",
        min_length=1,
        max_length=1000,
        examples=["My iPhone 7 battery dies every 2 hours after the iOS 11 update. How can I fix this?"],
    )

    @field_validator("message")
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Message cannot be empty or whitespace-only.")
        if len(s) > 1000:
            raise ValueError("Message exceeds maximum allowable length of 1,000 characters.")
        return s


@app.get("/health", tags=["Monitoring"], summary="Service health check")
def health_check() -> Dict[str, Any]:
    """Returns the operational status of the service, model configuration, and Groq connectivity."""
    try:
        # Verify Groq client initialization without making an API call
        _ = get_groq_client()
        groq_configured = True
    except Exception as e:
        logger.error("Health check Groq initialization error: %s", e)
        groq_configured = False

    return {
        "status": "healthy" if groq_configured else "degraded",
        "service": "apple-support-agent",
        "provider": "Groq",
        "model": DEFAULT_MODEL,
        "groq_configured": groq_configured,
        "collection": "apple_support_precedents",
        "index_size": 3000,
    }


@app.post(
    "/handle_message",
    response_model=AgentResponse,
    tags=["Agent"],
    summary="Handle a customer support inquiry end-to-end",
)
def handle_customer_message(request: MessageRequest) -> AgentResponse:
    """
    Executes the autonomous agent pipeline for a customer inquiry:
    1. Intent Classification (via Groq LLM few-shot prompt)
    2. Precedent Retrieval (via ChromaDB 3,000 precedent index)
    3. Grounded Reply Drafting
    4. Self-Critique Grounding Verification
    5. Rule-Based Escalation Decision
    """
    try:
        response = handle_message(customer_message=request.message)
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Error executing agent pipeline for inquiry: %s", request.message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agent processing error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.agent.api:app", host="127.0.0.1", port=8000, reload=False)
