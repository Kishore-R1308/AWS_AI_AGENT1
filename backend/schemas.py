from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# AWS CONNECTION
# ============================================================

class AWSConnectRequest(BaseModel):
    access_key: str = Field(
        ...,
        min_length=1,
    )

    secret_key: str = Field(
        ...,
        min_length=1,
    )

    region: str = Field(
        ...,
        min_length=1,
    )

    role_arn: str | None = None


class AWSConnectResponse(BaseModel):
    connected: bool

    account_id: str | None = None

    arn: str | None = None

    region: str | None = None

    message: str

    session_id: str | None = None


# ============================================================
# CHAT
# ============================================================

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
    )

    conversation_id: str = Field(
        ...,
        min_length=1,
    )

    message: str = Field(
        ...,
        min_length=1,
    )


class ChatResponse(BaseModel):
    """
    Response returned by the main /chat endpoint.
    """

    answer: str

    intent: str

    service: str | None = None

    # Structured RCA information
    rca: dict[str, Any] | None = None

    # Recommended actions
    recommendations: list[str] | None = None

    # Conversation-aware resolved query.
    #
    # Example:
    # User: "Why is my EC2 instance failing?"
    # Follow-up: "What about that one?"
    #
    # The agent can resolve the second query into something
    # meaningful using conversation history.
    resolved_query: str | None = None

    # AWS resources explicitly grounded by the agent.
    #
    # Example:
    # ["i-0123456789abcdef0"]
    resource_refs: list[str] = Field(
        default_factory=list
    )


# ============================================================
# CONVERSATIONS
# ============================================================

class ConversationCreateRequest(BaseModel):
    account_id: str = Field(
        ...,
        min_length=1,
    )


class ConversationResponse(BaseModel):
    id: str

    account_id: str

    title: str

    created_at: datetime

    updated_at: datetime


# ============================================================
# CONVERSATION MESSAGES
# ============================================================

class ConversationMessageResponse(BaseModel):
    id: int

    role: str

    content: str

    intent: str | None = None

    service: str | None = None

    rca: dict[str, Any] | None = None

    recommendations: list[str] | None = None

    created_at: datetime


# ============================================================
# AGENT PLANNER
# ============================================================

class Plan(BaseModel):
    """
    Structured output used by the agent planner.
    """

    intent: str

    tools: list[str] = Field(
        default_factory=list
    )