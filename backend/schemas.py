from typing import List, Optional

from pydantic import BaseModel


class AWSConnectRequest(BaseModel):
    access_key: str
    secret_key: str
    region: str
    role_arn: str


class AWSConnectResponse(BaseModel):
    connected: bool
    account_id: str
    arn: str
    region: str
    message: str
    session_id: str


class ChatRequest(BaseModel):
    # AWS authentication session. Do not use this as the chat ID.
    session_id: str
    message: str
    conversation_id: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    service: Optional[str] = None


class ConversationCreateRequest(BaseModel):
    account_id: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    intent: Optional[str] = None
    service: Optional[str] = None
    created_at: str


class Plan(BaseModel):
    intent: str
    tools: List[str]
