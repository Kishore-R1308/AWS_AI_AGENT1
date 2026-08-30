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
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    service: Optional[str] = None

class Plan(BaseModel):
    intent: str
    tools: List[str]
