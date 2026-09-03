import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.rag import retrieve_context
from backend.agent import run_agent
from backend.aws_auth import AWS_SESSIONS, connect_aws
from backend.database import Base, engine, get_db
from backend.models import ChatMessage, Conversation, ConversationMessage
from backend.schemas import (
    AWSConnectRequest,
    AWSConnectResponse,
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AWS AI Agent",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/rag-test")
def rag_test(q: str):
    context = retrieve_context(q)
    return {"query": q, "context": context}


@app.get("/")
def root():
    return {"message": "AWS AI Agent API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/aws/connect", response_model=AWSConnectResponse)
def aws_connect(request: AWSConnectRequest):
    session_id = str(uuid.uuid4())

    try:
        return connect_aws(
            session_id=session_id,
            access_key=request.access_key,
            secret_key=request.secret_key,
            region=request.region,
            role_arn=request.role_arn,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"AWS connection failed: {exc}",
        )


@app.get("/qdrant-raw-test")
def qdrant_raw_test():
    from backend.rag import get_qdrant_client, COLLECTION_NAME

    client = get_qdrant_client()

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2,
        with_payload=True,
        with_vectors=False,
    )

    return {
        "collection": COLLECTION_NAME,
        "points": [
            {
                "id": str(point.id),
                "payload": point.payload,
            }
            for point in points
        ],
    }


# -------------------------------------------------------------------
# ChatGPT-style conversation history
# -------------------------------------------------------------------

def _get_aws_account(session_id: str) -> str:
    aws_session = AWS_SESSIONS.get(session_id)

    if not aws_session:
        raise HTTPException(
            status_code=400,
            detail="Invalid AWS session. Please connect to AWS first.",
        )

    account_id = aws_session.get("account_id")

    if not account_id:
        raise HTTPException(
            status_code=400,
            detail="AWS account ID not found for this session.",
        )

    return account_id


def _get_conversation(
    db: Session,
    conversation_id: str,
    account_id: str,
) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.account_id == account_id,
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


def _make_title(message: str) -> str:
    # Simple deterministic title: no extra LLM call.
    title = " ".join(message.strip().split())

    if not title:
        return "New Chat"

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    return title


@app.post("/conversations")
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
):
    conversation = Conversation(
        id=str(uuid.uuid4()),
        account_id=request.account_id,
        title="New Chat",
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": str(conversation.created_at),
        "updated_at": str(conversation.updated_at),
    }


@app.get("/conversations/{account_id}")
def list_conversations(
    account_id: str,
    db: Session = Depends(get_db),
):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.account_id == account_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    return [
        {
            "id": item.id,
            "title": item.title,
            "created_at": str(item.created_at),
            "updated_at": str(item.updated_at),
        }
        for item in conversations
    ]


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.conversation_id == conversation_id
        )
        .order_by(ConversationMessage.created_at.asc())
        .all()
    )

    return [
        {
            "id": item.id,
            "role": item.role,
            "content": item.content,
            "intent": item.intent,
            "service": item.service,
            "created_at": str(item.created_at),
        }
        for item in messages
    ]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully."}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    account_id = _get_aws_account(request.session_id)

    conversation = _get_conversation(
        db,
        request.conversation_id,
        account_id,
    )

    try:
        result = run_agent(
            session_id=request.session_id,
            query=request.message,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # First user message becomes the chat title.
    existing_user_message = (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.conversation_id == conversation.id,
            ConversationMessage.role == "user",
        )
        .first()
    )

    if not existing_user_message:
        conversation.title = _make_title(request.message)

    user_record = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )

    assistant_record = ConversationMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        intent=result.get("intent"),
        service=result.get("service"),
    )

    db.add(user_record)
    db.add(assistant_record)

    conversation.updated_at = datetime.utcnow()

    db.commit()

    return result


# Legacy endpoint retained so old clients do not break.
@app.get("/history/{account_id}")
def legacy_history(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(ChatMessage.account_id == account_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return [
        {
            "id": record.id,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "intent": record.intent,
            "service": record.service,
            "created_at": str(record.created_at),
        }
        for record in records
    ]
