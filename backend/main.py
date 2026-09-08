import json
import uuid
from typing import Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.agent import run_agent
from backend.aws_auth import AWS_SESSIONS, connect_aws
from backend.database import Base, engine, get_db
from backend.models import (
    ChatMessage,
    Conversation,
    ConversationMessage,
)
from backend.rag import (
    retrieve_context,
    get_qdrant_client,
    QDRANT_COLLECTION_NAME,
)
from backend.schemas import (
    AWSConnectRequest,
    AWSConnectResponse,
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationResponse,
    ConversationMessageResponse,
)


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AWS AI Agent",
    description=(
        "AI-powered AWS monitoring, knowledge, "
        "RAG and root cause analysis assistant."
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AWS AI Agent API",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# AWS SESSION HELPERS
# ============================================================

def _get_aws_account(session_id: str):
    """
    Get the AWS session/account associated with a session ID.
    """

    aws_session = AWS_SESSIONS.get(session_id)

    if not aws_session:
        raise HTTPException(
            status_code=401,
            detail="AWS session not found or expired.",
        )

    account_id = aws_session.get("account_id")

    if not account_id:
        raise HTTPException(
            status_code=401,
            detail="AWS account information is unavailable.",
        )

    return aws_session


# ============================================================
# CONVERSATION HELPERS
# ============================================================

def _get_conversation(
    db: Session,
    conversation_id: str,
    account_id: str,
):
    """
    Retrieve a conversation only if it belongs to
    the authenticated AWS account.
    """

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == str(conversation_id),
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
    """
    Generate a conversation title from the first
    user message.
    """

    title = " ".join(
        message.strip().split()
    )

    if not title:
        return "New Chat"

    max_length = 60

    if len(title) <= max_length:
        return title

    return (
        title[:max_length]
        .rstrip()
        + "..."
    )


def _build_conversation_history(messages):
    """
    Convert stored conversation messages into text
    for the planner and RCA nodes.

    Only actual conversation content is included.
    """

    history = []

    for message in messages:
        role = message.role or "user"
        content = message.content or ""

        history.append(
            f"{role}: {content}"
        )

    return "\n".join(history)


# ============================================================
# RESPONSE NORMALIZATION HELPERS
# ============================================================

def _normalize_rca(value) -> dict[str, Any] | None:
    """
    Convert the agent's RCA output into the structure
    expected by ChatResponse.

    If the value is already a dictionary, preserve it.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if not text:
        return None

    return {
        "analysis": text,
    }


def _normalize_recommendations(
    value,
) -> list[str] | None:
    """
    Convert the agent's recommendation output into a list.

    Supports:
    - list output
    - numbered text
    - plain text
    """

    if value is None:
        return None

    if isinstance(value, list):
        result = []

        for item in value:
            text = str(item).strip()

            if text:
                result.append(text)

        return result or None

    text = str(value).strip()

    if not text:
        return None

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    recommendations = []

    current = ""

    for line in lines:

        # ----------------------------------------------------
        # Start of numbered recommendation
        # ----------------------------------------------------

        if (
            len(line) >= 2
            and line[0].isdigit()
            and (
                line[1] == "."
                or line[1] == ")"
            )
        ):
            if current:
                recommendations.append(
                    current.strip()
                )

            current = line

        else:
            if current:
                current += " " + line
            else:
                current = line

    if current:
        recommendations.append(
            current.strip()
        )

    if not recommendations:
        recommendations = [text]

    return recommendations


def _serialize_for_database(value):
    """
    Convert structured values into JSON text for
    SQLite/SQLAlchemy Text columns.
    """

    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def _deserialize_rca(value):
    """
    Convert stored RCA JSON back into a dictionary.
    """

    if not value:
        return None

    if isinstance(value, dict):
        return value

    try:
        parsed = json.loads(value)

        if isinstance(parsed, dict):
            return parsed

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    return {
        "analysis": str(value),
    }


def _deserialize_recommendations(value):
    """
    Convert stored recommendation JSON back into a list.
    """

    if not value:
        return None

    if isinstance(value, list):
        return value

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return [
                str(item)
                for item in parsed
            ]

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    # --------------------------------------------------------
    # Backward compatibility with old
    # plain-text recommendation records.
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in str(value).splitlines()
        if line.strip()
    ]

    return lines or None


# ============================================================
# AWS CONNECT
# ============================================================

@app.post(
    "/aws/connect",
    response_model=AWSConnectResponse,
)
def aws_connect(
    request: AWSConnectRequest,
):
    """
    Connect to AWS using the supplied credentials.
    """

    session_id = str(
        uuid.uuid4()
    )

    try:
        result = connect_aws(
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

    return {
        "connected": True,
        "account_id": result.get(
            "account_id"
        ),
        "arn": result.get(
            "arn"
        ),
        "region": result.get(
            "region"
        ),
        "message": result.get(
            "message",
            "AWS connected successfully.",
        ),
        "session_id": session_id,
    }


# ============================================================
# RAG TEST
# ============================================================

@app.get("/rag-test")
def rag_test(q: str):
    """
    Test Qdrant RAG retrieval.
    """

    try:
        context = retrieve_context(
            q,
            8,
        )

        return {
            "query": q,
            "context": context or "",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval failed: {exc}",
        )


# ============================================================
# QDRANT RAW TEST
# ============================================================

@app.get("/qdrant-raw-test")
def qdrant_raw_test():
    """
    Inspect a small number of raw points from Qdrant.
    """

    try:
        client = get_qdrant_client()

        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=2,
            with_payload=True,
            with_vectors=False,
        )

        return {
            "collection": QDRANT_COLLECTION_NAME,
            "count": len(points),
            "points": [
                {
                    "id": point.id,
                    "payload": point.payload,
                }
                for point in points
            ],
            "next_offset": next_offset,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant test failed: {exc}",
        )


# ============================================================
# CREATE CONVERSATION
# ============================================================

@app.post(
    "/conversations",
    response_model=ConversationResponse,
)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
):
    """
    Create exactly one new conversation.

    The frontend should call this only when the user
    explicitly creates a new chat.
    """

    conversation_id = str(
        uuid.uuid4()
    )

    conversation = Conversation(
        id=conversation_id,
        account_id=request.account_id,
        title="New Chat",
    )

    try:
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to create conversation: "
                f"{exc}"
            ),
        )

    return conversation


# ============================================================
# LIST CONVERSATIONS
# ============================================================

@app.get(
    "/conversations/{account_id}",
    response_model=list[ConversationResponse],
)
def list_conversations(
    account_id: str,
    db: Session = Depends(get_db),
):
    """
    Return conversations belonging to an AWS account.
    """

    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.account_id
            == account_id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )

    return conversations


# ============================================================
# GET CONVERSATION MESSAGES
# ============================================================

@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
)
def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    """
    Return all messages belonging to a conversation.
    """

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id
            == str(conversation_id)
        )
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
            ConversationMessage.conversation_id
            == str(conversation_id)
        )
        .order_by(
            ConversationMessage.created_at.asc()
        )
        .all()
    )

    result = []

    for message in messages:
        result.append(
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "intent": message.intent,
                "service": message.service,
                "rca": _deserialize_rca(
                    message.rca
                ),
                "recommendations": (
                    _deserialize_recommendations(
                        message.recommendations
                    )
                ),
                "created_at": message.created_at,
            }
        )

    return result


# ============================================================
# DELETE CONVERSATION
# ============================================================

@app.delete(
    "/conversations/{conversation_id}"
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete a conversation and its messages.
    """

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id
            == str(conversation_id)
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    try:
        db.delete(conversation)
        db.commit()

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete conversation: "
                f"{exc}"
            ),
        )

    return {
        "success": True,
        "message": "Conversation deleted.",
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Main AWS AI Agent chat endpoint.

    Flow:

        User
          ↓
        Validate AWS session
          ↓
        Load conversation
          ↓
        Build conversation history
          ↓
        Agent planner
          ↓
        Knowledge / Monitoring / RCA
          ↓
        Save response
          ↓
        Return answer
    """

    # --------------------------------------------------------
    # Validate AWS session
    # --------------------------------------------------------

    aws_session = _get_aws_account(
        request.session_id
    )

    account_id = aws_session.get(
        "account_id"
    )

    # --------------------------------------------------------
    # Validate conversation
    # --------------------------------------------------------

    conversation_id = str(
        request.conversation_id
    ).strip()

    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail="conversation_id is required.",
        )

    conversation = _get_conversation(
        db=db,
        conversation_id=conversation_id,
        account_id=account_id,
    )

    # --------------------------------------------------------
    # Validate user message
    # --------------------------------------------------------

    query = request.message.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # --------------------------------------------------------
    # Load previous messages
    # --------------------------------------------------------

    previous_messages = (
        db.query(ConversationMessage)
        .filter(
            ConversationMessage.conversation_id
            == conversation.id
        )
        .order_by(
            ConversationMessage.created_at.asc()
        )
        .all()
    )

    conversation_history = (
        _build_conversation_history(
            previous_messages
        )
    )

    # --------------------------------------------------------
    # Run AI agent
    # --------------------------------------------------------

    try:
        result = run_agent(
            session_id=request.session_id,
            query=query,
            conversation_history=conversation_history,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        )

    # --------------------------------------------------------
    # Extract agent result
    # --------------------------------------------------------

    answer = str(
        result.get(
            "answer",
            "",
        )
    )

    intent = str(
        result.get(
            "intent",
            "KNOWLEDGE",
        )
    )

    service = result.get(
        "service"
    )

    if service:
        service = str(
            service
        )

    rca = _normalize_rca(
        result.get("rca")
    )

    recommendations = (
        _normalize_recommendations(
            result.get(
                "recommendations"
            )
        )
    )

    # --------------------------------------------------------
    # Extract resolved query and resource references
    # --------------------------------------------------------

    resolved_query = result.get(
        "resolved_query",
        query,
    )

    if resolved_query is None:
        resolved_query = query

    resolved_query = str(
        resolved_query
    ).strip()

    resource_refs = result.get(
        "resource_refs",
        [],
    )

    if resource_refs is None:
        resource_refs = []

    elif isinstance(resource_refs, str):
        resource_refs = [
            resource_refs
        ]

    elif not isinstance(
        resource_refs,
        list,
    ):
        resource_refs = [
            str(resource_refs)
        ]

    resource_refs = [
        str(ref).strip()
        for ref in resource_refs
        if str(ref).strip()
    ]

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    user_message = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content=query,
        intent=None,
        service=None,
        rca=None,
        recommendations=None,
    )

    db.add(user_message)

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    assistant_message = ConversationMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        intent=intent,
        service=service or None,
        rca=_serialize_for_database(
            rca
        ),
        recommendations=(
            _serialize_for_database(
                recommendations
            )
        ),
    )

    db.add(assistant_message)

    # --------------------------------------------------------
    # Set title from first user message
    # --------------------------------------------------------

    if not previous_messages:
        conversation.title = _make_title(
            query
        )

    # --------------------------------------------------------
    # Update conversation timestamp
    # --------------------------------------------------------

    from datetime import datetime

    conversation.updated_at = (
        datetime.utcnow()
    )

    # --------------------------------------------------------
    # Commit conversation
    # --------------------------------------------------------

    try:
        db.add(conversation)

        db.commit()

        db.refresh(
            assistant_message
        )

        db.refresh(
            conversation
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save conversation: "
                f"{exc}"
            ),
        )

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "answer": answer,
        "intent": intent,
        "service": service or None,
        "rca": rca,
        "recommendations": recommendations,
        "resolved_query": resolved_query,
        "resource_refs": resource_refs,
    }


# ============================================================
# LEGACY CHAT HISTORY
# ============================================================

@app.get(
    "/history/{account_id}"
)
def get_legacy_history(
    account_id: str,
    db: Session = Depends(get_db),
):
    """
    Legacy endpoint retained for backward compatibility.
    """

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id
            == account_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": message.id,
            "session_id": message.session_id,
            "user_message": message.user_message,
            "assistant_message": message.assistant_message,
            "intent": message.intent,
            "service": message.service,
            "rca": _deserialize_rca(
                message.rca
            ),
            "recommendations": (
                _deserialize_recommendations(
                    message.recommendations
                )
            ),
            "created_at": message.created_at,
        }
        for message in messages
    ]