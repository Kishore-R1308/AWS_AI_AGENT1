import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from rag import retrieve_context

from agent import run_agent
from aws_auth import AWS_SESSIONS, connect_aws
from database import Base, engine, get_db
from models import ChatMessage
from schemas import (
    AWSConnectRequest,
    AWSConnectResponse,
    ChatRequest,
    ChatResponse,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AWS AI Agent",
    version="1.0.0",
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

    return {
        "query": q,
        "context": context
    }


@app.get("/")
def root():
    return {
        "message": "AWS AI Agent API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post(
    "/aws/connect",
    response_model=AWSConnectResponse,
)
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
    from rag import get_qdrant_client, COLLECTION_NAME

    client = get_qdrant_client()

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2,
        with_payload=True,
        with_vectors=False,
    )

    result = []

    for point in points:
        result.append({
            "id": str(point.id),
            "payload": point.payload,
        })

    return {
        "collection": COLLECTION_NAME,
        "points": result,
    }

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
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

    aws_sessions=AWS_SESSIONS.get(request.session_id)
    if not aws_sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id. Please connect to AWS first.",
        )

    account_id=aws_sessions.get("account_id")
    if not account_id:
        raise HTTPException(
            status_code=400,
            detail="Account ID not found for the given session_id.",
        )

    record = ChatMessage(
        account_id=account_id,
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=result["answer"],
        intent=result["intent"],
        service=result.get("service"),
    )

    db.add(record)
    db.commit()

    return result


@app.get("/history/{account_id}")
def history(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
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
