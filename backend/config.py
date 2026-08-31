import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# =========================
# GROQ
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# =========================
# SQLITE
# =========================

DATABASE_URL = "sqlite:///./aws_agent.db"


# =========================
# QDRANT CLOUD
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# =========================
# AWS
# =========================

AWS_SESSION_DURATION = int(
    os.getenv(
        "AWS_SESSION_DURATION",
        "3600",
    )
)