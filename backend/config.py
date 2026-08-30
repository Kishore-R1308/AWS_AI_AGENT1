import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(BASE_DIR / ".env")


# =========================
# Groq / LLM Configuration
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# =========================
# SQLite Configuration
# =========================

DATABASE_URL = "sqlite:///./aws_agent.db"


# =========================
# Qdrant Cloud Configuration
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# =========================
# AWS Configuration
# =========================

AWS_SESSION_DURATION = int(
    os.getenv(
        "AWS_SESSION_DURATION",
        "3600",
    )
)