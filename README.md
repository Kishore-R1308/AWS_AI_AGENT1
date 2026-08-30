# AWS AI Agent

A demo project combining:

- Streamlit frontend
- FastAPI backend
- LangGraph agent
- Groq LLM
- Hugging Face embeddings
- ChromaDB RAG
- Boto3 AWS monitoring tools
- AWS STS AssumeRole cross-account authentication
- PostgreSQL chat history

## 1. Prerequisites

Install:

- Python 3.11+
- Docker Desktop
- VS Code
- AWS account
- Groq API key

## 2. Configure environment

Copy `.env.example` to `.env`.

PowerShell:

```powershell
copy .env.example .env
```

Edit `.env` and add your Groq API key.

## 3. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install packages

```powershell
pip install -r backend\requirements.txt
```

## 5. Start PostgreSQL

```powershell
docker compose up -d
```

## 6. Build the RAG database

```powershell
python scripts\ingest_rag.py
```

The first run downloads the Hugging Face embedding model.

## 7. Start FastAPI

Terminal 1:

```powershell
.venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

API documentation:

http://localhost:8000/docs

## 8. Start Streamlit

Terminal 2:

```powershell
.venv\Scripts\activate
streamlit run frontend\app.py
```

Open:

http://localhost:8501

## 9. AWS cross-account role

The target AWS account should contain a read-only IAM role such as:

AIAgentReadOnlyRole

The role trust policy must trust the source account/principal.

The source identity must have permission to call:

sts:AssumeRole

Suggested demo permissions:

- sts:GetCallerIdentity
- ec2:DescribeInstances
- ec2:DescribeRegions
- ec2:DescribeVolumes
- s3:ListAllMyBuckets
- s3:GetBucketLocation
- s3:ListBucket
- rds:DescribeDBInstances
- rds:DescribeDBClusters

Use least privilege. Do not use AdministratorAccess.

## 10. Demo queries

Knowledge / RAG:

- How do I create an EC2 instance?
- What is an S3 bucket?
- How does RDS Multi-AZ work?
- How do IAM roles work?

Monitoring / Boto3:

- Which EC2 instances are running?
- Show my S3 buckets.
- Which S3 bucket has the highest storage?
- Show my RDS databases.

## Security note

This demo stores temporary STS credentials in backend memory for simplicity.

Do not use this credential storage approach in production.

Production should use a secure authentication/session mechanism and least-privilege IAM.

The S3 storage demo calculates object sizes by listing objects. This is appropriate only for a small demo account. Production should use S3 Storage Lens, Inventory, CloudWatch, or another scalable mechanism.
