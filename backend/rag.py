from __future__ import annotations

import hashlib
import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from langchain_qdrant import QdrantVectorStore

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
)


# ============================================================
# CONFIGURATION
# ============================================================
COLLECTION_NAME = "aws_documents"

# all-MiniLM-L6-v2 embedding dimension
VECTOR_SIZE = 384

# Chunk configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

# Retrieval configuration
DEFAULT_RETRIEVAL_K = 8
FETCH_K = 30


# ============================================================
# GLOBAL OBJECTS
# ============================================================

_embeddings = None
_vectorstore = None
_qdrant_client = None
_splitter = None


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    """
    Load the embedding model only once.
    """

    global _embeddings

    if _embeddings is None:

        print("=" * 70)
        print("LOADING EMBEDDING MODEL")
        print("=" * 70)

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",

            model_kwargs={
                "device": "cpu",
            },

            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

        print(
            "Embedding model loaded successfully."
        )

    return _embeddings


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """
    Create Qdrant Cloud client only once.
    """

    global _qdrant_client

    if _qdrant_client is None:

        if not QDRANT_URL:
            raise ValueError(
                "QDRANT_URL environment variable is not set."
            )

        if not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_API_KEY environment variable is not set."
            )

        print("=" * 70)
        print("CONNECTING TO QDRANT CLOUD")
        print("=" * 70)

        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

        print(
            "Connected to Qdrant Cloud."
        )

    return _qdrant_client


# ============================================================
# ENSURE COLLECTION
# ============================================================

def ensure_collection():
    """
    Create the Qdrant collection if it does not already exist.
    """

    client = get_qdrant_client()

    collections = client.get_collections()

    collection_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME not in collection_names:

        print("=" * 70)
        print("CREATING QDRANT COLLECTION")
        print("=" * 70)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:

        print(
            f"Qdrant collection already exists: "
            f"{COLLECTION_NAME}"
        )


# ============================================================
# TEXT SPLITTER
# ============================================================

def get_splitter():
    """
    Create the text splitter only once.
    """

    global _splitter

    if _splitter is None:

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,

            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    return _splitter


# ============================================================
# VECTOR STORE
# ============================================================

def get_vectorstore():
    """
    Connect to the Qdrant Cloud vector store.
    """

    global _vectorstore

    if _vectorstore is None:

        print("=" * 70)
        print("LOADING QDRANT VECTOR STORE")
        print("=" * 70)

        ensure_collection()

        _vectorstore = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )

        print(
            f"Qdrant collection loaded: "
            f"{COLLECTION_NAME}"
        )

    return _vectorstore


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# DOCUMENT ID
# ============================================================

def make_document_id(
    source: str,
    page: int | None,
    chunk_index: int,
    text: str,
    metadata: dict | None = None,
) -> str:

    metadata = metadata or {}

    stable_metadata = {
        "content_type": metadata.get(
            "content_type",
            "text",
        ),

        "image_index": metadata.get(
            "image_index",
        ),

        "table_index": metadata.get(
            "table_index",
        ),

        "block_index": metadata.get(
            "block_index",
        ),

        "image_id": metadata.get(
            "image_id",
        ),

        "table_id": metadata.get(
            "table_id",
        ),

        "image_fingerprint": metadata.get(
            "image_fingerprint",
        ),

        "page": page,
    }

    metadata_string = repr(
        sorted(
            stable_metadata.items()
        )
    )

    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{metadata_string}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(
    documents: Iterable[Document],
) -> list[Document]:

    splitter = get_splitter()

    chunks: list[Document] = []

    for document in documents:

        text = normalize_text(
            document.page_content
        )

        if not text:
            continue

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        split_texts = splitter.split_text(
            text
        )

        for chunk_index, chunk_text in enumerate(
            split_texts
        ):

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            metadata = {
                **document.metadata,

                "source": source,

                "page": page,

                "chunk_index": chunk_index,

                "chunk_size": len(
                    chunk_text
                ),
            }

            metadata["chunk_id"] = make_document_id(
                source=source,
                page=page,
                chunk_index=chunk_index,
                text=chunk_text,
                metadata=metadata,
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    return chunks


# ============================================================
# SOURCE FILTER
# ============================================================

def get_source_filter(source: str):
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(
                    value=source
                ),
            )
        ]
    )


# ============================================================
# REMOVE DOCUMENT BY SOURCE
# ============================================================

def delete_documents_by_source(
    source: str,
) -> int:

    if not source:
        return 0

    client = get_qdrant_client()

    ensure_collection()

    try:

        print()
        print("=" * 70)
        print("REMOVING OLD DOCUMENT CHUNKS")
        print("=" * 70)

        print(
            f"Source: {source}"
        )

        source_filter = get_source_filter(
            source
        )

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=source_filter,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )

        if not points:

            print(
                "No existing chunks found."
            )

            return 0

        ids = [
            point.id
            for point in points
        ]

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids,
        )

        print(
            f"Deleted chunks: {len(ids)}"
        )

        return len(ids)

    except Exception as exc:

        print(
            "Could not delete old document chunks."
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Error: {exc}"
        )

        raise


# ============================================================
# CHECK SOURCE EXISTS
# ============================================================

def source_exists(
    source: str,
) -> bool:

    if not source:
        return False

    client = get_qdrant_client()

    ensure_collection()

    try:

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=get_source_filter(
                source
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return bool(points)

    except Exception as exc:

        print(
            f"Could not check source "
            f"{source}: {exc}"
        )

        return False


# ============================================================
# GET SOURCE CHUNK COUNT
# ============================================================

def get_source_chunk_count(
    source: str,
) -> int:

    if not source:
        return 0

    client = get_qdrant_client()

    ensure_collection()

    try:

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=get_source_filter(
                source
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )

        return len(points)

    except Exception as exc:

        print(
            f"Could not count chunks "
            f"for {source}: {exc}"
        )

        return 0


# ============================================================
# ADD DOCUMENTS
# ============================================================

def add_documents(
    documents: list[Document],
    batch_size: int = 16,
) -> int:

    if not documents:

        print(
            "No documents received for indexing."
        )

        return 0

    print()
    print("=" * 70)
    print("SPLITTING DOCUMENTS")
    print("=" * 70)

    chunks = split_documents(
        documents
    )

    if not chunks:

        print(
            "No chunks generated."
        )

        return 0

    print(
        f"Total chunks generated: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE IDS
    # --------------------------------------------------------

    unique_chunks: list[Document] = []

    seen_ids: set[str] = set()

    duplicate_count = 0

    for document in chunks:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if not chunk_id:

            chunk_id = hashlib.sha256(
                document.page_content.encode(
                    "utf-8"
                )
            ).hexdigest()

            document.metadata[
                "chunk_id"
            ] = chunk_id

        if chunk_id in seen_ids:

            duplicate_count += 1

            continue

        seen_ids.add(
            chunk_id
        )

        unique_chunks.append(
            document
        )

    chunks = unique_chunks

    print(
        f"Duplicate chunks removed: "
        f"{duplicate_count}"
    )

    print(
        f"Unique chunks to index: "
        f"{len(chunks)}"
    )

    if not chunks:
        return 0

    vectorstore = get_vectorstore()

    total = len(chunks)

    print()
    print("=" * 70)
    print("STARTING QDRANT INDEXING")
    print("=" * 70)

    indexed_count = 0

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        unique_batch = []

        batch_ids = set()

        for document in batch:

            chunk_id = document.metadata[
                "chunk_id"
            ]

            if chunk_id in batch_ids:
                continue

            batch_ids.add(
                chunk_id
            )

            unique_batch.append(
                document
            )

        batch = unique_batch

        if not batch:
            continue

        ids = [
            document.metadata[
                "chunk_id"
            ]
            for document in batch
        ]

        vectorstore.add_documents(
            documents=batch,
            ids=ids,
        )

        indexed_count += len(batch)

        print(
            f"Indexed "
            f"{indexed_count}/{total} chunks"
        )

    print()
    print("=" * 70)
    print("QDRANT INDEXING COMPLETED")
    print("=" * 70)

    print(
        f"Total unique chunks indexed: "
        f"{indexed_count}"
    )

    return indexed_count


# ============================================================
# REPLACE DOCUMENT
# ============================================================

def replace_document(
    source: str,
    documents: list[Document],
    batch_size: int = 16,
) -> int:

    if not source:

        raise ValueError(
            "Source cannot be empty."
        )

    print()
    print("=" * 70)
    print("REPLACING DOCUMENT")
    print("=" * 70)

    print(
        f"Source: {source}"
    )

    deleted_count = (
        delete_documents_by_source(
            source
        )
    )

    print(
        f"Old chunks deleted: "
        f"{deleted_count}"
    )

    if not documents:

        print(
            "No new documents generated."
        )

        return 0

    indexed_count = add_documents(
        documents=documents,
        batch_size=batch_size,
    )

    print()
    print(
        f"Replacement completed for: "
        f"{source}"
    )

    print(
        f"New chunks indexed: "
        f"{indexed_count}"
    )

    return indexed_count


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:

    if not query:
        return []

    vectorstore = get_vectorstore()

    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=FETCH_K,
        lambda_mult=0.65,
    )


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> str:

    documents = retrieve_documents(
        query=query,
        k=k,
    )

    if not documents:

        return (
            "No relevant AWS documentation "
            "was found."
        )

    parts = []

    for index, document in enumerate(
        documents,
        1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        content_type = document.metadata.get(
            "content_type",
            "text",
        )

        parts.append(
            f"SOURCE {index}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content Type: {content_type}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        parts
    )


# ============================================================
# RETRIEVED SOURCES
# ============================================================

def get_retrieved_sources(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[dict]:

    documents = retrieve_documents(
        query=query,
        k=k,
    )

    sources = []

    for document in documents:

        sources.append(
            {
                "source": document.metadata.get(
                    "source"
                ),

                "page": document.metadata.get(
                    "page"
                ),

                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),

                "content_type": document.metadata.get(
                    "content_type",
                    "text",
                ),
            }
        )

    return sources


# ============================================================
# COLLECTION RESET
# ============================================================

def reset_collection():

    global _vectorstore

    client = get_qdrant_client()

    try:

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        _vectorstore = None

        print()
        print("=" * 70)
        print("QDRANT COLLECTION RESET")
        print("=" * 70)

        print(
            f"Deleted collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception as exc:

        print(
            "Could not reset Qdrant collection:"
        )

        print(
            str(exc)
        )