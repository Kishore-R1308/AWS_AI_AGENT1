from __future__ import annotations

import hashlib
import re
import uuid
from typing import Iterable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from langchain_qdrant import QdrantVectorStore

from backend.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = QDRANT_COLLECTION_NAME

VECTOR_SIZE = 384

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

DEFAULT_RETRIEVAL_K = 8


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
    Load the HuggingFace embedding model once and reuse it.
    """

    global _embeddings

    if _embeddings is None:

        if not EMBEDDING_MODEL_NAME:
            raise ValueError(
                "EMBEDDING_MODEL_NAME environment variable "
                "is not set."
            )

        print("=" * 70)
        print("LOADING EMBEDDING MODEL")
        print("=" * 70)

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

        print(
            f"Embedding model loaded: "
            f"{EMBEDDING_MODEL_NAME}"
        )

    return _embeddings


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """
    Create and cache the Qdrant Cloud client.
    """

    global _qdrant_client

    if _qdrant_client is None:

        if not QDRANT_URL:
            raise ValueError(
                "QDRANT_URL environment variable "
                "is not set."
            )

        if not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_API_KEY environment variable "
                "is not set."
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
    Create the Qdrant collection if it does not exist.
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
    Create and cache the recursive text splitter.
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
    Create and cache the LangChain Qdrant vector store.
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
    """
    Clean extracted document text before chunking.
    """

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
    """
    Generate a deterministic SHA-256 ID for each chunk.
    """

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

    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(
    documents: Iterable[Document],
) -> list[Document]:
    """
    Normalize and split documents into chunks while
    preserving their metadata.
    """

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
                "chunk_size": len(chunk_text),
            }

            chunk_id = make_document_id(
                source=source,
                page=page,
                chunk_index=chunk_index,
                text=chunk_text,
                metadata=metadata,
            )

            metadata["chunk_id"] = chunk_id

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
    """
    Build a Qdrant filter for a specific document source.

    LangChain Qdrant stores Document metadata under
    the `metadata` payload field.
    """

    return Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(
                    value=source,
                ),
            )
        ]
    )


# ============================================================
# DELETE DOCUMENTS BY SOURCE
# ============================================================

def delete_documents_by_source(
    source: str,
) -> int:
    """
    Delete all chunks belonging to a source.
    """

    if not source:
        return 0

    client = get_qdrant_client()

    ensure_collection()

    print()
    print("=" * 70)
    print("REMOVING OLD DOCUMENT CHUNKS")
    print("=" * 70)

    print(
        f"Source: {source}"
    )

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
# SOURCE EXISTS
# ============================================================

def source_exists(
    source: str,
) -> bool:
    """
    Check whether a source already exists in Qdrant.
    """

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
# SOURCE CHUNK COUNT
# ============================================================

def get_source_chunk_count(
    source: str,
) -> int:
    """
    Return the number of chunks belonging to a source.
    """

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
    """
    Split, deduplicate and insert documents into Qdrant.
    """

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
    # DEDUPLICATE CHUNKS
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
    indexed_count = 0

    print()
    print("=" * 70)
    print("STARTING QDRANT INDEXING")
    print("=" * 70)

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

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
    """
    Remove all existing chunks for a source and index
    the new document chunks.
    """

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
# PAYLOAD EXTRACTION
# ============================================================

def _extract_document_from_payload(
    payload: dict,
) -> Document | None:
    """
    Convert a Qdrant payload into a LangChain Document.

    Handles the standard LangChain Qdrant payload format:

        {
            "page_content": "...",
            "metadata": {...}
        }

    and also supports flatter/custom payload structures
    for compatibility with existing collections.
    """

    if not payload:
        return None

    # --------------------------------------------------------
    # STANDARD LANGCHAIN FORMAT
    # --------------------------------------------------------

    if "page_content" in payload:

        text = payload.get(
            "page_content"
        )

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        if text:

            return Document(
                page_content=str(text),
                metadata=metadata,
            )

    # --------------------------------------------------------
    # CUSTOM / LEGACY FORMAT
    # --------------------------------------------------------

    text = (
        payload.get("text")
        or payload.get("content")
        or payload.get("page_content")
    )

    if not text:
        return None

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    # Preserve top-level fields if present
    for key in (
        "source",
        "document_name",
        "page",
        "content_type",
        "chunk_id",
        "chunk_index",
    ):

        if key in payload and key not in metadata:

            metadata[key] = payload[key]

    return Document(
        page_content=str(text),
        metadata=metadata,
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:
    """
    Retrieve the most relevant documents from Qdrant.
    """

    if not query:
        return []

    if k <= 0:
        return []

    client = get_qdrant_client()

    ensure_collection()

    query_vector = get_embeddings().embed_query(
        query
    )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    documents = []

    for point in result.points:

        payload = point.payload or {}

        document = (
            _extract_document_from_payload(
                payload
            )
        )

        if document is None:
            continue

        # Keep similarity score available
        document.metadata[
            "score"
        ] = getattr(
            point,
            "score",
            None,
        )

        # Preserve Qdrant point ID
        document.metadata[
            "qdrant_point_id"
        ] = point.id

        documents.append(
            document
        )

    return documents


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> str:
    """
    Convert retrieved Qdrant documents into context for the LLM.
    """

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
            document.metadata.get(
                "document_name",
                "Unknown source",
            ),
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        content_type = document.metadata.get(
            "content_type",
            "text",
        )

        score = document.metadata.get(
            "score"
        )

        score_text = (
            f"{score:.4f}"
            if isinstance(
                score,
                (int, float),
            )
            else "N/A"
        )

        parts.append(
            f"SOURCE {index}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content Type: {content_type}\n"
            f"Relevance Score: {score_text}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# RETRIEVED SOURCES
# ============================================================

def get_retrieved_sources(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[dict]:
    """
    Return source metadata for retrieved documents.
    """

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

                "score": document.metadata.get(
                    "score"
                ),

                "qdrant_point_id": document.metadata.get(
                    "qdrant_point_id"
                ),
            }
        )

    return sources


# ============================================================
# COLLECTION RESET
# ============================================================

def reset_collection():
    """
    Delete the entire Qdrant collection.

    The collection will be recreated automatically the next
    time ensure_collection() or get_vectorstore() is called.
    """

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

        raise
