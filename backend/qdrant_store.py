import os
import uuid
import sys
import atexit
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from .config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    VECTOR_SIZE,
    TOP_K
)
from .embeddings import embed_text

# Suppress sentence-transformers / qdrant shutdown noise on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_client = None


def _cleanup_qdrant():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(_cleanup_qdrant)


def get_qdrant_client() -> QdrantClient:
    """
    Returns a connected QdrantClient instance.
    If valid QDRANT_URL (not 'local' or empty) and API key exist, tries cloud;
    otherwise uses local embedded Qdrant on disk at './qdrant_db'.
    """
    global _client
    if _client is not None:
        return _client

    use_cloud = (
        QDRANT_URL
        and QDRANT_URL.lower() != "local"
        and not QDRANT_URL.startswith("local")
        and QDRANT_API_KEY
    )

    if use_cloud:
        try:
            clean_url = QDRANT_URL.rstrip("/")
            client_candidate = QdrantClient(
                url=clean_url,
                api_key=QDRANT_API_KEY,
                timeout=3.0,
                check_compatibility=False
            )
            client_candidate.get_collections()
            _client = client_candidate
            print("[Qdrant] Connected to Qdrant Cloud cluster.")
            return _client
        except Exception:
            print("[Qdrant] Cloud unavailable. Falling back to local embedded storage at './qdrant_db'.")

    os.makedirs("./qdrant_db", exist_ok=True)
    _client = QdrantClient(path="./qdrant_db", check_compatibility=False)
    print("[Qdrant] Using local persistent Qdrant database at './qdrant_db'.")
    return _client


def create_collection():
    client = get_qdrant_client()
    collections = client.get_collections()
    existing_collections = [c.name for c in collections.collections]

    if QDRANT_COLLECTION not in existing_collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"[Qdrant] Created collection: {QDRANT_COLLECTION}")


def store_documents(documents: List[Dict[str, Any]]):
    client = get_qdrant_client()
    create_collection()
    points = []

    for document in documents:
        text = document["text"]
        vector = embed_text(text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "source": document.get("source", "knowledge_base"),
                "chunk_id": document.get("chunk_id", "")
            }
        )
        points.append(point)

    if points:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        print(f"[Qdrant] Stored {len(points)} document chunks in '{QDRANT_COLLECTION}'.")


def search_documents(query: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    create_collection()

    query_vector = embed_text(query)
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    results = []
    for point in response.points:
        results.append({
            "text": point.payload.get("text", "") if point.payload else "",
            "source": point.payload.get("source", "") if point.payload else "",
            "chunk_id": point.payload.get("chunk_id", "") if point.payload else "",
            "score": point.score
        })

    return results