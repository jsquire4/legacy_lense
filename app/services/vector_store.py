"""Qdrant vector store operations."""

import logging
import uuid

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)
from functools import lru_cache

from app.config import get_settings
from app.services.chunker import Chunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    api_key = settings.QDRANT_API_KEY or None
    return QdrantClient(url=settings.QDRANT_URL, api_key=api_key, timeout=30)


@lru_cache
def get_async_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    api_key = settings.QDRANT_API_KEY or None
    return AsyncQdrantClient(url=settings.QDRANT_URL, api_key=api_key, timeout=30)


def _resolve_collection(collection_name: str | None = None) -> str:
    """Resolve collection name, defaulting to settings."""
    if collection_name:
        return collection_name
    return get_settings().QDRANT_COLLECTION_NAME


def _format_hits(points) -> list[dict]:
    """Convert Qdrant points to hit dicts."""
    return [
        {
            "id": point.id,
            "score": point.score,
            "text": point.payload.get("text", ""),
            "metadata": {k: v for k, v in point.payload.items() if k != "text"},
        }
        for point in points
    ]


def ensure_collection(collection_name: str | None = None, embedding_dim: int | None = None) -> None:
    """Create the collection if it doesn't exist, with payload indexes."""
    settings = get_settings()
    client = get_qdrant_client()
    coll = _resolve_collection(collection_name)
    dim = embedding_dim or settings.EMBEDDING_DIM

    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if coll not in existing_names:
        try:
            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            for field_name in ("unit_type", "file_path", "unit_name", "called_by"):
                client.create_payload_index(
                    collection_name=coll,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("Created collection '%s' (dim=%d) with payload indexes", coll, dim)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Collection '%s' was created concurrently", coll)
            else:
                raise
    else:
        logger.info("Collection '%s' already exists", coll)


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]],
                   collection_name: str | None = None) -> None:
    """Upsert chunks with their embeddings into Qdrant."""
    client = get_qdrant_client()
    coll = _resolve_collection(collection_name)

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": chunk.text,
                **chunk.metadata,
            },
        ))

    # Upload in batches
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=coll,
            points=batch,
        )
        logger.info("Upserted batch %d-%d of %d points", i, i + len(batch), len(points))


def delete_collection(collection_name: str) -> bool:
    """Delete a Qdrant collection. Returns True if it existed."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name)
        logger.info("Deleted collection '%s'", collection_name)
        return True
    return False


# --- Sync versions (used by ingestion + eval) ---

def search(query_embedding: list[float], top_k: int = 8,
           collection_name: str | None = None) -> list[dict]:
    """Search for similar vectors using query_points."""
    client = get_qdrant_client()
    coll = _resolve_collection(collection_name)

    results = client.query_points(
        collection_name=coll,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


def search_by_name(query_embedding: list[float], unit_name: str, top_k: int = 3,
                   collection_name: str | None = None) -> list[dict]:
    """Search with a unit_name filter for exact routine matching."""
    client = get_qdrant_client()
    coll = _resolve_collection(collection_name)

    results = client.query_points(
        collection_name=coll,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="unit_name", match=MatchValue(value=unit_name))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


# --- Async versions (used by API endpoints) ---

async def async_search(query_embedding: list[float], top_k: int = 8,
                       collection_name: str | None = None) -> list[dict]:
    """Async search for similar vectors."""
    client = get_async_qdrant_client()
    coll = _resolve_collection(collection_name)

    results = await client.query_points(
        collection_name=coll,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


async def async_search_by_name(query_embedding: list[float], unit_name: str, top_k: int = 3,
                               collection_name: str | None = None) -> list[dict]:
    """Async search with a unit_name filter."""
    client = get_async_qdrant_client()
    coll = _resolve_collection(collection_name)

    results = await client.query_points(
        collection_name=coll,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="unit_name", match=MatchValue(value=unit_name))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


async def async_search_by_caller(query_embedding: list[float], routine_name: str,
                                  top_k: int = 5, collection_name: str | None = None) -> list[dict]:
    """Find chunks whose called_routines include the given routine name."""
    client = get_async_qdrant_client()
    coll = _resolve_collection(collection_name)

    results = await client.query_points(
        collection_name=coll,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="called_routines", match=MatchValue(value=routine_name))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)
