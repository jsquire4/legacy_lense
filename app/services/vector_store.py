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
    return QdrantClient(url=settings.QDRANT_URL, api_key=api_key)


@lru_cache
def get_async_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    api_key = settings.QDRANT_API_KEY or None
    return AsyncQdrantClient(url=settings.QDRANT_URL, api_key=api_key)


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


def ensure_collection() -> None:
    """Create the collection if it doesn't exist, with payload indexes."""
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if collection_name not in existing_names:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            client.create_payload_index(
                collection_name=collection_name,
                field_name="unit_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="file_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="unit_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created collection '%s' with payload indexes", collection_name)
        except Exception as e:
            # Handle TOCTOU race — collection may have been created concurrently
            if "already exists" in str(e).lower():
                logger.info("Collection '%s' was created concurrently", collection_name)
            else:
                raise
    else:
        logger.info("Collection '%s' already exists", collection_name)


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert chunks with their embeddings into Qdrant."""
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

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
            collection_name=collection_name,
            points=batch,
        )
        logger.info("Upserted batch %d-%d of %d points", i, i + len(batch), len(points))


# --- Sync versions (used by ingestion + eval) ---

def search(query_embedding: list[float], top_k: int = 8) -> list[dict]:
    """Search for similar vectors using query_points."""
    settings = get_settings()
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


def search_by_name(query_embedding: list[float], unit_name: str, top_k: int = 3) -> list[dict]:
    """Search with a unit_name filter for exact routine matching."""
    settings = get_settings()
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="unit_name", match=MatchValue(value=unit_name))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


# --- Async versions (used by API endpoints) ---

async def async_search(query_embedding: list[float], top_k: int = 8) -> list[dict]:
    """Async search for similar vectors."""
    settings = get_settings()
    client = get_async_qdrant_client()

    results = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)


async def async_search_by_name(query_embedding: list[float], unit_name: str, top_k: int = 3) -> list[dict]:
    """Async search with a unit_name filter."""
    settings = get_settings()
    client = get_async_qdrant_client()

    results = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="unit_name", match=MatchValue(value=unit_name))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return _format_hits(results.points)
