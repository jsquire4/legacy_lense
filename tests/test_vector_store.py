"""Tests for the vector store service (mocked, no Qdrant calls)."""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.services.vector_store import (
    ensure_collection,
    upsert_chunks,
    search,
    search_by_name,
    async_search,
    async_search_by_name,
    async_search_by_caller,
    delete_collection,
    get_async_qdrant_client,
)
from app.services.chunker import Chunk


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_ensure_collection_creates_new(mock_settings, mock_client_fn):
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test_collection"
    settings.EMBEDDING_DIM = 1536
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client_fn.return_value = mock_client

    ensure_collection()
    mock_client.create_collection.assert_called_once()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_ensure_collection_skips_existing(mock_settings, mock_client_fn):
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test_collection"
    mock_settings.return_value = settings

    existing = MagicMock()
    existing.name = "test_collection"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [existing]
    mock_client_fn.return_value = mock_client

    ensure_collection()
    mock_client.create_collection.assert_not_called()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_upsert_chunks(mock_settings, mock_client_fn):
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client

    chunks = [
        Chunk(text="test text", metadata={"file_path": "test.f", "unit_name": "TEST"}),
    ]
    embeddings = [[0.1] * 1536]

    upsert_chunks(chunks, embeddings)
    mock_client.upsert.assert_called_once()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_search_with_explicit_collection_name(mock_settings, mock_client_fn):
    """search with collection_name uses it directly (line 43)."""
    mock_settings.return_value = MagicMock(QDRANT_COLLECTION_NAME="default")
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = []
    mock_client_fn.return_value = mock_client

    results = search([0.1] * 1536, top_k=5, collection_name="custom-collection")
    assert results == []
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "custom-collection"


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_search_uses_query_points(mock_settings, mock_client_fn):
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_point = MagicMock()
    mock_point.id = "abc"
    mock_point.score = 0.95
    mock_point.payload = {"text": "source code", "file_path": "test.f"}

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [mock_point]
    mock_client_fn.return_value = mock_client

    results = search([0.1] * 1536, top_k=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.95
    mock_client.query_points.assert_called_once()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_search_by_name_uses_filter(mock_settings, mock_client_fn):
    """search_by_name calls query_points with unit_name filter."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_point = MagicMock()
    mock_point.id = "dgesv-1"
    mock_point.score = 0.98
    mock_point.payload = {"text": "DGESV code", "unit_name": "DGESV", "file_path": "dgesv.f"}

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [mock_point]
    mock_client_fn.return_value = mock_client

    results = search_by_name([0.1] * 1536, "DGESV", top_k=3)
    assert len(results) == 1
    assert results[0]["metadata"]["unit_name"] == "DGESV"
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert "query_filter" in call_kwargs


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_ensure_collection_creates_payload_indexes(mock_settings, mock_client_fn):
    """ensure_collection creates payload indexes when creating new collection."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    settings.EMBEDDING_DIM = 1536
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client_fn.return_value = mock_client

    ensure_collection()
    assert mock_client.create_payload_index.call_count >= 1


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_ensure_collection_toctou_race(mock_settings, mock_client_fn):
    """ensure_collection handles 'already exists' race gracefully."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "toctou_test"
    settings.EMBEDDING_DIM = 1536
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection.side_effect = Exception("Collection already exists")
    mock_client_fn.return_value = mock_client

    ensure_collection()
    mock_client.create_collection.assert_called_once()


@patch("app.services.vector_store.get_settings")
def test_get_qdrant_client_returns_client(mock_settings):
    """get_qdrant_client returns QdrantClient (covers lines 28-29)."""
    settings = MagicMock()
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = "test-key"
    mock_settings.return_value = settings

    from app.services.vector_store import get_qdrant_client

    get_qdrant_client.cache_clear()
    try:
        client = get_qdrant_client()
        from qdrant_client import QdrantClient

        assert isinstance(client, QdrantClient)
    finally:
        get_qdrant_client.cache_clear()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_ensure_collection_raises_on_other_error(mock_settings, mock_client_fn):
    """ensure_collection re-raises when create_collection fails with non-already-exists (line 72)."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "raise_test"
    settings.EMBEDDING_DIM = 1536
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection.side_effect = Exception("Connection refused")
    mock_client_fn.return_value = mock_client

    with pytest.raises(Exception, match="Connection refused"):
        ensure_collection()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_upsert_chunks_multi_batch(mock_settings, mock_client_fn):
    """upsert_chunks uploads in batches of 100."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client

    chunks = [
        Chunk(text=f"chunk {i}", metadata={"file_path": "test.f", "unit_name": f"R{i}"})
        for i in range(150)
    ]
    embeddings = [[0.1] * 1536] * 150

    upsert_chunks(chunks, embeddings)
    assert mock_client.upsert.call_count == 2


@patch("app.services.vector_store.get_settings")
def test_get_async_qdrant_client_returns_client(mock_settings):
    """get_async_qdrant_client returns AsyncQdrantClient (covers lines 35-37)."""
    settings = MagicMock()
    settings.QDRANT_URL = "http://localhost:6333"
    settings.QDRANT_API_KEY = ""
    mock_settings.return_value = settings

    get_async_qdrant_client.cache_clear()
    try:
        client = get_async_qdrant_client()
        from qdrant_client import AsyncQdrantClient

        assert isinstance(client, AsyncQdrantClient)
    finally:
        get_async_qdrant_client.cache_clear()


@patch("app.services.vector_store.get_async_qdrant_client")
@patch("app.services.vector_store.get_settings")
@pytest.mark.asyncio
async def test_async_search(mock_settings, mock_client_fn):
    """async_search calls query_points and returns formatted hits."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_point = MagicMock()
    mock_point.id = "async-1"
    mock_point.score = 0.88
    mock_point.payload = {"text": "async code", "file_path": "async.f"}

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=mock_result)
    mock_client_fn.return_value = mock_client

    results = await async_search([0.1] * 1536, top_k=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.88
    mock_client.query_points.assert_called_once()


@patch("app.services.vector_store.get_async_qdrant_client")
@patch("app.services.vector_store.get_settings")
@pytest.mark.asyncio
async def test_async_search_by_name(mock_settings, mock_client_fn):
    """async_search_by_name calls query_points with unit_name filter."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_point = MagicMock()
    mock_point.id = "dgesv-async"
    mock_point.score = 0.99
    mock_point.payload = {"text": "DGESV code", "unit_name": "DGESV", "file_path": "dgesv.f"}

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=mock_result)
    mock_client_fn.return_value = mock_client

    results = await async_search_by_name([0.1] * 1536, "DGESV", top_k=3)
    assert len(results) == 1
    assert results[0]["metadata"]["unit_name"] == "DGESV"
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert "query_filter" in call_kwargs


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_delete_collection_not_found_returns_false(mock_settings, mock_client_fn):
    """delete_collection returns False when collection does not exist (lines 126-132)."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "lapack"
    mock_settings.return_value = settings

    mock_coll = MagicMock()
    mock_coll.name = "other-collection"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_coll]
    mock_client_fn.return_value = mock_client

    result = delete_collection("missing-collection")
    assert result is False
    mock_client.delete_collection.assert_not_called()


@patch("app.services.vector_store.get_qdrant_client")
@patch("app.services.vector_store.get_settings")
def test_delete_collection_found_returns_true(mock_settings, mock_client_fn):
    """delete_collection returns True and deletes when collection exists (lines 128-131)."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "lapack"
    mock_settings.return_value = settings

    mock_coll = MagicMock()
    mock_coll.name = "lapack-text-embedding-3-small"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_coll]
    mock_client_fn.return_value = mock_client

    result = delete_collection("lapack-text-embedding-3-small")
    assert result is True
    mock_client.delete_collection.assert_called_once_with("lapack-text-embedding-3-small")


@patch("app.services.vector_store.get_async_qdrant_client")
@patch("app.services.vector_store.get_settings")
@pytest.mark.asyncio
async def test_async_search_by_caller(mock_settings, mock_client_fn):
    """async_search_by_caller uses called_routines filter (lines 208-220)."""
    settings = MagicMock()
    settings.QDRANT_COLLECTION_NAME = "test"
    mock_settings.return_value = settings

    mock_point = MagicMock()
    mock_point.id = "caller-1"
    mock_point.score = 0.92
    mock_point.payload = {"text": "calls DGEMM", "called_routines": ["DGEMM"], "file_path": "x.f"}

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=mock_result)
    mock_client_fn.return_value = mock_client

    results = await async_search_by_caller([0.1] * 1536, "DGEMM", top_k=5)
    assert len(results) == 1
    assert results[0]["metadata"]["called_routines"] == ["DGEMM"]
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert "query_filter" in call_kwargs
