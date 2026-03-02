"""Tests for the vector store service (mocked, no Qdrant calls)."""

from unittest.mock import patch, MagicMock
import pytest

from app.services.vector_store import ensure_collection, upsert_chunks, search
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
