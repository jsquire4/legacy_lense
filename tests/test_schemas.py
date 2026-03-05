"""Tests for Pydantic schemas."""

import pytest

from app.schemas import QueryRequest, CapabilityRequest, TrialRequest, _validate_embedding_model


def test_validate_embedding_model_unknown_raises():
    """_validate_embedding_model raises ValueError for unknown model (line 8)."""
    with pytest.raises(ValueError, match="Unknown embedding model: 'bad-model'"):
        _validate_embedding_model("bad-model")


def test_validate_embedding_model_none_returns_none():
    assert _validate_embedding_model(None) is None


def test_validate_embedding_model_known_returns_value():
    assert _validate_embedding_model("text-embedding-3-small") == "text-embedding-3-small"


def test_query_request_rejects_unknown_embedding_model():
    """QueryRequest validation rejects unknown embedding_model."""
    with pytest.raises(ValueError, match="Unknown embedding model"):
        QueryRequest(query="test", embedding_model="unknown-embedding")
