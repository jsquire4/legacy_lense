"""Tests for app.embedding_registry (embedding model registry)."""

import pytest

from app.embedding_registry import (
    collection_name_for_model,
    get_model_info,
)


def test_collection_name_for_model_known():
    assert collection_name_for_model("lapack", "text-embedding-3-small") == "lapack-text-embedding-3-small"
    assert collection_name_for_model("lapack", "voyage-code-3") == "lapack-voyage-code-3"


def test_collection_name_for_model_unknown():
    """Fallback branch: unknown model uses model_name as suffix."""
    assert collection_name_for_model("lapack", "custom-model") == "lapack-custom-model"


def test_get_model_info_raises_keyerror():
    with pytest.raises(KeyError, match="unknown"):
        get_model_info("unknown")
