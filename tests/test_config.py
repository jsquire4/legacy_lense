"""Tests for app config."""

import pytest

from app.config import Settings, get_settings


def test_get_settings_returns_settings_instance(monkeypatch):
    """get_settings() returns a Settings instance with required env vars."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.OPENAI_API_KEY == "test-key"
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_COLLECTION_NAME == "lapack"
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
        assert settings.CHAT_MODEL == "gpt-4o"
    finally:
        get_settings.cache_clear()
