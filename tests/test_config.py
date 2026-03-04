"""Tests for app config."""

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
        assert settings.VOYAGE_API_KEY == ""
        assert settings.GEMINI_API_KEY == ""
        assert settings.COHERE_API_KEY == ""
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_COLLECTION_NAME == "lapack-text-embedding-3-small"
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
        assert settings.CHAT_MODEL == "gpt-4.1-nano"
    finally:
        get_settings.cache_clear()
