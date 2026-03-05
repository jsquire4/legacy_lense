"""Tests for app config."""

from app.config import Settings, get_settings


def test_get_settings_returns_settings_instance():
    """Settings constructed with explicit values and no .env file has correct defaults."""
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        QDRANT_URL="http://localhost:6333",
        QDRANT_API_KEY="test-qdrant-key",
    )
    assert isinstance(settings, Settings)
    assert settings.OPENAI_API_KEY == "test-key"
    assert settings.VOYAGE_API_KEY == ""
    assert settings.GEMINI_API_KEY == ""
    assert settings.COHERE_API_KEY == ""
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.QDRANT_COLLECTION_NAME == "lapack-text-embedding-3-small"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.CHAT_MODEL == "gpt-4.1-nano"
