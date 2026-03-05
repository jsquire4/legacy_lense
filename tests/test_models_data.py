"""Tests for app.models_data (LLM model pricing and provider routing)."""

import pytest

from app.models_data import (
    get_provider,
    is_reasoning_model,
    uses_legacy_max_tokens,
)


def test_get_provider_known_model():
    assert get_provider("gpt-4o-mini") == "openai"
    assert get_provider("gemini-2.5-flash") == "gemini"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model: 'unknown'"):
        get_provider("unknown")


def test_is_reasoning_model_gpt5():
    assert is_reasoning_model("gpt-5-mini") is True
    assert is_reasoning_model("gpt-5.2") is True


def test_is_reasoning_model_non_gpt5():
    assert is_reasoning_model("gpt-4o") is False
    assert is_reasoning_model("gpt-4.1-nano") is False


def test_is_reasoning_model_unknown_returns_false():
    """ValueError branch: unknown model returns False."""
    assert is_reasoning_model("unknown") is False


def test_uses_legacy_max_tokens_gpt35():
    assert uses_legacy_max_tokens("gpt-3.5-turbo") is True


def test_uses_legacy_max_tokens_other():
    assert uses_legacy_max_tokens("gpt-4o") is False
    assert uses_legacy_max_tokens("gemini-2.5-flash") is False


def test_uses_legacy_max_tokens_unknown_returns_false():
    """ValueError branch: unknown model returns False."""
    assert uses_legacy_max_tokens("unknown") is False
