"""Tests for app.services.gemini_helpers."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.gemini_helpers import (
    is_gemini_reasoning_model,
    build_gemini_config,
    retry_on_rate_limit,
    retry_on_rate_limit_sync,
)


def test_is_gemini_reasoning_model_true():
    assert is_gemini_reasoning_model("gemini-2.5-pro") is True


def test_is_gemini_reasoning_model_false():
    """Non-reasoning Gemini model returns False (line 47)."""
    assert is_gemini_reasoning_model("gemini-2.5-flash") is False
    assert is_gemini_reasoning_model("gemini-1.5-pro") is False


@pytest.mark.asyncio
async def test_retry_on_rate_limit_retries_then_succeeds():
    """retry_on_rate_limit retries on 429 then succeeds (lines 77-80)."""
    call_count = 0

    @retry_on_rate_limit
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert call_count == 2


def test_retry_on_rate_limit_sync_retries_then_succeeds():
    """retry_on_rate_limit_sync retries on 429 then succeeds (lines 94-99)."""
    call_count = 0

    @retry_on_rate_limit_sync
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert call_count == 2


def test_retry_on_rate_limit_sync_exhausts_raises():
    """retry_on_rate_limit_sync re-raises after exhausting retries (line 101)."""
    @retry_on_rate_limit_sync
    def always_429():
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError, match="429 RESOURCE_EXHAUSTED"):
        always_429()
