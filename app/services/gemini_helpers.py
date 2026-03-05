"""Shared Gemini client and message conversion utilities."""

import asyncio
import logging
from functools import lru_cache, wraps

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # doubles each retry: 2, 4, 8, 16, 32s


@lru_cache
def get_gemini_client():
    from google import genai
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to use Gemini generation")
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def messages_to_gemini(messages: list[dict]) -> tuple[str | None, list]:
    """Convert OpenAI-style messages to Gemini format.

    Returns (system_instruction, contents) where contents is a list of
    Content objects for the Gemini SDK.
    """
    from google.genai import types

    system_instruction = None
    contents = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"] or ""
        else:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"] or "")]))

    return system_instruction, contents


def is_gemini_reasoning_model(model: str) -> bool:
    """Return True if the Gemini model is a thinking/reasoning model that rejects temperature."""
    return model.startswith("gemini-2.5-pro")


def build_gemini_config(system_instruction: str | None, max_completion_tokens: int, temperature: float = 0.1, model: str | None = None):
    """Build a GenerateContentConfig for Gemini."""
    from google.genai import types
    kwargs = dict(
        system_instruction=system_instruction,
        max_output_tokens=max_completion_tokens,
    )
    if model is None or not is_gemini_reasoning_model(model):
        kwargs["temperature"] = temperature
    return types.GenerateContentConfig(**kwargs)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a Gemini 429 / RESOURCE_EXHAUSTED error."""
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def retry_on_rate_limit(func):
    """Decorator: retry an async function on Gemini rate-limit errors with exponential backoff."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for attempt in range(_MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning("Gemini rate limited, retrying in %.0fs (attempt %d/%d): %s",
                                   delay, attempt + 1, _MAX_RETRIES, e)
                    await asyncio.sleep(delay)
                else:
                    raise
    return wrapper


def retry_on_rate_limit_sync(func):
    """Decorator: retry a sync function on Gemini rate-limit errors with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        for attempt in range(_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning("Gemini rate limited, retrying in %.0fs (attempt %d/%d): %s",
                                   delay, attempt + 1, _MAX_RETRIES, e)
                    time.sleep(delay)
                else:
                    raise
    return wrapper
