"""Shared test utilities — factories, parsers, and helpers."""

import json
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

def parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts.

    Handles both multi-event text blocks (separated by blank lines)
    and individual SSE event strings.
    """
    events: list[dict] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event and data is not None:
            events.append({"event": event, "data": data})
    return events


def parse_single_sse(text: str) -> dict:
    """Parse a single SSE event string into {event, data}."""
    event = data = None
    for line in text.strip().split("\n"):
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
    return {"event": event, "data": data}


async def collect_sse_events(gen) -> list[dict]:
    """Collect all SSE events from an async generator."""
    events = []
    async for raw in gen:
        events.append(parse_single_sse(raw))
    return events


# ---------------------------------------------------------------------------
# OpenAI mock factories
# ---------------------------------------------------------------------------

def make_openai_stream_chunk(
    content: str | None = None,
    usage: dict | None = None,
    empty_choices: bool = False,
) -> MagicMock:
    """Create a mock OpenAI streaming chunk.

    Args:
        content: Delta content text. None means delta.content is None.
        usage: Dict with prompt_tokens, completion_tokens, total_tokens.
               None means chunk.usage is None.
        empty_choices: If True, chunk.choices is an empty list.
    """
    chunk = MagicMock()
    if usage is not None:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
        chunk.usage.total_tokens = usage.get("total_tokens", 0)
    else:
        chunk.usage = None

    if empty_choices:
        chunk.choices = []
    else:
        choice = MagicMock()
        choice.delta.content = content
        chunk.choices = [choice]

    return chunk


def make_openai_response(
    content: str = "",
    usage: dict | None = None,
    empty_choices: bool = False,
    content_none: bool = False,
) -> MagicMock:
    """Create a mock OpenAI non-streaming response.

    Args:
        content: Message content text.
        usage: Dict with prompt_tokens, completion_tokens, total_tokens.
               None means response.usage is None.
        empty_choices: If True, response.choices is an empty list.
        content_none: If True, message.content is None.
    """
    response = MagicMock()
    if usage is not None:
        response.usage = MagicMock()
        response.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        response.usage.completion_tokens = usage.get("completion_tokens", 0)
        response.usage.total_tokens = usage.get("total_tokens", 0)
    else:
        response.usage = None

    if empty_choices:
        response.choices = []
    else:
        choice = MagicMock()
        choice.message.content = None if content_none else content
        response.choices = [choice]

    return response


# ---------------------------------------------------------------------------
# Gemini mock factories
# ---------------------------------------------------------------------------

def make_gemini_response(
    text: str = "",
    usage_metadata: dict | None = None,
) -> MagicMock:
    """Create a mock Gemini response.

    Args:
        text: Response text.
        usage_metadata: Dict with prompt_token_count, candidates_token_count,
                        total_token_count. None means usage_metadata is None.
    """
    response = MagicMock()
    response.text = text
    if usage_metadata is not None:
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = usage_metadata.get("prompt_token_count", 0)
        response.usage_metadata.candidates_token_count = usage_metadata.get("candidates_token_count", 0)
        response.usage_metadata.total_token_count = usage_metadata.get("total_token_count", 0)
    else:
        response.usage_metadata = None
    return response


# ---------------------------------------------------------------------------
# Embedding settings factory
# ---------------------------------------------------------------------------

def mock_embedding_settings(
    model: str = "text-embedding-3-small",
    max_tokens: int = 8191,
) -> MagicMock:
    """Create a mock settings object for embedding tests."""
    settings = MagicMock()
    settings.EMBEDDING_MODEL = model
    settings.MAX_CHUNK_TOKENS = max_tokens
    return settings


# ---------------------------------------------------------------------------
# API test factories
# ---------------------------------------------------------------------------

def make_retrieve_result(
    chunks=None, strategy="vector", expanded_names=None,
):
    """Build a standard retrieve() return value for API tests."""
    if chunks is None:
        chunks = [
            {"id": "abc123", "text": "test", "score": 0.9,
             "metadata": {"file_path": "test.f"}, "_match_type": "vector"}
        ]
    return {
        "chunks": chunks,
        "expanded_names": expanded_names or [],
        "retrieval_strategy": strategy,
    }


def make_generate_result(
    answer="Test answer", citations=None, model="gpt-4o-mini", token_usage=None,
):
    """Build a standard generate_answer() return value for API tests."""
    return {
        "answer": answer,
        "citations": citations or [],
        "model": model,
        "token_usage": token_usage or {},
    }
