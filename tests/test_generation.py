"""Tests for the generation service (mocked, no API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.generation import (
    _assemble_context,
    _build_citation_fallback,
    _build_messages,
    _extract_citations_from_text,
    _get_generation_client,
    _strip_markdown,
    generate_answer,
    generate_answer_stream,
)


def test_assemble_context_empty():
    """_assemble_context returns empty string for empty chunks."""
    assert _assemble_context([]) == ""


def test_assemble_context_within_budget():
    """_assemble_context joins chunks within budget."""
    chunks = [
        {"text": "Chunk one", "metadata": {}},
        {"text": "Chunk two", "metadata": {}},
    ]
    result = _assemble_context(chunks, budget=1000)
    assert "Chunk one" in result
    assert "Chunk two" in result
    assert "---" in result


def test_assemble_context_truncates_beyond_budget():
    """_assemble_context stops when budget exceeded."""
    chunk_text = "word " * 500
    chunks = [
        {"text": chunk_text, "metadata": {}},
        {"text": "Chunk two that should not fit", "metadata": {}},
    ]
    result = _assemble_context(chunks, budget=100)
    assert "Chunk two" not in result


def test_extract_citations_from_text():
    """_extract_citations_from_text finds file:line patterns."""
    text = "See dgesv.f:10-20 and dgetrf.f:5 for details."
    result = _extract_citations_from_text(text)
    assert "dgesv.f:10-20" in result
    assert "dgetrf.f:5" in result


def test_extract_citations_deduplicates():
    """_extract_citations_from_text deduplicates."""
    text = "dgesv.f:10 dgesv.f:10 dgesv.f:10"
    result = _extract_citations_from_text(text)
    assert result.count("dgesv.f:10") == 1


def test_extract_citations_f90():
    """_extract_citations_from_text matches .f90 suffix."""
    text = "See dnrm2.f90:123"
    result = _extract_citations_from_text(text)
    assert "dnrm2.f90:123" in result


def test_build_citation_fallback_with_line_range():
    """_build_citation_fallback uses start_line and end_line when present."""
    chunks = [
        {"metadata": {"file_path": "/path/dgesv.f", "start_line": 10, "end_line": 50}},
    ]
    result = _build_citation_fallback(chunks)
    assert "dgesv.f:10-50" in result


def test_build_citation_fallback_filename_only():
    """_build_citation_fallback uses filename when no line range."""
    chunks = [
        {"metadata": {"file_path": "/path/dgesv.f"}},
    ]
    result = _build_citation_fallback(chunks)
    assert "dgesv.f" in result


def test_build_citation_fallback_without_file_path():
    """_build_citation_fallback skips chunks without file_path."""
    chunks = [{"metadata": {}}]
    result = _build_citation_fallback(chunks)
    assert result == []


def test_strip_markdown():
    """_strip_markdown removes headers, bold, bullets, numbered lists."""
    assert _strip_markdown("### Foo") == "Foo"
    assert _strip_markdown("**bold**") == "bold"
    assert _strip_markdown("- item") == "item"
    assert _strip_markdown("1. first") == "first"
    assert _strip_markdown("  * nested") == "nested"


@patch("app.services.generation.get_settings")
def test_get_generation_client_returns_client(mock_settings):
    """_get_generation_client returns AsyncOpenAI client."""
    settings = MagicMock()
    settings.OPENAI_API_KEY = "test-key"
    mock_settings.return_value = settings

    _get_generation_client.cache_clear()
    try:
        from openai import AsyncOpenAI

        client = _get_generation_client()
        assert isinstance(client, AsyncOpenAI)
    finally:
        _get_generation_client.cache_clear()


def test_build_messages_uses_capability_prompt():
    """_build_messages uses CAPABILITIES when capability given."""
    chunks = [{"text": "code", "metadata": {}}]
    msgs = _build_messages("Explain", chunks, "explain_code")
    assert msgs[0]["role"] == "system"
    assert "explain" in msgs[0]["content"].lower()


def test_build_messages_uses_default_prompt():
    """_build_messages uses DEFAULT_SYSTEM_PROMPT when capability is None."""
    chunks = [{"text": "code", "metadata": {}}]
    msgs = _build_messages("What?", chunks, None)
    assert "LegacyLens" in msgs[0]["content"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_empty_chunks(mock_settings, mock_client_fn):
    """generate_answer returns fallback when chunks empty."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    result = await generate_answer("What is DGESV?", [], None)
    assert "don't have sufficient context" in result["answer"]
    assert result["citations"] == []
    mock_client_fn.return_value.chat.completions.create.assert_not_called()


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_extracted_citations_skip_fallback(mock_settings, mock_client_fn):
    """generate_answer skips citation fallback when LLM output has citations (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "See dgesv.f:10-20 for details."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}}]
    result = await generate_answer("What?", chunks, None)
    assert "dgesv.f:10-20" in result["citations"]
    assert "Sources:" not in result["answer"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_with_chunks(mock_settings, mock_client_fn):
    """generate_answer returns answer from LLM."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "DGESV solves Ax=b using LU."
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 120

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [
        {"text": "SUBROUTINE DGESV", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    result = await generate_answer("What is DGESV?", chunks, None)
    assert "DGESV" in result["answer"]
    assert result["token_usage"]["prompt_tokens"] == 100
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_citation_fallback(mock_settings, mock_client_fn):
    """generate_answer adds citation fallback when LLM omits."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "DGESV solves linear systems."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [
        {"text": "SUBROUTINE DGESV", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    result = await generate_answer("What is DGESV?", chunks, None)
    assert "Sources:" in result["answer"]
    assert "dgesv.f:1-50" in result["citations"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_uses_default_prompt_when_no_capability(mock_settings, mock_client_fn):
    """generate_answer uses DEFAULT_SYSTEM_PROMPT when capability is None."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert "Answer" in result["answer"]
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "LegacyLens" in system_content


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_uses_capability_prompt(mock_settings, mock_client_fn):
    """generate_answer uses capability-specific prompt when capability given."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Explained."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("Explain", chunks, "explain_code")
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert "explaining" in system_content.lower() or "explain" in system_content.lower()


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_empty_chunks(mock_settings, mock_client_fn):
    """generate_answer_stream yields fallback for empty chunks."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    events = [e async for e in generate_answer_stream("What is DGESV?", [], None)]
    assert len(events) == 2
    assert events[0]["type"] == "token"
    assert "don't have sufficient context" in events[0]["token"]
    assert events[1]["type"] == "done"


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_uses_capability_prompt(mock_settings, mock_client_fn):
    """generate_answer_stream uses CAPABILITIES prompt when capability in CAPABILITIES."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Answer."

    mock_client = AsyncMock()
    # AsyncOpenAI stream returns an async iterator
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("Explain this", chunks, "explain_code")]
    assert events[-1]["type"] == "done"
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "explain" in system_content.lower()


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_uses_default_prompt(mock_settings, mock_client_fn):
    """generate_answer_stream uses DEFAULT_SYSTEM_PROMPT when capability is None."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Streamed."

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"
    call_args = mock_client.chat.completions.create.call_args
    assert "LegacyLens" in call_args.kwargs["messages"][0]["content"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_with_chunks(mock_settings, mock_client_fn):
    """generate_answer_stream yields token and done events."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk1 = MagicMock()
    mock_chunk1.usage = None
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Hello "

    mock_chunk2 = MagicMock()
    mock_chunk2.usage = MagicMock()
    mock_chunk2.usage.prompt_tokens = 50
    mock_chunk2.usage.completion_tokens = 10
    mock_chunk2.usage.total_tokens = 60
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "world"

    mock_chunk3 = MagicMock()
    mock_chunk3.usage = None
    mock_chunk3.choices = [MagicMock()]
    mock_chunk3.choices[0].delta.content = None

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk1
        yield mock_chunk2
        yield mock_chunk3
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) >= 2
    assert any("Hello" in e["token"] for e in token_events)
    assert any("world" in e["token"] for e in token_events)
    assert events[-1]["type"] == "done"


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_citation_fallback(mock_settings, mock_client_fn):
    """generate_answer_stream adds citation suffix when LLM omits."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Answer without citations."

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [
        {"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Sources:" in e["token"] for e in token_events)
    assert events[-1]["citations"] == ["dgesv.f:1-50"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_empty_choices(mock_settings, mock_client_fn):
    """generate_answer handles response with empty choices (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert result["answer"] == ""
    assert result["citations"] == []


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_message_content_none(mock_settings, mock_client_fn):
    """generate_answer handles message.content is None (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert result["answer"] == ""


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_citation_fallback_empty_chunks(mock_settings, mock_client_fn):
    """generate_answer with chunks without file_path yields no citation suffix (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer without citations."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert "Sources:" not in result["answer"]
    assert result["citations"] == []


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_track_ttft_stream_chunk_empty_choices(mock_settings, mock_client_fn):
    """_generate_with_ttft handles stream chunks with empty choices (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk1 = MagicMock()
    mock_chunk1.usage = None
    mock_chunk1.choices = []

    mock_chunk2 = MagicMock()
    mock_chunk2.usage = MagicMock()
    mock_chunk2.usage.prompt_tokens = 50
    mock_chunk2.usage.completion_tokens = 10
    mock_chunk2.usage.total_tokens = 60
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "Answer"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk1
        yield mock_chunk2
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer" in result["answer"]


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_track_ttft_stream_chunk_no_content(mock_settings, mock_client_fn):
    """_generate_with_ttft handles stream chunks with choices but delta.content None (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk1 = MagicMock()
    mock_chunk1.usage = None
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = None

    mock_chunk2 = MagicMock()
    mock_chunk2.usage = MagicMock()
    mock_chunk2.usage.prompt_tokens = 50
    mock_chunk2.usage.completion_tokens = 10
    mock_chunk2.usage.total_tokens = 60
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "Answer"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk1
        yield mock_chunk2
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer" in result["answer"]
    assert "ttft_ms" in result


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_track_ttft(mock_settings, mock_client_fn):
    """generate_answer with track_ttft=True returns ttft_ms and uses streaming."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk1 = MagicMock()
    mock_chunk1.usage = None
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Answer"

    mock_chunk2 = MagicMock()
    mock_chunk2.usage = MagicMock()
    mock_chunk2.usage.prompt_tokens = 50
    mock_chunk2.usage.completion_tokens = 10
    mock_chunk2.usage.total_tokens = 60
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = " text"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk1
        yield mock_chunk2
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer text" in result["answer"]
    assert "ttft_ms" in result
    assert isinstance(result["ttft_ms"], (int, float))


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_delta_content_empty(mock_settings, mock_client_fn):
    """generate_answer_stream skips yield when delta.content is empty (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk1 = MagicMock()
    mock_chunk1.usage = None
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Hi"

    mock_chunk2 = MagicMock()
    mock_chunk2.usage = None
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = ""  # empty string

    mock_chunk3 = MagicMock()
    mock_chunk3.usage = MagicMock()
    mock_chunk3.usage.prompt_tokens = 10
    mock_chunk3.usage.completion_tokens = 5
    mock_chunk3.usage.total_tokens = 15
    mock_chunk3.choices = [MagicMock()]
    mock_chunk3.choices[0].delta.content = "!"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk1
        yield mock_chunk2
        yield mock_chunk3
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Hi" in e["token"] for e in token_events)
    assert any("!" in e["token"] for e in token_events)


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_chunk_no_choices(mock_settings, mock_client_fn):
    """generate_answer_stream handles stream chunks with empty choices (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk_empty = MagicMock()
    mock_chunk_empty.usage = None
    mock_chunk_empty.choices = []

    mock_chunk_content = MagicMock()
    mock_chunk_content.usage = MagicMock()
    mock_chunk_content.usage.prompt_tokens = 10
    mock_chunk_content.usage.completion_tokens = 5
    mock_chunk_content.usage.total_tokens = 15
    mock_chunk_content.choices = [MagicMock()]
    mock_chunk_content.choices[0].delta.content = "Done"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk_empty
        yield mock_chunk_content
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_citation_fallback_no_file_path(mock_settings, mock_client_fn):
    """generate_answer_stream with chunks without file_path yields no citation suffix (branch coverage)."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Answer."

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert not any("Sources:" in e["token"] for e in token_events)


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_reasoning_model_uses_reasoning_effort(mock_settings, mock_client_fn):
    """generate_answer passes reasoning_effort instead of temperature for gpt-5 models."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-5-nano"
    mock_settings.return_value = settings

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer."
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("What?", chunks, None)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_track_ttft_reasoning_model(mock_settings, mock_client_fn):
    """_generate_with_ttft passes reasoning_effort for gpt-5 models."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-5-mini"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = MagicMock()
    mock_chunk.usage.prompt_tokens = 50
    mock_chunk.usage.completion_tokens = 10
    mock_chunk.usage.total_tokens = 60
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Answer"

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("What?", chunks, None, track_ttft=True)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs


@patch("app.services.generation._get_generation_client")
@patch("app.services.generation.get_settings")
@pytest.mark.asyncio
async def test_generate_answer_stream_reasoning_model(mock_settings, mock_client_fn):
    """generate_answer_stream passes reasoning_effort for gpt-5 models."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-5.2"
    mock_settings.return_value = settings

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Answer."

    mock_client = AsyncMock()
    async def async_iter():
        yield mock_chunk
    mock_client.chat.completions.create.return_value = async_iter()
    mock_client_fn.return_value = mock_client

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs
