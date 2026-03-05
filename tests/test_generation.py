"""Tests for the generation service (mocked, no API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_async_iter
from tests.helpers import make_openai_stream_chunk, make_openai_response, make_gemini_response

from app.services.generation import (
    _assemble_context,
    _build_citation_fallback,
    _build_messages,
    _extract_citations_from_text,
    _get_generation_client,
    _messages_to_gemini,
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


# --- generate_answer tests (using mock_gen_settings fixture) ---

@pytest.mark.asyncio
async def test_generate_answer_empty_chunks(mock_gen_settings):
    """generate_answer returns fallback when chunks empty."""
    settings, mock_client = mock_gen_settings
    result = await generate_answer("What is DGESV?", [], None)
    assert "don't have sufficient context" in result["answer"]
    assert result["citations"] == []
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_answer_extracted_citations_skip_fallback(mock_gen_settings):
    """generate_answer skips citation fallback when LLM output has citations."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(
        content="See dgesv.f:10-20 for details.",
    )

    chunks = [{"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}}]
    result = await generate_answer("What?", chunks, None)
    assert "dgesv.f:10-20" in result["citations"]
    assert "Sources:" not in result["answer"]


@pytest.mark.asyncio
async def test_generate_answer_with_chunks(mock_gen_settings):
    """generate_answer returns answer from LLM."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(
        content="DGESV solves Ax=b using LU.",
        usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    )

    chunks = [
        {"text": "SUBROUTINE DGESV", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    result = await generate_answer("What is DGESV?", chunks, None)
    assert "DGESV" in result["answer"]
    assert result["token_usage"]["prompt_tokens"] == 100
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_citation_fallback(mock_gen_settings):
    """generate_answer adds citation fallback when LLM omits."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(
        content="DGESV solves linear systems.",
    )

    chunks = [
        {"text": "SUBROUTINE DGESV", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    result = await generate_answer("What is DGESV?", chunks, None)
    assert "Sources:" in result["answer"]
    assert "dgesv.f:1-50" in result["citations"]


@pytest.mark.asyncio
async def test_generate_answer_uses_default_prompt_when_no_capability(mock_gen_settings):
    """generate_answer uses DEFAULT_SYSTEM_PROMPT when capability is None."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(content="Answer.")

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert "Answer" in result["answer"]
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "LegacyLens" in system_content


@pytest.mark.asyncio
async def test_generate_answer_uses_capability_prompt(mock_gen_settings):
    """generate_answer uses capability-specific prompt when capability given."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(content="Explained.")

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("Explain", chunks, "explain_code")
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert "explaining" in system_content.lower() or "explain" in system_content.lower()


# --- generate_answer_stream tests ---

@pytest.mark.asyncio
async def test_generate_answer_stream_empty_chunks(mock_gen_settings):
    """generate_answer_stream yields fallback for empty chunks."""
    events = [e async for e in generate_answer_stream("What is DGESV?", [], None)]
    assert len(events) == 2
    assert events[0]["type"] == "token"
    assert "don't have sufficient context" in events[0]["token"]
    assert events[1]["type"] == "done"


@pytest.mark.asyncio
async def test_generate_answer_stream_uses_capability_prompt(mock_gen_settings):
    """generate_answer_stream uses CAPABILITIES prompt when capability in CAPABILITIES."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer."),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("Explain this", chunks, "explain_code")]
    assert events[-1]["type"] == "done"
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "explain" in system_content.lower()


@pytest.mark.asyncio
async def test_generate_answer_stream_uses_default_prompt(mock_gen_settings):
    """generate_answer_stream uses DEFAULT_SYSTEM_PROMPT when capability is None."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Streamed."),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"
    call_args = mock_client.chat.completions.create.call_args
    assert "LegacyLens" in call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_generate_answer_stream_with_chunks(mock_gen_settings):
    """generate_answer_stream yields token and done events."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Hello "),
        make_openai_stream_chunk(content="world", usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}),
        make_openai_stream_chunk(content=None),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) >= 2
    assert any("Hello" in e["token"] for e in token_events)
    assert any("world" in e["token"] for e in token_events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_generate_answer_stream_citation_fallback(mock_gen_settings):
    """generate_answer_stream adds citation suffix when LLM omits."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer without citations."),
    )

    chunks = [
        {"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}},
    ]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Sources:" in e["token"] for e in token_events)
    assert events[-1]["citations"] == ["dgesv.f:1-50"]


@pytest.mark.asyncio
async def test_generate_answer_empty_choices(mock_gen_settings):
    """generate_answer handles response with empty choices."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(empty_choices=True)

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert result["answer"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_generate_answer_message_content_none(mock_gen_settings):
    """generate_answer handles message.content is None."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(content_none=True)

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert result["answer"] == ""


@pytest.mark.asyncio
async def test_generate_answer_citation_fallback_empty_chunks(mock_gen_settings):
    """generate_answer with chunks without file_path yields no citation suffix."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_openai_response(
        content="Answer without citations.",
    )

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None)
    assert "Sources:" not in result["answer"]
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_generate_answer_track_ttft_stream_chunk_empty_choices(mock_gen_settings):
    """_generate_with_ttft handles stream chunks with empty choices."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(empty_choices=True),
        make_openai_stream_chunk(content="Answer", usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer" in result["answer"]


@pytest.mark.asyncio
async def test_generate_answer_track_ttft_stream_chunk_no_content(mock_gen_settings):
    """_generate_with_ttft handles stream chunks with choices but delta.content None."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content=None),
        make_openai_stream_chunk(content="Answer", usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer" in result["answer"]
    assert "ttft_ms" in result


@pytest.mark.asyncio
async def test_generate_answer_track_ttft(mock_gen_settings):
    """generate_answer with track_ttft=True returns ttft_ms and uses streaming."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer"),
        make_openai_stream_chunk(content=" text", usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True)
    assert "Answer text" in result["answer"]
    assert "ttft_ms" in result
    assert isinstance(result["ttft_ms"], (int, float))


@pytest.mark.asyncio
async def test_generate_answer_stream_delta_content_empty(mock_gen_settings):
    """generate_answer_stream skips yield when delta.content is empty."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Hi"),
        make_openai_stream_chunk(content=""),  # empty string
        make_openai_stream_chunk(content="!", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Hi" in e["token"] for e in token_events)
    assert any("!" in e["token"] for e in token_events)


@pytest.mark.asyncio
async def test_generate_answer_stream_chunk_no_choices(mock_gen_settings):
    """generate_answer_stream handles stream chunks with empty choices."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(empty_choices=True),
        make_openai_stream_chunk(content="Done", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_generate_answer_stream_citation_fallback_no_file_path(mock_gen_settings):
    """generate_answer_stream with chunks without file_path yields no citation suffix."""
    settings, mock_client = mock_gen_settings
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer."),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    token_events = [e for e in events if e["type"] == "token"]
    assert not any("Sources:" in e["token"] for e in token_events)


@pytest.mark.asyncio
async def test_generate_answer_reasoning_model_uses_reasoning_effort(mock_gen_settings):
    """generate_answer passes reasoning_effort instead of temperature for gpt-5 models."""
    settings, mock_client = mock_gen_settings
    settings.CHAT_MODEL = "gpt-5-nano"
    mock_client.chat.completions.create.return_value = make_openai_response(content="Answer.")

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("What?", chunks, None)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_generate_answer_track_ttft_reasoning_model(mock_gen_settings):
    """_generate_with_ttft passes reasoning_effort for gpt-5 models."""
    settings, mock_client = mock_gen_settings
    settings.CHAT_MODEL = "gpt-5-mini"
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer", usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}),
    )

    chunks = [{"text": "code", "metadata": {}}]
    await generate_answer("What?", chunks, None, track_ttft=True)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs


@pytest.mark.asyncio
async def test_generate_answer_stream_reasoning_model(mock_gen_settings):
    """generate_answer_stream passes reasoning_effort for gpt-5 models."""
    settings, mock_client = mock_gen_settings
    settings.CHAT_MODEL = "gpt-5.2"
    mock_client.chat.completions.create.return_value = make_async_iter(
        make_openai_stream_chunk(content="Answer."),
    )

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None)]
    assert events[-1]["type"] == "done"
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in call_kwargs


# --- Gemini generation tests ---

def test_messages_to_gemini_extracts_system():
    """_messages_to_gemini separates system message from content."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    system, contents = _messages_to_gemini(messages)
    assert system == "You are helpful."
    assert len(contents) == 1
    assert contents[0].role == "user"


@pytest.mark.asyncio
async def test_generate_answer_gemini_dispatch(mock_gemini_gen_settings):
    """generate_answer dispatches to Gemini for gemini-* models."""
    settings, mock_client = mock_gemini_gen_settings

    mock_client.aio.models.generate_content = AsyncMock(return_value=make_gemini_response(
        text="Gemini answer about dgesv.f:10-20.",
        usage_metadata={"prompt_token_count": 100, "candidates_token_count": 20, "total_token_count": 120},
    ))

    chunks = [{"text": "SUBROUTINE DGESV", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}}]
    result = await generate_answer("What is DGESV?", chunks, None, model="gemini-2.5-flash")

    assert "Gemini answer" in result["answer"]
    assert result["model"] == "gemini-2.5-flash"
    assert result["token_usage"]["prompt_tokens"] == 100
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_gemini_empty_chunks(mock_gemini_gen_settings):
    """generate_answer returns fallback for Gemini when chunks empty."""
    result = await generate_answer("What?", [], None, model="gemini-2.5-flash")
    assert "don't have sufficient context" in result["answer"]


@pytest.mark.asyncio
async def test_generate_answer_gemini_citation_fallback(mock_gemini_gen_settings):
    """generate_answer adds citation fallback for Gemini when LLM omits citations."""
    settings, mock_client = mock_gemini_gen_settings

    mock_client.aio.models.generate_content = AsyncMock(return_value=make_gemini_response(
        text="DGESV solves linear systems.",
    ))

    chunks = [{"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}}]
    result = await generate_answer("What?", chunks, None, model="gemini-2.5-flash")
    assert "Sources:" in result["answer"]
    assert "dgesv.f:1-50" in result["citations"]


@pytest.mark.asyncio
async def test_generate_answer_stream_gemini(mock_gemini_gen_settings):
    """generate_answer_stream works with Gemini models."""
    settings, mock_client = mock_gemini_gen_settings

    usage = MagicMock()
    usage.prompt_token_count = 50
    usage.candidates_token_count = 10
    usage.total_token_count = 60

    async def fake_async_iter():
        yield MagicMock(text="Hello ", usage_metadata=None)
        yield MagicMock(text="world", usage_metadata=usage)

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_async_iter())

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None, model="gemini-2.5-flash")]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Hello" in e["token"] for e in token_events)
    assert any("world" in e["token"] for e in token_events)
    assert events[-1]["type"] == "done"
    assert events[-1]["token_usage"]["total_tokens"] == 60


@pytest.mark.asyncio
async def test_generate_answer_gemini_track_ttft(mock_gemini_gen_settings):
    """generate_answer with track_ttft=True works for Gemini models."""
    settings, mock_client = mock_gemini_gen_settings

    usage = MagicMock()
    usage.prompt_token_count = 50
    usage.candidates_token_count = 10
    usage.total_token_count = 60

    async def fake_async_iter():
        yield MagicMock(text="Answer", usage_metadata=usage)

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_async_iter())

    chunks = [{"text": "code", "metadata": {}}]
    result = await generate_answer("What?", chunks, None, track_ttft=True, model="gemini-2.5-flash")
    assert "Answer" in result["answer"]
    assert "ttft_ms" in result
    assert isinstance(result["ttft_ms"], (int, float))


@pytest.mark.asyncio
async def test_generate_answer_stream_gemini_error(mock_gemini_gen_settings):
    """generate_answer_stream yields error event when Gemini stream raises."""
    settings, mock_client = mock_gemini_gen_settings

    async def failing_stream():
        raise RuntimeError("Gemini API unavailable")
        yield  # noqa: F841 — unreachable yield makes this an async generator

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=failing_stream())

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None, model="gemini-2.5-flash")]
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "Gemini API unavailable" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_generate_answer_stream_gemini_citation_fallback(mock_gemini_gen_settings):
    """generate_answer_stream adds citation suffix for Gemini when LLM omits."""
    settings, mock_client = mock_gemini_gen_settings

    async def fake_stream():
        yield MagicMock(text="Answer without citations.", usage_metadata=None)

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

    chunks = [{"text": "code", "metadata": {"file_path": "dgesv.f", "start_line": 1, "end_line": 50}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None, model="gemini-2.5-flash")]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("Sources:" in e["token"] for e in token_events)
    assert events[-1]["type"] == "done"
    assert "dgesv.f:1-50" in events[-1]["citations"]


# --- Audit issue #6: Gemini safety block in streaming ---

@pytest.mark.asyncio
async def test_gemini_generate_stream_safety_block(mock_gemini_gen_settings):
    """_gemini_generate_stream handles ValueError from chunk.text (safety block)."""
    settings, mock_client = mock_gemini_gen_settings

    chunk_blocked = MagicMock()
    type(chunk_blocked).text = property(lambda self: (_ for _ in ()).throw(ValueError("safety block")))
    chunk_blocked.usage_metadata = None

    chunk_ok = MagicMock()
    chunk_ok.text = "OK"
    chunk_ok.usage_metadata = None

    async def fake_stream():
        yield chunk_blocked
        yield chunk_ok

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

    chunks = [{"text": "code", "metadata": {}}]
    events = [e async for e in generate_answer_stream("What?", chunks, None, model="gemini-2.5-flash")]
    token_events = [e for e in events if e["type"] == "token"]
    # The blocked chunk should produce empty text (skipped), OK chunk should appear
    assert any("OK" in e["token"] for e in token_events)
    assert events[-1]["type"] == "done"


# --- Audit issue #18: Gemini non-stream error propagation ---

@pytest.mark.asyncio
async def test_gemini_generate_nonstream_error(mock_gemini_gen_settings):
    """generate_answer propagates exception when Gemini non-streaming call raises."""
    settings, mock_client = mock_gemini_gen_settings

    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=RuntimeError("Gemini API down")
    )

    chunks = [{"text": "code", "metadata": {}}]
    with pytest.raises(RuntimeError, match="Gemini API down"):
        await generate_answer("What?", chunks, None, model="gemini-2.5-flash")


# --- Audit issue #22: messages_to_gemini edge cases ---

def test_messages_to_gemini_no_system():
    """_messages_to_gemini returns None system_instruction for user-only messages."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    system, contents = _messages_to_gemini(messages)
    assert system is None
    assert len(contents) == 1


def test_messages_to_gemini_assistant_role():
    """_messages_to_gemini maps assistant role to 'model'."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    system, contents = _messages_to_gemini(messages)
    assert system == "You are helpful."
    assert len(contents) == 2
    assert contents[1].role == "model"
