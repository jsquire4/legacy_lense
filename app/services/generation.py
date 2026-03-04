"""LLM answer generation with citation enforcement."""

import logging
import re
import time
from pathlib import Path

import tiktoken

from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings
from app.models_data import is_reasoning_model, uses_legacy_max_tokens
from app.services.capabilities import CAPABILITIES, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")


@lru_cache
def _get_generation_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1)

CONTEXT_TOKEN_BUDGET = 3000


def _token_limit_key(model: str) -> str:
    """Return the correct API parameter name for the token limit."""
    return "max_tokens" if uses_legacy_max_tokens(model) else "max_completion_tokens"


def _build_llm_kwargs(
    model: str, messages: list[dict], max_completion_tokens: int, stream: bool = False,
) -> dict:
    """Build kwargs dict for chat.completions.create, handling model-specific params."""
    kwargs = dict(model=model, messages=messages)
    kwargs[_token_limit_key(model)] = max_completion_tokens
    if is_reasoning_model(model):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = 0.1
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _assemble_context(
    chunks: list[dict], budget: int = CONTEXT_TOKEN_BUDGET,
) -> str:
    """Fit max complete chunks within token budget."""
    if not chunks:
        return ""

    context_parts = []
    total_tokens = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = _count_tokens(chunk_text)

        if total_tokens + chunk_tokens > budget:
            break

        context_parts.append(chunk_text)
        total_tokens += chunk_tokens

    # If no chunks fit within budget but chunks exist, hard-truncate the first one
    if not context_parts and chunks:
        first_text = chunks[0].get("text", "")
        tokens = _encoder.encode(first_text)[:budget]
        context_parts.append(_encoder.decode(tokens))
        logger.warning("Hard-truncated first chunk to %d tokens (budget=%d)", len(tokens), budget)

    return "\n\n---\n\n".join(context_parts)


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from LLM output."""
    # Headers: ### Foo → Foo
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Bold/italic: **foo** or *foo* → foo
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Bullet lists: - foo or * foo → foo
    text = re.sub(r'^[\s]*[-*]\s+', '', text, flags=re.MULTILINE)
    # Numbered lists: 1. foo → foo
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text


def _extract_citations_from_text(text: str) -> list[str]:
    """Extract file:line citations from generated text."""
    pattern = r'[\w/.-]+\.(?:f(?:90|95|03|08)?|for|fpp):\d+(?:-\d+)?'
    return list(set(re.findall(pattern, text, re.IGNORECASE)))


def _build_citation_fallback(chunks: list[dict]) -> list[str]:
    """Build citation list from chunk metadata when LLM omits them."""
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        if file_path:
            fname = Path(file_path).name
            if start and end:
                citations.append(f"{fname}:{start}-{end}")
            else:
                citations.append(fname)
    return list(dict.fromkeys(citations))  # dedupe preserving order


def _build_messages(
    query: str,
    chunks: list[dict],
    capability: str | None = None,
    context_budget: int = CONTEXT_TOKEN_BUDGET,
):
    """Build the messages list for the LLM call."""
    context = _assemble_context(chunks, budget=context_budget)

    if capability and capability in CAPABILITIES:
        system_prompt = CAPABILITIES[capability]
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    user_message = f"""Context from the LAPACK Fortran codebase:

{context}

---

Question: {query}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def generate_answer(
    query: str,
    chunks: list[dict],
    capability: str | None = None,
    max_completion_tokens: int = 2048,
    context_budget: int = CONTEXT_TOKEN_BUDGET,
    track_ttft: bool = False,
    model: str | None = None,
) -> dict:
    """Generate an answer using retrieved context chunks.

    When track_ttft=True, uses streaming to measure time-to-first-token
    (returned as ttft_ms in the result dict).
    """
    settings = get_settings()
    client = _get_generation_client()
    resolved_model = model or settings.CHAT_MODEL

    if not chunks:
        logger.warning("No chunks retrieved for query: %.80s", query)
        return {
            "answer": "I don't have sufficient context from the LAPACK codebase to answer this question. Try rephrasing your query or asking about a specific routine.",
            "citations": [],
            "model": resolved_model,
            "token_usage": {},
        }

    messages = _build_messages(query, chunks, capability, context_budget)

    if track_ttft:
        answer_text, token_usage, ttft_ms = await _generate_with_ttft(
            client, resolved_model, messages, max_completion_tokens,
        )
    else:
        kwargs = _build_llm_kwargs(resolved_model, messages, max_completion_tokens)
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("LLM generation failed (model=%s): %s", resolved_model, e)
            raise

        answer_text = ""
        if len(response.choices) > 0:
            message = response.choices[0].message
            if message.content is not None:
                answer_text = message.content

        token_usage = {}
        if response.usage is not None:
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        ttft_ms = None

    citations = _extract_citations_from_text(answer_text)

    # Citation enforcement fallback
    if not citations:
        citations = _build_citation_fallback(chunks[:5])
        if citations:
            answer_text += "\n\nSources: " + ", ".join(citations)

    result = {
        "answer": answer_text,
        "citations": citations,
        "model": resolved_model,
        "token_usage": token_usage,
    }
    if ttft_ms is not None:
        result["ttft_ms"] = ttft_ms
    return result


async def _generate_with_ttft(client, model, messages, max_completion_tokens):
    """Stream a completion and measure time-to-first-token."""
    t0 = time.time()
    kwargs = _build_llm_kwargs(model, messages, max_completion_tokens, stream=True)
    try:
        stream = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error("LLM streaming (ttft) failed (model=%s): %s", model, e)
        raise

    accumulated = []
    token_usage = {}
    ttft_ms = None

    async for chunk in stream:
        if chunk.usage is not None:
            token_usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                if ttft_ms is None:
                    ttft_ms = round((time.time() - t0) * 1000, 1)
                accumulated.append(delta.content)

    return "".join(accumulated), token_usage, ttft_ms


async def generate_answer_stream(
    query: str,
    chunks: list[dict],
    capability: str | None = None,
    max_completion_tokens: int = 2048,
    context_budget: int = CONTEXT_TOKEN_BUDGET,
    model: str | None = None,
):
    """Stream answer tokens, yielding dicts for each event.

    Yields:
        {"type": "token", "token": str}           — per content delta
        {"type": "done", "citations": list, "token_usage": dict} — final
    """
    settings = get_settings()
    client = _get_generation_client()
    resolved_model = model or settings.CHAT_MODEL

    if not chunks:
        logger.warning("No chunks retrieved for query: %.80s", query)
        yield {
            "type": "token",
            "token": "I don't have sufficient context from the LAPACK codebase to answer this question. Try rephrasing your query or asking about a specific routine.",
        }
        yield {"type": "done", "citations": [], "token_usage": {}}
        return

    messages = _build_messages(query, chunks, capability, context_budget)

    kwargs = _build_llm_kwargs(resolved_model, messages, max_completion_tokens, stream=True)

    try:
        stream = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error("LLM stream generation failed (model=%s): %s", resolved_model, e)
        yield {"type": "error", "message": str(e)}
        return

    accumulated = []
    token_usage = {}

    async for chunk in stream:
        if chunk.usage is not None:
            token_usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }

        if chunk.choices:
            delta = chunk.choices[0].delta
            content = delta.content
            if content:
                accumulated.append(content)
                yield {"type": "token", "token": content}

    full_text = "".join(accumulated)
    citations = _extract_citations_from_text(full_text)

    if not citations:
        citations = _build_citation_fallback(chunks[:5])
        if citations:
            suffix = "\n\nSources: " + ", ".join(citations)
            yield {"type": "token", "token": suffix}

    yield {"type": "done", "citations": citations, "token_usage": token_usage}
