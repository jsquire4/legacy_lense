"""Multi-provider embedding service with batching and token pre-truncation."""

import asyncio
import logging
from functools import lru_cache

import tiktoken
from openai import AsyncOpenAI, OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiktoken helpers
# ---------------------------------------------------------------------------

@lru_cache
def _get_encoder(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


def _encoder_for_model(model_name: str | None = None):
    """Get the tiktoken encoder for an embedding model, or None for non-OpenAI."""
    if model_name:
        try:
            from app.embedding_registry import get_model_info
            info = get_model_info(model_name)
            if info.tokenizer is None:
                return None
            return _get_encoder(info.tokenizer)
        except KeyError:
            logger.warning("Unknown embedding model '%s' in _encoder_for_model, using cl100k_base", model_name)
    return _get_encoder("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int, encoder=None) -> str:
    if encoder is None:
        return text
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def _maybe_truncate(texts: list[str], max_tokens: int, encoder) -> list[str]:
    """Truncate texts only when a tiktoken encoder is available."""
    if encoder is None:
        return texts
    return [_truncate_to_tokens(t, max_tokens, encoder) for t in texts]


# ---------------------------------------------------------------------------
# Client factories (all @lru_cache)
# ---------------------------------------------------------------------------

@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3)


@lru_cache
def get_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3)


@lru_cache
def _get_voyage_client():
    import voyageai
    settings = get_settings()
    if not settings.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required to use Voyage AI embeddings")
    return voyageai.Client(api_key=settings.VOYAGE_API_KEY)


@lru_cache
def _get_async_voyage_client():
    import voyageai
    settings = get_settings()
    if not settings.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required to use Voyage AI embeddings")
    return voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY)


def _get_gemini_client():
    from app.services.gemini_helpers import get_gemini_client
    return get_gemini_client()


@lru_cache
def _get_cohere_client():
    import cohere
    settings = get_settings()
    if not settings.COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is required to use Cohere embeddings")
    return cohere.Client(api_key=settings.COHERE_API_KEY)


@lru_cache
def _get_async_cohere_client():
    import cohere
    settings = get_settings()
    if not settings.COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is required to use Cohere embeddings")
    return cohere.AsyncClient(api_key=settings.COHERE_API_KEY)


# ---------------------------------------------------------------------------
# Sync batch embed functions (ingestion path)
# ---------------------------------------------------------------------------

_OPENAI_BATCH = 512
_VOYAGE_BATCH = 50
_GEMINI_BATCH = 100
_COHERE_BATCH = 96


def _batched_embed(texts, model, batch_size, client_factory, embed_call):
    """Generic batched embedding loop."""
    client = client_factory()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(embed_call(client, batch, model))
    return all_embeddings


def _openai_embed_call(client, batch, model):
    response = client.embeddings.create(model=model, input=batch)
    return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


def _voyage_embed_call(client, batch, model):
    result = client.embed(batch, model=model, input_type="document")
    return result.embeddings


def _gemini_embed_call(client, batch, model):
    result = client.models.embed_content(model=model, contents=batch)
    return [e.values for e in result.embeddings]


def _cohere_embed_call(client, batch, model):
    result = client.embed(
        texts=batch, model=model,
        input_type="search_document", embedding_types=["float"],
    )
    return result.embeddings.float_


_SYNC_DISPATCH = {
    "openai": lambda texts, model: _batched_embed(texts, model, _OPENAI_BATCH, get_openai_client, _openai_embed_call),
    "voyage": lambda texts, model: _batched_embed(texts, model, _VOYAGE_BATCH, _get_voyage_client, _voyage_embed_call),
    "gemini": lambda texts, model: _batched_embed(texts, model, _GEMINI_BATCH, _get_gemini_client, _gemini_embed_call),
    "cohere": lambda texts, model: _batched_embed(texts, model, _COHERE_BATCH, _get_cohere_client, _cohere_embed_call),
}


# ---------------------------------------------------------------------------
# Async single embed functions (query path)
# ---------------------------------------------------------------------------

async def _openai_embed_single(text: str, model: str) -> list[float]:
    client = get_async_openai_client()
    response = await client.embeddings.create(model=model, input=[text])
    return response.data[0].embedding


async def _voyage_embed_single(text: str, model: str) -> list[float]:
    client = _get_async_voyage_client()
    result = await client.embed([text], model=model, input_type="query")
    return result.embeddings[0]


async def _gemini_embed_single(text: str, model: str) -> list[float]:
    client = _get_gemini_client()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: client.models.embed_content(model=model, contents=[text])
    )
    return result.embeddings[0].values


async def _cohere_embed_single(text: str, model: str) -> list[float]:
    client = _get_async_cohere_client()
    result = await client.embed(
        texts=[text],
        model=model,
        input_type="search_query",
        embedding_types=["float"],
    )
    return result.embeddings.float_[0]


_ASYNC_DISPATCH = {
    "openai": _openai_embed_single,
    "voyage": _voyage_embed_single,
    "gemini": _gemini_embed_single,
    "cohere": _cohere_embed_single,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_embed_cache: dict[tuple[str, str], list[float]] = {}
_EMBED_CACHE_MAX = 128


def _resolve_model(model: str | None) -> tuple[str, str, int]:
    """Return (resolved_model_name, provider, max_tokens). Defaults to settings."""
    from app.embedding_registry import get_model_info
    settings = get_settings()
    resolved = model or settings.EMBEDDING_MODEL
    try:
        info = get_model_info(resolved)
        return resolved, info.provider, info.max_tokens
    except KeyError:
        logger.warning("Unknown embedding model '%s', falling back to OpenAI provider", resolved)
        return resolved, "openai", settings.MAX_CHUNK_TOKENS


async def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string with in-memory cache."""
    resolved_model, provider, max_tokens = _resolve_model(model)
    cache_key = (resolved_model, text)

    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    encoder = _encoder_for_model(resolved_model)
    truncated = _truncate_to_tokens(text, max_tokens, encoder)

    embed_fn = _ASYNC_DISPATCH[provider]
    embedding = await embed_fn(truncated, resolved_model)

    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        oldest = next(iter(_embed_cache))
        del _embed_cache[oldest]
    _embed_cache[cache_key] = embedding

    return embedding


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of texts. Handles batching, truncation, and provider dispatch."""
    resolved_model, provider, max_tokens = _resolve_model(model)
    encoder = _encoder_for_model(resolved_model)

    truncated = _maybe_truncate(texts, max_tokens, encoder)

    embed_fn = _SYNC_DISPATCH[provider]
    all_embeddings = embed_fn(truncated, resolved_model)

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)} embeddings for {len(texts)} texts"
        )

    return all_embeddings
