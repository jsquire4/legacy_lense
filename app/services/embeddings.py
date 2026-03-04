"""OpenAI embedding service with batching and token pre-truncation."""

import logging
from functools import lru_cache

import tiktoken
from openai import AsyncOpenAI, OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 512


@lru_cache
def _get_encoder(encoding_name: str = "cl100k_base"):
    return tiktoken.get_encoding(encoding_name)


def _encoder_for_model(model_name: str | None = None):
    """Get the tiktoken encoder for an embedding model."""
    if model_name:
        try:
            from app.embedding_registry import get_model_info
            info = get_model_info(model_name)
            return _get_encoder(info.tokenizer)
        except KeyError:
            pass
    return _get_encoder("cl100k_base")


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3)


@lru_cache
def get_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3)


def _truncate_to_tokens(text: str, max_tokens: int, encoder=None) -> str:
    enc = encoder or _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# --- Async query path (used by API endpoints) ---

_embed_cache: dict[tuple[str, str], list[float]] = {}
_EMBED_CACHE_MAX = 128


async def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string with in-memory cache."""
    settings = get_settings()
    resolved_model = model or settings.EMBEDDING_MODEL
    cache_key = (resolved_model, text)

    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    client = get_async_openai_client()
    encoder = _encoder_for_model(resolved_model)
    truncated = _truncate_to_tokens(text, settings.MAX_CHUNK_TOKENS, encoder)

    response = await client.embeddings.create(
        model=resolved_model,
        input=[truncated],
    )
    embedding = response.data[0].embedding

    # Evict oldest if full
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        oldest = next(iter(_embed_cache))
        del _embed_cache[oldest]
    _embed_cache[cache_key] = embedding

    return embedding


# --- Sync batch path (used by ingestion scripts) ---

def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of texts using OpenAI's embedding API. Handles batching and truncation."""
    settings = get_settings()
    client = get_openai_client()
    resolved_model = model or settings.EMBEDDING_MODEL
    max_tokens = settings.MAX_CHUNK_TOKENS
    encoder = _encoder_for_model(resolved_model)

    # Pre-truncate all texts
    truncated = [_truncate_to_tokens(t, max_tokens, encoder) for t in texts]

    all_embeddings = []
    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=resolved_model,
            input=batch,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(all_embeddings)} embeddings for {len(texts)} texts"
        )

    return all_embeddings
