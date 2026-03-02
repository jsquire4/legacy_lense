"""OpenAI embedding service with batching and token pre-truncation."""

import logging
from functools import lru_cache

import tiktoken
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")

BATCH_SIZE = 512


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=3)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI's embedding API. Handles batching and truncation."""
    settings = get_settings()
    client = get_openai_client()
    max_tokens = settings.MAX_CHUNK_TOKENS

    # Pre-truncate all texts
    truncated = [_truncate_to_tokens(t, max_tokens) for t in texts]

    all_embeddings = []
    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
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
