"""Embedding model registry for multi-model support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelInfo:
    name: str
    provider: str          # "openai", "voyage", "gemini", "cohere"
    dimensions: int
    tokenizer: str | None  # tiktoken encoding name; None for providers that tokenize server-side
    max_tokens: int
    collection_suffix: str  # for Qdrant collection naming


EMBEDDING_MODELS: dict[str, EmbeddingModelInfo] = {
    "text-embedding-3-small": EmbeddingModelInfo(
        name="text-embedding-3-small",
        provider="openai",
        dimensions=1536,
        tokenizer="cl100k_base",
        max_tokens=8191,
        collection_suffix="text-embedding-3-small",
    ),
    "text-embedding-3-large": EmbeddingModelInfo(
        name="text-embedding-3-large",
        provider="openai",
        dimensions=3072,
        tokenizer="cl100k_base",
        max_tokens=8191,
        collection_suffix="text-embedding-3-large",
    ),
    "text-embedding-ada-002": EmbeddingModelInfo(
        name="text-embedding-ada-002",
        provider="openai",
        dimensions=1536,
        tokenizer="cl100k_base",
        max_tokens=8191,
        collection_suffix="text-embedding-ada-002",
    ),
    # --- Voyage AI ---
    "voyage-code-3": EmbeddingModelInfo(
        name="voyage-code-3",
        provider="voyage",
        dimensions=1024,
        tokenizer=None,
        max_tokens=32000,
        collection_suffix="voyage-code-3",
    ),
    "voyage-4-large": EmbeddingModelInfo(
        name="voyage-4-large",
        provider="voyage",
        dimensions=1024,
        tokenizer=None,
        max_tokens=32000,
        collection_suffix="voyage-4-large",
    ),
    "voyage-4": EmbeddingModelInfo(
        name="voyage-4",
        provider="voyage",
        dimensions=1024,
        tokenizer=None,
        max_tokens=32000,
        collection_suffix="voyage-4",
    ),
    "voyage-4-lite": EmbeddingModelInfo(
        name="voyage-4-lite",
        provider="voyage",
        dimensions=1024,
        tokenizer=None,
        max_tokens=32000,
        collection_suffix="voyage-4-lite",
    ),
    # --- Google Gemini ---
    "gemini-embedding-001": EmbeddingModelInfo(
        name="gemini-embedding-001",
        provider="gemini",
        dimensions=3072,
        tokenizer=None,
        max_tokens=2048,
        collection_suffix="gemini-embedding-001",
    ),
    # --- Cohere ---
    "embed-v4.0": EmbeddingModelInfo(
        name="embed-v4.0",
        provider="cohere",
        dimensions=1536,
        tokenizer=None,
        max_tokens=128000,
        collection_suffix="embed-v4-0",
    ),
}


def collection_name_for_model(base: str, model_name: str) -> str:
    """Derive a Qdrant collection name from a base name and embedding model."""
    info = EMBEDDING_MODELS.get(model_name)
    if info:
        return f"{base}-{info.collection_suffix}"
    return f"{base}-{model_name}"


def get_model_info(model_name: str) -> EmbeddingModelInfo:
    """Get model info by name. Raises KeyError if not found."""
    return EMBEDDING_MODELS[model_name]
