"""Embedding model registry for multi-model support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelInfo:
    name: str
    provider: str          # "openai"
    dimensions: int
    tokenizer: str         # tiktoken encoding name
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
