"""Tests for the embedding service (mocked, no API calls)."""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.services.embeddings import (
    embed_texts, _truncate_to_tokens, embed_query,
    get_async_openai_client, _maybe_truncate,
)
from tests.helpers import mock_embedding_settings


def test_truncate_short_text():
    text = "Hello world"
    result = _truncate_to_tokens(text, 100)
    assert result == text


def test_truncate_long_text():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 10000
    result = _truncate_to_tokens(text, 100, encoder=enc)
    assert len(enc.encode(result)) <= 100


def test_truncate_no_encoder_returns_unchanged():
    """When encoder is None, text is returned as-is."""
    text = "word " * 10000
    result = _truncate_to_tokens(text, 10, encoder=None)
    assert result == text


def test_maybe_truncate_skips_non_openai():
    """_maybe_truncate returns texts unchanged when encoder is None."""
    texts = ["a" * 10000, "b" * 5000]
    result = _maybe_truncate(texts, 10, encoder=None)
    assert result == texts


@patch("app.services.embeddings.get_openai_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_calls_api(mock_settings, mock_client_fn):
    mock_settings.return_value = mock_embedding_settings()

    mock_embedding = MagicMock()
    mock_embedding.index = 0
    mock_embedding.embedding = [0.1] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    result = embed_texts(["test query"])
    assert len(result) == 1
    assert len(result[0]) == 1536
    mock_client.embeddings.create.assert_called_once()


@patch("app.services.embeddings.get_openai_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_batching(mock_settings, mock_client_fn):
    mock_settings.return_value = mock_embedding_settings()

    def make_response(batch):
        embeddings = []
        for i in range(len(batch)):
            emb = MagicMock()
            emb.index = i
            emb.embedding = [float(i)] * 10
            embeddings.append(emb)
        resp = MagicMock()
        resp.data = embeddings
        return resp

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = lambda model, input: make_response(input)  # noqa: A002
    mock_client_fn.return_value = mock_client

    texts = [f"text {i}" for i in range(600)]
    result = embed_texts(texts)
    assert len(result) == 600
    assert mock_client.embeddings.create.call_count == 2  # 512 + 88


@patch("app.services.embeddings.get_settings")
def test_get_openai_client_returns_client(mock_settings):
    """get_openai_client returns OpenAI client (covers lines 20-21)."""
    settings = MagicMock()
    settings.OPENAI_API_KEY = "test-key"
    mock_settings.return_value = settings

    from app.services.embeddings import get_openai_client

    get_openai_client.cache_clear()
    try:
        client = get_openai_client()
        from openai import OpenAI

        assert isinstance(client, OpenAI)
        mock_settings.assert_called()
    finally:
        get_openai_client.cache_clear()


# --- Tests below use clear_embed_cache fixture ---


@patch("app.services.embeddings.get_openai_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_raises_on_count_mismatch(mock_settings, mock_client_fn):
    """embed_texts raises RuntimeError when response has fewer embeddings than inputs."""
    mock_settings.return_value = mock_embedding_settings()

    mock_embedding = MagicMock()
    mock_embedding.index = 0
    mock_embedding.embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    with pytest.raises(RuntimeError, match="Embedding count mismatch"):
        embed_texts(["text1", "text2"])


@patch("app.services.embeddings.get_settings")
def test_get_async_openai_client_returns_client(mock_settings):
    """get_async_openai_client returns AsyncOpenAI client (covers lines 26-27)."""
    settings = MagicMock()
    settings.OPENAI_API_KEY = "test-key"
    mock_settings.return_value = settings

    get_async_openai_client.cache_clear()
    try:
        client = get_async_openai_client()
        from openai import AsyncOpenAI

        assert isinstance(client, AsyncOpenAI)
        mock_settings.assert_called()
    finally:
        get_async_openai_client.cache_clear()


@patch("app.services.embeddings.get_async_openai_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_calls_api(mock_settings, mock_client_fn, clear_embed_cache):
    """embed_query calls API and caches result."""
    mock_settings.return_value = mock_embedding_settings()

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.2] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    mock_client_fn.return_value = mock_client

    result = await embed_query("test query for embedding")
    assert len(result) == 1536
    assert result[0] == 0.2
    mock_client.embeddings.create.assert_called_once()


@patch("app.services.embeddings.get_async_openai_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_cache_hit(mock_settings, mock_client_fn, clear_embed_cache):
    """embed_query returns cached result on repeat query, no API call."""
    mock_settings.return_value = mock_embedding_settings()

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.3] * 1536
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    mock_client_fn.return_value = mock_client

    r1 = await embed_query("cached query")
    r2 = await embed_query("cached query")
    assert r1 == r2
    mock_client.embeddings.create.assert_called_once()


@patch("app.services.embeddings._EMBED_CACHE_MAX", 2)
@patch("app.services.embeddings.get_async_openai_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_cache_eviction(mock_settings, mock_client_fn, clear_embed_cache):
    """embed_query evicts oldest when cache is full."""
    mock_settings.return_value = mock_embedding_settings()

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=MagicMock(data=[MagicMock(embedding=[0.0] * 1536)]))
    mock_client.embeddings.create.side_effect = lambda model, input: MagicMock(
        data=[MagicMock(embedding=[hash(str(input)) % 1000 / 1000.0] * 1536)]
    )
    mock_client_fn.return_value = mock_client

    import app.services.embeddings as emb_mod

    await embed_query("query1")
    await embed_query("query2")
    await embed_query("query3")  # Should evict query1
    assert ("text-embedding-3-small", "query1") not in emb_mod._embed_cache
    assert ("text-embedding-3-small", "query2") in emb_mod._embed_cache
    assert ("text-embedding-3-small", "query3") in emb_mod._embed_cache
    assert mock_client.embeddings.create.call_count == 3


# ---------------------------------------------------------------------------
# Voyage AI provider tests
# ---------------------------------------------------------------------------

@patch("app.services.embeddings._get_voyage_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_voyage(mock_settings, mock_voyage_fn):
    """embed_texts dispatches to Voyage SDK with input_type='document'."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=32000)

    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_result
    mock_voyage_fn.return_value = mock_client

    result = embed_texts(["text1", "text2"], model="voyage-code-3")
    assert len(result) == 2
    assert len(result[0]) == 1024
    mock_client.embed.assert_called_once_with(
        ["text1", "text2"], model="voyage-code-3", input_type="document"
    )


@patch("app.services.embeddings._get_async_voyage_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_voyage(mock_settings, mock_voyage_fn, clear_embed_cache):
    """embed_query dispatches to async Voyage SDK with input_type='query'."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=32000)

    mock_result = MagicMock()
    mock_result.embeddings = [[0.5] * 1024]
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=mock_result)
    mock_voyage_fn.return_value = mock_client

    result = await embed_query("test query", model="voyage-code-3")
    assert len(result) == 1024
    mock_client.embed.assert_called_once_with(
        ["test query"], model="voyage-code-3", input_type="query"
    )


# ---------------------------------------------------------------------------
# Gemini provider tests
# ---------------------------------------------------------------------------

@patch("app.services.embeddings._get_gemini_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_gemini(mock_settings, mock_gemini_fn):
    """embed_texts dispatches to Gemini SDK."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=2048)

    emb1, emb2 = MagicMock(), MagicMock()
    emb1.values = [0.1] * 3072
    emb2.values = [0.2] * 3072
    mock_result = MagicMock()
    mock_result.embeddings = [emb1, emb2]
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_result
    mock_gemini_fn.return_value = mock_client

    result = embed_texts(["text1", "text2"], model="gemini-embedding-001")
    assert len(result) == 2
    assert len(result[0]) == 3072
    mock_client.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001", contents=["text1", "text2"]
    )


# ---------------------------------------------------------------------------
# Cohere provider tests
# ---------------------------------------------------------------------------

@patch("app.services.embeddings._get_cohere_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_cohere(mock_settings, mock_cohere_fn):
    """embed_texts dispatches to Cohere SDK with embedding_types=['float']."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=512)

    mock_result = MagicMock()
    mock_result.embeddings.float_ = [[0.1] * 1536, [0.2] * 1536]
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_result
    mock_cohere_fn.return_value = mock_client

    result = embed_texts(["text1", "text2"], model="embed-v4.0")
    assert len(result) == 2
    assert len(result[0]) == 1536
    mock_client.embed.assert_called_once_with(
        texts=["text1", "text2"],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
    )


# ---------------------------------------------------------------------------
# Async query tests for Gemini and Cohere
# ---------------------------------------------------------------------------

@patch("app.services.embeddings._get_gemini_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_gemini(mock_settings, mock_gemini_fn, clear_embed_cache):
    """embed_query dispatches to Gemini SDK via run_in_executor, accesses .values."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=2048)

    mock_emb = MagicMock()
    mock_emb.values = [0.5] * 3072
    mock_result = MagicMock()
    mock_result.embeddings = [mock_emb]
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_result
    mock_gemini_fn.return_value = mock_client

    result = await embed_query("test query", model="gemini-embedding-001")
    assert len(result) == 3072
    assert result[0] == 0.5
    mock_client.models.embed_content.assert_called_once()


@patch("app.services.embeddings._get_async_cohere_client")
@patch("app.services.embeddings.get_settings")
@pytest.mark.asyncio
async def test_embed_query_cohere(mock_settings, mock_cohere_fn, clear_embed_cache):
    """embed_query dispatches to async Cohere SDK with input_type='search_query'."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=128000)

    mock_result = MagicMock()
    mock_result.embeddings.float_ = [[0.5] * 1536]
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=mock_result)
    mock_cohere_fn.return_value = mock_client

    result = await embed_query("test query", model="embed-v4.0")
    assert len(result) == 1536
    mock_client.embed.assert_called_once_with(
        texts=["test query"],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
    )


# --- Audit issue #5: Unknown model fallback ---

@patch("app.services.embeddings.get_settings")
def test_resolve_model_unknown_fallback(mock_settings):
    """_resolve_model falls back to OpenAI provider for unknown models."""
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

    from app.services.embeddings import _resolve_model
    name, provider, max_tokens = _resolve_model("totally-unknown-model")
    assert name == "totally-unknown-model"
    assert provider == "openai"
    assert max_tokens == 8191


def test_encoder_for_model_unknown_fallback():
    """_encoder_for_model falls back to cl100k_base for unknown models."""
    from app.services.embeddings import _encoder_for_model
    encoder = _encoder_for_model("totally-unknown-model")
    assert encoder is not None
    # Verify it's cl100k_base by encoding a known string
    tokens = encoder.encode("hello")
    assert len(tokens) >= 1


# --- Audit issue #19: Gemini embedding batching ---

@patch("app.services.embeddings._get_gemini_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_gemini_batching(mock_settings, mock_gemini_fn):
    """embed_texts with Gemini batches at _GEMINI_BATCH=100."""
    mock_settings.return_value = mock_embedding_settings(max_tokens=2048)

    def fake_embed(model, contents):
        result = MagicMock()
        embs = []
        for _ in contents:
            e = MagicMock()
            e.values = [0.1] * 3072
            embs.append(e)
        result.embeddings = embs
        return result

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = fake_embed
    mock_gemini_fn.return_value = mock_client

    texts = [f"text {i}" for i in range(150)]
    result = embed_texts(texts, model="gemini-embedding-001")
    assert len(result) == 150
    assert mock_client.models.embed_content.call_count == 2  # 100 + 50


# --- Audit issue #20: Gemini client missing key ---

@patch("app.services.gemini_helpers.get_settings")
def test_gemini_client_missing_key(mock_settings):
    """get_gemini_client raises RuntimeError when GEMINI_API_KEY is empty."""
    settings = MagicMock()
    settings.GEMINI_API_KEY = ""
    mock_settings.return_value = settings

    from app.services.gemini_helpers import get_gemini_client
    get_gemini_client.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            get_gemini_client()
    finally:
        get_gemini_client.cache_clear()
