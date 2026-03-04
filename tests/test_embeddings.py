"""Tests for the embedding service (mocked, no API calls)."""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.services.embeddings import embed_texts, _truncate_to_tokens, embed_query, get_async_openai_client


def test_truncate_short_text():
    text = "Hello world"
    result = _truncate_to_tokens(text, 100)
    assert result == text


def test_truncate_long_text():
    text = "word " * 10000
    result = _truncate_to_tokens(text, 100)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(result)) <= 100


@patch("app.services.embeddings.get_openai_client")
@patch("app.services.embeddings.get_settings")
def test_embed_texts_calls_api(mock_settings, mock_client_fn):
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
    settings = MagicMock()
    settings.EMBEDDING_MODEL = "text-embedding-3-small"
    settings.MAX_CHUNK_TOKENS = 8191
    mock_settings.return_value = settings

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
