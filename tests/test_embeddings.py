"""Tests for the embedding service (mocked, no API calls)."""

from unittest.mock import patch, MagicMock
import pytest

from app.services.embeddings import embed_texts, _truncate_to_tokens


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
