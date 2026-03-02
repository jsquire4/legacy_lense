"""Tests for the retrieval service (mocked, no API/Qdrant calls)."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.retrieval import _extract_routine_name, retrieve


def test_extract_routine_name_found():
    """_extract_routine_name extracts LAPACK routine from query."""
    assert _extract_routine_name("What is DGESV?") == "DGESV"
    assert _extract_routine_name("Explain DGETRF") == "DGETRF"
    assert _extract_routine_name("How does DGEMM work?") == "DGEMM"


def test_extract_routine_name_not_found():
    """_extract_routine_name returns None when no match."""
    assert _extract_routine_name("How does LU decomposition work?") is None
    assert _extract_routine_name("no match here") is None


def test_extract_routine_name_sdczi_prefix():
    """_extract_routine_name only matches S/D/C/Z/I prefix."""
    assert _extract_routine_name("What is DGESV?") == "DGESV"
    assert _extract_routine_name("What is ZGEMM?") == "ZGEMM"


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
def test_retrieve_embed_failure(mock_embed, mock_search_by_name, mock_search):
    """retrieve returns failed strategy when embed_texts returns empty."""
    mock_embed.return_value = []

    result = retrieve("query", top_k=5)
    assert result["chunks"] == []
    assert result["retrieval_strategy"] == "failed"
    mock_search.assert_not_called()


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
def test_retrieve_name_match(mock_embed, mock_search_by_name, mock_search):
    """retrieve uses name_match when routine name in query."""
    mock_embed.return_value = [[0.1] * 1536]
    mock_search_by_name.return_value = [
        {"id": "n1", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV"}},
    ]
    mock_search.return_value = [
        {"id": "v1", "score": 0.8, "text": "t", "metadata": {}},
    ]

    result = retrieve("What is DGESV?", top_k=5)
    assert result["retrieval_strategy"] == "name_match"
    assert len(result["chunks"]) >= 1
    assert any(c.get("_match_type") == "name" for c in result["chunks"])
    mock_search_by_name.assert_called()


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
@patch("app.services.retrieval.get_openai_client")
def test_retrieve_query_expansion(mock_client_fn, mock_embed, mock_search_by_name, mock_search):
    """retrieve uses query_expansion for conceptual query."""
    mock_embed.return_value = [[0.1] * 1536]
    mock_search_by_name.return_value = [
        {"id": "e1", "score": 0.9, "text": "t", "metadata": {"unit_name": "DGETRF"}},
    ]
    mock_search.return_value = []

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "DGETRF DGETRF2 DGESV"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    with patch("app.services.retrieval.get_settings") as mock_settings:
        mock_settings.return_value.CHAT_MODEL = "gpt-4o-mini"
        result = retrieve("How does LU decomposition work?", top_k=5)

    assert result["retrieval_strategy"] == "query_expansion"
    assert len(result["expanded_names"]) >= 1
    assert any(c.get("_match_type") == "expansion" for c in result["chunks"])


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
@patch("app.services.retrieval.get_openai_client")
def test_retrieve_expansion_exception_returns_empty(mock_client_fn, mock_embed, mock_search_by_name, mock_search):
    """_expand_query returns [] on exception, retrieval continues with vector."""
    mock_embed.return_value = [[0.1] * 1536]
    mock_client_fn.return_value.chat.completions.create.side_effect = Exception("API error")
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.8, "text": "t", "metadata": {}},
    ]

    result = retrieve("How does LU work?", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert len(result["chunks"]) >= 1


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
def test_retrieve_call_graph_follow(mock_embed, mock_search_by_name, mock_search):
    """retrieve follows call graph from name-matched results."""
    mock_embed.return_value = [[0.1] * 1536]
    mock_search_by_name.side_effect = [
        [{"id": "n1", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV", "called_routines": ["DGETRF", "DGETRS"]}}],
        [{"id": "c1", "score": 0.9, "text": "t", "metadata": {"unit_name": "DGETRF"}}],
        [{"id": "c2", "score": 0.85, "text": "t", "metadata": {"unit_name": "DGETRS"}}],
    ]
    mock_search.return_value = []

    result = retrieve("What is DGESV?", top_k=5)
    assert any(c.get("_match_type") == "call_graph" for c in result["chunks"])
    assert mock_search_by_name.call_count >= 2


@patch("app.services.retrieval.search")
@patch("app.services.retrieval.search_by_name")
@patch("app.services.retrieval.embed_texts")
@patch("app.services.retrieval._expand_query")
def test_retrieve_vector_merge_dedup(mock_expand, mock_embed, mock_search_by_name, mock_search):
    """retrieve merges vector results and deduplicates by id."""
    mock_embed.return_value = [[0.1] * 1536]
    mock_expand.return_value = []
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.9, "text": "t1", "metadata": {}},
        {"id": "v2", "score": 0.8, "text": "t2", "metadata": {}},
    ]

    result = retrieve("conceptual query", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert len(result["chunks"]) == 2
    ids = [c["id"] for c in result["chunks"]]
    assert len(ids) == len(set(ids))
