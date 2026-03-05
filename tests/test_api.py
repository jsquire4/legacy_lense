"""Smoke tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES
from tests.helpers import make_retrieve_result, make_generate_result
from tests.helpers import parse_sse_events

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_response_cache():
    """Clear the response cache before and after every test to prevent leaks."""
    import app.main as main_mod
    main_mod._RESPONSE_CACHE.clear()
    yield
    main_mod._RESPONSE_CACHE.clear()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "LegacyLens" in response.text


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint(mock_retrieve, mock_generate):
    mock_retrieve.return_value = make_retrieve_result()
    mock_generate.return_value = make_generate_result(citations=["test.f:1-10"])

    response = client.post("/api/query", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "latency_ms" in data


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_hit(mock_retrieve, mock_generate):
    """Second identical query returns cached response; retrieve/generate called once."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "c1", "text": "cached", "score": 0.9, "metadata": {"file_path": "x.f"}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result(answer="Cached answer", citations=["x.f:1-5"])

    payload = {"query": "Cache me please", "top_k": 8}
    r1 = client.post("/api/query", json=payload)
    r2 = client.post("/api/query", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["answer"] == r2.json()["answer"]
    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once()


@patch("app.main._CACHE_TTL", 0)
@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_ttl_expiry(mock_retrieve, mock_generate):
    """Cache entry expires after TTL; next request fetches fresh data (covers del path)."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result()

    payload = {"query": "TTL expiry test query"}
    r1 = client.post("/api/query", json=payload)
    assert r1.status_code == 200
    assert mock_retrieve.call_count == 1

    r2 = client.post("/api/query", json=payload)
    assert r2.status_code == 200
    assert mock_retrieve.call_count == 2


@patch("app.main._CACHE_MAX", 3)
@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_eviction(mock_retrieve, mock_generate):
    """Cache evicts oldest when exceeding _CACHE_MAX."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result()

    import app.main as main_mod

    for i in range(5):
        client.post("/api/query", json={"query": f"Distinct query {i} unique"})
    assert mock_retrieve.call_count == 5
    assert len(main_mod._RESPONSE_CACHE) <= 3


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_capability_endpoint(mock_retrieve, mock_generate):
    mock_retrieve.return_value = make_retrieve_result()
    mock_generate.return_value = make_generate_result(answer="Explained")

    response = client.post(
        "/api/capabilities/explain_code",
        json={"query": "Explain DGESV"},
    )
    assert response.status_code == 200


def test_unknown_capability():
    response = client.post(
        "/api/capabilities/nonexistent",
        json={"query": "test"},
    )
    assert response.status_code == 404


def test_unknown_capability_stream():
    """Capability stream endpoint returns 404 for unknown capability."""
    response = client.post(
        "/api/capabilities/nonexistent/stream",
        json={"query": "test"},
    )
    assert response.status_code == 404


def test_query_request_validation_empty_query():
    """QueryRequest rejects empty query."""
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 422


def test_query_request_validation_top_k_bounds():
    """QueryRequest enforces top_k between 1 and 20."""
    response_lo = client.post("/api/query", json={"query": "test", "top_k": 0})
    assert response_lo.status_code == 422
    response_hi = client.post("/api/query", json={"query": "test", "top_k": 21})
    assert response_hi.status_code == 422


@patch("app.main.setup_logging")
def test_lifespan_calls_setup_logging(mock_setup_logging):
    """Lifespan context calls setup_logging on startup."""
    with TestClient(app) as c:
        c.get("/health")
    mock_setup_logging.assert_called()



@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = make_retrieve_result()

    async def async_gen(*args, **kwargs):
        yield {"type": "token", "token": "Hello"}
        yield {"type": "token", "token": " world"}
        yield {"type": "done", "citations": ["test.f:1-10"], "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/query/stream", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    assert "retrieval" in event_types
    assert "token" in event_types
    assert "done" in event_types


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_skips_unknown_event_types(mock_retrieve, mock_stream):
    """Stream generator skips events with type other than token/done (branch coverage)."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
    )

    async def async_gen(*args, **kwargs):
        yield {"type": "unknown"}  # skipped
        yield {"type": "token", "token": "Hi"}
        yield {"type": "done", "citations": [], "token_usage": {}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/query/stream", json={"query": "test"})
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["event"] == "done"


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_capability_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = make_retrieve_result()

    async def async_gen(*args, **kwargs):
        yield {"type": "token", "token": "Explained"}
        yield {"type": "done", "citations": [], "token_usage": {}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/capabilities/explain_code/stream", json={"query": "Explain DGESV"})
    assert response.status_code == 200

    events = parse_sse_events(response.text)
    assert events[-1]["event"] == "done"


# --- Eval tests (patches point to app.services.eval_runner) ---

_EVAL_RETRIEVE = make_retrieve_result(
    chunks=[{"id": "x1", "text": "t", "score": 0.9, "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}],
    strategy="name_match",
)


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_returns_sse(mock_retrieve):
    mock_retrieve.return_value = _EVAL_RETRIEVE
    response = client.get("/api/eval/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_chunks_without_file_path(mock_retrieve):
    """Eval stream handles chunks with no file_path in metadata (branch coverage)."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[
            {"id": "x1", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "name"},
            {"id": "x2", "text": "t", "score": 0.8, "metadata": {"file_path": "dgesv.f"}, "_match_type": "vector"},
        ],
    )

    response = client.get("/api/eval/stream")
    events = parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(EVAL_QUERIES)
    first = progress_events[0]["data"]
    assert "retrieved_files" in first


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_deduplicates_file_paths(mock_retrieve):
    """Eval stream deduplicates retrieved_files when multiple chunks share file_path (branch coverage)."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[
            {"id": "x1", "text": "t", "score": 0.9, "metadata": {"file_path": "/path/dgesv.f"}, "_match_type": "name"},
            {"id": "x2", "text": "t2", "score": 0.85, "metadata": {"file_path": "/other/dgesv.f"}, "_match_type": "vector"},
        ],
    )

    response = client.get("/api/eval/stream")
    events = parse_sse_events(response.text)
    first_progress = [e for e in events if e["event"] == "progress"][0]["data"]
    assert first_progress["retrieved_files"].count("dgesv.f") == 1


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_emits_progress_events(mock_retrieve):
    mock_retrieve.return_value = _EVAL_RETRIEVE

    response = client.get("/api/eval/stream")
    events = parse_sse_events(response.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(EVAL_QUERIES)

    first = progress_events[0]["data"]
    assert "query" in first
    assert "capability" in first
    assert "recall_at_5" in first
    assert "latency_ms" in first
    assert "retrieved_files" in first
    assert "expected_files" in first
    assert first["index"] == 0
    # New retrieval metrics
    assert "mrr" in first
    assert "ndcg_at_5" in first
    assert "negative_oracle_pass" in first
    assert "precision_at_1" in first
    assert "precision_at_3" in first
    assert "recall_at_1" in first
    assert "recall_at_3" in first
    assert "difficulty" in first


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_summary_is_last_event(mock_retrieve):
    mock_retrieve.return_value = _EVAL_RETRIEVE

    response = client.get("/api/eval/stream")
    events = parse_sse_events(response.text)

    assert events[-1]["event"] == "summary"
    summary = events[-1]["data"]
    assert "avg_recall_at_5" in summary
    assert "avg_latency_ms" in summary
    assert summary["total_queries"] == len(EVAL_QUERIES)
    assert 0.0 <= summary["avg_recall_at_5"] <= 1.0
    # New summary metrics
    assert "avg_mrr" in summary
    assert "avg_ndcg_at_5" in summary
    assert "negative_oracle_pass_rate" in summary
    assert "by_difficulty" in summary
    by_diff = summary["by_difficulty"]
    assert isinstance(by_diff, dict)
    # Should have at least one difficulty tier
    for tier_data in by_diff.values():
        assert "avg_recall_at_5" in tier_data
        assert "avg_mrr" in tier_data
        assert "count" in tier_data


@patch("app.services.embeddings.embed_query", new_callable=AsyncMock)
@patch("app.services.eval_runner.generate_answer", new_callable=AsyncMock)
@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_e2e_eval_stream_endpoint(mock_retrieve, mock_generate, mock_embed):
    mock_retrieve.return_value = _EVAL_RETRIEVE
    # Answer must be >=200 chars and contain ALL keywords for each query.
    # Use a kitchen-sink answer that covers all keyword sets.
    mock_generate.return_value = make_generate_result(
        answer=(
            "DGESV solves a linear system of equations using LU factorization with partial pivoting. "
            "It calls DGETRF and DGETRS for the factorization and back-substitution steps. "
            "The singular bidiagonal decomposition is computed via DGESVD with JOBU parameter. "
            "DGEMM performs matrix multiply with alpha and transpose options. "
            "The pivot selection uses blocked algorithms. DLANGE computes the Frobenius norm of a matrix. "
            "Cholesky factorization for symmetric positive definite matrices is done by DPOTRF. "
            "DGELS solves least squares problems using QR factorization and orthogonal transformations. "
            "Error checking validates INFO and XERBLA handles dimension and LDA constraints. "
            "Workspace query via LWORK returns optimal workspace size. "
            "Loop and block and cache optimizations improve BLAS performance. "
            "The driver routines handle substitution and validation of singular systems. "
            "IDAMAX selects pivots. This routine does not exist in LAPACK source."
        ),
        citations=["dgesv.f:1-50"],
    )
    # Mock embed_query to return a fixed vector
    mock_embed.return_value = [1.0, 0.0, 0.0]

    response = client.get("/api/eval/e2e/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(E2E_EVAL_QUERIES)

    first = progress_events[0]["data"]
    assert "passed" in first
    assert "capability" in first
    assert "checks" in first
    assert "latency_ms" in first
    assert "answer_preview" in first
    assert "citations" in first
    assert isinstance(first["citations"], list)
    # New E2E fields
    assert "is_hallucination_probe" in first
    assert "citation_is_fallback" in first

    assert events[-1]["event"] == "summary"
    summary = events[-1]["data"]
    assert "pass_rate" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert summary["total_queries"] == len(E2E_EVAL_QUERIES)
    # New summary fields
    assert "avg_similarity" in summary
    assert "hallucination_probe_pass_rate" in summary
    assert "citation_fallback_count" in summary


@patch("app.services.embeddings.embed_query", new_callable=AsyncMock)
@patch("app.services.eval_runner.generate_answer", new_callable=AsyncMock)
@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_e2e_eval_stream_includes_failed_checks(mock_retrieve, mock_generate, mock_embed):
    """E2E eval stream includes progress events when checks fail (branch coverage)."""
    mock_retrieve.return_value = _EVAL_RETRIEVE
    mock_generate.return_value = make_generate_result(answer="Short.")
    mock_embed.return_value = [1.0, 0.0, 0.0]

    response = client.get("/api/eval/e2e/stream")
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) >= 1
    failed = [p for p in progress_events if not p["data"].get("passed", True)]
    assert len(failed) >= 1


# --- Model endpoint tests ---

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    assert len(models) >= 11  # 9 OpenAI + 2 Gemini
    names = [m["name"] for m in models]
    assert "gpt-4o-mini" in names
    assert "gpt-4o" in names
    defaults = [m for m in models if m.get("default")]
    assert len(defaults) == 1
    for m in models:
        assert "input_cost_per_1m" in m
        assert "output_cost_per_1m" in m
        assert "available" in m
        assert "provider" in m


@patch("app.main.get_settings")
def test_models_endpoint_available_field(mock_settings):
    """Models endpoint returns available=True/False based on API keys."""
    settings = MagicMock()
    settings.OPENAI_API_KEY = "sk-test"
    settings.GEMINI_API_KEY = ""
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    response = client.get("/api/models")
    data = response.json()
    models = data["models"]

    openai_models = [m for m in models if m["provider"] == "openai"]
    gemini_models = [m for m in models if m["provider"] == "gemini"]

    assert all(m["available"] for m in openai_models)
    assert all(not m["available"] for m in gemini_models)


# --- Trial CRUD endpoint tests ---

@patch("app.main.save_trial")
def test_create_trial_endpoint(mock_save):
    mock_save.return_value = 1
    response = client.post("/api/trials", json={
        "model": "gpt-4o-mini",
        "eval_type": "retrieval",
        "avg_recall_at_5": 0.85,
        "total_queries": 10,
    })
    assert response.status_code == 200
    assert response.json() == {"id": 1}
    mock_save.assert_called_once()


@patch("app.main.list_trials")
def test_list_trials_endpoint(mock_list):
    mock_list.return_value = [
        {"id": 1, "model": "gpt-4o-mini", "eval_type": "retrieval"},
    ]
    response = client.get("/api/trials")
    assert response.status_code == 200
    assert len(response.json()["trials"]) == 1


@patch("app.main.delete_trial")
def test_delete_trial_endpoint(mock_delete):
    mock_delete.return_value = True
    response = client.delete("/api/trials/1")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}


@patch("app.main.delete_trial")
def test_delete_trial_not_found(mock_delete):
    mock_delete.return_value = False
    response = client.delete("/api/trials/999")
    assert response.status_code == 404


# --- Collections endpoint ---

@patch("app.main.get_qdrant_client")
def test_collections_endpoint(mock_get_client):
    """Collections endpoint returns only ingested embedding models."""
    mock_client = MagicMock()
    coll1 = MagicMock()
    coll1.name = "lapack-text-embedding-3-small"
    coll2 = MagicMock()
    coll2.name = "lapack-voyage-code-3"
    coll3 = MagicMock()
    coll3.name = "unrelated-collection"
    mock_client.get_collections.return_value = MagicMock(collections=[coll1, coll2, coll3])
    mock_get_client.return_value = mock_client

    response = client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    names = [m["name"] for m in data["models"]]
    assert "text-embedding-3-small" in names
    assert "voyage-code-3" in names
    assert len(names) == 2  # unrelated-collection not included


@patch("app.main.get_qdrant_client")
def test_collections_endpoint_qdrant_error(mock_get_client):
    """Collections endpoint returns empty list when get_qdrant_client raises."""
    mock_get_client.side_effect = Exception("Connection refused")
    response = client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []
    assert "error" in data


@patch("app.main.get_qdrant_client")
def test_collections_endpoint_qdrant_get_collections_error(mock_get_client):
    """Collections endpoint returns empty list when get_collections() raises."""
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = Exception("Timeout")
    mock_get_client.return_value = mock_client
    response = client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []
    assert "error" in data


@patch("app.main.get_qdrant_client")
def test_collections_endpoint_no_matches(mock_get_client):
    """Collections endpoint returns empty when no collections match the registry."""
    mock_client = MagicMock()
    coll = MagicMock()
    coll.name = "unrelated-collection"
    mock_client.get_collections.return_value = MagicMock(collections=[coll])
    mock_get_client.return_value = mock_client
    response = client.get("/api/collections")
    assert response.status_code == 200
    assert response.json() == {"models": []}


@patch("app.main.get_qdrant_client")
def test_collections_endpoint_empty_qdrant(mock_get_client):
    """Collections endpoint returns empty when Qdrant has no collections."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_get_client.return_value = mock_client
    response = client.get("/api/collections")
    assert response.status_code == 200
    assert response.json() == {"models": []}


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_with_embedding_model(mock_retrieve, mock_generate):
    """Query endpoint passes embedding_model through to retrieve."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {"file_path": "t.f"}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result()

    response = client.post("/api/query", json={
        "query": "test embedding model param",
        "embedding_model": "voyage-code-3",
    })
    assert response.status_code == 200
    _, kwargs = mock_retrieve.call_args
    assert kwargs.get("collection_name") == "lapack-voyage-code-3"
    assert kwargs.get("embedding_model") == "voyage-code-3"


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_cache_key_differs_by_embedding_model(mock_retrieve, mock_generate):
    """Same query with different embedding_model values produces separate cache entries."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {"file_path": "t.f"}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result()

    payload_a = {"query": "cache key emb test", "embedding_model": "text-embedding-3-small"}
    payload_b = {"query": "cache key emb test", "embedding_model": "voyage-code-3"}
    client.post("/api/query", json=payload_a)
    client.post("/api/query", json=payload_b)
    assert mock_retrieve.call_count == 2  # not cached across different embedding models


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_with_embedding_model(mock_retrieve, mock_stream):
    """Stream endpoint passes embedding_model through to retrieve."""
    mock_retrieve.return_value = make_retrieve_result()

    async def async_gen(*args, **kwargs):
        yield {"type": "token", "token": "Hi"}
        yield {"type": "done", "citations": [], "token_usage": {}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/query/stream", json={
        "query": "test stream emb model",
        "embedding_model": "voyage-code-3",
    })
    assert response.status_code == 200
    _, kwargs = mock_retrieve.call_args
    assert kwargs.get("collection_name") == "lapack-voyage-code-3"
    assert kwargs.get("embedding_model") == "voyage-code-3"


def test_query_endpoint_rejects_unknown_embedding_model():
    """Query endpoint returns 422 for unknown embedding_model."""
    response = client.post("/api/query", json={
        "query": "test unknown model",
        "embedding_model": "nonexistent-model",
    })
    assert response.status_code == 422


# --- Embedding models endpoint ---

@patch("app.main.get_settings")
def test_embedding_models_endpoint(mock_settings):
    """Embedding models endpoint returns all registered models with availability."""
    settings = MagicMock()
    settings.OPENAI_API_KEY = "sk-test"
    settings.VOYAGE_API_KEY = ""
    settings.GEMINI_API_KEY = ""
    settings.COHERE_API_KEY = ""
    mock_settings.return_value = settings

    response = client.get("/api/embedding-models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    names = [m["name"] for m in models]
    assert "text-embedding-3-small" in names
    assert "voyage-code-3" in names
    # OpenAI models available, Voyage not
    openai_models = [m for m in models if m["provider"] == "openai"]
    voyage_models = [m for m in models if m["provider"] == "voyage"]
    assert all(m["available"] for m in openai_models)
    assert all(not m["available"] for m in voyage_models)


# --- Model param on query/eval endpoints ---

@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_with_model(mock_retrieve, mock_generate):
    """Query endpoint passes model param through."""
    mock_retrieve.return_value = make_retrieve_result(
        chunks=[{"id": "x", "text": "t", "score": 0.9, "metadata": {"file_path": "t.f"}, "_match_type": "vector"}],
    )
    mock_generate.return_value = make_generate_result(model="gpt-4o")

    response = client.post("/api/query", json={"query": "test model param", "model": "gpt-4o"})
    assert response.status_code == 200
    mock_retrieve.assert_called_once()
    _, kwargs = mock_retrieve.call_args
    assert kwargs.get("model") == "gpt-4o"


@patch("app.services.eval_runner.retrieve", new_callable=AsyncMock)
def test_eval_stream_with_model_param(mock_retrieve):
    """Eval stream endpoint accepts model query param."""
    mock_retrieve.return_value = _EVAL_RETRIEVE
    response = client.get("/api/eval/stream?model=gpt-4o")
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert any(e["event"] == "summary" for e in events)


@patch("app.main.ingest_stream_generator")
def test_ingest_stream_returns_sse(mock_gen):
    """Ingest stream endpoint returns SSE response."""
    from app.sse import sse_event
    async def fake_gen(embedding_model):
        yield sse_event("progress", {"phase": "parsing", "files": 1, "units": 1, "chunks": 1})
        yield sse_event("summary", {"embedding_model": "text-embedding-3-small", "dimensions": 1536, "files_processed": 1, "chunks_ingested": 1, "ingestion_time_sec": 1.0, "chunks_per_sec": 1.0})
    mock_gen.side_effect = fake_gen
    response = client.get("/api/ingest/stream?embedding_model=text-embedding-3-small")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    assert "progress" in event_types
    assert "summary" in event_types
    summary = [e for e in events if e["event"] == "summary"][0]["data"]
    assert summary["embedding_model"] == "text-embedding-3-small"


@patch("app.main.save_trial")
def test_create_ingestion_trial_endpoint(mock_save):
    mock_save.return_value = 1
    response = client.post("/api/trials", json={
        "model": "text-embedding-3-small",
        "eval_type": "ingestion",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "ingestion_time_sec": 45.2,
        "chunks_ingested": 2407,
        "files_processed": 2300,
    })
    assert response.status_code == 200
    assert response.json() == {"id": 1}
    call_data = mock_save.call_args[0][0]
    assert call_data["ingestion_time_sec"] == 45.2
    assert call_data["chunks_ingested"] == 2407
    assert call_data["files_processed"] == 2300


@patch("app.main._build_response", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_warm_cache_success(mock_build):
    """_warm_cache calls _build_response for each warmup query * model."""
    mock_build.return_value = MagicMock()

    from app.main import _warm_cache, _WARMUP_QUERIES, _WARMUP_MODELS
    await _warm_cache()
    assert mock_build.call_count == len(_WARMUP_QUERIES) * len(_WARMUP_MODELS)


@patch("app.main._build_response", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_warm_cache_handles_errors(mock_build):
    """_warm_cache continues when a query fails."""
    mock_build.side_effect = Exception("API down")

    from app.main import _warm_cache, _WARMUP_QUERIES, _WARMUP_MODELS
    await _warm_cache()  # should not raise
    assert mock_build.call_count == len(_WARMUP_QUERIES) * len(_WARMUP_MODELS)


# --- Audit issue #2: embedding_model validation on eval endpoints ---

def test_eval_stream_rejects_unknown_embedding_model():
    """Eval stream returns 422 for invalid embedding_model."""
    response = client.get("/api/eval/stream?embedding_model=fake")
    assert response.status_code == 422


def test_e2e_eval_stream_rejects_unknown_embedding_model():
    """E2E eval stream returns 422 for invalid embedding_model."""
    response = client.get("/api/eval/e2e/stream?embedding_model=fake")
    assert response.status_code == 422


# --- Audit issue #3: expand endpoint tests ---

@patch("app.main._expand_query", new_callable=AsyncMock)
@patch("app.main._extract_routine_name")
def test_expand_endpoint_with_routine_name(mock_extract, mock_expand):
    """Expand endpoint returns empty expanded_names when query contains a routine name."""
    mock_extract.return_value = "DGESV"
    response = client.post("/api/expand", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    data = response.json()
    assert data["expanded_names"] == []
    assert "query_hash" in data
    mock_expand.assert_not_called()


@patch("app.main._expand_query", new_callable=AsyncMock)
@patch("app.main._extract_routine_name")
def test_expand_endpoint_with_conceptual_query(mock_extract, mock_expand):
    """Expand endpoint returns expanded names for conceptual queries."""
    mock_extract.return_value = None
    mock_expand.return_value = ["DGETRF", "DGESV"]
    response = client.post("/api/expand", json={"query": "How does LU work?"})
    assert response.status_code == 200
    data = response.json()
    assert data["expanded_names"] == ["DGETRF", "DGESV"]
    assert "query_hash" in data
    mock_expand.assert_called_once()


# --- Audit issue #4: query stream retrieval error ---

@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_endpoint_retrieve_error(mock_retrieve):
    """Stream endpoint yields error SSE event when retrieve raises."""
    mock_retrieve.side_effect = Exception("Qdrant timeout")
    response = client.post("/api/query/stream", json={"query": "test error"})
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
    assert "Qdrant timeout" in error_events[0]["data"]["message"]
