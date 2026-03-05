# Plan: 100% Test Coverage

**Status: ACHIEVED** — 100% coverage (1,747/1,747 statements), 344 tests passing.

---

## Phase 1: New Test Modules (Low Effort, High Impact)

### 1.1 `tests/test_models_data.py` (new file)

**Target:** `app/models_data.py` — 4 missed lines (48–49, 60–61)

| Test | Purpose |
|------|---------|
| `test_get_provider_known_model` | `get_provider("gpt-4o-mini")` → `"openai"` |
| `test_get_provider_unknown_raises` | `get_provider("unknown")` → raises `ValueError` |
| `test_is_reasoning_model_gpt5` | `is_reasoning_model("gpt-5-mini")` → `True` |
| `test_is_reasoning_model_non_gpt5` | `is_reasoning_model("gpt-4o")` → `False` |
| `test_is_reasoning_model_unknown_returns_false` | `is_reasoning_model("unknown")` → `False` (ValueError branch) |
| `test_uses_legacy_max_tokens_gpt35` | `uses_legacy_max_tokens("gpt-3.5-turbo")` → `True` |
| `test_uses_legacy_max_tokens_other` | `uses_legacy_max_tokens("gpt-4o")` → `False` |
| `test_uses_legacy_max_tokens_unknown_returns_false` | `uses_legacy_max_tokens("unknown")` → `False` (ValueError branch) |

---

### 1.2 `tests/test_embedding_registry.py` (new file)

**Target:** `app/embedding_registry.py` — 1 missed line (100)

| Test | Purpose |
|------|---------|
| `test_collection_name_for_model_known` | `collection_name_for_model("lapack", "text-embedding-3-small")` → `"lapack-text-embedding-3-small"` |
| `test_collection_name_for_model_unknown` | `collection_name_for_model("lapack", "custom-model")` → `"lapack-custom-model"` (fallback branch) |
| `test_get_model_info_raises_keyerror` | `get_model_info("unknown")` → raises `KeyError` |

---

## Phase 2: Schemas & Eval Data (Quick Wins)

### 2.1 `tests/test_eval_data.py` — add 1 test

**Target:** `app/eval_data.py` line 439

| Test | Purpose |
|------|---------|
| `test_ndcg_zero_k_returns_zero` | `compute_ndcg_at_k(["a.f"], ["a.f"], k=0)` → `0.0` (hits `idcg == 0.0` branch) |

---

### 2.2 `tests/test_api.py` — schema validation

**Target:** `app/schemas.py` line 8 (ValueError in `_validate_embedding_model`)

| Test | Purpose |
|------|---------|
| `test_query_request_rejects_unknown_embedding_model` | POST `/api/query` with `embedding_model: "unknown"` → 422 with validation error |

*Note: `test_query_endpoint_rejects_unknown_embedding_model` may already cover this via a different path. Verify and add if the validator branch is not hit.*

---

## Phase 3: main.py (API Error Paths)

**Target:** `app/main.py` — 8 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 171–172 | `_build_response` retrieval exception handler | `test_query_endpoint_retrieval_failure_raises` — mock `retrieve` to raise, assert 500 and error propagation |
| 291–292, 306–308 | `_stream_generator` retrieval exception | `test_query_stream_retrieval_failure_emits_error` — mock `retrieve` to raise, assert SSE `error` event |
| 472 | `list_trials_endpoint` with `eval_type` filter | `test_list_trials_endpoint_with_eval_type_filter` — `GET /api/trials?eval_type=retrieval` with mixed trial types, assert filtered result |

---

## Phase 4: Services — Chunker, Parser, Retrieval, Trial Store

### 4.1 `tests/test_chunker.py` — 2 lines (85, 104)

**Lines 85, 104:** `called_by` branch in `_build_chunk_header` and `_build_metadata`

| Test | Purpose |
|------|---------|
| `test_chunk_header_includes_called_by` | Chunk with `called_by` in metadata → header contains `CALLED_BY:` |
| `test_chunk_metadata_includes_called_by` | Unit with `called_by` → metadata dict has `called_by` key |

*Inspect chunker to ensure `called_by` is passed through; add tests that trigger these branches.*

---

### 4.2 `tests/test_parser.py` — 5 lines (109–110, 140, 149–150)

**Lines 109–110:** `AttributeError`/`IndexError`/`TypeError` in fixed-form span extraction  
**Line 140:** `_get_fparser2_span` returns `None` when content is empty  
**Lines 149–150:** `_get_fparser2_span` exception handler

| Test | Purpose |
|------|---------|
| `test_parse_fixed_form_span_extraction_fallback` | Fortran with malformed span (e.g. `item.content[-1].item.span` missing) → falls back to `item.item.span[1]` |
| `test_get_fparser2_span_empty_content` | Node with empty content → returns `None` |
| `test_get_fparser2_span_exception_fallback` | Node that raises on span access → returns `None` |

*May require exporting `_get_fparser2_span` for direct testing or using fixtures that produce the edge cases.*

---

### 4.3 `tests/test_retrieval.py` — 4 lines (70–72, 112)

**Lines 70–72:** Expansion cache hit + `logger.info`  
**Line 112:** `_EXPANSION_CACHE.popitem` when cache exceeds max size

| Test | Purpose |
|------|---------|
| `test_expand_query_cache_hit_returns_cached` | Call `expand_query` twice with same query → second call returns cached result, no API call |
| `test_expand_query_cache_eviction` | Call `expand_query` with many unique queries to exceed `_EXPANSION_CACHE_MAX` → cache evicts oldest |

---

### 4.4 `tests/test_trial_store.py` — 2 lines (63–64)

**Lines 63–64:** Migration `OperationalError` (column already exists race)

| Test | Purpose |
|------|---------|
| `test_migration_operational_error_ignored` | Patch `conn.execute(ddl)` to raise `OperationalError` for one migration → connection still works, no failure |

---

## Phase 5: Embeddings (API Key / Client Error Paths)

**Target:** `app/services/embeddings.py` — 22 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 73–77 | `_get_voyage_client` missing `VOYAGE_API_KEY` | `test_get_voyage_client_missing_key_raises` |
| 82–86 | `_get_async_voyage_client` missing key | `test_get_async_voyage_client_missing_key_raises` |
| 90–91 | `_get_cohere_client` missing key | `test_get_cohere_client_missing_key_raises` |
| 96–100 | `_get_async_cohere_client` missing key | `test_get_async_cohere_client_missing_key_raises` |
| 105–109 | `_get_gemini_client` via `_get_gemini_client` (gemini_helpers) | Covered by `test_gemini_client_missing_key` — verify it calls through embeddings |

*Strategy:* Patch `get_settings` to return settings with empty API keys, call the client getters, assert `RuntimeError`. Clear `lru_cache` before/after to avoid cross-test pollution.

---

## Phase 6: Eval Runner

**Target:** `app/services/eval_runner.py` — 11 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 50–51 | `run_eval` with `embedding_model` → `collection_name_for_model` | Already in `test_eval_stream_with_model_param` — verify coverage |
| 158–159 | Same for e2e eval | Verify e2e tests pass `embedding_model` |
| 193–208 | E2E eval generation exception handler | `test_e2e_eval_stream_generation_failure_emits_progress` — mock `generate_answer` to raise, assert SSE progress event with `passed: False`, `error` in checks |
| 225–226 | E2E embedding similarity exception | `test_e2e_eval_stream_embedding_similarity_failure` — mock `embed_query` to raise for golden/answer embedding, assert eval continues (warning path) |

---

## Phase 7: Generation (Exception Handlers)

**Target:** `app/services/generation.py` — 15 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 183–184 | `_gemini_generate` `ValueError` on `response.text` | Mock Gemini response with `text` that raises `ValueError` when accessed |
| 268–270 | `_generate_with_ttft` OpenAI exception | Mock `client.chat.completions.create` to raise |
| 306–308 | `_gemini_generate_with_ttft` exception | Mock `_gemini_generate_stream` to raise |
| 344–346 | `generate_answer_stream` Gemini exception | Mock Gemini stream to raise, assert `error` event |
| 404–407 | `generate_answer_stream` OpenAI exception | Mock OpenAI stream create to raise, assert `error` event |

---

## Phase 8: Gemini Helpers

**Target:** `app/services/gemini_helpers.py` — 11 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 21 | `get_gemini_client` missing `GEMINI_API_KEY` | `test_get_gemini_client_missing_key_raises` |
| 47 | `is_gemini_reasoning_model` for non-reasoning model | `test_is_gemini_reasoning_model_false` — `is_gemini_reasoning_model("gemini-2.5-flash")` → `False` |
| 77–80 | `retry_on_rate_limit` retry path | Mock async func to raise 429 twice then succeed |
| 94–101 | `retry_on_rate_limit_sync` retry path | Mock sync func to raise 429 twice then succeed |

---

## Phase 9: Ingest Runner

**Target:** `app/services/ingest_runner.py` — 16 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 31–40 | `ingest_stream_generator` unknown model, lock held, missing data dir, no files | Already have `test_unknown_model_emits_error`, `test_missing_data_dir_emits_error`, `test_no_files_found_emits_error` — add `test_ingest_lock_held_emits_error` (mock `_ingest_lock.locked()` → True) |
| 50–51 | No files found after `_find_fortran_files` | Covered by `test_no_files_found_emits_error` |
| 130–139 | Rate limit retry + progress event, or final error | `test_ingest_rate_limit_retry_emits_progress` — mock `embed_texts` to raise 429 on first call, succeed on second; assert progress event. `test_ingest_embed_final_failure_after_retries` — 429 three times → error event |

---

## Phase 10: Vector Store

**Target:** `app/services/vector_store.py` — 12 missed lines

| Lines | Code Path | Test |
|-------|-----------|------|
| 43 | `_resolve_collection(None)` → `get_settings().QDRANT_COLLECTION_NAME` | Call `search`/`ensure_collection` with no `collection_name` (default) — may already be covered; verify |
| 126–132 | `delete_collection` when collection does not exist | `test_delete_collection_not_found_returns_false` — mock `get_collections` so target collection not in list, assert `delete_collection("missing")` → `False` |
| 208–220 | `async_search_by_caller` | `test_async_search_by_caller` — mock `query_points` with `called_routines` filter, assert formatted hits |

*Add `async_search_by_caller` to test imports and write async test.*

---

## Implementation Order

1. **Phase 1** — New modules (`test_models_data`, `test_embedding_registry`) — ~30 min  
2. **Phase 2** — Eval data + schemas — ~15 min  
3. **Phase 3** — main.py API tests — ~30 min  
4. **Phase 4** — Chunker, parser, retrieval, trial_store — ~45 min  
5. **Phase 5** — Embeddings client errors — ~20 min  
6. **Phase 6** — Eval runner — ~30 min  
7. **Phase 7** — Generation exceptions — ~45 min  
8. **Phase 8** — Gemini helpers — ~30 min  
9. **Phase 9** — Ingest runner — ~30 min  
10. **Phase 10** — Vector store — ~20 min  

**Total estimate:** ~4–5 hours.

---

## Verification

After each phase:

```bash
python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=100
```

Add to CI (e.g. `pyproject.toml` or `Makefile`):

```toml
[tool.pytest.ini_options]
addopts = "--cov=app --cov-fail-under=100
```

---

## Notes

- **LRU cache:** Clear `lru_cache` before/after tests that change `get_settings` or client behavior to avoid leakage.
- **Async tests:** Use `@pytest.mark.asyncio` and ensure event loop is configured.
- **Private functions:** Prefer integration tests that trigger branches via public API. If needed, test private helpers via `__import__` or by temporarily exposing them.
