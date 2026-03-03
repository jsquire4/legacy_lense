# Test Coverage Remediation Plan — 100% Target

**Status:** ✅ **Complete** — 100% coverage achieved (136 tests)

**Baseline (after fixing 11 failing tests):** 92% total coverage (71 lines missing)  
**Target:** 100% coverage across `app/`

---

## Summary by Module

| Module | Current | Missing Lines | Effort |
|--------|---------|---------------|--------|
| app/main.py | 96% | 40-44, 51, 148-149 | Low |
| app/services/embeddings.py | 72% | 26-27, 45-64 | Low |
| app/services/generation.py | 81% | 23-24, 59-66, 147, 188, 194-224 | Medium |
| app/services/retrieval.py | 84% | 39-59, 132 | Medium |
| app/services/vector_store.py | 85% | 35-37, 163-172, 177-189 | Low |

---

## Phase 1: app/main.py (96% → 100%)

**Missing:** `_cache_get` hit path, `_cache_put` eviction, cache hit in `_build_response`

### 1.1 Cache hit in `_build_response` (lines 148-149)

- **What:** When the same query/top_k/capability is requested twice, `_cache_get` returns a cached response.
- **How:** Add a test that calls the query endpoint twice with identical params and asserts the second response is from cache (e.g. via timing or a cache-hit indicator if added).
- **File:** `tests/test_api.py`
- **Approach:** Use `TestClient` to POST `/api/query` twice with same body. Optionally patch `_cache_get` to verify it was called and returned non-None on second call.

### 1.2 `_cache_get` TTL hit (lines 40-44)

- **What:** When key exists and `time.time() - ts < _CACHE_TTL`, return cached response and move key to end.
- **How:** Same as 1.1 — the cache hit path exercises this. Ensure the test doesn’t wait for TTL expiry.

### 1.3 `_cache_put` eviction (line 51)

- **What:** When `len(_RESPONSE_CACHE) > _CACHE_MAX` (64), `popitem(last=False)` evicts oldest.
- **How:** Add a test that makes 65+ distinct queries (different query strings) to fill the cache and trigger eviction. Mock `retrieve` and `generate_answer` to avoid real API calls.
- **File:** `tests/test_api.py`

---

## Phase 2: app/services/embeddings.py (72% → 100%)

**Missing:** `get_async_openai_client`, `embed_query` (cache hit, API path, cache eviction)

### 2.1 `get_async_openai_client` (lines 26-27)

- **What:** Returns cached `AsyncOpenAI` client.
- **How:** Add `test_get_async_openai_client_returns_client` in `tests/test_embeddings.py`, similar to `test_get_openai_client_returns_client`. Patch `get_settings` and assert return type.

### 2.2 `embed_query` (lines 45-64)

- **What:** Single-query embedding with in-memory cache; cache hit, API call, cache eviction when `len(_embed_cache) >= 128`.
- **How:**
  - **Cache hit:** Call `embed_query` twice with same text; second call should not hit API (mock `client.embeddings.create` and assert called once).
  - **Cache eviction:** Call `embed_query` 129 times with distinct texts (mock API). Verify 129th call triggers eviction (oldest removed). May need to patch `_EMBED_CACHE_MAX` to 2 for a simpler test.

---

## Phase 3: app/services/generation.py (81% → 100%)

**Missing:** `_get_generation_client`, `_strip_markdown`, `_build_messages` branches, `track_ttft`, `_generate_with_ttft`, edge cases in `generate_answer`/`generate_answer_stream`

### 3.1 `_get_generation_client` (lines 23-24)

- **What:** Returns cached `AsyncOpenAI` client for generation.
- **How:** Add test that calls `generate_answer` with chunks (already done) — client is used. For direct coverage, add a unit test that imports and calls `_get_generation_client()` with mocked `get_settings`.

### 3.2 `_strip_markdown` (lines 59-66)

- **What:** Strips markdown (headers, bold, bullets, numbered lists) from LLM output.
- **How:** Add `test_strip_markdown_*` in `tests/test_generation.py`. Test headers (`### Foo`), bold (`**foo**`), bullets (`- item`), numbered lists (`1. item`). This is currently unused in the main flow but exists for future use — ensure it’s covered.

### 3.3 `_build_messages` (lines 99-114)

- **What:** Builds system + user messages; uses `CAPABILITIES[capability]` or `DEFAULT_SYSTEM_PROMPT`.
- **How:** Already covered indirectly by `generate_answer`/`generate_answer_stream` tests. Verify coverage report; if still missing, add a direct unit test for `_build_messages` (may need to export it or test via generation).

### 3.4 `track_ttft=True` and `_generate_with_ttft` (lines 147, 194-224)

- **What:** When `track_ttft=True`, uses streaming and measures time-to-first-token; `ttft_ms` in result.
- **How:** Add `test_generate_answer_track_ttft` that calls `generate_answer(..., track_ttft=True)` with mocked streaming client. Assert `ttft_ms` in result and that streaming path was used.

### 3.5 `generate_answer` / `generate_answer_stream` edge cases (line 188)

- **What:** Line 188 is `if ttft_ms is not None: result["ttft_ms"] = ttft_ms`.
- **How:** Covered by 3.4.

---

## Phase 4: app/services/retrieval.py (84% → 100%)

**Missing:** `_expand_query` (success + exception), line 132 (dead branch)

### 4.1 `_expand_query` success (lines 39-58)

- **What:** Uses LLM to expand conceptual query into LAPACK routine names.
- **How:** `_expand_query` is never called in the current `retrieve` flow — `expanded_names` is always `[]`. The branch `if routine_name` does name-match; the `else` does nothing. The `early_vector_results` branch (line 132) is dead because `expanded_names` is never set.
- **Options:**
  - **A:** Add a code path that calls `_expand_query` for conceptual queries (e.g. when `routine_name` is None) and uses the result. Then add tests for that path.
  - **B:** Add a direct unit test for `_expand_query` with mocked `get_async_openai_client` and `get_settings`. Test success (returns names) and exception (returns `[]`).

### 4.2 `_expand_query` exception (lines 56-58)

- **How:** Patch `client.chat.completions.create` to raise. Assert `_expand_query` returns `[]`.

### 4.3 Line 132 (dead branch)

- **What:** `vector_results = early_vector_results` — unreachable because `expanded_names` is never populated.
- **Recommendation:** Remove dead code or implement the conceptual expansion path. For 100% coverage without refactor, this line is unreachable; consider `# pragma: no cover` or refactoring to make it reachable.

---

## Phase 5: app/services/vector_store.py (85% → 100%)

**Missing:** `get_async_qdrant_client`, `async_search`, `async_search_by_name`

### 5.1 `get_async_qdrant_client` (lines 35-37)

- **What:** Returns cached `AsyncQdrantClient`.
- **How:** Add `test_get_async_qdrant_client_returns_client` mirroring `test_get_qdrant_client_returns_client`. Patch `get_settings`, assert return type.

### 5.2 `async_search` (lines 163-172)

- **What:** Async vector search.
- **How:** `retrieve` calls `async_search` when `routine_name or not expanded_names` (always true). Integration-style tests may already hit this. Add a unit test that mocks `get_async_qdrant_client` and `client.query_points`, then calls `async_search` and asserts `_format_hits`-style output.

### 5.3 `async_search_by_name` (lines 177-189)

- **What:** Async search with `unit_name` filter.
- **How:** `test_retrieve_name_match` and similar tests likely hit this. Add a direct unit test in `tests/test_vector_store.py` that mocks the async client and calls `async_search_by_name`, asserting the filter is used.

---

## Implementation Order

1. **Phase 2** (embeddings) — small, isolated tests.
2. **Phase 5** (vector_store) — small, isolated tests.
3. **Phase 3** (generation) — `_strip_markdown`, `track_ttft` tests.
4. **Phase 1** (main) — cache tests.
5. **Phase 4** (retrieval) — `_expand_query` unit tests; decide on dead branch (remove or refactor).

---

## Out of Scope (for app/ coverage)

- `scripts/evaluate.py`, `scripts/ingest.py` — not part of `app/`; coverage is for `app/` only.
- Third-party and stdlib code.

---

## Verification

After each phase:

```bash
python -m pytest tests/ --cov=app --cov-report=term-missing -q
```

Target: `TOTAL` line showing `100%` for `app/`.

Note: Branch coverage is disabled in `.coveragerc`; use line coverage for the 100% target.
