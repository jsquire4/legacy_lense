# Refactoring Plan: God Module Decomposition + Test Fixture Debt + Frontend Dedup

## Tactical Approach: Hybrid Bottom-Up

**Rationale**: Refactor test fixtures for files with no source changes first (zero risk), then extract source modules validated by existing verbose tests, then simplify those tests, then frontend last.

---

## Phase 0: Baseline

### Task 0.1: Run existing test suite
- `python -m pytest tests/ -v` — record pass count and timing.
- All tests must pass before any changes.

---

## Phase 1: Test Fixture Debt — Zero-Risk Files

These files have NO corresponding source changes, so fixture refactoring is purely cosmetic.

### Task 1.1: `tests/test_parser.py` — Extract `tmp_fortran_file` fixture

**Current state**: 15+ tests repeat `tempfile.NamedTemporaryFile` + `path.unlink()` in try/finally.

**Action**: Add to `tests/conftest.py`:
```python
@pytest.fixture
def tmp_fortran_file():
    paths = []
    def _make(content: str, suffix: str = ".f") -> Path:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content.encode() if isinstance(content, str) else content)
            p = Path(f.name)
            paths.append(p)
            return p
    yield _make
    for p in paths:
        p.unlink(missing_ok=True)
```

Replace all 15+ occurrences in `test_parser.py`.

**Verify**: `pytest tests/test_parser.py -v`

### Task 1.2: `tests/test_embeddings.py` — Extract `clear_embed_cache` fixture

**Current state**: 3 tests use `emb_mod._embed_cache.clear()` in try/finally.

**Action**: Add to `tests/conftest.py`:
```python
@pytest.fixture
def clear_embed_cache():
    import app.services.embeddings as emb_mod
    emb_mod._embed_cache.clear()
    yield
    emb_mod._embed_cache.clear()
```

Replace the 3 try/finally blocks.

**Verify**: `pytest tests/test_embeddings.py -v`

### Task 1.3: Full suite check
`pytest tests/ -v` — must match baseline.

---

## Phase 2: `generation.py` — Extract `_build_llm_kwargs()`

### Task 2.1: Extract shared kwargs builder

**Current state**: The kwargs-building block is duplicated across 3 functions:
- `generate_answer()` (lines 167-175)
- `_generate_with_ttft()` (lines 221-231)
- `generate_answer_stream()` (lines 288-298)

This duplication already caused 3 bug-fix commits.

**Action**: Add helper after `_build_messages()` (~line 131):
```python
def _build_llm_kwargs(
    model: str, messages: list[dict],
    max_completion_tokens: int, stream: bool = False,
) -> dict:
    kwargs = dict(model=model, messages=messages)
    kwargs[_token_limit_key(model)] = max_completion_tokens
    if is_reasoning_model(model):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = 0.1
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs
```

Replace all 3 call sites with one-liners.

**Verify**: `pytest tests/test_generation.py -v` — all 24 tests, especially reasoning model tests.

---

## Phase 3: Test Fixture Debt — `test_generation.py`

### Task 3.1: Extract shared settings mock and async_iter helper

**Current state**:
- 4-line settings mock block repeated 12+ times
- `async def async_iter(): yield chunk` closure in 9+ tests

**Action**: Add to `tests/conftest.py`:
```python
@pytest.fixture
def mock_gen_settings():
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    return settings

def make_async_iter(*chunks):
    async def _iter():
        for chunk in chunks:
            yield chunk
    return _iter()
```

Replace all occurrences in `test_generation.py`. Tests with non-default models keep inline settings but use the fixture pattern.

**Verify**: `pytest tests/test_generation.py -v`

---

## Phase 4: `retrieval.py` — Extract Strategy Helpers

### Task 4.1: Extract 4 retrieval strategies to named helpers

**Current state**: `retrieve()` is 96 lines (lines 106-201) with 4 inline strategies.

**Action**: Extract above `retrieve()`:
1. `_name_match_search()` — direct name lookup (lines 125-133)
2. `_expansion_search()` — LLM expansion + fan-out (lines 134-147)
3. `_call_graph_search()` — one-hop call graph follow (lines 150-167)
4. `_caller_search()` — impact analysis caller lookup (lines 170-179)

Each helper receives and mutates `seen_ids` set for deduplication (preserving existing behavior). `retrieve()` becomes a ~40-line orchestrator.

**Verify**: `pytest tests/test_retrieval.py -v` — all 13 tests.

---

## Phase 5: Test Fixture Debt — `test_retrieval.py`

### Task 5.1: Extract shared `@patch` stack as fixture

**Current state**: 10/13 tests use the same 3-decorator `@patch` stack.

**Action**: Add to `tests/conftest.py`:
```python
@pytest.fixture
def retrieval_mocks():
    with patch("app.services.retrieval.embed_query", new_callable=AsyncMock) as mock_embed, \
         patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock) as mock_name, \
         patch("app.services.retrieval.async_search", new_callable=AsyncMock) as mock_search:
        yield {"embed": mock_embed, "name_search": mock_name, "vector_search": mock_search}
```

Replace decorator stacks in 10 tests.

**Verify**: `pytest tests/test_retrieval.py -v`

---

## Phase 6: `main.py` — God Module Decomposition

### Task 6.1: Extract Pydantic models to `app/schemas.py`

**Current state**: 8 Pydantic models inline in main.py (lines 101-147 + line 484).

**Action**: Create `app/schemas.py` with all 8 models. Update `main.py` imports.

**Verify**: `pytest tests/test_api.py -v`

### Task 6.2: Deduplicate chunk-building in `_stream_generator`

**Current state**: `_build_chunk_details()` (lines 156-170) builds ChunkDetail objects, but `_stream_generator` (lines 266-278) re-implements the same logic inline.

**Action**: Modify `_stream_generator` to call `_build_chunk_details()` and add the `"text"` field.

**Verify**: `pytest tests/test_api.py -v` — specifically streaming tests.

### Task 6.3: Extract eval generators to `app/services/eval_runner.py`

**Current state**: Two 50+ line async generators in main.py:
- `_eval_stream_generator()` (lines 339-395)
- `_e2e_eval_stream_generator()` (lines 402-464)

**Action**: Create `app/services/eval_runner.py` with both generators plus their constants (`RETRIEVAL_BATCH_SIZE`, `E2E_MAX_TOKENS`, `E2E_CONTEXT_BUDGET`) and a local `_sse_event` helper. Update main.py endpoints to import from the new module.

**Critical**: Update `@patch` paths in `test_api.py` eval tests from `app.main.retrieve` → `app.services.eval_runner.retrieve` (and same for `generate_answer`).

**Verify**: `pytest tests/test_api.py -v` — all eval-related tests.

### Task 6.4: Full suite check
`pytest tests/ -v`

---

## Phase 7: Test Fixture Debt — `test_api.py`

### Task 7.1: Extract mock return value factories

**Current state**: retrieve/generate mock-return dicts copy-pasted in 15+ tests.

**Action**: Add to `tests/conftest.py`:
```python
def make_retrieve_result(chunks=None, strategy="vector", expanded_names=None):
    if chunks is None:
        chunks = [{"id": "abc123", "text": "test", "score": 0.9,
                    "metadata": {"file_path": "test.f"}, "_match_type": "vector"}]
    return {"chunks": chunks, "expanded_names": expanded_names or [], "retrieval_strategy": strategy}

def make_generate_result(answer="Test answer", citations=None, model="gpt-4o-mini", token_usage=None):
    return {"answer": answer, "citations": citations or [], "model": model, "token_usage": token_usage or {}}
```

Replace 15+ inline dicts. Tests with custom values pass overrides.

**Verify**: `pytest tests/test_api.py -v`

---

## Phase 8: Frontend Dedup

### Task 8.1: Extract shared eval runner helper in `index.html`

**Current state**: `runEvals()` and `runE2EEvals()` share identical patterns (disable button, clear UI, accumulators, readSSE, progress/summary handling, re-enable button).

**Action**: Extract `runEvalStream({btnId, statusId, url, onProgress, onSummary, onComplete, buildRow, ...})` helper. Both functions become thin wrappers passing their specific config.

**Verify**: Manual testing — run both eval types, verify progress and summary display.

---

## Phase 9: Final Validation

### Task 9.1: Full test suite — `pytest tests/ -v`
### Task 9.2: Import graph check — no circular imports
```bash
python -c "from app.main import app; print('OK')"
python -c "from app.services.eval_runner import eval_stream_generator; print('OK')"
python -c "from app.schemas import QueryResponse; print('OK')"
```
### Task 9.3: Line count audit — verify expected reductions

---

## New Files

| File | Purpose |
|------|---------|
| `app/schemas.py` | Pydantic request/response models from main.py |
| `app/services/eval_runner.py` | Eval stream generators from main.py |

## Modified Files

| File | Changes |
|------|---------|
| `app/main.py` | Remove schemas, eval generators, dedup chunk building |
| `app/services/generation.py` | Extract `_build_llm_kwargs()`, replace 3 call sites |
| `app/services/retrieval.py` | Extract 4 strategy helpers, simplify `retrieve()` |
| `app/static/index.html` | Extract `runEvalStream()` shared helper |
| `tests/conftest.py` | Add 7 fixtures/helpers |
| `tests/test_generation.py` | Use shared settings mock + `make_async_iter` |
| `tests/test_api.py` | Use factories, update eval test patch paths |
| `tests/test_parser.py` | Use `tmp_fortran_file` fixture |
| `tests/test_retrieval.py` | Use `retrieval_mocks` fixture |
| `tests/test_embeddings.py` | Use `clear_embed_cache` fixture |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Circular import from eval_runner | It imports leaf services only, not main.py. Verified by Task 9.2. |
| Test patch paths break after extraction | Task 6.3 explicitly updates paths. Grep for old paths before committing. |
| `_build_llm_kwargs` omits a param | Existing 24 generation tests validate all branches. |
| `seen_ids` mutation in retrieval helpers | Passed by reference, matching existing behavior. 13 tests validate. |
| Frontend `runEvalStream` breaks events | Manual testing required (no automated frontend tests). |

## Execution Order

**Sequential**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

Each phase ends with a test suite run. No phase starts until the prior phase's tests pass.
