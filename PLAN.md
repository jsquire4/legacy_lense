# Implementation Plan: LegacyLens

## Chosen Approach

Custom RAG pipeline (no LangChain/LlamaIndex) using Python/FastAPI, OpenAI SDK for embeddings + generation, Qdrant Cloud for vector storage, and fparser for Fortran parsing. LAPACK repository as the target codebase. Single-service deployment on Railway with a minimal web UI.

---

## 1. Project Structure

```
leglen/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app, routes, static mount, health check
│   ├── config.py              # pydantic-settings, centralized env config
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parser.py          # Fortran parsing (fparser1 + fparser2)
│   │   ├── chunker.py         # Custom chunking logic + token enforcement
│   │   ├── embeddings.py      # OpenAI embedding calls (centralized client)
│   │   ├── vector_store.py    # Qdrant operations (query_points, upsert)
│   │   ├── retrieval.py       # Query → embed → search pipeline
│   │   ├── generation.py      # LLM answer generation with citations
│   │   └── capabilities.py    # 4 code-understanding capabilities
│   └── static/
│       └── index.html         # Minimal web UI
├── scripts/
│   ├── ingest.py              # CLI ingestion script
│   └── evaluate.py            # Evaluation harness (Precision@5)
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_api.py
├── docs/
│   ├── architecture.md
│   └── cost_analysis.md
├── logs/                      # JSON logs (gitignored)
├── .env
├── .gitignore
├── Dockerfile
├── requirements.txt
└── Gauntlet_Docs/
    ├── ll_reqs.md
    ├── presearch.md
    └── concept_brief.md
```

---

## 2. Implementation Tasks (Build Order)

### Phase 1: Foundation (Steps 1-5) — Target: ~4 hours

#### Step 1: Config + Project Scaffold
- **Files:** `app/__init__.py`, `app/config.py`, `app/services/__init__.py`, `app/main.py` (skeleton), `requirements.txt`
- **Details:**
  - `config.py`: pydantic-settings `Settings` class with `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `EMBEDDING_DIM`, `MAX_CHUNK_TOKENS`
  - `@lru_cache` on `get_settings()` — read `.env` once
  - `main.py` skeleton: `FastAPI()` instance + `GET /health` returning `{"status": "ok"}`
  - `requirements.txt`: fastapi, uvicorn[standard], pydantic-settings, python-dotenv, openai, qdrant-client, fparser, tiktoken, aiofiles
- **Depends on:** Nothing
- **Complexity:** Small

#### Step 2: Fortran Parser Service
- **Files:** `app/services/parser.py`
- **Details:**
  - `ParsedUnit` dataclass: `name`, `kind` (SUBROUTINE/FUNCTION/MODULE/PROGRAM/RAW), `source_text`, `doc_comments`, `file_path`, `start_line`, `end_line`, `called_routines` (list[str])
  - Route by extension: `.f` → fparser1 (`isfree=False, isstrict=False`), `.f90` → fparser2 (`std="f2003"`)
  - fparser1: use `parse()` with `analyze=True`, fallback to `analyze=False`. Extract from `tree.a.external_subprogram`. Line numbers from `.item.span`
  - fparser2: use `ParserFactory().create(std="f2003")`, walk AST for `Subroutine_Subprogram`, `Function_Subprogram`, `Module`
  - **Graceful fallback**: if parsing fails completely, return a single `RAW` unit containing the entire file text. Never crash the ingest pipeline
  - Extract `*>` Doxygen-style doc comments separately from code body
- **Depends on:** Step 1
- **Complexity:** Large — **highest risk step**, budget extra time
- **Risk mitigation:** RAW fallback ensures ingest always completes even if parser chokes

#### Step 3: Chunker Service
- **Files:** `app/services/chunker.py`
- **Details:**
  - `Chunk` dataclass: `text`, `metadata` dict (file_path, unit_name, kind, start_line, end_line, chunk_index, doc_comments)
  - `chunk_units(units: list[ParsedUnit]) -> list[Chunk]`
  - Token counting via `tiktoken.get_encoding("cl100k_base")` — cache the encoder
  - **Hard cap: 8191 tokens** (text-embedding-3-small limit). Enforce in the chunker, not downstream
  - Units under cap → single chunk. Units over cap → sliding window split (500 token windows, 50 token overlap)
  - LAPACK reality: most `.f` files are one-proc-per-file and under the cap. Windowed fallback is a safety net
  - Embedding text format per chunk includes structured metadata header:
    ```
    FILE: src/dgesv.f
    UNIT_TYPE: SUBROUTINE
    UNIT_NAME: DGESV
    LINES: 42-210
    IDENTIFIERS: A, B, IPIV, INFO

    <code body>
    ```
  - No empty chunks — validate before emitting
- **Depends on:** Step 2 (ParsedUnit dataclass)
- **Complexity:** Small-Medium
- **Test-first candidate:** write chunk contract tests before implementation

#### Step 4: Embedding Service
- **Files:** `app/services/embeddings.py`
- **Details:**
  - Centralized OpenAI client: module-level singleton via `get_openai_client()` reading from settings
  - `embed_texts(texts: list[str]) -> list[list[float]]`
  - Batch up to 512 texts per API call (conservative; API limit is 2048)
  - Validate: no empty strings, assert `len(result) == len(texts)`
  - Use `is not None` for all response field checks (lesson from past projects)
  - Token truncation before embedding: use tiktoken to pre-truncate to 8191 tokens
- **Depends on:** Step 1
- **Complexity:** Small
- **Test-first candidate:** model name and vector dimension are load-bearing constants

#### Step 5: Vector Store Service
- **Files:** `app/services/vector_store.py`
- **Details:**
  - Module-level Qdrant client singleton
  - `ensure_collection()`: check `collection_exists()`, create with `VectorParams(size=1536, distance=Distance.COSINE)` if not. Create payload indexes on `unit_type` and `file_path` for filtered search
  - `upsert_chunks(chunks, embeddings)`: batch upsert via `upload_points()` with `batch_size=100, max_retries=3`
  - `search(query_vector, top_k=8, filters=None)`: use `client.query_points()` (**not** deprecated `client.search()`). Return list of dicts with `score`, `text`, `metadata`
  - Point IDs: UUID4 (simple, collision-free)
  - Store raw chunk text in payload as `content` field for retrieval display
- **Depends on:** Step 1
- **Complexity:** Small-Medium

### Phase 2: Pipeline Integration (Steps 6-8) — Target: ~4 hours

#### Step 6: Ingest Script
- **Files:** `scripts/ingest.py`
- **Details:**
  - CLI via `argparse`: `--data-dir`, `--extensions` (default `.f .f90`), `--batch-size`, `--dry-run`
  - Walk data dir recursively, filter by extension
  - Pipeline: `parse_file()` → `chunk_units()` → collect all chunks → `embed_texts()` → `upsert_chunks()`
  - Progress reporting to stdout (file count, chunk count)
  - `--dry-run`: parse + chunk only, print stats, skip API calls
  - Call `ensure_collection()` at startup
  - Log completion stats as structured JSON
  - **Must complete 10k+ LOC ingestion in under 5 minutes** (requirement)
- **Depends on:** Steps 2, 3, 4, 5
- **Complexity:** Small (wiring) but **medium-high debugging risk** — first integration point

#### Step 7: Retrieval Service
- **Files:** `app/services/retrieval.py`
- **Details:**
  - `retrieve(query: str, top_k: int = 8) -> list[dict]`
  - Embed query → vector search → return ranked results
  - Results include score, text, and all metadata
  - Simple and thin — all complexity is in the services it calls
- **Depends on:** Steps 4, 5
- **Complexity:** Small

#### Step 8: Generation Service
- **Files:** `app/services/generation.py`
- **Details:**
  - `generate_answer(query: str, context_chunks: list[dict], capability: str = "explain") -> dict`
  - **Binary-search prompt truncation**: fit the largest number of complete chunks within a `MAX_CONTEXT_TOKENS` budget (~6000 tokens). Never truncate mid-chunk — drop trailing chunks
  - System prompt enforces: technical tone, no speculation, mandatory citations as `file.f:START-END`, explicit "insufficient context" when applicable
  - Context assembly: chunks labeled as `[SOURCE 1] file.f lines 10-80:\n<code>`
  - Response structure: `{"answer": str, "citations": list[dict], "model": str, "token_usage": dict}`
  - Citation enforcement: if model response lacks file references, append sources from context metadata
  - Check `response.choices is not None` and `len(response.choices) > 0` (lesson learned)
- **Depends on:** Step 7
- **Complexity:** Medium — prompt engineering will iterate

### Phase 3: Interface + Deploy (Steps 9-10) — Target: ~3 hours

#### Step 9: FastAPI Endpoints + Web UI
- **Files:** `app/main.py` (full), `app/static/index.html`
- **Details:**
  - Routes:
    - `GET /health` → `{"status": "ok"}`
    - `POST /api/query` → body `{"query": str, "top_k": int}`, returns `{"answer", "citations", "sources", "latency_ms"}`
    - `POST /api/capabilities/{capability}` → same body, routes to specific capability
    - `GET /` → serves index.html
  - Pydantic request/response models
  - Error handling: wrap service calls in try/except, return clean error responses
  - CORSMiddleware enabled
  - `index.html`: single-file vanilla HTML/CSS/JS (<150 lines). Text input, submit button, results panel showing answer + cited code snippets. `fetch("/api/query")` on submit. No framework
  - Mount static files with `html=True` for index fallback
- **Depends on:** Steps 7, 8
- **Complexity:** Small-Medium

#### Step 10: Railway Deployment
- **Files:** `Dockerfile`
- **Details:**
  - `FROM python:3.12-slim`
  - `COPY requirements.txt . && RUN pip install --no-cache-dir -r requirements.txt`
  - `COPY . .`
  - `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}` (shell form for $PORT expansion)
  - Set Railway env vars: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
  - Generate public domain in Railway dashboard
  - **Ingest is run locally before deploy** — Railway only serves the query API
  - Add `/health` to Railway healthcheck config
- **Depends on:** Step 9
- **Complexity:** Small

---

**>>> MVP GATE — Steps 1-10 must be complete <<<**

---

### Phase 4: Capabilities + Polish (Steps 11-15) — Target: ~6 hours

#### Step 11: Code Understanding Capabilities
- **Files:** `app/services/capabilities.py`, update routes in `app/main.py`
- **Details:**
  - All 4 capabilities share retrieval + generation core, differing only in system prompt
  - `explain_code(query)` — plain-English explanation + pseudocode of retrieved procedures
  - `generate_docs(query)` — produce structured documentation (parameters, purpose, algorithm description)
  - `detect_patterns(query)` — identify algorithmic patterns across retrieved chunks (BLAS calls, loop structures, error handling)
  - `map_dependencies(procedure_name)` — retrieve chunks mentioning the procedure, extract calls/called-by relationships
  - Each wraps `generation.generate_answer()` with a specialized system prompt — minimal new code, maximum reuse
  - Add capability selector to web UI (dropdown or tabs)
- **Depends on:** Steps 7, 8, 9
- **Complexity:** Medium

#### Step 12: Evaluation Harness
- **Files:** `scripts/evaluate.py`
- **Details:**
  - 15+ hardcoded queries covering: known routines (DGEMM, DGETRF, DGESV, DPOTRF), algorithmic questions, dependency questions, edge cases
  - For each: call `retrieve(query, top_k=5)`, record retrieved files, measure latency
  - Precision@5: compare retrieved files against manually-defined relevant files
  - Output: table (query, P@5, latency_ms) + summary (mean P@5, mean latency)
  - Write results as JSON to `logs/eval_results.json`
  - Target: P@5 > 0.7, latency < 3s
- **Depends on:** Steps 6, 7 (needs ingested data)
- **Complexity:** Medium (requires knowing LAPACK to define ground truth)

#### Step 13: Observability Logging
- **Files:** add logging to all services, optional `app/logging_config.py`
- **Details:**
  - Custom JSON formatter for Python's `logging` module
  - Log events: `query_received`, `retrieval_complete` (with chunk IDs + scores), `generation_complete` (with token usage), `error`
  - Fields per query log: timestamp, query text, retrieved chunk IDs, similarity scores, latency_ms, token usage (input/output), error flags
  - `LOG_LEVEL` from settings
  - In production: logs to stdout (Railway captures)
- **Depends on:** All services exist
- **Complexity:** Small

#### Step 14: Cost Analysis Document
- **Files:** `docs/cost_analysis.md`
- **Details:**
  - Embedding cost: total ingested tokens × $0.02/1M (text-embedding-3-small)
  - Query cost: avg tokens per query × $0.15/1M input + $0.60/1M output (gpt-4o-mini)
  - Qdrant: free tier for <= 1GB; note paid tier pricing
  - Projections for 100 / 1,000 / 10,000 / 100,000 users with stated assumptions (queries/user/day, avg tokens)
  - Table format per the assignment requirements
- **Depends on:** Steps 3, 8 (need real token counts)
- **Complexity:** Small

#### Step 15: Architecture Justification Document
- **Files:** `docs/architecture.md`
- **Details:**
  - Justify: vector DB (Qdrant), embedding model, chunking strategy, retrieval pipeline, generation approach, deployment architecture
  - For each: option chosen, alternatives considered, deciding factor
  - Include ASCII architecture diagram: LAPACK → parser → chunker → embedder → Qdrant ← retriever → generator → FastAPI → Web UI
  - Reference evaluation results as evidence
- **Depends on:** All steps
- **Complexity:** Small

---

## 3. Testing Strategy

### Test-First (write before implementation)
- **Chunker**: chunk boundaries, metadata shape, no empty chunks, token cap enforcement
- **Embeddings**: correct model name, vector dimension, batch splitting
- **Vector store**: collection dimensions, payload metadata, `query_points` usage

### Test-After
- **Generation**: prompt engineering iterates too fast for stable tests early on
- **FastAPI endpoints**: smoke tests after routes exist
- **Retrieval**: mostly glue code; once sub-services are tested, this is trivial

### Evaluation Harness as Integration Tests
- 15+ queries with known-good source files
- `pytest -m eval` runs against real APIs with ingested data
- P@5 drop from a code change = something broke → bisect with unit tests
- Standard tests: `pytest -m "not eval"` — fast, mocked, no API calls

### Key Test Infrastructure
- `pytest` + `pytest-mock` for unit tests
- `FastAPI.TestClient` for endpoint smoke tests
- `tmp_path` fixture for Fortran parsing tests (inline source snippets)
- `conftest.py` with small LAPACK code samples

---

## 4. Error Handling

### Failure Modes
| Failure | Where | Handling |
|---------|-------|----------|
| fparser crashes on a file | Parser service | Catch, log, return RAW fallback unit with entire file text |
| Empty/comment-only file | Parser service | Return empty list, skip in ingestion |
| Chunk exceeds token limit | Chunker | Hard cap enforced — windowed split, never exceed 8191 |
| OpenAI rate limit (429) | Embeddings/Generation | Built-in retry in openai SDK (`max_retries=3`) + exponential backoff |
| OpenAI timeout | Embeddings/Generation | SDK retry handles it; log the event |
| Qdrant connection failure | Vector store | Retry with backoff; surface error to user on query path |
| Qdrant upsert failure | Vector store | `upload_points(max_retries=3)`; log failed batch |
| Empty retrieval results | Retrieval | Return empty list; generation handles gracefully |
| LLM hallucinates / no citations | Generation | Citation enforcement: append sources from context metadata |
| Context window overflow | Generation | Binary-search truncation ensures fit |
| Railway PORT mismatch | Deployment | Shell-form CMD with `${PORT:-8000}` |

### Error Boundaries
- **Parser**: catches all exceptions per file, never crashes pipeline
- **Ingest script**: catches per-file errors, continues with remaining files, reports failures at end
- **API endpoints**: try/except around service calls, return structured error JSON with HTTP 500
- **No stack traces leaked** to end users

---

## 5. Execution Strategy

### Critical Path
```
Config → Parser → Chunker → Ingest → Retrieval → Generation → API/UI → Deploy
  1        2        3         6         7           8          9       10
```

### Parallelizable Steps
- Steps 2, 4, 5 can start simultaneously after Step 1
- Step 3 can start once Step 2's `ParsedUnit` dataclass is defined
- Steps 11, 12, 13 are fully independent post-MVP
- Steps 14, 15 can be written any time after Step 8

### Time Budget (34 hours)
| Phase | Steps | Target Hours | Cumulative |
|-------|-------|-------------|------------|
| Foundation | 1-5 | 4h | 4h |
| Pipeline Integration | 6-8 | 4h | 8h |
| Interface + Deploy | 9-10 | 3h | 11h |
| **MVP GATE** | | | **~11h** |
| Capabilities | 11 | 3h | 14h |
| Eval + Observability | 12-13 | 3h | 17h |
| Documentation | 14-15 | 2h | 19h |
| Buffer / Debugging | | 5h | 24h |
| Sleep | | ~8h | 32h |
| Final polish / fixes | | 2h | 34h |

### LAPACK Clone
- Clone `Reference-LAPACK/lapack` into `data/lapack/` (gitignored)
- Add `data/` to `.gitignore`
- Ingest targets `data/lapack/SRC/` and `data/lapack/BLAS/SRC/`

### Incremental Delivery
1. After Step 6: verify ingest works locally with `--dry-run`, then real ingest
2. After Step 9: test full query flow locally before deploying
3. After Step 10: MVP deployed — everything after is additive
4. After Step 12: run eval, iterate on prompts/chunking if P@5 is low

---

## 6. Definition of Done

- [ ] Full LAPACK repository ingested (10k+ LOC, 50+ files)
- [ ] Semantic search returns relevant chunks with file/line references
- [ ] Answer generation produces citation-backed explanations
- [ ] 4 code-understanding capabilities working (explain, docs, patterns, dependencies)
- [ ] Web UI functional (query input → answer display with citations)
- [ ] Deployed and publicly accessible on Railway
- [ ] Evaluation harness: 15+ queries, P@5 > 0.7, latency < 3s
- [ ] Structured JSON observability logging
- [ ] Cost analysis document (100 / 1k / 10k / 100k users)
- [ ] Architecture justification document
- [ ] All unit tests passing
- [ ] Clean repo with no committed secrets
