# LegacyLens Architecture

## Overview

LegacyLens is a RAG (Retrieval-Augmented Generation) application that makes the LAPACK Fortran codebase queryable via natural language. Users ask questions and receive citation-backed answers grounded in actual source code, with support for multiple LLM and embedding providers and a rigorous built-in evaluation system.

## System Architecture

```
User Query → LLM Expansion → Embed (multi-provider) → Hybrid Search (5 strategies) → Context Assembly (3K tokens) → LLM Generation → Citation-enforced Response
```

### Components

1. **FastAPI Web Server** (`app/main.py`) — Serves the web UI, REST API, SSE streaming endpoints, eval harnesses, model registry, and trial CRUD
2. **Fortran Parser** (`app/services/parser.py`) — Parses .f and .f90 files using fparser1/fparser2 with RAW fallback
3. **Chunker** (`app/services/chunker.py`) — Splits parsed units into token-capped chunks (8191-token limit, 50-token sliding window overlap)
4. **Embedding Service** (`app/services/embeddings.py`) — Multi-provider embeddings: OpenAI, Voyage AI, Google Gemini, Cohere with per-provider batch sizes and token pre-truncation
5. **Embedding Registry** (`app/embedding_registry.py`) — 9 embedding models across 4 providers with dimensions, tokenizers, and collection name mapping
6. **Vector Store** (`app/services/vector_store.py`) — Manages Qdrant operations with payload indexes on unit_name, file_path, called_by for hybrid retrieval
7. **Retrieval Service** (`app/services/retrieval.py`) — Five-strategy hybrid retrieval: name match, LLM query expansion (cached), call-graph following, caller impact analysis, vector similarity
8. **Generation Service** (`app/services/generation.py`) — Multi-provider LLM generation (OpenAI + Gemini) with citation enforcement, streaming, TTFT tracking, and reasoning model support
9. **Capabilities** (`app/services/capabilities.py`) — 7 specialized code understanding prompts (explain_code, generate_docs, detect_patterns, map_dependencies, impact_analysis, extract_business_rules, plus general-purpose)
10. **Model Registry** (`app/models_data.py`) — 11 models across 2 providers (OpenAI + Gemini) with pricing data, reasoning model detection, and legacy API detection
11. **Eval Runner** (`app/services/eval_runner.py`) — SSE streaming eval harness for retrieval (77 queries) and E2E (27 queries) with difficulty tiers and hallucination probes
12. **Trial Store** (`app/services/trial_store.py`) — SQLite-backed storage for eval trial results with schema migrations
13. **Logging** (`app/logging_config.py`) — Structured JSON logging with rotating file handler

## Model Support

### LLM Models (11)

| Model | Provider | Type | Default |
|-------|----------|------|---------|
| GPT-3.5-turbo | OpenAI | Legacy (uses `max_tokens`) | |
| GPT-4o-mini | OpenAI | Standard | |
| GPT-4o | OpenAI | Standard | |
| GPT-4.1-nano | OpenAI | Standard | Yes |
| GPT-4.1-mini | OpenAI | Standard | |
| GPT-4.1 | OpenAI | Standard | |
| GPT-5-nano | OpenAI | Reasoning | |
| GPT-5-mini | OpenAI | Reasoning | |
| GPT-5.2 | OpenAI | Reasoning | |
| Gemini 2.5 Flash | Google | Standard | |
| Gemini 2.5 Pro | Google | Standard | |

### Embedding Models (9)

| Model | Provider | Dimensions | Max Tokens |
|-------|----------|------------|------------|
| text-embedding-3-small | OpenAI | 1536 | 8191 |
| text-embedding-3-large | OpenAI | 3072 | 8191 |
| text-embedding-ada-002 | OpenAI | 1536 | 8191 |
| voyage-code-3 | Voyage AI | 1024 | 32000 |
| voyage-4-large | Voyage AI | 1024 | 32000 |
| voyage-4 | Voyage AI | 1024 | 32000 |
| voyage-4-lite | Voyage AI | 1024 | 32000 |
| gemini-embedding-001 | Google | 3072 | 2048 |
| embed-v4.0 | Cohere | 1536 | 128000 |

Each embedding model gets its own Qdrant collection (e.g., `lapack_voyage-code-3`), allowing head-to-head retrieval quality comparison via the eval harness.

### Model-specific API Handling

- **Legacy models** (GPT-3.5-turbo): Use `max_tokens` parameter, standard `temperature`
- **Standard models** (GPT-4o/4.1 series, Gemini): Use `max_completion_tokens`, standard `temperature`
- **Reasoning models** (GPT-5 series): Use `max_completion_tokens`, `reasoning_effort="low"` instead of `temperature`

### Cache Warming

On startup, the app pre-caches responses for default queries across cheap models for instant first-query performance.

## Key Design Decisions

### Why self-hosted Qdrant (not Qdrant Cloud)?
- Co-located on Railway internal network — ~10-30ms per query vs ~100-200ms with external hosting
- Multiple collections (one per embedding model) for head-to-head comparison
- No free-tier storage limits when running multiple embedding model collections

### Why fparser (not regex)?
- Handles both fixed-form (.f) and free-form (.f90) Fortran
- Provides AST for reliable boundary detection
- Regex alone hits ~80-90% accuracy but breaks on continuation lines and bare END statements

### Why file-level chunking?
- LAPACK follows strict one-procedure-per-file convention for .f files
- Most files fit within the 8191-token embedding limit as a single chunk
- Only ~55 out of 2300+ files require splitting (the largest routines)
- Sliding window with overlap handles oversized files

### Why hybrid retrieval?
- Pure vector search misses exact routine matches for queries like "What does DGESV do?"
- Name matching via payload filter gives exact hits with perfect recall for known routines
- LLM query expansion bridges conceptual queries ("How does SVD work?") to routine names (DGESVD, DGESDD)
- Call-graph following retrieves implementation dependencies (one hop) for better context
- Vector similarity fills remaining slots for broad coverage

### Why multi-provider embeddings?
- Different embedding models excel at different retrieval tasks — code-specialized models (Voyage) vs general-purpose (OpenAI) vs high-dimensional (Gemini)
- Per-collection storage in Qdrant allows ingesting once per model and comparing retrieval quality via the eval harness
- Default is OpenAI text-embedding-3-small (1536-dim, $0.02/1M tokens) — best cost/quality tradeoff for the code-heavy corpus

### Why no framework (LangChain, LlamaIndex)?
- Custom pipeline maximizes understanding of RAG mechanics
- Direct use of provider SDKs + Qdrant client keeps dependencies minimal
- Full control over chunking strategy, context assembly, and citation enforcement

### Why 3000-token context budget?
- Fits maximum number of complete chunks within budget
- Avoids mid-chunk truncation that loses context
- Leaves room for system prompt and 2048-token response within model context windows
- Unified across all models for fair comparison

## Data Flow

### Ingestion Pipeline
```
Fortran Files → Parser (fparser1/fparser2) → ParsedUnits → Chunker → Chunks → Embeddings → Qdrant
```

- 2,272 files parsed into 2,304 units (some .f90 files yield multiple units)
- 2,334 chunks after splitting oversized units
- 0 parse errors (RAW fallback catches all failures)

### Query Pipeline
```
User Query
  ├─ Embed query (swappable: OpenAI / Voyage / Gemini / Cohere)
  ├─ Detect routine name? → Name-match search (Qdrant payload filter, top-3)
  │   └─ Follow call-graph one hop (called_routines metadata)
  ├─ Impact analysis? → Caller lookup (called_by metadata filter, top-5)
  ├─ Conceptual query? → LLM query expansion (5-8 names, 256-entry LRU cache) → Search each
  └─ Vector similarity search (cosine, top_k=5)
      ↓
  Merge & deduplicate → Context assembly (3K token budget) → LLM (2048 response tokens) → Citation enforcement → Response
```

## Evaluation

### Retrieval Evals (77 queries, 3 difficulty tiers)
- Tests that the right files are retrieved per capability domain
- **Easy** (~32): named routines, direct matches (e.g., "What does DGESV do?")
- **Medium** (~30): less famous routines, multi-file expected sets (e.g., "How does LAPACK compute eigenvalues?")
- **Hard** (~15): variant disambiguation, multi-file chains (e.g., "Compare DGESVD vs DGESDD vs DGESVDX")
- Metrics per query: Precision@K (1/3/5), Recall@K (1/3/5), MRR, NDCG@5, negative oracle penalty
- P@5 displayed as percentage of theoretical maximum (accounts for structural ceiling when |expected| < k)
- Summary includes per-difficulty breakdown (avg recall, avg MRR, count)
- Embedding-model selectable — tests any ingested collection

### E2E Generation Evals (27 queries)
- **21 normal queries** with source-verified golden answers across all 7 capabilities
- **6 hallucination probes** for nonexistent routines (DFAKE, DGESVM, QGESV, DLASOR, DGEMN, ZLATRS2) — must trigger refusal
- Quality gates (all must pass for overall pass):
  - Citation presence and relevance (cited files match expected files)
  - Citation fallback detection (flags auto-generated citations vs LLM-cited)
  - Minimum answer length (200 chars)
  - ALL-keyword matching (every keyword must appear, source-verified against LAPACK code)
  - Embedding similarity gate (cosine similarity between generated and golden answer ≥ 0.75)
  - Refusal detection (auto-fails normal answers that dodge; auto-passes probe refusals)
- Run against any LLM + embedding model combination
- Unified token budgets: 3000 context tokens, 2048 response tokens

### Trial Storage
- SQLite-backed trial results with schema migrations for new metric columns
- Auto-saves after each eval run
- Trial history tables with per-metric breakdown, best-value highlighting, and separate embedding/LLM columns

## Observability

Each query response includes:
- **Retrieval details**: per-chunk rank, file name, routine name, similarity score, match type (name/expansion/call_graph/vector)
- **Query expansion**: list of LLM-expanded routine names (when triggered)
- **Timing**: retrieval, generation, and total latency (displayed in seconds)
- **Token usage**: prompt, completion, and total tokens

Structured JSON logs are written to `logs/legacylens.jsonl` with rotating file handler (5 MB x 3 backups).

## Deployment

- **Application**: Railway single service (Python 3.12-slim Dockerfile, non-root user)
- **Vector DB**: Qdrant self-hosted as a separate Railway service (co-located on internal network)
- **Local dev**: `docker compose up --build` (app + Qdrant)
- **Ingestion**: Runs locally via `scripts/ingest.py` — parse, chunk, embed, upsert per collection
- **CI/CD**: Auto-deploys on push to GitHub

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Parser failure | RAW fallback — entire file becomes one unit |
| Oversized chunks | Sliding window split with 50-token overlap |
| OpenAI rate limits | SDK built-in retry (max_retries=1) |
| Qdrant failures | Batch upsert with retry |
| Empty retrieval | Generation responds "insufficient context" |
| Missing citations | Enforcement layer appends sources from chunk metadata |
| Query expansion failure | Graceful fallback to vector-only search |
| Legacy model API mismatch | Dynamic parameter selection (max_tokens vs max_completion_tokens) |
| Reasoning model constraints | Conditional reasoning_effort instead of temperature |
| Trial DB not writable | Falls back to /tmp/data/ on Railway |
