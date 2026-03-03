# LegacyLens Architecture

## Overview

LegacyLens is a RAG (Retrieval-Augmented Generation) application that makes the LAPACK Fortran codebase queryable via natural language. Users ask questions and receive citation-backed answers grounded in actual source code, with support for multiple OpenAI models and built-in evaluation tooling.

## System Architecture

```
User Query → FastAPI → Hybrid Retrieval → Context Assembly → LLM Generation → Citation-enforced Response
```

### Components

1. **FastAPI Web Server** (`app/main.py`) — Serves the web UI, REST API, streaming endpoints, eval harnesses, model registry, and trial CRUD
2. **Fortran Parser** (`app/services/parser.py`) — Parses .f and .f90 files using fparser1/fparser2 with RAW fallback
3. **Chunker** (`app/services/chunker.py`) — Splits parsed units into token-capped chunks (8191 token limit)
4. **Embedding Service** (`app/services/embeddings.py`) — Generates embeddings via OpenAI text-embedding-3-small
5. **Vector Store** (`app/services/vector_store.py`) — Manages Qdrant collection operations
6. **Retrieval Service** (`app/services/retrieval.py`) — Hybrid retrieval: name match + query expansion + call-graph + vector similarity
7. **Generation Service** (`app/services/generation.py`) — LLM answer generation with citation enforcement, reasoning model support, and legacy/modern API parameter handling
8. **Capabilities** (`app/services/capabilities.py`) — 6 specialized code understanding prompts
9. **Model Registry** (`app/models_data.py`) — 9 OpenAI models with pricing data, reasoning model detection, and legacy API detection
10. **Trial Store** (`app/services/trial_store.py`) — SQLite-backed storage for eval trial results
11. **Logging** (`app/logging_config.py`) — Structured JSON logging with rotating file handler

## Model Support

### Supported Models (9)

| Model | Type | Default |
|-------|------|---------|
| GPT-3.5-turbo | Legacy (uses `max_tokens`) | |
| GPT-4o-mini | Standard | |
| GPT-4o | Standard | |
| GPT-4.1-nano | Standard | Yes |
| GPT-4.1-mini | Standard | |
| GPT-4.1 | Standard | |
| GPT-5-nano | Reasoning | |
| GPT-5-mini | Reasoning | |
| GPT-5.2 | Reasoning | |

### Model-specific API Handling

- **Legacy models** (GPT-3.5-turbo): Use `max_tokens` parameter, standard `temperature`
- **Standard models** (GPT-4o/4.1 series): Use `max_completion_tokens`, standard `temperature`
- **Reasoning models** (GPT-5 series): Use `max_completion_tokens`, `reasoning_effort="low"` instead of `temperature`

### Cache Warming

On startup, the app pre-caches responses for 7 default queries across 3 cheap models (GPT-4.1-nano, GPT-4o-mini, GPT-5-nano) — 21 cached responses for instant first-query performance.

## Key Design Decisions

### Why Qdrant Cloud (not self-hosted)?
- Same client code, zero infrastructure management
- Free tier provides 1GB (sufficient for LAPACK embeddings)
- Avoids Railway volume permission issues with self-hosted Qdrant

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

### Why no framework (LangChain, LlamaIndex)?
- Custom pipeline maximizes understanding of RAG mechanics
- Direct use of OpenAI SDK + Qdrant client keeps dependencies minimal
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
  ├─ Embed query (text-embedding-3-small)
  ├─ Detect routine name? → Name-match search (Qdrant payload filter)
  │   └─ Follow call-graph one hop
  ├─ No name? → LLM query expansion → Search each expanded name
  └─ Vector similarity search (cosine, top_k=8)
      ↓
  Merge & deduplicate → Context assembly (3K token budget) → LLM (2048 response tokens) → Citation enforcement → Response
```

## Evaluation

### Retrieval Evals (37 queries)
- Tests that the right files are retrieved per capability domain
- Measures Recall@5 and retrieval latency
- Embedding-based — model-independent

### E2E Generation Evals (21 queries)
- Tests full retrieve + generate pipeline with quality checks
- Per-query checks: citations present, minimum answer length, expected keywords, refusal detection
- Refusal detection auto-fails answers containing "insufficient context", "I don't know", etc.
- Model-selectable — run against any of the 9 supported models
- Unified token budgets: 3000 context tokens, 2048 response tokens

### Trial Storage
- SQLite-backed trial results (ephemeral on Railway, persistent locally)
- Auto-saves after each eval run
- Trial history table with per-model comparison and best-value highlighting

## Observability

Each query response includes:
- **Retrieval details**: per-chunk rank, file name, routine name, similarity score, match type (name/expansion/call_graph/vector)
- **Query expansion**: list of LLM-expanded routine names (when triggered)
- **Timing**: retrieval, generation, and total latency (displayed in seconds)
- **Token usage**: prompt, completion, and total tokens

Structured JSON logs are written to `logs/legacylens.jsonl` with rotating file handler (5 MB x 3 backups).

## Deployment

- **Application**: Railway single service (Dockerfile)
- **Vector DB**: Qdrant Cloud
- **Local dev**: `docker compose up --build`
- **Ingestion**: Runs locally via `scripts/ingest.py`
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
