# LegacyLens Architecture

## Overview

LegacyLens is a RAG (Retrieval-Augmented Generation) application that makes the LAPACK Fortran codebase queryable via natural language. Users ask questions and receive citation-backed answers grounded in actual source code.

## System Architecture

```
User Query → FastAPI → Hybrid Retrieval → Context Assembly → LLM Generation → Citation-enforced Response
```

### Components

1. **FastAPI Web Server** (`app/main.py`) — Serves the web UI and REST API with observability data
2. **Fortran Parser** (`app/services/parser.py`) — Parses .f and .f90 files using fparser
3. **Chunker** (`app/services/chunker.py`) — Splits parsed units into token-capped chunks
4. **Embedding Service** (`app/services/embeddings.py`) — Generates embeddings via OpenAI
5. **Vector Store** (`app/services/vector_store.py`) — Manages Qdrant collection operations
6. **Retrieval Service** (`app/services/retrieval.py`) — Hybrid retrieval: name match + query expansion + call-graph + vector
7. **Generation Service** (`app/services/generation.py`) — LLM answer generation with citation enforcement
8. **Capabilities** (`app/services/capabilities.py`) — 6 specialized code understanding prompts
9. **Logging** (`app/logging_config.py`) — Structured JSON logging with rotating file handler

## Key Design Decisions

### Why Qdrant Cloud (not self-hosted)?
- Same client code, zero infrastructure management
- Free tier provides 1GB (sufficient for LAPACK embeddings)
- Avoids Railway volume permission issues with self-hosted Qdrant

### Why fparser (not regex)?
- Handles both fixed-form (.f) and free-form (.f90) Fortran
- Provides AST for reliable boundary detection
- Active maintenance (last updated Sept 2025)
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

### Why 6000-token context budget?
- Fits maximum number of complete chunks within budget
- Avoids mid-chunk truncation that loses context
- Leaves room for system prompt and response within gpt-4o's context window

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
  Merge & deduplicate → Context assembly (6K token budget) → gpt-4o → Citation enforcement → Response
```

## Observability

Each query response includes:
- **Retrieval details**: per-chunk rank, file name, routine name, similarity score, match type (name/expansion/call_graph/vector)
- **Query expansion**: list of LLM-expanded routine names (when triggered)
- **Timing**: retrieval_ms, generation_ms, total_ms
- **Token usage**: prompt, completion, and total tokens

Structured JSON logs are written to `logs/legacylens.jsonl` with rotating file handler (5 MB x 3 backups).

## Deployment

- **Application**: Railway single service (Dockerfile auto-detected)
- **Vector DB**: Qdrant (Railway, internal network)
- **Local dev**: `docker compose up --build` (one command)
- **Ingestion**: Runs locally via `scripts/ingest.py`
- **CI/CD**: Auto-deploys on push to GitHub

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Parser failure | RAW fallback — entire file becomes one unit |
| Oversized chunks | Sliding window split with 50-token overlap |
| OpenAI rate limits | SDK built-in retry (max_retries=3) |
| Qdrant failures | Batch upsert with retry |
| Empty retrieval | Generation responds "insufficient context" |
| Missing citations | Enforcement layer appends sources from chunk metadata |
| Query expansion failure | Graceful fallback to vector-only search |
