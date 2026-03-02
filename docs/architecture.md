# LegacyLens Architecture

## Overview

LegacyLens is a RAG (Retrieval-Augmented Generation) application that makes the LAPACK Fortran codebase queryable via natural language. Users ask questions and receive citation-backed answers grounded in actual source code.

## System Architecture

```
User Query → FastAPI → Embed Query → Qdrant Search → Context Assembly → LLM Generation → Response
```

### Components

1. **FastAPI Web Server** (`app/main.py`) — Serves the web UI and API endpoints
2. **Fortran Parser** (`app/services/parser.py`) — Parses .f and .f90 files using fparser
3. **Chunker** (`app/services/chunker.py`) — Splits parsed units into token-capped chunks
4. **Embedding Service** (`app/services/embeddings.py`) — Generates embeddings via OpenAI
5. **Vector Store** (`app/services/vector_store.py`) — Manages Qdrant collection operations
6. **Retrieval Service** (`app/services/retrieval.py`) — Query → embed → search pipeline
7. **Generation Service** (`app/services/generation.py`) — LLM answer generation with citations
8. **Capabilities** (`app/services/capabilities.py`) — Specialized code understanding prompts

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

### Why no framework (LangChain, LlamaIndex)?
- Custom pipeline maximizes understanding of RAG mechanics
- Direct use of OpenAI SDK + Qdrant client keeps dependencies minimal
- Full control over chunking strategy, context assembly, and citation enforcement

### Why binary-search context truncation?
- Fits maximum number of complete chunks within token budget
- Avoids mid-chunk truncation that loses context
- 6000-token context budget leaves room for system prompt and response

## Data Flow

### Ingestion Pipeline
```
Fortran Files → Parser (fparser1/fparser2) → ParsedUnits → Chunker → Chunks → Embeddings → Qdrant
```

- 2300 files parsed into 2304 units (some .f90 files yield multiple units)
- 2407 chunks after splitting oversized units
- 0 parse errors (RAW fallback catches all failures)

### Query Pipeline
```
User Query → Embed → Qdrant Search (top_k=8) → Context Assembly → GPT-4o-mini → Citation-enforced Answer
```

## Deployment

- **Application**: Railway single service (Python/FastAPI)
- **Vector DB**: Qdrant Cloud (external, free tier)
- **Ingestion**: Runs locally via `scripts/ingest.py`
- **Serving**: Railway serves query API only (no ingestion on deploy)

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Parser failure | RAW fallback — entire file becomes one unit |
| Oversized chunks | Sliding window split with 50-token overlap |
| OpenAI rate limits | SDK built-in retry (max_retries=3) |
| Qdrant failures | Batch upsert with retry |
| Empty retrieval | Generation responds "insufficient context" |
| Missing citations | Enforcement layer appends sources from chunk metadata |
