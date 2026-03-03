# LegacyLens

A RAG application for querying the LAPACK Fortran codebase via natural language. Ask questions about routines, algorithms, and dependencies — get citation-backed answers grounded in actual source code.

## Features

- **Hybrid retrieval**: name matching, LLM query expansion, call-graph following, and vector similarity
- **6 specialized capabilities**: code explanation, documentation generation, pattern detection, dependency mapping, impact analysis, business rule extraction
- **9 model support**: GPT-3.5-turbo through GPT-5.2 with automatic API parameter handling for legacy, standard, and reasoning models
- **Model comparison**: built-in E2E eval harness with trial storage for comparing model quality and latency
- **Cache warming**: pre-cached responses for default queries across 3 cheap models for instant performance
- **Citation enforcement**: every answer includes file:line references
- **Refusal detection**: auto-fails eval answers that dodge the question
- **Observability**: per-query timing, token usage, chunk scores, and retrieval strategy details
- **Structured logging**: rotating JSON log files

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- Qdrant Cloud instance (free tier works)

### Local Development

1. Clone and install:
   ```bash
   git clone <repo-url> && cd leglen
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Fill in: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
   ```

3. Ingest the LAPACK codebase (one-time):
   ```bash
   git clone https://github.com/Reference-LAPACK/lapack.git data/lapack
   python scripts/ingest.py
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open http://localhost:8000

### Docker

```bash
docker compose up --build
```

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok"}`.

### `POST /api/query`
General-purpose query. Supports optional `model` parameter.

```json
{
  "query": "What does DGESV do?",
  "top_k": 8,
  "model": "gpt-4.1-nano"
}
```

Response includes `answer`, `citations`, `latency_ms`, `retrieval_details`, `token_usage`, and `timing`.

### `POST /api/capabilities/{capability}`
Specialized query. Capabilities: `explain_code`, `generate_docs`, `detect_patterns`, `map_dependencies`, `impact_analysis`, `extract_business_rules`.

### `GET /api/models`
Returns all supported models with pricing and default indicator.

### `POST /api/trials` / `GET /api/trials` / `DELETE /api/trials/{id}`
CRUD for eval trial results.

### Streaming Endpoints

`POST /api/query/stream` and `POST /api/capabilities/{capability}/stream` return SSE streams with `retrieval`, `token`, and `done` events.

## Architecture

```
Query → Embed → Hybrid Search → Context Assembly (3K tokens) → LLM (2048 response tokens) → Citation-enforced Response
```

Hybrid search merges:
1. **Name match** — Qdrant payload filter for detected routine names
2. **Query expansion** — LLM identifies relevant routine names for conceptual queries
3. **Call-graph following** — one-hop expansion of called routines
4. **Vector similarity** — cosine similarity fills remaining slots

See [docs/architecture.md](docs/architecture.md) for full details.

## Evaluation

### Retrieval Evals (37 queries)
- Recall@5 and latency per query
- Embedding-based, model-independent

### E2E Generation Evals (21 queries)
- Quality checks: citations, answer length, keywords, refusal detection
- Run against any supported model
- Results auto-saved to trial history for comparison

```bash
python scripts/evaluate.py
```

## Deployment

Deployed on Railway with auto-deploy on push. Qdrant Cloud for vector storage.

**Live**: https://legacylense-production.up.railway.app

Environment variables required:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12, FastAPI |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| LLM | 9 OpenAI models (default: GPT-4.1-nano) |
| Vector DB | Qdrant Cloud |
| Parser | fparser (AST-based Fortran parsing) |
| Trial Storage | SQLite |
| Deployment | Railway + Docker |
