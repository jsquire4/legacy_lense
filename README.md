# LegacyLens

A RAG application for querying the LAPACK Fortran codebase via natural language. Ask questions about routines, algorithms, and dependencies — get citation-backed answers grounded in actual source code.

## Features

- **Hybrid retrieval**: name matching, LLM query expansion, call-graph following, and vector similarity
- **5 specialized capabilities**: code explanation, documentation generation, pattern detection, dependency mapping, impact analysis
- **Citation enforcement**: every answer includes file:line references
- **Observability**: per-query timing, token usage, chunk scores, and retrieval strategy details
- **Structured logging**: rotating JSON log files in `logs/`

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

This builds the image and starts the server on port 8000. Requires a `.env` file with your API keys.

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok"}`.

### `POST /api/query`
General-purpose query.

```json
{
  "query": "What does DGESV do?",
  "top_k": 8
}
```

Response includes `answer`, `citations`, `latency_ms`, `retrieval_details`, `token_usage`, and `timing`.

### `POST /api/capabilities/{capability}`
Specialized query. Capabilities: `explain_code`, `generate_docs`, `detect_patterns`, `map_dependencies`, `impact_analysis`.

Same request/response format as `/api/query`.

### Streaming Endpoints

`POST /api/query/stream` and `POST /api/capabilities/{capability}/stream` return SSE streams with `retrieval`, `token`, and `done` events for real-time UI rendering.

## Architecture

```
Query → Embed → Hybrid Search → Context Assembly (6K tokens) → gpt-4o → Citation-enforced Response
```

Hybrid search merges:
1. **Name match** — Qdrant payload filter for detected routine names
2. **Query expansion** — LLM identifies relevant routine names for conceptual queries
3. **Call-graph following** — one-hop expansion of called routines
4. **Vector similarity** — cosine similarity fills remaining slots

See [docs/architecture.md](docs/architecture.md) for full details.

## Evaluation

```bash
python scripts/evaluate.py
```

Results (37 queries):
- **Recall@5**: >0.90 (target: >0.70)
- **Avg retrieval**: <500ms
- **Avg end-to-end**: <3s (target: <3s)

## Deployment

Deployed on Railway with auto-deploy on push. Qdrant runs as a co-located Railway service for minimal retrieval latency.

**Live**: https://legacylense-production.up.railway.app

Environment variables required:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

## Cost

- **Ingestion**: ~$0.09 one-time (4.7M tokens embedded)
- **Per query**: ~$0.021 (gpt-4o)
- **Infrastructure**: $5/mo Railway (app + Qdrant)

See [docs/cost_analysis.md](docs/cost_analysis.md) for scaling projections.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12, FastAPI |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| LLM | gpt-4o |
| Vector DB | Qdrant (Railway, internal network) |
| Parser | fparser (AST-based Fortran parsing) |
| Deployment | Railway + Docker |
