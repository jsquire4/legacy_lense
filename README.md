# LegacyLens

A RAG application for querying the LAPACK Fortran codebase via natural language. Ask questions about routines, algorithms, and dependencies — get citation-backed answers grounded in actual source code.

## Features

- **Hybrid retrieval**: name matching, LLM query expansion, call-graph following, and vector similarity
- **4 specialized capabilities**: code explanation, documentation generation, pattern detection, dependency mapping
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
Specialized query. Capabilities: `explain_code`, `generate_docs`, `detect_patterns`, `map_dependencies`.

Same request/response format as `/api/query`.

## Architecture

```
Query → Embed → Hybrid Search → Context Assembly (6K tokens) → gpt-4o-mini → Citation-enforced Response
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

Results (15 queries):
- **Recall@5**: 0.96 (target: >0.70)
- **Avg latency**: <3s (target: <3s)

## Deployment

Deployed on Railway with auto-deploy on push. Qdrant Cloud provides the vector database (free tier, 1 GB).

Environment variables required:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

## Cost

- **Ingestion**: ~$0.09 one-time (4.7M tokens embedded)
- **Per query**: ~$0.0016 (gpt-4o-mini)
- **Infrastructure**: $5/mo Railway + free Qdrant Cloud

See [docs/cost_analysis.md](docs/cost_analysis.md) for scaling projections.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12, FastAPI |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| LLM | gpt-4o-mini |
| Vector DB | Qdrant Cloud |
| Parser | fparser (AST-based Fortran parsing) |
| Deployment | Railway + Docker |
