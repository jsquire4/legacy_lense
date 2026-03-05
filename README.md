# LegacyLens

A RAG application for querying the LAPACK Fortran codebase via natural language. Ask questions about routines, algorithms, and dependencies — get citation-backed answers grounded in the actual source code.

## Features

- **Hybrid retrieval**: name matching, LLM query expansion, call-graph following, caller impact analysis, and vector similarity
- **7 specialized capabilities**: code explanation, documentation generation, pattern detection, dependency mapping, impact analysis, business rule extraction, plus general-purpose
- **Multi-provider LLMs**: OpenAI (GPT-3.5-turbo through GPT-5.2) and Google Gemini (2.5-flash, 2.5-pro) with automatic API parameter handling
- **Multi-provider embeddings**: OpenAI, Voyage AI, Google Gemini, and Cohere — per-collection model selection with separate Qdrant collections
- **Rigorous eval system**: 77 retrieval queries across 3 difficulty tiers with MRR, NDCG@5, P@K, R@K, and negative oracle metrics; 27 E2E queries with embedding similarity, source-verified golden answers, and 6 hallucination probes
- **Model comparison**: built-in eval harness with trial storage for comparing model quality and latency across embedding + LLM combinations
- **Cache warming**: pre-cached responses for default queries across cheap models for instant performance
- **Citation enforcement**: every answer includes file:line references with fallback detection
- **Refusal detection**: auto-fails eval answers that dodge the question
- **Observability**: per-query timing, token usage, chunk scores, retrieval strategy details, and TTFT tracking
- **Structured logging**: rotating JSON log files

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for Qdrant)
- OpenAI API key (required — used for default embeddings and generation)
- Optional provider keys for additional models:
  - `VOYAGE_API_KEY` — Voyage AI embeddings (voyage-code-3, voyage-4-*)
  - `GEMINI_API_KEY` — Google Gemini embeddings + LLM generation
  - `COHERE_API_KEY` — Cohere embeddings (embed-v4.0)

### Local Development

1. Clone and install:
   ```bash
   git clone <repo-url> && cd leglen
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start Qdrant:
   ```bash
   docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Required: OPENAI_API_KEY, QDRANT_URL=http://localhost:6333
   # Optional: VOYAGE_API_KEY, GEMINI_API_KEY, COHERE_API_KEY
   ```

4. Ingest the LAPACK codebase (one-time):
   ```bash
   git clone https://github.com/Reference-LAPACK/lapack.git data/lapack
   python scripts/ingest.py
   # To ingest with a different embedding model:
   # python scripts/ingest.py --embedding-model voyage-code-3
   ```

5. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Open http://localhost:8000

### Docker (app + Qdrant together)

```bash
docker compose up --build
```

This starts both the app and Qdrant. You'll still need to run `scripts/ingest.py` separately to populate the vector store.

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
Query → Expand (LLM) → Embed (multi-provider) → Hybrid Search → Context Assembly (3K tokens) → LLM Generation → Citation-enforced Response
```

Hybrid search merges five strategies in parallel:
1. **Name match** — Qdrant payload filter for detected routine names (top-3)
2. **Query expansion** — LLM identifies 5-8 relevant routine names for conceptual queries (cached, 256-entry LRU)
3. **Call-graph following** — one-hop expansion via `called_routines` metadata
4. **Caller impact analysis** — reverse lookup via `called_by` metadata (top-5)
5. **Vector similarity** — cosine similarity fills remaining slots

See [docs/architecture.md](docs/architecture.md) for full details.

## Evaluation

### Retrieval Evals (77 queries, 3 difficulty tiers)
- Metrics: Precision@K (1/3/5), Recall@K (1/3/5), MRR, NDCG@5, negative oracle penalty
- P@5 displayed as percentage of theoretical maximum to account for structural ceiling
- Difficulty breakdown (easy/medium/hard) with per-tier stats
- Embedding-based — tests any embedding model via collection selection

### E2E Generation Evals (27 queries)
- 21 normal queries with source-verified golden answers and ALL-keyword matching
- 6 hallucination probes for nonexistent routines (must trigger refusal)
- Embedding similarity gate: cosine similarity between generated and golden answer embeddings (threshold 0.75)
- Citation relevance checking against expected files, with fallback detection
- Run against any supported LLM + embedding model combination
- Results auto-saved to trial history with per-metric breakdown

```bash
python scripts/evaluate.py
```

## Deployment

Deployed on Railway with auto-deploy on push. App and Qdrant run as separate Railway services on the same internal network for low-latency retrieval.

**Live**: https://legacylense-production.up.railway.app

Environment variables required:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (optional for self-hosted)
- `VOYAGE_API_KEY` (optional, for Voyage AI embeddings)
- `GEMINI_API_KEY` (optional, for Gemini embeddings/generation)
- `COHERE_API_KEY` (optional, for Cohere embeddings)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Embeddings | OpenAI (1536/3072-dim), Voyage AI (1024-dim), Gemini (3072-dim), Cohere (1536-dim) |
| LLM | OpenAI (GPT-3.5-turbo through GPT-5.2) + Google Gemini (2.5-flash, 2.5-pro) |
| Vector DB | Qdrant (self-hosted on Railway) |
| Parser | fparser (AST-based Fortran parsing with RAW fallback) |
| Trial Storage | SQLite with schema migrations |
| Deployment | Railway (app + Qdrant) + Docker Compose (local) |
