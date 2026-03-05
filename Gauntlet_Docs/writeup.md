# LegacyLens — Technical Writeup

## Technologies Used

### Core Stack

**Python 3.12 / FastAPI** — First Python project. Chose FastAPI for its async support, automatic OpenAPI docs, and minimal boilerplate. The entire backend is a single `app/main.py` with service modules — no framework overhead.

**Embeddings (4 providers)** — Swappable embedding models with per-collection Qdrant storage:
- OpenAI `text-embedding-3-small` (1536-dim) and `text-embedding-3-large` (3072-dim)
- Voyage AI `voyage-code-3`, `voyage-4-large`, `voyage-4`, `voyage-4-lite` (1024-dim)
- Google Gemini `gemini-embedding-001` (3072-dim)
- Cohere `embed-v4.0` (1536-dim)

Each embedding model gets its own Qdrant collection, so you can ingest once per model and compare retrieval quality head-to-head via the eval harness.

**LLMs (2 providers)** — Defaults to `gpt-4.1-nano` for generation, but supports 11 models total — OpenAI (`gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5.2`) and Google Gemini (`gemini-2.5-flash`, `gemini-2.5-pro`) — via a runtime model selector. This lets users compare end-to-end response latency, answer quality, and citation accuracy across providers, model generations, sizes, and reasoning abilities. The quality ceiling is the retrieved context, not the model.

**Qdrant** — Purpose-built vector DB with payload filtering. This was the key enabler for hybrid retrieval — exact name matches via payload filter AND cosine similarity in the same system. Self-hosted on Railway, co-located with the app on the same internal network for low-latency retrieval (~10-30ms per query vs ~100-200ms with external hosting).

**fparser** — Python library for parsing Fortran ASTs. Handles both fixed-form FORTRAN 77 (.f) and free-form Fortran 90 (.f90). This was critical for syntax-aware chunking — regex alone breaks on continuation lines, comment keywords, and bare END statements. fparser gives real AST boundaries.

### Supporting Tools

- **tiktoken** — Token counting for chunk size enforcement (8191-token embedding limit)
- **Docker / Docker Compose** — Local dev with co-located Qdrant
- **Railway** — Deployment (app + Qdrant as separate services on internal network)
- **highlight.js** — Fortran syntax highlighting in the web UI

## Edge Cases Considered

### What if a file can't be parsed?

The Fortran parser (fparser) doesn't handle every file perfectly — some have non-standard syntax or unusual formatting. Rather than skipping those files and losing data, the system falls back to ingesting the entire file as-is. This is why we hit 0 files dropped across 2,272 source files. Nothing gets left behind.

### What if a file is too big to embed in one piece?

OpenAI embeddings have a hard token limit (8,191 tokens). Most LAPACK routines fit in one chunk, but about 55 of the largest routines don't — some are 20,000+ tokens. These get split into overlapping pieces so nothing is lost at the seam where one chunk ends and the next begins.

### What if someone asks about a specific routine by name?

When a user types "What does DGEMM do?", the system detects that DGEMM is a routine name and does an exact lookup in the database rather than relying on fuzzy similarity search. This gives perfect results for known routines. Without this, a similarity search might return the wrong precision variant (SGEMM instead of DGEMM) or unrelated routines that happen to have similar descriptions.

### What if someone asks a conceptual question with no routine name?

A query like "How does SVD work?" doesn't contain any routine name to look up. So the system asks the LLM to identify which LAPACK routines are relevant (DGESVD, DGESDD, etc.) and then searches for each one. This bridges the gap between how humans think about math and how LAPACK organizes its code.

### What if a routine calls other routines that matter?

High-level LAPACK routines delegate most of their work to lower-level helpers. If you ask about DGESV (solve a linear system), the answer isn't complete without understanding DGETRF (the factorization it calls). The system automatically pulls in one level of called routines so the LLM has the full picture.

### What if multiple search strategies find the same chunk?

The system searches five different ways (name match, expansion, call-graph, caller impact, similarity). These often return overlapping results. Duplicates are removed, keeping only the highest-scored instance of each chunk.

### What if nothing relevant is found?

Rather than making something up, the LLM is instructed to say it doesn't have enough context. This was a deliberate design choice — wrong answers with fake citations are worse than no answer at all.

### What if the LLM doesn't cite its sources?

The system prompt demands file:line citations for every claim. If the LLM skips them anyway, a post-processing step attaches the source file references from the chunks that were actually retrieved.

### What if too much context is stuffed into the prompt?

There's a hard budget of 3,000 tokens for retrieved context. Chunks are added in order of relevance until the budget runs out. If a chunk would push past the limit, it's excluded entirely — no partial routines, since a half-complete function confuses the LLM more than it helps.

## Most Difficult Problem: Latency

Latency was the hardest engineering problem. The target was <3s end-to-end, but early versions were hitting 12-15 seconds per query.

### The Bottleneck Stack

A single query involves multiple sequential network calls:

1. **Embed the query** (~200ms) — OpenAI API call
2. **LLM query expansion** (~500-800ms) — OpenAI API call (only for conceptual queries)
3. **Vector search** (~50-200ms) — Qdrant query
4. **Name-match searches** (~50-100ms each) — Qdrant payload filter queries
5. **Call-graph searches** (~50-100ms each) — Additional Qdrant queries per called routine
6. **LLM generation** (~1-2s) — OpenAI API call

Run sequentially, this easily exceeds 3s. With query expansion triggering 5-8 individual name searches, it was blowing past 10s.

### What Fixed It

**1. Async pipeline with concurrent search** — The biggest win. Instead of searching expanded names one-at-a-time, all name-match searches run concurrently via `asyncio.gather()`. This collapsed 5-8 sequential Qdrant calls (~500-800ms each) into a single parallel batch (~200ms total).

**2. Co-located Qdrant on Railway** — Moved from Qdrant Cloud (external network, ~100-200ms per query) to a Railway-hosted Qdrant instance on the same internal network (~10-30ms per query). This shaved ~150ms off every single Qdrant call, and with hybrid retrieval making 3-10 calls per query, it added up fast.

**3. Streaming generation** — Didn't reduce actual latency, but eliminated perceived latency. The first token appears in ~300ms, so the user sees the answer building in real-time while the full generation takes 1-2s. Retrieval details are sent as the first SSE event, so the chunk sidebar populates instantly.

**4. Context budget enforcement** — Early versions stuffed too many tokens into the generation prompt, causing gpt-4o to take longer. Capping context at 3,000 tokens kept generation time predictable at 1-2s.

**5. Eval pipeline optimization** — The eval harness (77 retrieval + 27 E2E queries) uses batched `asyncio.gather()` (5 queries at a time to balance throughput and API limits). Not user-facing latency, but critical for rapid iteration on retrieval quality.

