# Concept Brief: LegacyLens

## Problem Statement
Students and developers encountering legacy Fortran codebases (like LAPACK) have no easy way to understand what the code does, how routines relate to each other, or what patterns the codebase follows. Reading fixed-form FORTRAN 77 is a lost art. LegacyLens uses RAG to make legacy code queryable in natural language.

## Vision
A publicly accessible web application where a user types a natural language question about the LAPACK codebase and gets back a precise, citation-backed answer with file paths and line references. The system supports multiple code-understanding capabilities: explanation, documentation generation, pattern detection, and dependency mapping.

## Scope

### In Scope (MVP — 24-hour gate)
- Full ingestion of Reference-LAPACK/lapack repository
- Syntax-aware chunking using `fparser` for structure detection + custom chunking logic
- Embedding generation via OpenAI `text-embedding-3-small` (1536d)
- Vector storage in Qdrant Cloud (free tier)
- Semantic similarity search (cosine, top_k=8)
- Answer generation via OpenAI `gpt-4o-mini` with citation enforcement
- Minimal web UI (FastAPI + static HTML page with query input and results display)
- Deployed on Railway (single service)

### In Scope (Full submission — post-MVP)
- 4 code-understanding capabilities (build order):
  1. Code explanation (semantic + pseudocode)
  2. Documentation generation
  3. Pattern detection
  4. Dependency mapping
- Evaluation harness (15+ manual queries, Precision@5, latency tracking)
- Structured JSON observability logging
- Cost analysis projections (100 / 1k / 10k / 100k users)
- Architecture justification document

### Stretch Goals
- Business rule extraction (capability 5)
- Impact analysis (capability 6)
- Web UI polish

### Out of Scope
- Re-ranking / hybrid search
- Advanced HNSW tuning
- Real-time incremental ingestion
- Authentication / multi-tenancy
- Local/self-hosted LLM or embedding models

## Key Decisions Made

1. **Qdrant Cloud over self-hosted**: Same client code, zero infrastructure management. Self-hosting on Railway only adds deployment pain with no learning upside. Free tier provides 1GB storage.

2. **`fparser` for Fortran parsing**: Actively maintained (Sept 2025), handles both fixed-form (.f) and free-form (.f90). Provides AST for boundary detection. Custom chunking logic built on top for learning purposes.

3. **One-file-one-chunk as primary strategy**: LAPACK follows a strict one-procedure-per-file convention for .f files. Chunking largely reduces to file-level splitting with metadata extraction. Fallback splitting needed only for oversized units and multi-procedure .f90 files.

4. **No framework abstractions**: Custom pipeline using OpenAI SDK + Qdrant Python client directly. Maximizes understanding of RAG mechanics. Aligns with rubric emphasis on understanding over framework usage.

5. **FastAPI + static HTML**: Simplest path to a publicly accessible interface. Satisfies deployment requirement without UI framework overhead. Dead simple for someone coming from JS/TS.

6. **Capabilities ordered by difficulty**: Explanation → docs → patterns → dependencies. First two are essentially prompt variations on the same retrieval. Latter two require more metadata and multi-chunk reasoning.

## Open Questions
- Exact chunk size threshold for fallback splitting (token count vs line count?)
- Whether to embed doc-comment headers separately from code bodies or together
- Optimal context assembly strategy (how many chunks, truncation policy)
- These will be resolved during implementation with empirical testing

## Technical Context
- **Language**: Python (new to user — code must be clean and readable)
- **Stack**: FastAPI, OpenAI SDK, Qdrant Python client, fparser
- **Deployment**: Railway (single service), Qdrant Cloud (free tier)
- **Source data**: Reference-LAPACK/lapack — Fortran 77/90, ~10k+ LOC, 50+ files
- **User background**: Strong JS/TS, first Python project, first RAG project

## Research Findings

### Deployment (Qdrant + Railway)
- Railway has a one-click Qdrant template but self-hosting brings volume permission issues, no healthcheck, deployment downtime on redeploy
- Qdrant Cloud free tier: 1GB, no credit card, auto-suspends after 1 week inactivity (just ping it)
- Recommended path: Qdrant Cloud for vector DB + Railway single-service for Python app

### Fortran Parsing
- LAPACK .f files are fixed-form FORTRAN 77, one procedure per file — chunking is largely file-level
- Minority .f90 files are free-form and may contain multiple procedures
- `fparser` (pip install fparser) is the strongest Python parser: full AST, active maintenance, handles both formats
- Regex alone hits ~80-90% accuracy on LAPACK but breaks on continuation lines, comment keywords, bare END statements
- tree-sitter-fortran exists but the fixed-form grammar is abandoned

## Risk Assessment

1. **Retrieval quality on mathematical code**: Embeddings may not capture numerical/mathematical semantics well. Mitigation: rich metadata in embedded content (identifiers, parameters, comments), manual eval early.

2. **24-hour MVP pressure**: Tight timeline with new language + new domain. Mitigation: strict build order, MVP-first mentality, no premature optimization.

3. **fparser edge cases**: Beta library, may hit parsing failures on some LAPACK files. Mitigation: graceful fallback to file-level chunking if AST parsing fails on a file.

4. **Qdrant Cloud free tier limits**: 1GB storage, auto-suspension. Mitigation: LAPACK embeddings should fit well within 1GB; keep cluster active during project window.

5. **Python inexperience**: User is JS/TS-native. Mitigation: keep code simple, well-commented, Pythonic patterns that map to familiar JS concepts.
