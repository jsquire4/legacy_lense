# LegacyLens – Pre-Search Document

## Phase 1: Define Your Constraints

### 1) Scale & Load Profile

**Target Codebase:**  
Reference-LAPACK/lapack (Fortran numerical linear algebra library)  
https://github.com/Reference-LAPACK/lapack

**Language:**  
Fortran

**Approximate Size:**  
Large-scale scientific codebase (significantly exceeds 10,000 LOC and 50+ files requirement)

**Indexing Scope (MVP):**  
Full repository indexing (100% of Fortran source files)

**Expected Usage (MVP):**  
Low volume – single developer usage and demo queries

**Ingestion Strategy:**  
Batch ingestion during initial indexing

**Latency Target:**  
< 3 seconds end-to-end query response

---

### 2) Budget & Cost Ceiling

**Embedding Model:**  
OpenAI `text-embedding-3-small` (1536 dimensions)

**LLM for Answer Generation:**  
OpenAI `gpt-4o-mini`  
Upgrade path to higher-tier model if reasoning quality requires it.

**Vector Database:**  
Qdrant (self-hosted via Docker)

**Cost Philosophy:**
- Use hosted APIs for embeddings and LLM for rapid development
- Keep vector database local for learning and cost control
- Accept modest API spend during development; optimize later if scaling

---

### 3) Time to Ship

**MVP Timeline:**  
24-hour delivery window

**MVP Requirements:**
- Ingest full LAPACK repository
- Syntax-aware chunking (Fortran PROGRAM / MODULE / SUBROUTINE / FUNCTION)
- Generate embeddings for all chunks
- Store embeddings in vector database
- Semantic similarity search
- Natural language query interface (CLI)
- Return relevant code snippets with file and line references
- Basic answer generation using retrieved context
- Deployed and publicly accessible

**Interface Decision:**  
CLI-only for MVP

---

### 4) Data Sensitivity

The selected codebase (LAPACK) is open source.

For the purposes of this assignment, data sensitivity is not a concern. Code snippets may be sent to external embedding and LLM APIs during development.

If applied to proprietary or regulated codebases in the future, the system could be adapted to use locally hosted embedding models and LLMs.

---

### 5) Team & Skill Constraints

**RAG Experience:**  
No prior hands-on experience building RAG pipelines.

**Vector Database Experience:**  
No prior production experience with vector databases.

**Implementation Approach:**
- Direct OpenAI API calls for embeddings and LLM
- Direct Qdrant client integration
- Custom retrieval pipeline (no LangChain or LlamaIndex)

Framework abstractions are intentionally avoided for MVP to maximize understanding of indexing, retrieval, and scoring mechanics.

---

## Phase 2: Architecture Discovery

### 6) Vector Database Selection

**Selected Database:**  
Qdrant (self-hosted via Docker)

**Rationale:**  
Qdrant is a purpose-built vector database supporting high-performance approximate nearest neighbor (ANN) search with explicit control over distance metrics (cosine similarity), HNSW indexing, persistent storage, and strong metadata filtering.

This aligns with the goal of building a mini production-quality system while learning internal mechanics of vector retrieval.

---

### 7) Embedding Strategy

**Embedding Model:**  
OpenAI `text-embedding-3-small` (1536 dimensions)

**Vector Dimension:**  
1536 (consistent across ingestion and query time)

**Embedding Content Strategy:**

Each chunk will embed both code and structured metadata.

Embedded content per chunk will include:
- File path
- Module / subroutine / function name
- Line range
- Extracted identifiers (variables, parameters, called subroutines)
- Comments (if present)
- Code body

Example embedding format:

FILE: src/lapack/dgesv.f
UNIT_TYPE: SUBROUTINE
UNIT_NAME: DGESV
LINES: 42-210
IDENTIFIERS: A, B, IPIV, INFO

<code body>

**Rationale:**  
Including structural metadata improves retrieval quality for function-level queries, dependency tracing, and variable-based search.

---

### 8) Chunking Approach

**Primary Strategy:**  
Chunk by Fortran program units:
- PROGRAM
- MODULE
- SUBROUTINE
- FUNCTION

Each program unit will be treated as a semantic boundary and indexed as an independent chunk.

**Fallback Strategy for Large Units:**  
If a program unit exceeds a defined size threshold (e.g., token or line count), it will be split into subchunks using fixed-size windows with overlap (e.g., 200–400 lines with 30–50 line overlap).

This ensures:
- Preservation of semantic structure
- Improved retrieval precision
- Reduced context truncation during answer generation
- Better handling of large LAPACK routines

**Metadata Captured Per Chunk:**
- File path
- Unit type (PROGRAM / MODULE / SUBROUTINE / FUNCTION)
- Unit name
- Start and end line numbers
- Extracted called subroutines (for future dependency analysis)

**Rationale:**  
Program-unit-aware chunking maintains meaningful boundaries aligned with Fortran structure, while fallback splitting ensures performance and context control for oversized routines. This balances semantic integrity with practical retrieval constraints.

---

### 9) Retrieval Pipeline

**Query Flow:**
1. User submits natural language query via CLI.
2. Query is embedded using the same embedding model (`text-embedding-3-small`).
3. Vector similarity search is executed against Qdrant collection.
4. Top-k results are retrieved (`top_k = 8`).
5. Retrieved chunks are assembled into context for answer generation.
6. LLM generates explanation with citations.

**Similarity Metric:**  
Cosine similarity (aligned with embedding model design).

**Context Assembly Strategy:**  
- Retrieved chunks are concatenated in ranked order.
- If necessary, include limited surrounding lines for additional context.
- Preserve file path and line references for citation.

**Re-ranking:**  
No secondary re-ranking for MVP. Retrieval relies on embedding similarity only.

**Low-Confidence Handling:**  
If similarity scores are low or results appear weak, the system will:
- Return the top matches with transparency
- Indicate possible ambiguity
- Encourage refinement of the query

**Rationale:**  
A simple, transparent retrieval pipeline reduces system complexity while allowing inspection of search behavior and scoring during development.

---

### 10) Answer Generation

**LLM Model:**  
OpenAI `gpt-4o-mini`

**Generation Strategy:**
The LLM will synthesize responses strictly from retrieved context. No external knowledge or assumptions about LAPACK internals are permitted.

**Citation Policy:**
- All substantive claims must include file and line range citations.
- Citation format: `path/to/file.f:START_LINE-END_LINE`.
- If insufficient evidence exists in retrieved context, the model must explicitly state this.

**Prompt Discipline:**
The system prompt will enforce:
- Deterministic, technical tone
- No speculation
- No invented functions or variables
- Explicit citation requirements

**Failure Handling:**
If the context does not adequately answer the question, the response will:
- State that context is insufficient
- List the most relevant retrieved chunks
- Avoid fabricating explanations

**Rationale:**  
Scientific and numerical code demands precision. Enforcing citation-backed explanations ensures reliability and builds trust with technical users.

---

### 11) Framework Selection

**Framework Decision:**  
Custom implementation (no LangChain, no LlamaIndex)

**Rationale:**
- Maximize understanding of embedding, indexing, and retrieval mechanics
- Avoid abstraction layers that obscure vector search behavior
- Maintain full control over chunk formatting, metadata schema, scoring, and context assembly
- Reduce unnecessary complexity for a CLI-based MVP

The system will directly integrate:
- OpenAI SDK for embeddings and generation
- Qdrant Python client for vector storage and search
- Custom retrieval and context assembly logic

This aligns with the goal of building a mini production-quality system while deeply understanding RAG fundamentals.

---

## Phase 3: Post-Stack Refinement

### 12) Failure Mode Analysis

The system acknowledges common RAG failure modes without attempting to solve all advanced code-analysis challenges.

**Expected Failure Cases:**

- **Ambiguous Identifiers:** Common variable or routine names (e.g., `INFO`, `TEMP`, `MAIN`) may retrieve multiple semantically similar but irrelevant chunks.
- **Cross-File Dependencies:** Data flow across multiple files or modules may not be fully captured if relevant chunks are not retrieved within `top_k`.
- **Large Routine Context Loss:** Very large subroutines may require fallback chunking, potentially splitting logically connected code.
- **Short or Vague Queries:** Queries such as "error handling" or "initialization" may return broad or noisy matches.
- **Embedding Limitations:** Embeddings may not perfectly capture mathematical intent or low-level numerical semantics.

**Mitigation Strategy (MVP Scope):**
- Return top matches transparently with similarity scores.
- Require citation-backed answers.
- Indicate when context is insufficient.
- Encourage users to refine queries when ambiguity is detected.

Advanced static analysis (e.g., full call graph tracing or data-flow analysis) is intentionally out of scope for MVP.

---

### 13) Evaluation Strategy

The system will be evaluated on retrieval quality and overall response performance.

**Retrieval Metric:**  
Precision@5

Definition: Of the top 5 retrieved chunks, the proportion that are manually verified as relevant to the query.

**Evaluation Process:**
- Create a manual test set of approximately 15 queries.
- Queries will cover realistic use cases such as:
  - Entry point identification
  - Subroutine explanation
  - Dependency tracing (what does X call?)
  - Variable modification search
  - File I/O identification
  - Error handling patterns
- For each query, manually determine which chunks are relevant.
- Compute precision@5.

**Performance Metric:**  
End-to-end response time (from CLI submission to full answer output).

Target: < 3 seconds under normal development conditions.

**Rationale:**  
Precision@5 directly evaluates retrieval effectiveness, which is the foundation of RAG quality. End-to-end latency ensures the system remains usable and responsive.

---

### 14) Performance Optimization

Performance optimization will focus on practical improvements within MVP scope rather than advanced index tuning.

**Ingestion Optimizations:**
- Batch embedding requests to reduce API overhead.
- Parallel file reading during initial indexing.
- Avoid re-indexing unchanged files during development.

**Query-Time Optimizations:**
- Use `top_k = 8` to balance recall and latency.
- Limit context assembly size to prevent excessive LLM token usage.
- Cache query embeddings during evaluation runs when appropriate.

Advanced index parameter tuning (e.g., HNSW hyperparameters) is intentionally out of scope for MVP.

**Rationale:**  
The goal is to achieve reliable sub-3-second responses using sensible defaults before introducing deeper performance engineering.

---

### 15) Observability

The system will implement structured JSON logging to support debugging and performance analysis.

**Logged Fields Per Query:**
- Timestamp
- Raw user query
- Extracted query embedding ID (if cached)
- Retrieved chunk IDs
- Similarity scores
- End-to-end latency (milliseconds)
- LLM token usage (input/output tokens)
- Error or failure flags (if applicable)

Logs will be written in JSON format to allow later analysis or ingestion into external tools.

For MVP, logs will be written to a local file and optionally printed to console for inspection.

**Rationale:**  
Structured logging enables transparent debugging of retrieval quality and performance without introducing heavy observability infrastructure.

---

### 16) Deployment & DevOps

**Deployment Target:**  
Docker-based deployment on Railway.

The application (CLI interface wrapped in an API service) and Qdrant vector database will be containerized and deployed together.

**Architecture:**
- Docker container for application service
- Docker container for Qdrant
- Internal network communication between services
- Environment variables for OpenAI API keys and configuration

**Configuration Strategy:**
- Use a single Dockerfile for the application
- Use `docker-compose.yml` (or Railway equivalent) for multi-service configuration
- Explicitly define ports and persistent storage for Qdrant

**Secrets Management:**
- API keys stored as environment variables
- No secrets committed to repository

**Reproducibility:**
- All dependencies defined in `requirements.txt`
- Dockerfile defines exact runtime environment
- README includes clear local setup and deployment instructions

**Rationale:**  
Containerized deployment ensures environment consistency between local development and production. Deploying both API and Qdrant together satisfies the requirement that the MVP be publicly accessible while maintaining architectural clarity.

---

## Implementation Risks & Friction Areas

The following practical risks are identified to guide development and debugging:

### 1) Retrieval Quality Degradation

Risk:
- Poor identifier extraction
- Inconsistent chunk boundaries
- Embeddings failing to capture mathematical semantics
- Relevant chunks not appearing within `top_k`

Mitigation:
- Validate chunk metadata during ingestion.
- Manually inspect a sample of embeddings and retrieved results.
- Use evaluation queries early to measure precision@5 before full feature build-out.

---

### 2) Deployment Friction (Docker + Railway + Qdrant)

Risk:
- Misconfigured container networking
- Volume persistence issues for Qdrant
- Environment variable misconfiguration
- Port exposure errors

Mitigation:
- Validate multi-container setup locally with Docker Compose before deploying.
- Explicitly define volumes and exposed ports.
- Keep configuration minimal and deterministic.

---

### 3) Context Window Overflow

Risk:
- Concatenating multiple large chunks may exceed token limits.
- Excessive context reduces answer clarity and increases cost.

Mitigation:
- Enforce maximum chunk size during ingestion.
- Limit total assembled context size.
- Truncate low-relevance chunks if necessary.

---

### 4) Overengineering During MVP

Risk:
- Introducing advanced indexing tuning prematurely.
- Adding unnecessary framework abstractions.
- Expanding scope beyond evaluation requirements.

Mitigation:
- Prioritize passing MVP requirements first.
- Optimize only after measurable issues are observed.

These risks are acknowledged to maintain development focus and reduce time lost to avoidable friction.

