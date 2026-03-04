# LegacyLens – Project Requirements (Week 3)

This document converts the official assignment specification into structured Markdown format for reference during development.

---

## 1. Project Overview

You are required to build a Retrieval-Augmented Generation (RAG) application capable of translating legacy code (COBOL, Fortran, or similar) into:

- Clear semantic explanations
- Structured pseudocode

The system must ingest a large legacy codebase and support semantic search, contextual retrieval, and citation-backed answer generation.

---

## 2. Dataset Requirements

Your selected codebase must:

- Contain **at least 10,000 lines of code**
- Contain **at least 50 files**
- Be fully indexed (100% ingestion coverage)

You may use open-source legacy repositories.

---

## 3. MVP Requirements (24-Hour Gate)

To pass MVP, the system must include:

1. Ingestion of the selected legacy codebase
2. Syntax-aware chunking (e.g., function, paragraph, or program-unit boundaries)
3. Embedding generation for all chunks
4. Storage of embeddings in a vector database
5. Semantic similarity search
6. Natural language query interface (CLI or web)
7. Return of relevant code snippets with file and line references
8. Basic answer generation using retrieved context
9. Deployed and publicly accessible application

Failure to meet any of the above results in failing the MVP gate.

---

## 4. Required System Capabilities

Your final system must demonstrate at least **four (4) code-understanding capabilities**, such as:

- Code explanation (semantic + pseudocode)
- Dependency mapping (what calls what)
- Pattern detection
- Impact analysis (what breaks if X changes)
- Business rule extraction
- Documentation generation

Capabilities must operate using retrieval + generation, not static hardcoded rules.

---

## 5. Architecture Requirements

Your submission must include documented decisions for:

- Vector database selection
- Embedding model selection
- Chunking strategy
- Retrieval pipeline design
- Answer generation approach
- Deployment architecture

You must justify tradeoffs, not merely list tools.

---

## 6. Retrieval & Evaluation Standards

### Retrieval Expectations

- High-quality semantic retrieval
- Relevant chunks appearing within top-k results
- Correct file and line references

### Evaluation Metrics

You must evaluate retrieval performance using measurable metrics such as:

- Precision@5 (percentage of relevant chunks within top 5 results)
- End-to-end latency

Target expectations:

- >70% relevant chunks in top 5
- <3 seconds end-to-end response time

You must construct a manual evaluation set (approximately 15+ realistic queries).

---

## 7. Performance Requirements

- Ingest 10,000+ LOC in under 5 minutes
- End-to-end query latency under 3 seconds
- Full repository indexing

Performance should be measured and reported.

---

## 8. Observability Requirements

Your system must log and track:

- Queries
- Retrieved chunk IDs
- Similarity scores
- Latency
- Token usage
- Failure cases

Structured logging is recommended.

---

## 9. Cost Analysis Requirement

You must provide cost analysis projections for:

- 100 users
- 1,000 users
- 10,000 users
- 100,000 users

Include projected costs for:

- Embeddings
- LLM generation
- Vector database hosting

Clearly document assumptions (queries per user, token usage, etc.).

---

## 10. Deployment Requirement

The MVP must be:

- Deployed
- Publicly accessible
- Reproducible via documented setup steps

Secrets must not be committed to version control.

---

## 11. Submission Requirements

Your final submission must include:

- Working deployed application
- Completed Pre-Search document
- Architecture justification
- Evaluation results
- Cost projections
- Clean repository with reproducible setup

---

## 12. Scoring Emphasis

Evaluation prioritizes:

- Retrieval quality
- Citation correctness
- Architectural reasoning
- Measured performance
- Engineering maturity

Not prioritized:

- UI polish
- Overengineering
- Excessive abstraction layers

---

## 13. Common Failure Modes

Projects commonly fail due to:

- Poor chunking
- Weak retrieval quality
- No evaluation metrics
- No deployment
- Hallucinated explanations without citations
- Overreliance on frameworks without understanding mechanics

Avoid these pitfalls.

---

## 14. Recommended Build Order

1. Select dataset
2. Implement ingestion
3. Implement chunking
4. Store embeddings in vector DB
5. Implement semantic retrieval
6. Implement CLI query interface
7. Add answer generation
8. Add evaluation harness
9. Deploy
10. Add cost modeling and documentation

---

End of Requirements Document.

