"""Eval stream generators extracted from main.py."""

import asyncio
import logging
import time
from pathlib import Path

from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES, compute_recall_at_k, compute_precision_at_k, check_e2e_result
from app.services.retrieval import retrieve
from app.services.generation import generate_answer
from app.sse import sse_event as _sse_event

logger = logging.getLogger(__name__)

_RETRIEVAL_BATCH_SIZE = 5
_E2E_MAX_TOKENS = 2048
_E2E_CONTEXT_BUDGET = 3000


async def eval_stream_generator(model: str | None = None, embedding_model: str | None = None):
    """Async generator — batch retrieval evals (no LLM generation, safe to parallelize)."""
    total_recall = 0.0
    total_precision = 0.0
    total_latency = 0.0
    n = len(EVAL_QUERIES)

    # Derive collection name from embedding model if provided
    collection_name = None
    if embedding_model:
        from app.embedding_registry import collection_name_for_model
        collection_name = collection_name_for_model("lapack", embedding_model)

    for batch_start in range(0, n, _RETRIEVAL_BATCH_SIZE):
        batch = list(enumerate(EVAL_QUERIES))[batch_start:batch_start + _RETRIEVAL_BATCH_SIZE]

        async def run_one(i, item):
            query = item["query"]
            expected = item["expected_files"]
            capability = item.get("capability")
            t0 = time.time()
            result = await retrieve(query, top_k=5, model=model, capability=capability,
                                    collection_name=collection_name,
                                    embedding_model=embedding_model)
            latency_ms = round((time.time() - t0) * 1000, 1)
            retrieved_files = []
            seen = set()
            for chunk in result["chunks"]:
                fp = chunk.get("metadata", {}).get("file_path", "")
                if fp:
                    fname = Path(fp).name
                    if fname not in seen:
                        retrieved_files.append(fname)
                        seen.add(fname)
            recall = compute_recall_at_k(retrieved_files, expected, k=5)
            precision = compute_precision_at_k(retrieved_files, expected, k=5)
            return {
                "index": i, "query": query,
                "capability": capability,
                "precision_at_5": round(precision, 4),
                "recall_at_5": round(recall, 4), "latency_ms": latency_ms,
                "retrieved_files": retrieved_files[:5], "expected_files": expected,
            }

        results = await asyncio.gather(*[run_one(i, item) for i, item in batch])
        for r in sorted(results, key=lambda x: x["index"]):
            total_recall += r["recall_at_5"]
            total_precision += r["precision_at_5"]
            total_latency += r["latency_ms"]
            yield _sse_event("progress", r)

    yield _sse_event("summary", {
        "avg_precision_at_5": round(total_precision / n, 4) if n else 0,
        "avg_recall_at_5": round(total_recall / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
        "total_queries": n,
    })


async def e2e_eval_stream_generator(model: str | None = None):
    """Async generator — batch retrieval, then generate sequentially.

    Retrieval is fast (Qdrant + embed) and safe to fully parallelize.
    Generation runs one at a time with reduced token limits and a
    dedicated client (max_retries=1) to avoid hidden backoff delays.
    """
    n = len(E2E_EVAL_QUERIES)
    total_passed = 0
    total_latency = 0.0

    # Phase 1: batch all retrievals concurrently (fast)
    async def run_retrieval(i, item):
        return i, item, await retrieve(item["query"], top_k=5, model=model,
                                       capability=item.get("capability"))

    retrieval_results = await asyncio.gather(*[
        run_retrieval(i, item) for i, item in enumerate(E2E_EVAL_QUERIES)
    ])

    # Phase 2: generate sequentially with reduced limits
    for i, item, retrieval_result in sorted(retrieval_results, key=lambda x: x[0]):
        query = item["query"]
        capability = item.get("capability")
        checks = item["checks"]
        chunks = retrieval_result["chunks"]

        t0 = time.time()
        gen_result = await generate_answer(
            query, chunks, capability,
            max_completion_tokens=_E2E_MAX_TOKENS,
            context_budget=_E2E_CONTEXT_BUDGET,
            model=model,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        check_results = check_e2e_result(
            gen_result["answer"], gen_result["citations"], checks,
        )

        if check_results["pass"]:
            total_passed += 1
        total_latency += latency_ms

        yield _sse_event("progress", {
            "index": i,
            "query": query,
            "capability": capability,
            "passed": check_results["pass"],
            "checks": check_results,
            "latency_ms": latency_ms,
            "citations": gen_result["citations"],
            "answer_preview": gen_result["answer"][:500],
            "answer_length": len(gen_result["answer"]),
        })

    yield _sse_event("summary", {
        "total_queries": n,
        "passed": total_passed,
        "failed": n - total_passed,
        "pass_rate": round(total_passed / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
    })
