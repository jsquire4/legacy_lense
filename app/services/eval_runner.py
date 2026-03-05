"""Eval stream generators extracted from main.py."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path

from app.eval_data import (
    EVAL_QUERIES,
    E2E_EVAL_QUERIES,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_max_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_negative_oracle_penalty,
    check_e2e_result,
)
from app.services.retrieval import retrieve
from app.services.generation import generate_answer, _extract_citations_from_text
from app.sse import sse_event as _sse_event

logger = logging.getLogger(__name__)

_RETRIEVAL_BATCH_SIZE = 15
_E2E_MAX_TOKENS = 2048
_E2E_CONTEXT_BUDGET = 3000


async def eval_stream_generator(model: str | None = None, embedding_model: str | None = None):
    """Async generator — batch retrieval evals (no LLM generation, safe to parallelize)."""
    total_recall = 0.0
    total_precision = 0.0
    total_max_precision = 0.0
    total_latency = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_negative_oracle_pass = 0
    n = len(EVAL_QUERIES)

    # Per-difficulty accumulators
    by_difficulty: dict[str, dict] = defaultdict(lambda: {
        "total_recall_at_5": 0.0, "total_mrr": 0.0, "count": 0,
    })

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
            difficulty = item.get("difficulty", "unknown")
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
            recall_5 = compute_recall_at_k(retrieved_files, expected, k=5)
            precision_5 = compute_precision_at_k(retrieved_files, expected, k=5)
            max_precision_5 = compute_max_precision_at_k(expected, k=5)
            mrr = compute_mrr(retrieved_files, expected, k=5)
            ndcg_5 = compute_ndcg_at_k(retrieved_files, expected, k=5)
            neg_oracle = compute_negative_oracle_penalty(retrieved_files, expected, k=5)
            return {
                "index": i, "query": query,
                "capability": capability,
                "difficulty": difficulty,
                "precision_at_5": round(precision_5, 4),
                "max_precision_at_5": round(max_precision_5, 4),
                "precision_at_3": round(compute_precision_at_k(retrieved_files, expected, k=3), 4),
                "precision_at_1": round(compute_precision_at_k(retrieved_files, expected, k=1), 4),
                "recall_at_5": round(recall_5, 4),
                "recall_at_3": round(compute_recall_at_k(retrieved_files, expected, k=3), 4),
                "recall_at_1": round(compute_recall_at_k(retrieved_files, expected, k=1), 4),
                "mrr": round(mrr, 4),
                "ndcg_at_5": round(ndcg_5, 4),
                "negative_oracle_pass": neg_oracle,
                "latency_ms": latency_ms,
                "retrieved_files": retrieved_files[:5], "expected_files": expected,
            }

        results = await asyncio.gather(*[run_one(i, item) for i, item in batch])
        for r in sorted(results, key=lambda x: x["index"]):
            total_recall += r["recall_at_5"]
            total_precision += r["precision_at_5"]
            total_max_precision += r["max_precision_at_5"]
            total_latency += r["latency_ms"]
            total_mrr += r["mrr"]
            total_ndcg += r["ndcg_at_5"]
            if r["negative_oracle_pass"]:
                total_negative_oracle_pass += 1

            diff = r["difficulty"]
            by_difficulty[diff]["total_recall_at_5"] += r["recall_at_5"]
            by_difficulty[diff]["total_mrr"] += r["mrr"]
            by_difficulty[diff]["count"] += 1

            yield _sse_event("progress", r)

    # Build by_difficulty summary
    difficulty_summary = {}
    for diff, acc in by_difficulty.items():
        c = acc["count"]
        difficulty_summary[diff] = {
            "avg_recall_at_5": round(acc["total_recall_at_5"] / c, 4) if c else 0,
            "avg_mrr": round(acc["total_mrr"] / c, 4) if c else 0,
            "count": c,
        }

    yield _sse_event("summary", {
        "avg_precision_at_5": round(total_precision / n, 4) if n else 0,
        "avg_max_precision_at_5": round(total_max_precision / n, 4) if n else 0,
        "avg_recall_at_5": round(total_recall / n, 4) if n else 0,
        "avg_mrr": round(total_mrr / n, 4) if n else 0,
        "avg_ndcg_at_5": round(total_ndcg / n, 4) if n else 0,
        "negative_oracle_pass_rate": round(total_negative_oracle_pass / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
        "total_queries": n,
        "by_difficulty": difficulty_summary,
    })


async def e2e_eval_stream_generator(model: str | None = None, embedding_model: str | None = None):
    """Async generator — batch retrieval, then generate sequentially.

    Retrieval is fast (Qdrant + embed) and safe to fully parallelize.
    Generation runs one at a time with reduced token limits and a
    dedicated client (max_retries=1) to avoid hidden backoff delays.
    """
    n = len(E2E_EVAL_QUERIES)
    total_passed = 0
    total_latency = 0.0
    total_similarity = 0.0
    similarity_count = 0
    hallucination_probe_total = 0
    hallucination_probe_passed = 0
    citation_fallback_count = 0

    collection_name = None
    if embedding_model:
        from app.embedding_registry import collection_name_for_model
        collection_name = collection_name_for_model("lapack", embedding_model)

    # Phase 1: batch all retrievals concurrently (fast)
    async def run_retrieval(i, item):
        return i, item, await retrieve(item["query"], top_k=5, model=model,
                                       capability=item.get("capability"),
                                       collection_name=collection_name,
                                       embedding_model=embedding_model)

    retrieval_results = await asyncio.gather(*[
        run_retrieval(i, item) for i, item in enumerate(E2E_EVAL_QUERIES)
    ])

    # Phase 2: generate sequentially with reduced limits
    for i, item, retrieval_result in sorted(retrieval_results, key=lambda x: x[0]):
        query = item["query"]
        capability = item.get("capability")
        checks = item["checks"]
        chunks = retrieval_result["chunks"]
        is_probe = item.get("is_hallucination_probe", False)
        expected_files = item.get("expected_files")
        golden_answer = item.get("golden_answer", "")

        if is_probe:
            hallucination_probe_total += 1

        t0 = time.time()
        try:
            gen_result = await generate_answer(
                query, chunks, capability,
                max_completion_tokens=_E2E_MAX_TOKENS,
                context_budget=_E2E_CONTEXT_BUDGET,
                model=model,
            )
        except Exception as e:
            logger.error("E2E eval generation failed for query %d: %s", i, e)
            yield _sse_event("progress", {
                "index": i,
                "query": query,
                "capability": capability,
                "passed": False,
                "checks": {"pass": False, "error": str(e)},
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "citations": [],
                "answer_preview": f"[Generation error: {e}]",
                "answer_length": 0,
                "is_hallucination_probe": is_probe,
            })
            total_latency += round((time.time() - t0) * 1000, 1)
            continue
        latency_ms = round((time.time() - t0) * 1000, 1)

        answer = gen_result["answer"]
        citations = gen_result["citations"]

        # Embed golden answer and generated answer for similarity comparison
        answer_embedding = None
        golden_embedding = None
        if golden_answer:
            try:
                from app.services.embeddings import embed_query
                answer_emb_task = embed_query(answer)
                golden_emb_task = embed_query(golden_answer)
                answer_embedding, golden_embedding = await asyncio.gather(
                    answer_emb_task, golden_emb_task,
                )
            except Exception as e:
                logger.warning("Embedding for similarity failed: %s", e)

        # Detect citation fallback
        text_citations = _extract_citations_from_text(answer)
        citation_is_fallback = len(text_citations) == 0 and len(citations) > 0

        check_results = check_e2e_result(
            answer, citations, checks,
            answer_embedding=answer_embedding,
            golden_embedding=golden_embedding,
            expected_files=expected_files,
            citation_is_fallback=citation_is_fallback,
        )

        if check_results["pass"]:
            total_passed += 1
            if is_probe:
                hallucination_probe_passed += 1
        total_latency += latency_ms

        if citation_is_fallback:
            citation_fallback_count += 1

        # Track similarity
        sim_score = check_results.get("similarity_score")
        if sim_score is not None:
            total_similarity += sim_score
            similarity_count += 1

        progress_data = {
            "index": i,
            "query": query,
            "capability": capability,
            "passed": check_results["pass"],
            "checks": check_results,
            "latency_ms": latency_ms,
            "citations": citations,
            "answer_preview": answer[:500],
            "answer_length": len(answer),
            "is_hallucination_probe": is_probe,
            "citation_is_fallback": citation_is_fallback,
        }
        if sim_score is not None:
            progress_data["similarity_score"] = sim_score

        yield _sse_event("progress", progress_data)

    yield _sse_event("summary", {
        "total_queries": n,
        "passed": total_passed,
        "failed": n - total_passed,
        "pass_rate": round(total_passed / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
        "avg_similarity": round(total_similarity / similarity_count, 4) if similarity_count else 0,
        "hallucination_probe_pass_rate": round(
            hallucination_probe_passed / hallucination_probe_total, 4,
        ) if hallucination_probe_total else 0,
        "citation_fallback_count": citation_fallback_count,
    })
