"""Evaluation harness: Precision@5, Recall@5, and latency measurement."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval_data import EVAL_QUERIES, compute_recall_at_k, compute_precision_at_k
from app.services.retrieval import retrieve


async def run_eval(embedding_model: str | None = None, collection_name: str | None = None) -> dict:
    """Run retrieval evaluation and return results dict."""
    results = []
    total_p5 = 0.0
    total_r5 = 0.0
    total_latency = 0.0

    print(f"{'Query':<50} {'P@5':>6} {'R@5':>6} {'Latency':>10} {'Files Retrieved'}")
    print("-" * 120)

    for eval_item in EVAL_QUERIES:
        query = eval_item["query"]
        expected = eval_item["expected_files"]
        capability = eval_item.get("capability")

        start = time.time()
        result = await retrieve(query, top_k=5, capability=capability,
                                collection_name=collection_name,
                                embedding_model=embedding_model)
        chunks = result["chunks"]
        latency = (time.time() - start) * 1000

        retrieved_files = []
        for chunk in chunks:
            fp = chunk.get("metadata", {}).get("file_path", "")
            if fp:
                fname = Path(fp).name
                if fname not in retrieved_files:
                    retrieved_files.append(fname)

        p5 = compute_precision_at_k(retrieved_files, expected, k=5)
        r5 = compute_recall_at_k(retrieved_files, expected, k=5)
        total_p5 += p5
        total_r5 += r5
        total_latency += latency

        short_query = query[:48]
        files_str = ", ".join(retrieved_files[:5])
        print(f"{short_query:<50} {p5:>6.2f} {r5:>6.2f} {latency:>8.0f}ms {files_str}")

        results.append({
            "query": query,
            "capability": capability,
            "expected_files": expected,
            "retrieved_files": retrieved_files,
            "precision_at_5": round(p5, 4),
            "recall_at_5": round(r5, 4),
            "latency_ms": round(latency, 1),
        })

    n = len(EVAL_QUERIES)
    avg_p5 = total_p5 / n if n else 0.0
    avg_r5 = total_r5 / n if n else 0.0
    avg_latency = total_latency / n if n else 0.0

    print("-" * 120)
    print(f"{'AVERAGE':<50} {avg_p5:>6.2f} {avg_r5:>6.2f} {avg_latency:>8.0f}ms")
    print(f"\nPrecision@5: {avg_p5:.3f}")
    print(f"Recall@5: {avg_r5:.3f} (target: > 0.7)")
    print(f"Avg Latency: {avg_latency:.0f}ms (target: < 3000ms)")

    return {
        "embedding_model": embedding_model or "default",
        "collection_name": collection_name or "default",
        "queries": results,
        "summary": {
            "avg_precision_at_5": round(avg_p5, 3),
            "avg_recall_at_5": round(avg_r5, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "total_queries": n,
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Run LegacyLens retrieval evaluation")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model to use (default: from settings)")
    parser.add_argument("--collection-name", type=str, default=None,
                        help="Qdrant collection name override")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run eval for all embedding models with existing collections")
    parser.add_argument("--output", type=str, default="logs/eval_results.json",
                        help="Output JSON path (default: logs/eval_results.json)")
    args = parser.parse_args()

    if args.compare_all:
        from app.embedding_registry import EMBEDDING_MODELS, collection_name_for_model
        from qdrant_client import QdrantClient
        from app.config import get_settings

        settings = get_settings()
        api_key = settings.QDRANT_API_KEY or None
        client = QdrantClient(url=settings.QDRANT_URL, api_key=api_key, timeout=30)
        existing = {c.name for c in client.get_collections().collections}

        all_results = {}
        for model_name, info in EMBEDDING_MODELS.items():
            coll = collection_name_for_model("lapack", model_name)
            if coll not in existing:
                print(f"\nSkipping {model_name} — collection '{coll}' not found")
                continue
            print(f"\n{'='*80}")
            print(f"Evaluating: {model_name} (collection: {coll})")
            print(f"{'='*80}")
            result = await run_eval(embedding_model=model_name, collection_name=coll)
            all_results[model_name] = result

        # Summary comparison table
        if all_results:
            print(f"\n{'='*80}")
            print("COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(f"{'Model':<30} {'P@5':>8} {'R@5':>8} {'Latency':>10}")
            print("-" * 60)
            for model_name, result in all_results.items():
                s = result["summary"]
                print(f"{model_name:<30} {s['avg_precision_at_5']:>8.3f} {s['avg_recall_at_5']:>8.3f} {s['avg_latency_ms']:>8.0f}ms")

        output = {"models": all_results}
    else:
        output = await run_eval(
            embedding_model=args.embedding_model,
            collection_name=args.collection_name,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
