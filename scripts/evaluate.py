"""Evaluation harness: Precision@5 and latency measurement."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval_data import EVAL_QUERIES, compute_recall_at_k
from app.services.retrieval import retrieve


def main():
    results = []
    total_p5 = 0.0
    total_latency = 0.0

    print(f"{'Query':<50} {'R@5':>6} {'Latency':>10} {'Files Retrieved'}")
    print("-" * 110)

    for eval_item in EVAL_QUERIES:
        query = eval_item["query"]
        expected = eval_item["expected_files"]

        start = time.time()
        result = retrieve(query, top_k=5)
        chunks = result["chunks"]
        latency = (time.time() - start) * 1000

        retrieved_files = []
        for chunk in chunks:
            fp = chunk.get("metadata", {}).get("file_path", "")
            if fp:
                fname = Path(fp).name
                if fname not in retrieved_files:
                    retrieved_files.append(fname)

        p5 = compute_recall_at_k(retrieved_files, expected, k=5)
        total_p5 += p5
        total_latency += latency

        short_query = query[:48]
        files_str = ", ".join(retrieved_files[:5])
        print(f"{short_query:<50} {p5:>6.2f} {latency:>8.0f}ms {files_str}")

        results.append({
            "query": query,
            "expected_files": expected,
            "retrieved_files": retrieved_files,
            "recall_at_5": p5,
            "latency_ms": round(latency, 1),
        })

    n = len(EVAL_QUERIES)
    avg_p5 = total_p5 / n
    avg_latency = total_latency / n

    print("-" * 110)
    print(f"{'AVERAGE':<50} {avg_p5:>6.2f} {avg_latency:>8.0f}ms")
    print(f"\nRecall@5: {avg_p5:.3f} (target: > 0.7)")
    print(f"Avg Latency: {avg_latency:.0f}ms (target: < 3000ms)")

    # Save JSON results
    output = {
        "queries": results,
        "summary": {
            "avg_recall_at_5": round(avg_p5, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "total_queries": n,
        },
    }

    output_path = Path("logs/eval_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
