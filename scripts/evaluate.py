"""Evaluation harness: Precision@5 and latency measurement."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.retrieval import retrieve

# Test queries with expected source files (ground truth)
EVAL_QUERIES = [
    {
        "query": "What does DGESV do?",
        "expected_files": ["dgesv.f"],
    },
    {
        "query": "How does LU decomposition work in LAPACK?",
        "expected_files": ["dgetrf.f", "dgetrf2.f", "dgesv.f"],
    },
    {
        "query": "What is the DGEMM routine?",
        "expected_files": ["dgemm.f"],
    },
    {
        "query": "How does LAPACK compute eigenvalues?",
        "expected_files": ["dgeev.f", "dsyev.f"],
    },
    {
        "query": "What does DGETRF do?",
        "expected_files": ["dgetrf.f", "dgetrf2.f"],
    },
    {
        "query": "How does singular value decomposition work?",
        "expected_files": ["dgesvd.f", "dgesdd.f"],
    },
    {
        "query": "What is the BLAS DAXPY operation?",
        "expected_files": ["daxpy.f"],
    },
    {
        "query": "How does LAPACK solve least squares problems?",
        "expected_files": ["dgels.f", "dgelss.f", "dgelsd.f"],
    },
    {
        "query": "What does DTRSM do?",
        "expected_files": ["dtrsm.f"],
    },
    {
        "query": "How does Cholesky factorization work in LAPACK?",
        "expected_files": ["dpotrf.f", "dpotrf2.f"],
    },
    {
        "query": "What is DNRM2?",
        "expected_files": ["dnrm2.f90"],
    },
    {
        "query": "How does LAPACK handle workspace queries?",
        "expected_files": ["dgeev.f", "dgesdd.f", "dsyev.f", "dgeqrf.f", "dgeqp3.f"],
    },
    {
        "query": "What does DSCAL do?",
        "expected_files": ["dscal.f"],
    },
    {
        "query": "How does QR factorization work?",
        "expected_files": ["dgeqrf.f", "dgeqr2.f"],
    },
    {
        "query": "What is the IDAMAX function?",
        "expected_files": ["idamax.f"],
    },
]


def compute_recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute Recall@K: fraction of expected files found in top-K results."""
    if not expected_files:
        return 1.0
    top_k = retrieved_files[:k]
    found = sum(1 for exp in expected_files if exp in top_k)
    return found / len(expected_files)


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
        chunks = retrieve(query, top_k=5)
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
