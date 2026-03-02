"""Evaluation queries and metrics shared between the API and CLI harness."""

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
