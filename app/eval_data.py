"""Evaluation queries and metrics shared between the API and CLI harness."""

# --- Retrieval evals: test that the right files are retrieved per capability domain ---

EVAL_QUERIES = [
    # === General (no capability) ===
    {"query": "What does DGESV do?", "expected_files": ["dgesv.f"], "capability": None},
    {"query": "How does LU decomposition work in LAPACK?", "expected_files": ["dgetrf.f", "dgetrf2.f", "dgesv.f"], "capability": None},
    {"query": "What is the DGEMM routine?", "expected_files": ["dgemm.f"], "capability": None},
    {"query": "How does LAPACK compute eigenvalues?", "expected_files": ["dgeev.f", "dsyev.f"], "capability": None},
    {"query": "What does DGETRF do?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": None},
    {"query": "How does singular value decomposition work?", "expected_files": ["dgesvd.f", "dgesdd.f"], "capability": None},
    {"query": "What is the BLAS DAXPY operation?", "expected_files": ["daxpy.f"], "capability": None},

    # === Explain Code ===
    {"query": "Explain the algorithm in DGESV step by step", "expected_files": ["dgesv.f"], "capability": "explain_code"},
    {"query": "Explain what DLANGE computes and how", "expected_files": ["dlange.f"], "capability": "explain_code"},
    {"query": "Explain the blocked algorithm in DGETRF", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "explain_code"},
    {"query": "Explain how DGEQRF performs QR factorization", "expected_files": ["dgeqrf.f", "dgeqr2.f"], "capability": "explain_code"},
    {"query": "Explain the DGEMM matrix multiplication algorithm", "expected_files": ["dgemm.f"], "capability": "explain_code"},

    # === Generate Docs ===
    {"query": "Generate documentation for the DPOTRF routine", "expected_files": ["dpotrf.f", "dpotrf2.f"], "capability": "generate_docs"},
    {"query": "Generate documentation for DGESVD", "expected_files": ["dgesvd.f"], "capability": "generate_docs"},
    {"query": "Generate documentation for the DTRSM routine", "expected_files": ["dtrsm.f"], "capability": "generate_docs"},
    {"query": "Generate documentation for DSYEV", "expected_files": ["dsyev.f"], "capability": "generate_docs"},
    {"query": "Generate documentation for DGELS", "expected_files": ["dgels.f"], "capability": "generate_docs"},

    # === Detect Patterns ===
    {"query": "What programming patterns are used in DGESV?", "expected_files": ["dgesv.f"], "capability": "detect_patterns"},
    {"query": "What numerical patterns does DGEEV use?", "expected_files": ["dgeev.f"], "capability": "detect_patterns"},
    {"query": "Identify workspace query patterns in DSYEV", "expected_files": ["dsyev.f"], "capability": "detect_patterns"},
    {"query": "What error checking patterns does DGETRF use?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "detect_patterns"},
    {"query": "Detect loop and blocking patterns in DGEMM", "expected_files": ["dgemm.f"], "capability": "detect_patterns"},

    # === Map Dependencies ===
    {"query": "What routines does DGESV call?", "expected_files": ["dgesv.f", "dgetrf.f", "dgetrs.f"], "capability": "map_dependencies"},
    {"query": "Map the call dependencies of DGEEV", "expected_files": ["dgeev.f"], "capability": "map_dependencies"},
    {"query": "What BLAS routines does DGETRF depend on?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "map_dependencies"},
    {"query": "What routines does DGESVD call?", "expected_files": ["dgesvd.f"], "capability": "map_dependencies"},
    {"query": "Map the dependency chain of DGELS", "expected_files": ["dgels.f"], "capability": "map_dependencies"},

    # === Impact Analysis ===
    {"query": "What breaks if DGETRF is changed?", "expected_files": ["dgetrf.f", "dgesv.f"], "capability": "impact_analysis"},
    {"query": "What is the impact of modifying DGEMM?", "expected_files": ["dgemm.f"], "capability": "impact_analysis"},
    {"query": "What routines are affected if DLASSQ changes?", "expected_files": ["dlassq.f90"], "capability": "impact_analysis"},
    {"query": "What depends on DTRSM in LAPACK?", "expected_files": ["dtrsm.f"], "capability": "impact_analysis"},
    {"query": "What is the blast radius of changing DSCAL?", "expected_files": ["dscal.f"], "capability": "impact_analysis"},

    # === Extract Business Rules ===
    {"query": "What validation rules does DGESV enforce on its inputs?", "expected_files": ["dgesv.f"], "capability": "extract_business_rules"},
    {"query": "What are the convergence criteria in DGEEV?", "expected_files": ["dgeev.f"], "capability": "extract_business_rules"},
    {"query": "What workspace size rules does DSYEV use?", "expected_files": ["dsyev.f"], "capability": "extract_business_rules"},
    {"query": "What numerical guards does DLANGE implement?", "expected_files": ["dlange.f"], "capability": "extract_business_rules"},
    {"query": "What parameter constraints does DGEMM enforce?", "expected_files": ["dgemm.f"], "capability": "extract_business_rules"},
]

# --- E2E evals: test full retrieve+generate pipeline with quality checks ---

E2E_EVAL_QUERIES = [
    # === General ===
    {
        "query": "What does DGESV do?",
        "capability": None,
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["linear", "system", "solve"]},
    },
    {
        "query": "How does singular value decomposition work in LAPACK?",
        "capability": None,
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["singular", "value"]},
    },
    {
        "query": "What is the DGEMM routine?",
        "capability": None,
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["matrix", "multipl"]},
    },

    # === Explain Code ===
    {
        "query": "Explain the algorithm in DGETRF step by step",
        "capability": "explain_code",
        "checks": {"has_citations": True, "min_answer_length": 100, "expected_keywords": ["factor", "pivot"]},
    },
    {
        "query": "Explain what DLANGE computes and how",
        "capability": "explain_code",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["norm", "matrix"]},
    },
    {
        "query": "Explain the DGEMM matrix multiplication algorithm",
        "capability": "explain_code",
        "checks": {"has_citations": True, "min_answer_length": 100, "expected_keywords": ["matrix", "multipl"]},
    },

    # === Generate Docs ===
    {
        "query": "Generate documentation for DPOTRF",
        "capability": "generate_docs",
        "checks": {"has_citations": True, "min_answer_length": 100, "expected_keywords": ["Cholesky", "symmetric"]},
    },
    {
        "query": "Generate documentation for DGESVD",
        "capability": "generate_docs",
        "checks": {"has_citations": True, "min_answer_length": 100, "expected_keywords": ["singular", "value"]},
    },
    {
        "query": "Generate documentation for DGELS",
        "capability": "generate_docs",
        "checks": {"has_citations": True, "min_answer_length": 100, "expected_keywords": ["least", "square"]},
    },

    # === Detect Patterns ===
    {
        "query": "What programming patterns are used in DGESV?",
        "capability": "detect_patterns",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["error", "check"]},
    },
    {
        "query": "Identify workspace query patterns in DSYEV",
        "capability": "detect_patterns",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["workspace", "LWORK"]},
    },
    {
        "query": "Detect loop and blocking patterns in DGEMM",
        "capability": "detect_patterns",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["loop", "block"]},
    },

    # === Map Dependencies ===
    {
        "query": "What routines does DGESV call?",
        "capability": "map_dependencies",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["DGETRF", "DGETRS"]},
    },
    {
        "query": "What BLAS routines does DGETRF depend on?",
        "capability": "map_dependencies",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["DGEMM", "DTRSM"]},
    },
    {
        "query": "Map the dependency chain of DGELS",
        "capability": "map_dependencies",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["DGEQRF", "DTRSM"]},
    },

    # === Impact Analysis ===
    {
        "query": "What breaks if DGETRF is changed?",
        "capability": "impact_analysis",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["DGESV"]},
    },
    {
        "query": "What is the impact of modifying DGEMM?",
        "capability": "impact_analysis",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["performance", "routine"]},
    },
    {
        "query": "What depends on DTRSM in LAPACK?",
        "capability": "impact_analysis",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["triangular", "solve"]},
    },

    # === Extract Business Rules ===
    {
        "query": "What validation rules does DGESV enforce on its inputs?",
        "capability": "extract_business_rules",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["INFO", "N"]},
    },
    {
        "query": "What workspace size rules does DSYEV use?",
        "capability": "extract_business_rules",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["LWORK", "workspace"]},
    },
    {
        "query": "What parameter constraints does DGEMM enforce?",
        "capability": "extract_business_rules",
        "checks": {"has_citations": True, "min_answer_length": 50, "expected_keywords": ["dimension", "LDA"]},
    },
]


def compute_precision_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute Precision@K: fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_files[:k]
    found = sum(1 for f in top_k if f in expected_files)
    return found / k


def compute_recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute Recall@K: fraction of expected files found in top-K results."""
    if not expected_files:
        return 1.0
    top_k = retrieved_files[:k]
    found = sum(1 for exp in expected_files if exp in top_k)
    return found / len(expected_files)


def check_e2e_result(answer: str, citations: list[str], checks: dict) -> dict:
    """Evaluate an e2e generation result against quality checks.

    Returns dict with per-check pass/fail and an overall pass boolean.
    """
    results = {}
    answer_lower = answer.lower()

    # Detect refusal / cop-out answers
    _REFUSAL_PHRASES = [
        "insufficient context",
        "i don't know",
        "i do not know",
        "i cannot answer",
        "i can't answer",
        "no relevant context",
        "no lapack source",
        "not enough context",
        "cannot determine",
        "don't have sufficient context",
    ]
    results["not_refusal"] = not any(p in answer_lower for p in _REFUSAL_PHRASES)

    if checks.get("has_citations"):
        results["has_citations"] = len(citations) > 0

    if "min_answer_length" in checks:
        results["min_answer_length"] = len(answer) >= checks["min_answer_length"]

    if "expected_keywords" in checks:
        found = [kw for kw in checks["expected_keywords"] if kw.lower() in answer_lower]
        results["keywords_found"] = found
        results["keywords_pass"] = len(found) >= 1

    results["pass"] = all(
        v for k, v in results.items()
        if k not in ("keywords_found",) and isinstance(v, bool)
    )
    return results
