"""Evaluation queries and metrics shared between the API and CLI harness."""

import math

import numpy as np

# --- Retrieval evals: test that the right files are retrieved per capability domain ---

EVAL_QUERIES = [
    # ==========================================================================
    # === General (no capability) ===
    # ==========================================================================

    # --- Easy ---
    {"query": "What does DGESV do?", "expected_files": ["dgesv.f"], "capability": None, "difficulty": "easy"},
    {"query": "What is the DGEMM routine?", "expected_files": ["dgemm.f"], "capability": None, "difficulty": "easy"},
    {"query": "What is the BLAS DAXPY operation?", "expected_files": ["daxpy.f"], "capability": None, "difficulty": "easy"},
    {"query": "What does DDOT compute?", "expected_files": ["ddot.f"], "capability": None, "difficulty": "easy"},
    {"query": "What does DNRM2 compute?", "expected_files": ["dnrm2.f90"], "capability": None, "difficulty": "easy"},
    {"query": "What does DGEMV do?", "expected_files": ["dgemv.f"], "capability": None, "difficulty": "easy"},
    {"query": "What does DCOPY do?", "expected_files": ["dcopy.f"], "capability": None, "difficulty": "easy"},

    # --- Medium ---
    {"query": "How does LU decomposition work in LAPACK?", "expected_files": ["dgetrf.f", "dgetrf2.f", "dgesv.f"], "capability": None, "difficulty": "medium"},
    {"query": "How does LAPACK compute eigenvalues?", "expected_files": ["dgeev.f", "dsyev.f"], "capability": None, "difficulty": "medium"},
    {"query": "How does singular value decomposition work?", "expected_files": ["dgesvd.f", "dgesdd.f"], "capability": None, "difficulty": "medium"},
    {"query": "How does LAPACK solve banded linear systems?", "expected_files": ["dgbsv.f"], "capability": None, "difficulty": "medium"},
    {"query": "How does LAPACK compute condition numbers?", "expected_files": ["dgecon.f"], "capability": None, "difficulty": "medium"},

    # --- Hard ---
    {"query": "Compare DGESVD vs DGESDD vs DGESVDX for SVD computation", "expected_files": ["dgesvd.f", "dgesdd.f", "dgesvdx.f"], "capability": None, "difficulty": "hard"},
    {"query": "Compare symmetric eigensolvers DSYEV, DSYEVD, DSYEVR, DSYEVX", "expected_files": ["dsyev.f", "dsyevd.f", "dsyevr.f", "dsyevx.f"], "capability": None, "difficulty": "hard"},
    {"query": "What is DSGESV and how does mixed-precision iterative refinement work?", "expected_files": ["dsgesv.f"], "capability": None, "difficulty": "hard"},

    # ==========================================================================
    # === Explain Code ===
    # ==========================================================================

    # --- Easy ---
    {"query": "Explain the algorithm in DGESV step by step", "expected_files": ["dgesv.f"], "capability": "explain_code", "difficulty": "easy"},
    {"query": "Explain what DLANGE computes and how", "expected_files": ["dlange.f"], "capability": "explain_code", "difficulty": "easy"},
    {"query": "Explain the DGEMM matrix multiplication algorithm", "expected_files": ["dgemm.f"], "capability": "explain_code", "difficulty": "easy"},
    {"query": "Explain how DSCAL scales a vector", "expected_files": ["dscal.f"], "capability": "explain_code", "difficulty": "easy"},
    {"query": "Explain what DSWAP does", "expected_files": ["dswap.f"], "capability": "explain_code", "difficulty": "easy"},

    # --- Medium ---
    {"query": "Explain the blocked algorithm in DGETRF", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "explain_code", "difficulty": "medium"},
    {"query": "Explain how DGEQRF performs QR factorization", "expected_files": ["dgeqrf.f", "dgeqr2.f"], "capability": "explain_code", "difficulty": "medium"},
    {"query": "Explain how DGEES computes the Schur decomposition", "expected_files": ["dgees.f"], "capability": "explain_code", "difficulty": "medium"},
    {"query": "Explain how DGEHRD reduces a matrix to upper Hessenberg form", "expected_files": ["dgehrd.f"], "capability": "explain_code", "difficulty": "medium"},

    # --- Hard ---
    {"query": "Explain the differences between DGEQRF and DGEQRT compact WY representations", "expected_files": ["dgeqrf.f", "dgeqrt.f"], "capability": "explain_code", "difficulty": "hard"},
    {"query": "Explain the eigenvalue pipeline DSYTRD → DSTEQR → DORGTR", "expected_files": ["dsytrd.f", "dsteqr.f", "dorgtr.f"], "capability": "explain_code", "difficulty": "hard"},

    # ==========================================================================
    # === Generate Docs ===
    # ==========================================================================

    # --- Easy ---
    {"query": "Generate documentation for the DPOTRF routine", "expected_files": ["dpotrf.f", "dpotrf2.f"], "capability": "generate_docs", "difficulty": "easy"},
    {"query": "Generate documentation for DGESVD", "expected_files": ["dgesvd.f"], "capability": "generate_docs", "difficulty": "easy"},
    {"query": "Generate documentation for the DTRSM routine", "expected_files": ["dtrsm.f"], "capability": "generate_docs", "difficulty": "easy"},
    {"query": "Generate documentation for DSYEV", "expected_files": ["dsyev.f"], "capability": "generate_docs", "difficulty": "easy"},
    {"query": "Generate documentation for DGELS", "expected_files": ["dgels.f"], "capability": "generate_docs", "difficulty": "easy"},

    # --- Medium ---
    {"query": "Generate documentation for DGETRS", "expected_files": ["dgetrs.f"], "capability": "generate_docs", "difficulty": "medium"},
    {"query": "Generate documentation for DGETRI", "expected_files": ["dgetri.f"], "capability": "generate_docs", "difficulty": "medium"},
    {"query": "Generate documentation for DPOTRS", "expected_files": ["dpotrs.f"], "capability": "generate_docs", "difficulty": "medium"},
    {"query": "Generate documentation for DGTSV tridiagonal solver", "expected_files": ["dgtsv.f"], "capability": "generate_docs", "difficulty": "medium"},

    # --- Hard ---
    {"query": "Generate documentation for DGELSY as the modern replacement for DGELSX", "expected_files": ["dgelsy.f"], "capability": "generate_docs", "difficulty": "hard"},
    {"query": "Generate documentation for DGGSVD3", "expected_files": ["dggsvd3.f"], "capability": "generate_docs", "difficulty": "hard"},

    # ==========================================================================
    # === Detect Patterns ===
    # ==========================================================================

    # --- Easy ---
    {"query": "What programming patterns are used in DGESV?", "expected_files": ["dgesv.f"], "capability": "detect_patterns", "difficulty": "easy"},
    {"query": "Detect loop and blocking patterns in DGEMM", "expected_files": ["dgemm.f"], "capability": "detect_patterns", "difficulty": "easy"},
    {"query": "What patterns does IDAMAX use for finding pivot elements?", "expected_files": ["idamax.f"], "capability": "detect_patterns", "difficulty": "easy"},
    {"query": "What patterns does DLASWP use for row permutation?", "expected_files": ["dlaswp.f"], "capability": "detect_patterns", "difficulty": "easy"},

    # --- Medium ---
    {"query": "What numerical patterns does DGEEV use?", "expected_files": ["dgeev.f"], "capability": "detect_patterns", "difficulty": "medium"},
    {"query": "Identify workspace query patterns in DSYEV", "expected_files": ["dsyev.f"], "capability": "detect_patterns", "difficulty": "medium"},
    {"query": "What error checking patterns does DGETRF use?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "detect_patterns", "difficulty": "medium"},
    {"query": "What convergence patterns does DHSEQR use for QR iteration?", "expected_files": ["dhseqr.f"], "capability": "detect_patterns", "difficulty": "medium"},

    # --- Hard ---
    {"query": "Compare workspace query patterns between DSYEV and DSYEVD", "expected_files": ["dsyev.f", "dsyevd.f"], "capability": "detect_patterns", "difficulty": "hard"},
    {"query": "What tuning parameter patterns does ILAENV implement?", "expected_files": ["ilaenv.f"], "capability": "detect_patterns", "difficulty": "hard"},

    # ==========================================================================
    # === Map Dependencies ===
    # ==========================================================================

    # --- Easy ---
    {"query": "What routines does DGESV call?", "expected_files": ["dgesv.f", "dgetrf.f", "dgetrs.f"], "capability": "map_dependencies", "difficulty": "easy"},
    {"query": "What routines does DGESVD call?", "expected_files": ["dgesvd.f"], "capability": "map_dependencies", "difficulty": "easy"},
    {"query": "Map the dependency chain of DGELS", "expected_files": ["dgels.f"], "capability": "map_dependencies", "difficulty": "easy"},
    {"query": "What does DROT depend on?", "expected_files": ["drot.f"], "capability": "map_dependencies", "difficulty": "easy"},

    # --- Medium ---
    {"query": "Map the call dependencies of DGEEV", "expected_files": ["dgeev.f"], "capability": "map_dependencies", "difficulty": "medium"},
    {"query": "What BLAS routines does DGETRF depend on?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "map_dependencies", "difficulty": "medium"},
    {"query": "What routines does DSYSV call?", "expected_files": ["dsysv.f"], "capability": "map_dependencies", "difficulty": "medium"},
    {"query": "What does DPBTRF depend on for banded Cholesky?", "expected_files": ["dpbtrf.f"], "capability": "map_dependencies", "difficulty": "medium"},

    # --- Hard ---
    {"query": "Map the DGEEV dependency chain through DGEHRD and DHSEQR", "expected_files": ["dgeev.f", "dgehrd.f", "dhseqr.f"], "capability": "map_dependencies", "difficulty": "hard"},
    {"query": "What is the impact chain from DLARFG through factorization routines?", "expected_files": ["dlarfg.f"], "capability": "map_dependencies", "difficulty": "hard"},

    # ==========================================================================
    # === Impact Analysis ===
    # ==========================================================================

    # --- Easy ---
    {"query": "What breaks if DGETRF is changed?", "expected_files": ["dgetrf.f", "dgesv.f"], "capability": "impact_analysis", "difficulty": "easy"},
    {"query": "What is the impact of modifying DGEMM?", "expected_files": ["dgemm.f"], "capability": "impact_analysis", "difficulty": "easy"},
    {"query": "What depends on DTRSM in LAPACK?", "expected_files": ["dtrsm.f"], "capability": "impact_analysis", "difficulty": "easy"},
    {"query": "What is the blast radius of changing DSCAL?", "expected_files": ["dscal.f"], "capability": "impact_analysis", "difficulty": "easy"},

    # --- Medium ---
    {"query": "What routines are affected if DLASSQ changes?", "expected_files": ["dlassq.f90"], "capability": "impact_analysis", "difficulty": "medium"},
    {"query": "What is the impact of changing DTRTRI on triangular inverse operations?", "expected_files": ["dtrtri.f"], "capability": "impact_analysis", "difficulty": "medium"},
    {"query": "What breaks if DGEBAL balancing is modified?", "expected_files": ["dgebal.f"], "capability": "impact_analysis", "difficulty": "medium"},
    {"query": "What is the impact of modifying DSYR2K?", "expected_files": ["dsyr2k.f"], "capability": "impact_analysis", "difficulty": "medium"},

    # --- Hard ---
    {"query": "What is the blast radius of changing DLARTG rotation generation?", "expected_files": ["dlartg.f90"], "capability": "impact_analysis", "difficulty": "hard"},
    {"query": "What breaks if condition number requires factorization via DGECON+DGETRF?", "expected_files": ["dgecon.f", "dgetrf.f"], "capability": "impact_analysis", "difficulty": "hard"},

    # ==========================================================================
    # === Extract Business Rules ===
    # ==========================================================================

    # --- Easy ---
    {"query": "What validation rules does DGESV enforce on its inputs?", "expected_files": ["dgesv.f"], "capability": "extract_business_rules", "difficulty": "easy"},
    {"query": "What parameter constraints does DGEMM enforce?", "expected_files": ["dgemm.f"], "capability": "extract_business_rules", "difficulty": "easy"},
    {"query": "What does DGETRF do?", "expected_files": ["dgetrf.f", "dgetrf2.f"], "capability": "extract_business_rules", "difficulty": "easy"},

    # --- Medium ---
    {"query": "What are the convergence criteria in DGEEV?", "expected_files": ["dgeev.f"], "capability": "extract_business_rules", "difficulty": "medium"},
    {"query": "What workspace size rules does DSYEV use?", "expected_files": ["dsyev.f"], "capability": "extract_business_rules", "difficulty": "medium"},
    {"query": "What numerical guards does DLANGE implement?", "expected_files": ["dlange.f"], "capability": "extract_business_rules", "difficulty": "medium"},
    {"query": "What constraints does DPOSV enforce on positive definite systems?", "expected_files": ["dposv.f"], "capability": "extract_business_rules", "difficulty": "medium"},
    {"query": "What convergence rules does DSTEV use for tridiagonal eigenvalues?", "expected_files": ["dstev.f"], "capability": "extract_business_rules", "difficulty": "medium"},

    # --- Hard ---
    {"query": "What are the workspace rules for DGEDMD dynamic mode decomposition?", "expected_files": ["dgedmd.f90"], "capability": "extract_business_rules", "difficulty": "hard"},
    {"query": "What rules govern DPTSV positive definite tridiagonal solving?", "expected_files": ["dptsv.f"], "capability": "extract_business_rules", "difficulty": "hard"},
]

# --- E2E evals: test full retrieve+generate pipeline with quality checks ---

E2E_EVAL_QUERIES = [
    # === General ===
    {
        "query": "What does DGESV do?",
        "capability": None,
        "expected_files": ["dgesv.f"],
        "golden_answer": "DGESV solves a system of linear equations A*X = B by computing the LU factorization of A using partial pivoting with row interchanges, then solving the factored system. It calls DGETRF for factorization and DGETRS for the back-substitution.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["linear", "system", "factorization"]},
    },
    {
        "query": "How does singular value decomposition work in LAPACK?",
        "capability": None,
        "expected_files": ["dgesvd.f", "dgesdd.f", "sgesvd.f", "sgesdd.f"],
        "golden_answer": "LAPACK computes the SVD of a matrix A = U * Sigma * V^T using DGESVD (bidiagonalization + QR iteration) or DGESDD (divide-and-conquer for faster computation). Both routines reduce A to bidiagonal form first, then compute singular values and optionally singular vectors.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["singular", "bidiagonal", "decomposition"]},
    },
    {
        "query": "What is the DGEMM routine?",
        "capability": None,
        "expected_files": ["dgemm.f"],
        "golden_answer": "DGEMM performs one of the double-precision matrix-matrix operations C := alpha*op(A)*op(B) + beta*C, where op(X) is X or X^T. It is a Level 3 BLAS routine and the computational backbone of most LAPACK factorizations.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["matrix", "alpha", "BLAS"]},
    },

    # === Explain Code ===
    {
        "query": "Explain the algorithm in DGETRF step by step",
        "capability": "explain_code",
        "expected_files": ["dgetrf.f", "dgetrf2.f"],
        "golden_answer": "DGETRF computes the LU factorization of a general M-by-N matrix using partial pivoting with row interchanges. It uses a blocked algorithm: find the pivot in the current column, swap rows via DLASWP, then update the trailing submatrix using DGEMM and DTRSM.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["pivot", "LU", "DLASWP"]},
    },
    {
        "query": "Explain what DLANGE computes and how",
        "capability": "explain_code",
        "expected_files": ["dlange.f"],
        "golden_answer": "DLANGE computes the value of a matrix norm: one-norm (max column sum), infinity-norm (max row sum), Frobenius norm (sqrt of sum of squares), or max absolute element. It dispatches to different computational paths based on the NORM parameter.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["norm", "matrix", "Frobenius"]},
    },
    {
        "query": "Explain the DGEMM matrix multiplication algorithm",
        "capability": "explain_code",
        "expected_files": ["dgemm.f"],
        "golden_answer": "DGEMM implements C := alpha*op(A)*op(B) + beta*C where op(X) is X or X^T. It handles transpose options via the TRANSA/TRANSB parameters and includes early exits when alpha is zero or beta is one.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["alpha", "TRANSA", "op"]},
    },

    # === Generate Docs ===
    {
        "query": "Generate documentation for DPOTRF",
        "capability": "generate_docs",
        "expected_files": ["dpotrf.f", "dpotrf2.f"],
        "golden_answer": "DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A, producing either A = U^T*U (upper) or A = L*L^T (lower) depending on the UPLO parameter. It uses a blocked algorithm calling DPOTRF2 for unblocked panels.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["Cholesky", "symmetric", "positive"]},
    },
    {
        "query": "Generate documentation for DGESVD",
        "capability": "generate_docs",
        "expected_files": ["dgesvd.f"],
        "golden_answer": "DGESVD computes the singular value decomposition A = U * Sigma * V^T of a real M-by-N matrix. It reduces A to bidiagonal form, then uses QR iteration to compute singular values. The JOBU and JOBVT parameters control whether singular vectors are computed.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["singular", "bidiagonal", "JOBU"]},
    },
    {
        "query": "Generate documentation for DGELS",
        "capability": "generate_docs",
        "expected_files": ["dgels.f"],
        "golden_answer": "DGELS solves overdetermined or underdetermined linear systems using QR or LQ factorization. For overdetermined systems (M>N), it computes the least squares solution; for underdetermined (M<N), the minimum norm solution. It calls DGEQRF or DGELQF for factorization.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["least", "squares", "QR"]},
    },

    # === Detect Patterns ===
    {
        "query": "What programming patterns are used in DGESV?",
        "capability": "detect_patterns",
        "expected_files": ["dgesv.f"],
        "golden_answer": "DGESV uses input parameter checking with INFO return codes, delegates to DGETRF for factorization and DGETRS for solving, and follows the LAPACK convention of in-place computation. Parameter checking validates N>=0, NRHS>=0, and LDA>=max(1,N).",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["INFO", "DGETRF", "DGETRS"]},
    },
    {
        "query": "Identify workspace query patterns in DSYEV",
        "capability": "detect_patterns",
        "expected_files": ["dsyev.f"],
        "golden_answer": "DSYEV supports a workspace query pattern: when LWORK=-1, it returns the optimal workspace size in WORK(1) without performing computation. This allows callers to allocate optimal workspace before the actual call.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["workspace", "LWORK", "optimal"]},
    },
    {
        "query": "Detect loop and blocking patterns in DGEMM",
        "capability": "detect_patterns",
        "expected_files": ["dgemm.f"],
        "golden_answer": "DGEMM uses nested loops over matrix dimensions with early-exit optimizations when alpha is zero. It handles transpose combinations (NOTA, CONJA, etc.) via conditional branching, and calls XERBLA for parameter errors. The LSAME function is used to check character arguments.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["loop", "alpha", "transpose"]},
    },

    # === Map Dependencies ===
    {
        "query": "What routines does DGESV call?",
        "capability": "map_dependencies",
        "expected_files": ["dgesv.f", "dgetrf.f", "dgetrs.f"],
        "golden_answer": "DGESV calls DGETRF to compute the LU factorization with partial pivoting, then calls DGETRS to solve the triangular systems. DGETRF in turn calls DGETRF2 for panel factorization and uses DGEMM and DTRSM for blocked updates.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["DGETRF", "DGETRS", "LU"]},
    },
    {
        "query": "What BLAS routines does DGETRF depend on?",
        "capability": "map_dependencies",
        "expected_files": ["dgetrf.f", "dgetrf2.f"],
        "golden_answer": "DGETRF depends on DGEMM and DTRSM for the blocked algorithm updates, DLASWP for row interchanges, and calls DGETRF2 for the unblocked panel factorization. DGETRF2 in turn uses IDAMAX for pivot selection and DSCAL/DGER for column updates.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["DGEMM", "DTRSM", "pivot"]},
    },
    {
        "query": "Map the dependency chain of DGELS",
        "capability": "map_dependencies",
        "expected_files": ["dgels.f"],
        "golden_answer": "DGELS calls DGEQRF for QR factorization (overdetermined case) or DGELQF for LQ factorization (underdetermined case), then DORMQR/DORMLQ to apply orthogonal transformations, and DTRTRS to solve the triangular system.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["DGEQRF", "DTRTRS", "orthogonal"]},
    },

    # === Impact Analysis ===
    {
        "query": "What breaks if DGETRF is changed?",
        "capability": "impact_analysis",
        "expected_files": ["dgetrf.f", "dgesv.f"],
        "golden_answer": "DGETRF is called by DGESV, DGESVX, DGECON, DGETRI, and many other routines that require LU factorization. Changes to DGETRF would affect all linear system solvers, condition number estimators, and matrix inverse computations.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["DGESV", "LU", "factorization"]},
    },
    {
        "query": "What is the impact of modifying DGEMM?",
        "capability": "impact_analysis",
        "expected_files": ["dgemm.f"],
        "golden_answer": "DGEMM is the most critical BLAS Level 3 routine — virtually every LAPACK factorization algorithm depends on it for trailing submatrix updates. Modifying DGEMM impacts DGETRF, DPOTRF, DGEQRF, and the performance of the entire library.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["BLAS", "factorization", "DGETRF"]},
    },
    {
        "query": "What depends on DTRSM in LAPACK?",
        "capability": "impact_analysis",
        "expected_files": ["dtrsm.f"],
        "golden_answer": "DTRSM solves triangular matrix equations of the form op(A)*X = alpha*B or X*op(A) = alpha*B. It is a Level 3 BLAS routine used by DGETRF, DGETRS, DGETRI, and many factorization routines for solving triangular systems.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["triangular", "solve", "BLAS"]},
    },

    # === Extract Business Rules ===
    {
        "query": "What validation rules does DGESV enforce on its inputs?",
        "capability": "extract_business_rules",
        "expected_files": ["dgesv.f"],
        "golden_answer": "DGESV checks: N>=0, NRHS>=0, LDA>=max(1,N), LDB>=max(1,N). On violation it sets INFO<0 with the position of the illegal argument and calls XERBLA. On successful factorization, INFO=0; if the matrix is singular, INFO>0 indicates the diagonal element that is exactly zero.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["INFO", "XERBLA", "singular"]},
    },
    {
        "query": "What workspace size rules does DSYEV use?",
        "capability": "extract_business_rules",
        "expected_files": ["dsyev.f"],
        "golden_answer": "DSYEV requires LWORK >= max(1, 3*N-1) for minimum workspace. When LWORK=-1, it performs a workspace query returning the optimal size in WORK(1). Using optimal workspace improves performance through better blocking.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["LWORK", "workspace", "optimal"]},
    },
    {
        "query": "What parameter constraints does DGEMM enforce?",
        "capability": "extract_business_rules",
        "expected_files": ["dgemm.f"],
        "golden_answer": "DGEMM validates TRANSA/TRANSB are 'N','T', or 'C'; M,N,K >= 0; LDA >= max(1,M) or max(1,K) depending on transpose; LDB >= max(1,K) or max(1,N); LDC >= max(1,M). On violation, XERBLA is called with the parameter position.",
        "checks": {"has_citations": True, "min_answer_length": 200, "expected_keywords": ["dimension", "LDA", "XERBLA"]},
    },

    # === Hallucination Probes ===
    {
        "query": "What does the DFAKE routine do?",
        "capability": None,
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
    {
        "query": "Explain the DGESVM routine",
        "capability": "explain_code",
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
    {
        "query": "Generate documentation for QGESV",
        "capability": "generate_docs",
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
    {
        "query": "What does DLASOR compute?",
        "capability": None,
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
    {
        "query": "Map the dependencies of DGEMN",
        "capability": "map_dependencies",
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
    {
        "query": "What validation rules does ZLATRS2 enforce?",
        "capability": "extract_business_rules",
        "expected_files": [],
        "golden_answer": "",
        "is_hallucination_probe": True,
        "checks": {"expect_refusal": True, "min_answer_length": 0},
    },
]


# ---------------------------------------------------------------------------
# Retrieval metric functions
# ---------------------------------------------------------------------------

def compute_precision_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute Precision@K: fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_files[:k]
    found = sum(1 for f in top_k if f in expected_files)
    return found / k


def compute_max_precision_at_k(expected_files: list[str], k: int = 5) -> float:
    """Theoretical maximum Precision@K given the number of expected files.

    When |expected| < k, it's impossible to fill all k slots with relevant files,
    so the ceiling is |expected| / k.
    """
    if k == 0:
        return 0.0
    return min(len(expected_files), k) / k


def compute_recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute Recall@K: fraction of expected files found in top-K results."""
    if not expected_files:
        return 1.0
    top_k = retrieved_files[:k]
    found = sum(1 for exp in expected_files if exp in top_k)
    return found / len(expected_files)


def compute_reciprocal_rank(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Reciprocal Rank: 1/rank of first relevant result in top-K, or 0.

    The mean (MRR) is computed by averaging this value across queries.
    """
    top_k = retrieved_files[:k]
    for i, f in enumerate(top_k):
        if f in expected_files:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_files: list[str], expected_files: list[str], k: int = 5) -> float:
    """Compute NDCG@K with binary relevance.

    Returns 0.0 if expected_files is empty (nothing to rank).
    """
    if not expected_files:
        return 0.0
    top_k = retrieved_files[:k]

    # DCG: sum of 1/log2(i+2) for relevant items at position i
    dcg = 0.0
    for i, f in enumerate(top_k):
        if f in expected_files:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: all relevant items packed at top positions
    n_relevant = min(len(expected_files), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_negative_oracle_penalty(
    retrieved_files: list[str], expected_files: list[str], k: int = 5, threshold: int = 3,
) -> bool:
    """Pass if number of irrelevant files in top-K is at or below threshold."""
    top_k = retrieved_files[:k]
    irrelevant = sum(1 for f in top_k if f not in expected_files)
    return irrelevant <= threshold


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

def compute_embedding_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0 for zero vectors."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# E2E result checking
# ---------------------------------------------------------------------------

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
    "does not exist",
    "is not a real",
    "is not a recognized",
    "not a known",
    "no such routine",
    "could not find",
    "not found in",
]


def _is_refusal(answer_lower: str) -> bool:
    """Return True if the answer contains a refusal phrase."""
    return any(p in answer_lower for p in _REFUSAL_PHRASES)


def check_e2e_result(
    answer: str,
    citations: list[str],
    checks: dict,
    *,
    answer_embedding: list[float] | None = None,
    golden_embedding: list[float] | None = None,
    expected_files: list[str] | None = None,
    citation_is_fallback: bool = False,
) -> dict:
    """Evaluate an e2e generation result against quality checks.

    Returns dict with per-check pass/fail and an overall pass boolean.

    New keyword-only params are backward-compatible (all default to None/False).
    """
    results: dict = {}
    answer_lower = answer.lower()

    is_refusal = _is_refusal(answer_lower)

    # Hallucination probe: answer SHOULD be a refusal
    if checks.get("expect_refusal"):
        results["not_refusal"] = False  # not applicable for probes
        results["expect_refusal_pass"] = is_refusal
        results["is_hallucination_probe"] = True
        results["pass"] = is_refusal
        return results

    # Normal query: answer should NOT be a refusal
    results["not_refusal"] = not is_refusal

    if checks.get("has_citations"):
        results["has_citations"] = len(citations) > 0

    if "min_answer_length" in checks:
        results["min_answer_length"] = len(answer) >= checks["min_answer_length"]

    if "expected_keywords" in checks:
        expected_kws = checks["expected_keywords"]
        found = [kw for kw in expected_kws if kw.lower() in answer_lower]
        results["keywords_found"] = found
        # Require at least 2/3 of keywords (rounded up)
        required = -(-2 * len(expected_kws) // 3)  # ceiling division
        results["keywords_pass"] = len(found) >= required

    # Embedding similarity gate
    if answer_embedding is not None and golden_embedding is not None:
        sim = compute_embedding_similarity(answer_embedding, golden_embedding)
        threshold = checks.get("similarity_threshold", 0.60)
        results["similarity_score"] = round(sim, 4)
        results["similarity_pass"] = sim >= threshold

    # Citation relevance check
    if expected_files is not None and citations:
        # Extract just the filename from citation strings like "dgesv.f:1-50"
        cited_files = [c.split(":")[0].split("/")[-1] for c in citations]
        results["citation_relevant"] = any(cf in expected_files for cf in cited_files)
    elif expected_files is not None:
        results["citation_relevant"] = len(expected_files) == 0

    # Citation fallback flag
    results["citation_is_fallback"] = citation_is_fallback

    results["pass"] = all(
        v for k, v in results.items()
        if k not in ("keywords_found", "similarity_score", "citation_is_fallback", "is_hallucination_probe")
        and isinstance(v, bool)
    )
    return results
