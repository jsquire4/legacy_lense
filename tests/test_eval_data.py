"""Tests for eval_data module."""

from app.eval_data import (
    EVAL_QUERIES,
    E2E_EVAL_QUERIES,
    compute_precision_at_k,
    compute_max_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k,
    compute_negative_oracle_penalty,
    compute_embedding_similarity,
    check_e2e_result,
)


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

def test_compute_recall_at_k_empty_expected():
    assert compute_recall_at_k(["a.f", "b.f"], [], k=5) == 1.0


def test_compute_recall_at_k_partial_match():
    assert compute_recall_at_k(["dgesv.f", "other.f"], ["dgesv.f", "dgetrf.f"], k=5) == 0.5


def test_compute_recall_at_k_full_match():
    assert compute_recall_at_k(["dgesv.f", "dgetrf.f"], ["dgesv.f", "dgetrf.f"], k=5) == 1.0


def test_compute_recall_at_k_empty_retrieved():
    assert compute_recall_at_k([], ["dgesv.f", "dgetrf.f"], k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_precision_at_k
# ---------------------------------------------------------------------------

def test_compute_precision_at_k_all_relevant():
    assert compute_precision_at_k(["a.f", "b.f"], ["a.f", "b.f", "c.f"], k=2) == 1.0


def test_compute_precision_at_k_none_relevant():
    assert compute_precision_at_k(["x.f", "y.f"], ["a.f", "b.f"], k=2) == 0.0


def test_compute_precision_at_k_zero_k():
    assert compute_precision_at_k(["a.f"], ["a.f"], k=0) == 0.0


def test_compute_precision_at_k_partial():
    assert compute_precision_at_k(["a.f", "x.f", "b.f"], ["a.f", "b.f"], k=3) == 2 / 3


# ---------------------------------------------------------------------------
# compute_max_precision_at_k
# ---------------------------------------------------------------------------

def test_max_precision_single_expected():
    """1 expected file, k=5 → max is 0.2."""
    assert compute_max_precision_at_k(["a.f"], k=5) == 0.2


def test_max_precision_more_expected_than_k():
    """6 expected files, k=5 → max is 1.0."""
    assert compute_max_precision_at_k(["a.f", "b.f", "c.f", "d.f", "e.f", "f.f"], k=5) == 1.0


def test_max_precision_equal():
    """5 expected files, k=5 → max is 1.0."""
    assert compute_max_precision_at_k(["a.f", "b.f", "c.f", "d.f", "e.f"], k=5) == 1.0


def test_max_precision_zero_k():
    assert compute_max_precision_at_k(["a.f"], k=0) == 0.0


def test_max_precision_empty_expected():
    assert compute_max_precision_at_k([], k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_reciprocal_rank
# ---------------------------------------------------------------------------

def test_mrr_first_position():
    assert compute_reciprocal_rank(["a.f", "b.f", "c.f"], ["a.f"], k=5) == 1.0


def test_mrr_third_position():
    result = compute_reciprocal_rank(["x.f", "y.f", "a.f"], ["a.f"], k=5)
    assert abs(result - 1 / 3) < 1e-9


def test_mrr_not_found():
    assert compute_reciprocal_rank(["x.f", "y.f", "z.f"], ["a.f"], k=5) == 0.0


def test_mrr_multiple_relevant():
    """RR returns 1/rank of the FIRST relevant result."""
    result = compute_reciprocal_rank(["x.f", "a.f", "b.f"], ["a.f", "b.f"], k=5)
    assert abs(result - 0.5) < 1e-9


def test_mrr_empty_retrieved():
    assert compute_reciprocal_rank([], ["a.f"], k=5) == 0.0


def test_mrr_empty_expected():
    assert compute_reciprocal_rank(["a.f", "b.f"], [], k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    """All relevant items at top → NDCG = 1.0."""
    result = compute_ndcg_at_k(["a.f", "b.f", "x.f"], ["a.f", "b.f"], k=5)
    assert abs(result - 1.0) < 1e-9


def test_ndcg_reversed_ranking():
    """Relevant items pushed down → NDCG < 1.0."""
    result = compute_ndcg_at_k(["x.f", "y.f", "a.f", "b.f"], ["a.f", "b.f"], k=5)
    assert result < 1.0
    assert result > 0.0


def test_ndcg_no_relevant():
    assert compute_ndcg_at_k(["x.f", "y.f"], ["a.f"], k=5) == 0.0


def test_ndcg_empty_expected():
    assert compute_ndcg_at_k(["a.f", "b.f"], [], k=5) == 0.0


def test_ndcg_single_relevant_at_top():
    result = compute_ndcg_at_k(["a.f", "x.f"], ["a.f"], k=5)
    assert abs(result - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_negative_oracle_penalty
# ---------------------------------------------------------------------------

def test_negative_oracle_pass():
    """2 irrelevant in top-5 with threshold=3 → pass."""
    result = compute_negative_oracle_penalty(
        ["a.f", "x.f", "b.f", "y.f", "c.f"], ["a.f", "b.f", "c.f"], k=5, threshold=3,
    )
    assert result is True


def test_negative_oracle_fail():
    """5 irrelevant in top-5 → fail."""
    result = compute_negative_oracle_penalty(
        ["w.f", "x.f", "y.f", "z.f", "q.f"], ["a.f"], k=5, threshold=3,
    )
    assert result is False


def test_negative_oracle_at_threshold():
    """Exactly at threshold → pass."""
    result = compute_negative_oracle_penalty(
        ["x.f", "y.f", "z.f", "a.f", "b.f"], ["a.f", "b.f"], k=5, threshold=3,
    )
    assert result is True


# ---------------------------------------------------------------------------
# compute_embedding_similarity
# ---------------------------------------------------------------------------

def test_embedding_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert abs(compute_embedding_similarity(vec, vec) - 1.0) < 1e-9


def test_embedding_similarity_orthogonal():
    assert abs(compute_embedding_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_embedding_similarity_opposite():
    result = compute_embedding_similarity([1.0, 0.0], [-1.0, 0.0])
    assert abs(result - (-1.0)) < 1e-9


def test_embedding_similarity_zero_vector():
    assert compute_embedding_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert compute_embedding_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# check_e2e_result — not_refusal
# ---------------------------------------------------------------------------

def test_not_refusal_normal_answer():
    result = check_e2e_result("DGESV solves linear systems.", ["x.f"], {})
    assert result["not_refusal"] is True


def test_not_refusal_mixed_case_refusal():
    result = check_e2e_result("I Cannot Answer that question.", ["x.f"], {})
    assert result["not_refusal"] is False


def test_not_refusal_partial_phrase():
    result = check_e2e_result("There is Insufficient Context to answer.", ["x.f"], {})
    assert result["not_refusal"] is False


# ---------------------------------------------------------------------------
# check_e2e_result — hallucination probes
# ---------------------------------------------------------------------------

def test_hallucination_probe_refusal_passes():
    result = check_e2e_result(
        "I could not find any routine called DFAKE in the LAPACK source.",
        [], {"expect_refusal": True},
    )
    assert result["expect_refusal_pass"] is True
    assert result["pass"] is True
    assert result["is_hallucination_probe"] is True


def test_hallucination_probe_fabrication_fails():
    result = check_e2e_result(
        "DFAKE computes the determinant of a matrix using Gaussian elimination.",
        [], {"expect_refusal": True},
    )
    assert result["expect_refusal_pass"] is False
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# check_e2e_result — citation relevance
# ---------------------------------------------------------------------------

def test_citation_relevant_file_passes():
    result = check_e2e_result(
        "DGESV solves linear systems.", ["dgesv.f:1-50"],
        {"has_citations": True}, expected_files=["dgesv.f"],
    )
    assert result["citation_relevant"] is True


def test_citation_irrelevant_file_fails():
    result = check_e2e_result(
        "DGESV solves linear systems.", ["other.f:1-50"],
        {"has_citations": True}, expected_files=["dgesv.f"],
    )
    assert result["citation_relevant"] is False


def test_citation_relevance_no_expected_no_citations():
    """No expected files and no citations → citation_relevant should be True (empty expected)."""
    result = check_e2e_result("Answer", [], {}, expected_files=[])
    assert result["citation_relevant"] is True


# ---------------------------------------------------------------------------
# check_e2e_result — citation fallback
# ---------------------------------------------------------------------------

def test_citation_fallback_flag():
    result = check_e2e_result(
        "DGESV solves linear systems.", ["dgesv.f:1-50"],
        {"has_citations": True}, citation_is_fallback=True,
    )
    assert result["citation_is_fallback"] is True


def test_citation_no_fallback():
    result = check_e2e_result(
        "DGESV solves linear systems.", ["dgesv.f:1-50"],
        {"has_citations": True}, citation_is_fallback=False,
    )
    assert result["citation_is_fallback"] is False


# ---------------------------------------------------------------------------
# check_e2e_result — ALL-keywords
# ---------------------------------------------------------------------------

def test_keywords_all_must_match():
    """With 3 keywords, finding only 1 → fail (was pass under old 1-of-N logic)."""
    result = check_e2e_result(
        "This answer mentions linear but not the other two.",
        ["x.f"],
        {"expected_keywords": ["linear", "system", "solve"]},
    )
    assert result["keywords_pass"] is False
    assert result["pass"] is False


def test_keywords_all_found_passes():
    result = check_e2e_result(
        "DGESV solves a linear system of equations.",
        ["x.f"],
        {"expected_keywords": ["linear", "system", "solve"]},
    )
    assert result["keywords_pass"] is True


# ---------------------------------------------------------------------------
# check_e2e_result — embedding similarity
# ---------------------------------------------------------------------------

def test_embedding_similarity_gate_pass():
    result = check_e2e_result(
        "Answer text.", ["x.f"], {},
        answer_embedding=[1.0, 0.0, 0.0],
        golden_embedding=[1.0, 0.0, 0.0],
    )
    assert result["similarity_pass"] is True
    assert result["similarity_score"] == 1.0


def test_embedding_similarity_gate_fail():
    result = check_e2e_result(
        "Answer text.", ["x.f"], {},
        answer_embedding=[1.0, 0.0, 0.0],
        golden_embedding=[0.0, 1.0, 0.0],
    )
    assert result["similarity_pass"] is False
    assert result["similarity_score"] == 0.0


# ---------------------------------------------------------------------------
# check_e2e_result — backward compatibility
# ---------------------------------------------------------------------------

def test_check_e2e_result_all_pass():
    result = check_e2e_result(
        "DGESV solves linear systems using LU factorization.",
        ["dgesv.f:1-50"],
        {"has_citations": True, "min_answer_length": 20, "expected_keywords": ["linear", "LU"]},
    )
    assert result["has_citations"] is True
    assert result["min_answer_length"] is True
    assert result["keywords_pass"] is True
    assert result["pass"] is True


def test_check_e2e_result_fails_no_citations():
    result = check_e2e_result("Answer without citations.", [], {"has_citations": True})
    assert result["has_citations"] is False
    assert result["pass"] is False


def test_check_e2e_result_fails_min_length():
    result = check_e2e_result(
        "Short.", ["x.f"],
        {"has_citations": True, "min_answer_length": 100},
    )
    assert result["min_answer_length"] is False
    assert result["pass"] is False


def test_check_e2e_result_fails_keywords():
    result = check_e2e_result(
        "Answer about something else.", ["x.f"],
        {"has_citations": True, "expected_keywords": ["DGESV", "LU"]},
    )
    assert result["keywords_pass"] is False
    assert result["pass"] is False


def test_check_e2e_result_partial_checks():
    result = check_e2e_result("Answer", ["x.f"], {"has_citations": True})
    assert result["has_citations"] is True
    assert "min_answer_length" not in result
    assert "keywords_pass" not in result
    assert result["pass"] is True


def test_check_e2e_result_empty_checks():
    result = check_e2e_result("Answer", [], {})
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# check_e2e_result — summary arithmetic
# ---------------------------------------------------------------------------

def test_overall_pass_includes_similarity_and_citation():
    """Overall pass should fail if similarity_pass is False."""
    result = check_e2e_result(
        "Answer text here.", ["dgesv.f:1-50"],
        {"has_citations": True},
        answer_embedding=[1.0, 0.0],
        golden_embedding=[0.0, 1.0],
        expected_files=["dgesv.f"],
    )
    assert result["similarity_pass"] is False
    assert result["pass"] is False


def test_overall_pass_all_gates():
    """Overall pass when all gates pass."""
    result = check_e2e_result(
        "DGESV solves a linear system of equations.", ["dgesv.f:1-50"],
        {"has_citations": True, "min_answer_length": 10, "expected_keywords": ["linear", "system"]},
        answer_embedding=[1.0, 0.0],
        golden_embedding=[1.0, 0.0],
        expected_files=["dgesv.f"],
    )
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

def test_eval_queries_loaded():
    assert len(EVAL_QUERIES) >= 70
    for item in EVAL_QUERIES:
        assert "query" in item
        assert "expected_files" in item
        assert "difficulty" in item
        assert item["difficulty"] in ("easy", "medium", "hard")


def test_eval_queries_difficulty_distribution():
    easy = [q for q in EVAL_QUERIES if q["difficulty"] == "easy"]
    medium = [q for q in EVAL_QUERIES if q["difficulty"] == "medium"]
    hard = [q for q in EVAL_QUERIES if q["difficulty"] == "hard"]
    assert len(easy) >= 20
    assert len(medium) >= 20
    assert len(hard) >= 10


def test_e2e_eval_queries_loaded():
    assert len(E2E_EVAL_QUERIES) >= 20
    for item in E2E_EVAL_QUERIES:
        assert "query" in item
        assert "capability" in item
        assert "checks" in item


def test_e2e_eval_queries_have_golden_answers():
    """Non-probe E2E queries have golden_answer and expected_files."""
    for item in E2E_EVAL_QUERIES:
        if not item.get("is_hallucination_probe"):
            assert "golden_answer" in item, f"Missing golden_answer: {item['query']}"
            assert len(item["golden_answer"]) > 0
            assert "expected_files" in item


def test_e2e_hallucination_probes():
    probes = [q for q in E2E_EVAL_QUERIES if q.get("is_hallucination_probe")]
    assert len(probes) == 6
    for p in probes:
        assert p["checks"].get("expect_refusal") is True
        assert p["expected_files"] == []
        assert p["golden_answer"] == ""
