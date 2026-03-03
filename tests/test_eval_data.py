"""Tests for eval_data module."""

import pytest

from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES, compute_recall_at_k, check_e2e_result


def test_compute_recall_at_k_empty_expected():
    """compute_recall_at_k returns 1.0 when expected_files is empty."""
    assert compute_recall_at_k(["a.f", "b.f"], [], k=5) == 1.0


def test_compute_recall_at_k_partial_match():
    """compute_recall_at_k returns fraction when some expected found."""
    assert compute_recall_at_k(["dgesv.f", "other.f"], ["dgesv.f", "dgetrf.f"], k=5) == 0.5


def test_compute_recall_at_k_full_match():
    """compute_recall_at_k returns 1.0 when all expected found in top-k."""
    assert compute_recall_at_k(["dgesv.f", "dgetrf.f"], ["dgesv.f", "dgetrf.f"], k=5) == 1.0


def test_compute_recall_at_k_empty_retrieved():
    """compute_recall_at_k returns 0.0 when retrieved_files is empty but expected not."""
    assert compute_recall_at_k([], ["dgesv.f", "dgetrf.f"], k=5) == 0.0


def test_eval_queries_loaded():
    """EVAL_QUERIES has expected structure."""
    assert len(EVAL_QUERIES) >= 10
    for item in EVAL_QUERIES:
        assert "query" in item
        assert "expected_files" in item


def test_check_e2e_result_all_pass():
    """check_e2e_result passes when answer meets all checks."""
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
    """check_e2e_result fails when has_citations required but none provided."""
    result = check_e2e_result("Answer without citations.", [], {"has_citations": True})
    assert result["has_citations"] is False
    assert result["pass"] is False


def test_check_e2e_result_fails_min_length():
    """check_e2e_result fails when answer too short."""
    result = check_e2e_result(
        "Short.",
        ["x.f"],
        {"has_citations": True, "min_answer_length": 100},
    )
    assert result["min_answer_length"] is False
    assert result["pass"] is False


def test_check_e2e_result_fails_keywords():
    """check_e2e_result fails when no expected keyword found."""
    result = check_e2e_result(
        "Answer about something else.",
        ["x.f"],
        {"has_citations": True, "expected_keywords": ["DGESV", "LU"]},
    )
    assert result["keywords_pass"] is False
    assert result["pass"] is False


def test_check_e2e_result_partial_checks():
    """check_e2e_result handles checks with only some keys (branch coverage)."""
    result = check_e2e_result("Answer", ["x.f"], {"has_citations": True})
    assert result["has_citations"] is True
    assert "min_answer_length" not in result
    assert "keywords_pass" not in result
    assert result["pass"] is True


def test_check_e2e_result_empty_checks():
    """check_e2e_result handles empty checks dict."""
    result = check_e2e_result("Answer", [], {})
    assert result["pass"] is True


def test_e2e_eval_queries_loaded():
    """E2E_EVAL_QUERIES has expected structure with checks."""
    assert len(E2E_EVAL_QUERIES) >= 10
    for item in E2E_EVAL_QUERIES:
        assert "query" in item
        assert "capability" in item
        assert "checks" in item
