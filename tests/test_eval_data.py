"""Tests for eval_data module."""

import pytest

from app.eval_data import EVAL_QUERIES, compute_recall_at_k


def test_compute_recall_at_k_empty_expected():
    """compute_recall_at_k returns 1.0 when expected_files is empty."""
    assert compute_recall_at_k(["a.f", "b.f"], [], k=5) == 1.0


def test_compute_recall_at_k_partial_match():
    """compute_recall_at_k returns fraction when some expected found."""
    assert compute_recall_at_k(["dgesv.f", "other.f"], ["dgesv.f", "dgetrf.f"], k=5) == 0.5


def test_compute_recall_at_k_full_match():
    """compute_recall_at_k returns 1.0 when all expected found in top-k."""
    assert compute_recall_at_k(["dgesv.f", "dgetrf.f"], ["dgesv.f", "dgetrf.f"], k=5) == 1.0


def test_eval_queries_loaded():
    """EVAL_QUERIES has expected structure."""
    assert len(EVAL_QUERIES) >= 10
    for item in EVAL_QUERIES:
        assert "query" in item
        assert "expected_files" in item
