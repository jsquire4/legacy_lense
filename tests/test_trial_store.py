"""Tests for the SQLite trial store."""

import tempfile
from pathlib import Path

from app.services.trial_store import save_trial, list_trials, delete_trial


def _tmp_db():
    return Path(tempfile.mktemp(suffix=".db"))


def test_save_and_list_trial():
    db = _tmp_db()
    trial_id = save_trial({
        "model": "gpt-4o-mini",
        "eval_type": "retrieval",
        "avg_recall_at_5": 0.85,
        "avg_retrieval_latency_ms": 120.5,
        "total_queries": 10,
    }, db_path=db)
    assert isinstance(trial_id, int)
    assert trial_id >= 1

    trials = list_trials(db_path=db)
    assert len(trials) == 1
    assert trials[0]["model"] == "gpt-4o-mini"
    assert trials[0]["eval_type"] == "retrieval"
    assert trials[0]["avg_recall_at_5"] == 0.85


def test_list_trials_order():
    db = _tmp_db()
    save_trial({"model": "gpt-4o-mini", "eval_type": "retrieval"}, db_path=db)
    save_trial({"model": "gpt-4o", "eval_type": "e2e"}, db_path=db)
    trials = list_trials(db_path=db)
    assert len(trials) == 2
    # Most recent first
    assert trials[0]["model"] == "gpt-4o"
    assert trials[1]["model"] == "gpt-4o-mini"


def test_delete_trial():
    db = _tmp_db()
    trial_id = save_trial({"model": "gpt-4o", "eval_type": "e2e"}, db_path=db)
    assert delete_trial(trial_id, db_path=db) is True
    assert list_trials(db_path=db) == []


def test_delete_nonexistent_trial():
    db = _tmp_db()
    assert delete_trial(9999, db_path=db) is False


def test_save_trial_with_all_fields():
    db = _tmp_db()
    save_trial({
        "model": "gpt-5",
        "eval_type": "e2e",
        "avg_recall_at_5": 0.9,
        "pass_rate": 0.8,
        "avg_retrieval_latency_ms": 100.0,
        "avg_e2e_latency_ms": 2500.0,
        "total_queries": 20,
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
        "notes": "test run",
    }, db_path=db)
    trials = list_trials(db_path=db)
    t = trials[0]
    assert t["pass_rate"] == 0.8
    assert t["notes"] == "test run"
    assert t["input_cost_per_1m"] == 1.25


def test_save_trial_defaults():
    db = _tmp_db()
    save_trial({"model": "gpt-4o-mini", "eval_type": "retrieval"}, db_path=db)
    t = list_trials(db_path=db)[0]
    assert t["notes"] == ""
    assert t["pass_rate"] is None
    assert t["created_at"] is not None


def test_list_trials_empty():
    db = _tmp_db()
    assert list_trials(db_path=db) == []


def test_save_ingestion_trial():
    db = _tmp_db()
    save_trial({
        "model": "text-embedding-3-small",
        "eval_type": "ingestion",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "ingestion_time_sec": 45.2,
        "chunks_ingested": 2407,
        "files_processed": 2300,
    }, db_path=db)
    trials = list_trials(db_path=db)
    assert len(trials) == 1
    t = trials[0]
    assert t["eval_type"] == "ingestion"
    assert t["ingestion_time_sec"] == 45.2
    assert t["chunks_ingested"] == 2407
    assert t["files_processed"] == 2300
    assert t["embedding_model"] == "text-embedding-3-small"
    assert t["embedding_dimensions"] == 1536
