"""SQLite-backed storage for model comparison trial results."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Use /tmp for environments where the app dir may not be writable (e.g. Railway),
# fall back to project-local data/ for local development.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_DB = _PROJECT_ROOT / "data" / "trials.db"
DEFAULT_DB_PATH = _LOCAL_DB if _LOCAL_DB.parent.exists() and os.access(str(_LOCAL_DB.parent), os.W_OK) else Path("/tmp/data/trials.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    eval_type TEXT NOT NULL,
    avg_recall_at_5 REAL,
    pass_rate REAL,
    avg_retrieval_latency_ms REAL,
    avg_e2e_latency_ms REAL,
    total_queries INTEGER,
    input_cost_per_1m REAL,
    output_cost_per_1m REAL,
    notes TEXT DEFAULT ''
)
"""


def _get_conn(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    return conn


def save_trial(data: dict, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Insert a trial record and return its id."""
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO trials
               (created_at, model, eval_type, avg_recall_at_5, pass_rate,
                avg_retrieval_latency_ms, avg_e2e_latency_ms, total_queries,
                input_cost_per_1m, output_cost_per_1m, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("created_at", datetime.now(timezone.utc).isoformat()),
                data["model"],
                data["eval_type"],
                data.get("avg_recall_at_5"),
                data.get("pass_rate"),
                data.get("avg_retrieval_latency_ms"),
                data.get("avg_e2e_latency_ms"),
                data.get("total_queries"),
                data.get("input_cost_per_1m"),
                data.get("output_cost_per_1m"),
                data.get("notes", ""),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_trials(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all trials ordered by most recent first."""
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM trials ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_trial(trial_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete a trial by id. Returns True if a row was deleted."""
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute("DELETE FROM trials WHERE id = ?", (trial_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
