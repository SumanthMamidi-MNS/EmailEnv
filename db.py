# ═══ db.py ═══

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = "openenv.db"

_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT,
    score               REAL,
    steps               INTEGER,
    valid_actions       INTEGER,
    invalid_actions     INTEGER,
    cumulative_reward   REAL,
    breakdown           TEXT,
    agent_tier          TEXT,
    timestamp           TEXT
);
"""


# ── Internal helper ───────────────────────────────────────────────────────────

def _connect(path: str = _DEFAULT_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def init_db(path: str = _DEFAULT_PATH) -> None:
    """Create the episodes table if it does not already exist."""
    try:
        with _connect(path) as conn:
            conn.executescript(_DDL)
    except Exception:
        pass


def save_episode(
    task_id: str,
    score: float,
    steps: int,
    valid_actions: int,
    invalid_actions: int,
    cumulative_reward: float,
    breakdown: dict | list | str,
    agent_tier: str,
    path: str = _DEFAULT_PATH,
) -> int | None:
    """Persist one episode. Returns the new row id."""
    init_db(path)
    try:
        breakdown_str = (
            json.dumps(breakdown)
            if not isinstance(breakdown, str)
            else breakdown
        )
        ts = datetime.now(timezone.utc).isoformat()
        with _connect(path) as conn:
            cur = conn.execute(
                """
                INSERT INTO episodes
                    (task_id, score, steps, valid_actions, invalid_actions,
                     cumulative_reward, breakdown, agent_tier, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    score,
                    steps,
                    valid_actions,
                    invalid_actions,
                    cumulative_reward,
                    breakdown_str,
                    agent_tier,
                    ts,
                ),
            )
            return cur.lastrowid
    except Exception:
        return None


def get_all_episodes(path: str = _DEFAULT_PATH) -> list[dict]:
    """Return every episode row as a list of dicts."""
    init_db(path)
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_best_score(task_id: str, path: str = _DEFAULT_PATH) -> float | None:
    """Return the highest score recorded for a given task_id."""
    init_db(path)
    try:
        with _connect(path) as conn:
            row = conn.execute(
                "SELECT MAX(score) AS best FROM episodes WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row and row["best"] is not None:
            return float(row["best"])
        return None
    except Exception:
        return None


def get_episode_count(path: str = _DEFAULT_PATH) -> int:
    """Return total number of stored episodes."""
    init_db(path)
    try:
        with _connect(path) as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM episodes").fetchone()
        return int(row["cnt"]) if row else 0
    except Exception:
        return 0


def get_learning_curve(
    task_id: str, path: str = _DEFAULT_PATH
) -> list[dict]:
    """
    Return episodes for a task ordered chronologically.
    Each dict has: episode_number, score, timestamp.
    """
    init_db(path)
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                """
                SELECT id, score, timestamp
                FROM   episodes
                WHERE  task_id = ?
                ORDER  BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [
            {
                "episode_number": i + 1,
                "score": float(row["score"]),
                "timestamp": row["timestamp"],
            }
            for i, row in enumerate(rows)
        ]
    except Exception:
        return []


def clear_all(path: str = _DEFAULT_PATH) -> None:
    """Delete every row from the episodes table."""
    init_db(path)
    try:
        with _connect(path) as conn:
            conn.execute("DELETE FROM episodes")
    except Exception:
        pass