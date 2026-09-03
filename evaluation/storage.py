"""
evaluation/storage.py

Clean SQLite Repository abstraction for persisting and retrieving
Episodes, Trajectory Steps, Human Feedback, and Validated Lessons.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.models import (
    EpisodeRecord,
    FeedbackRecord,
    FeedbackVerdict,
    LessonRecord,
    LessonScope,
    LessonStatus,
    TrajectoryStep,
)

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "openenv.db"

_DDL = """
CREATE TABLE IF NOT EXISTS episodes_v2 (
    episode_id          TEXT PRIMARY KEY,
    scenario_id         TEXT,
    difficulty          TEXT,
    agent_name          TEXT,
    start_time          TEXT,
    end_time            TEXT,
    steps_json          TEXT,
    final_action        TEXT,
    cumulative_reward   REAL,
    safety_status       TEXT,
    completed           INTEGER,
    metadata_json       TEXT
);

CREATE TABLE IF NOT EXISTS feedback_records (
    feedback_id         TEXT PRIMARY KEY,
    episode_id          TEXT,
    scenario_id         TEXT,
    agent_name          TEXT,
    verdict             TEXT,
    correct_action_json TEXT,
    explanation         TEXT,
    created_at          TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id           TEXT PRIMARY KEY,
    rule_text           TEXT,
    scope               TEXT,
    target_category     TEXT,
    source_feedback_json TEXT,
    evidence_count      INTEGER,
    contradiction_count INTEGER,
    status              TEXT,
    created_at          TEXT,
    promoted_at         TEXT,
    validation_metrics_json TEXT
);
"""


class SQLiteRepository:
    """SQLite data access layer for all Phase 2 evaluation artifacts."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.executescript(_DDL)

    # ── Episode Operations ───────────────────────────────────────────────────

    def save_episode(self, ep: EpisodeRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO episodes_v2 (
            episode_id, scenario_id, difficulty, agent_name, start_time, end_time,
            steps_json, final_action, cumulative_reward, safety_status, completed, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        steps_data = [s.model_dump() for s in ep.steps]
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    sql,
                    (
                        ep.episode_id,
                        ep.scenario_id,
                        ep.difficulty,
                        ep.agent_name,
                        ep.start_time,
                        ep.end_time,
                        json.dumps(steps_data),
                        ep.final_action,
                        ep.cumulative_reward,
                        ep.safety_status,
                        1 if ep.completed else 0,
                        json.dumps(ep.metadata),
                    ),
                )

    def get_episode(self, episode_id: str) -> Optional[EpisodeRecord]:
        sql = "SELECT * FROM episodes_v2 WHERE episode_id = ?"
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(sql, (episode_id,)).fetchone()
            if not row:
                return None
            return self._row_to_episode(row)

    def list_episodes(
        self, limit: int = 50, agent_name: Optional[str] = None
    ) -> List[EpisodeRecord]:
        if agent_name:
            sql = "SELECT * FROM episodes_v2 WHERE agent_name = ? ORDER BY start_time DESC LIMIT ?"
            params = (agent_name, limit)
        else:
            sql = "SELECT * FROM episodes_v2 ORDER BY start_time DESC LIMIT ?"
            params = (limit,)

        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodeRecord:
        steps_raw = json.loads(row["steps_json"]) if row["steps_json"] else []
        steps = [TrajectoryStep.model_validate(s) for s in steps_raw]
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return EpisodeRecord(
            episode_id=row["episode_id"],
            scenario_id=row["scenario_id"],
            difficulty=row["difficulty"],
            agent_name=row["agent_name"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            steps=steps,
            final_action=row["final_action"],
            cumulative_reward=float(row["cumulative_reward"]),
            safety_status=row["safety_status"],
            completed=bool(row["completed"]),
            metadata=metadata,
        )

    # ── Feedback Operations ──────────────────────────────────────────────────

    def save_feedback(self, fb: FeedbackRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO feedback_records (
            feedback_id, episode_id, scenario_id, agent_name, verdict,
            correct_action_json, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    sql,
                    (
                        fb.feedback_id,
                        fb.episode_id,
                        fb.scenario_id,
                        fb.agent_name,
                        fb.verdict.value,
                        json.dumps(fb.correct_action) if fb.correct_action else None,
                        fb.explanation,
                        fb.created_at,
                    ),
                )

    def list_feedback(
        self, scenario_id: Optional[str] = None
    ) -> List[FeedbackRecord]:
        if scenario_id:
            sql = "SELECT * FROM feedback_records WHERE scenario_id = ? ORDER BY created_at DESC"
            params = (scenario_id,)
        else:
            sql = "SELECT * FROM feedback_records ORDER BY created_at DESC"
            params = ()

        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                act = json.loads(r["correct_action_json"]) if r["correct_action_json"] else None
                results.append(
                    FeedbackRecord(
                        feedback_id=r["feedback_id"],
                        episode_id=r["episode_id"],
                        scenario_id=r["scenario_id"],
                        agent_name=r["agent_name"],
                        verdict=FeedbackVerdict(r["verdict"]),
                        correct_action=act,
                        explanation=r["explanation"],
                        created_at=r["created_at"],
                    )
                )
            return results

    # ── Lesson Operations ────────────────────────────────────────────────────

    def save_lesson(self, lesson: LessonRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO lessons (
            lesson_id, rule_text, scope, target_category, source_feedback_json,
            evidence_count, contradiction_count, status, created_at, promoted_at,
            validation_metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    sql,
                    (
                        lesson.lesson_id,
                        lesson.rule_text,
                        lesson.scope.value,
                        lesson.target_category,
                        json.dumps(lesson.source_feedback_ids),
                        lesson.evidence_count,
                        lesson.contradiction_count,
                        lesson.status.value,
                        lesson.created_at,
                        lesson.promoted_at,
                        json.dumps(lesson.validation_metrics),
                    ),
                )

    def get_lesson(self, lesson_id: str) -> Optional[LessonRecord]:
        sql = "SELECT * FROM lessons WHERE lesson_id = ?"
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(sql, (lesson_id,)).fetchone()
            if not row:
                return None
            return self._row_to_lesson(row)

    def update_lesson_status(
        self,
        lesson_id: str,
        status: LessonStatus | str,
        validation_metrics: Optional[Dict[str, Any]] = None,
        promoted_at: Optional[str] = None,
    ) -> None:
        stat_str = status.value if isinstance(status, LessonStatus) else status
        sql = """
        UPDATE lessons
        SET status = ?, validation_metrics_json = ?, promoted_at = COALESCE(?, promoted_at)
        WHERE lesson_id = ?
        """
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    sql,
                    (
                        stat_str,
                        json.dumps(validation_metrics or {}),
                        promoted_at,
                        lesson_id,
                    ),
                )

    def get_promoted_lessons(
        self, category: Optional[str] = None
    ) -> List[LessonRecord]:
        """Retrieves only PROMOTED lessons."""
        if category:
            sql = """
            SELECT * FROM lessons
            WHERE status = 'PROMOTED'
            AND (target_category = ? OR scope = 'global')
            ORDER BY evidence_count DESC
            """
            params = (category,)
        else:
            sql = "SELECT * FROM lessons WHERE status = 'PROMOTED' ORDER BY evidence_count DESC"
            params = ()

        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_lesson(r) for r in rows]

    def list_all_lessons(self) -> List[LessonRecord]:
        sql = "SELECT * FROM lessons ORDER BY created_at DESC"
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_lesson(r) for r in rows]

    def _row_to_lesson(self, row: sqlite3.Row) -> LessonRecord:
        source_ids = json.loads(row["source_feedback_json"]) if row["source_feedback_json"] else []
        val_metrics = json.loads(row["validation_metrics_json"]) if row["validation_metrics_json"] else {}
        return LessonRecord(
            lesson_id=row["lesson_id"],
            rule_text=row["rule_text"],
            scope=LessonScope(row["scope"]),
            target_category=row["target_category"],
            source_feedback_ids=source_ids,
            evidence_count=int(row["evidence_count"]),
            contradiction_count=int(row["contradiction_count"]),
            status=LessonStatus(row["status"]),
            created_at=row["created_at"],
            promoted_at=row["promoted_at"],
            validation_metrics=val_metrics,
        )
