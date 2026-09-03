"""
tests/test_storage.py

Unit tests for SQLiteRepository persistence layer.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.models import (
    EpisodeRecord,
    FeedbackRecord,
    FeedbackVerdict,
    LessonRecord,
    LessonScope,
    LessonStatus,
    TrajectoryStep,
)
from evaluation.storage import SQLiteRepository


class TestSQLiteStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_eval.db"
        self.repo = SQLiteRepository(db_path=self.db_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_save_and_get_episode(self):
        step = TrajectoryStep(
            step_number=1,
            observation_summary={"step": 0},
            action={"action_type": "READ_EMAIL", "email_id": "test_01"},
            action_valid=True,
            reward=0.05,
            feedback="Opened unread email.",
        )
        ep = EpisodeRecord(
            episode_id="ep_123",
            scenario_id="starter_01",
            difficulty="STARTER",
            agent_name="BASELINE",
            steps=[step],
            final_action="ARCHIVE_EMAIL",
            cumulative_reward=0.55,
            safety_status="SAFE",
            completed=True,
        )

        self.repo.save_episode(ep)
        retrieved = self.repo.get_episode("ep_123")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.episode_id, "ep_123")
        self.assertEqual(len(retrieved.steps), 1)
        self.assertEqual(retrieved.steps[0].action_valid, True)
        self.assertEqual(retrieved.cumulative_reward, 0.55)

    def test_save_and_list_feedback(self):
        fb = FeedbackRecord(
            feedback_id="fb_001",
            episode_id="ep_123",
            scenario_id="starter_01",
            agent_name="BASELINE",
            verdict=FeedbackVerdict.INCORRECT,
            correct_action={"action_type": "REPLY_EMAIL", "email_id": "starter_01"},
            explanation="Should reply to contract requests.",
        )

        self.repo.save_feedback(fb)
        all_fb = self.repo.list_feedback()

        self.assertEqual(len(all_fb), 1)
        self.assertEqual(all_fb[0].feedback_id, "fb_001")
        self.assertEqual(all_fb[0].verdict, FeedbackVerdict.INCORRECT)

    def test_save_update_and_get_promoted_lessons(self):
        lesson = LessonRecord(
            lesson_id="les_001",
            rule_text="Always escalate active database outages.",
            scope=LessonScope.CATEGORY,
            target_category="production_incident",
            evidence_count=2,
            status=LessonStatus.CANDIDATE,
        )

        self.repo.save_lesson(lesson)

        # Before promotion, get_promoted_lessons should return empty
        promoted = self.repo.get_promoted_lessons(category="production_incident")
        self.assertEqual(len(promoted), 0)

        # Update to PROMOTED
        self.repo.update_lesson_status("les_001", status=LessonStatus.PROMOTED)
        promoted = self.repo.get_promoted_lessons(category="production_incident")
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0].lesson_id, "les_001")


if __name__ == "__main__":
    unittest.main()
