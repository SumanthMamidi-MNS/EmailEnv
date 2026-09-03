"""
tests/test_lessons.py

Unit tests for LessonGenerator pattern detection and evidence thresholds.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.feedback import FeedbackService
from evaluation.lessons import LessonGenerator
from evaluation.models import FeedbackVerdict, LessonStatus
from evaluation.storage import SQLiteRepository


class TestLessonGenerator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_lessons.db"
        self.repo = SQLiteRepository(db_path=self.db_path)
        self.feedback_svc = FeedbackService(repository=self.repo)
        self.generator = LessonGenerator(repository=self.repo, min_evidence=2)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_single_correction_does_not_create_candidate(self):
        self.feedback_svc.record_feedback(
            episode_id="ep_01",
            scenario_id="vendor_01",
            agent_name="HYBRID",
            verdict=FeedbackVerdict.INCORRECT,
            correct_action={"action_type": "REPLY_EMAIL", "email_id": "vendor_01", "body": "Confirmed."},
            explanation="Always reply to vendor inquiries.",
        )

        candidates = self.generator.generate_candidates()
        self.assertEqual(len(candidates), 0)

    def test_repeated_corrections_generate_candidate_lesson(self):
        self.feedback_svc.record_feedback(
            episode_id="ep_01",
            scenario_id="vendor_01",
            agent_name="HYBRID",
            verdict=FeedbackVerdict.INCORRECT,
            correct_action={"action_type": "REPLY_EMAIL", "email_id": "vendor_01", "body": "Confirmed."},
            explanation="Always reply to vendor inquiries.",
        )
        self.feedback_svc.record_feedback(
            episode_id="ep_02",
            scenario_id="vendor_01",
            agent_name="BASELINE",
            verdict=FeedbackVerdict.INCORRECT,
            correct_action={"action_type": "REPLY_EMAIL", "email_id": "vendor_01", "body": "Confirmed."},
            explanation="Always reply to vendor inquiries.",
        )

        candidates = self.generator.generate_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].evidence_count, 2)
        self.assertEqual(candidates[0].status, LessonStatus.CANDIDATE)


if __name__ == "__main__":
    unittest.main()
