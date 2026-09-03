"""
tests/test_feedback.py

Unit tests for FeedbackService and feedback schema validation.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.feedback import FeedbackService, FeedbackValidationError
from evaluation.models import FeedbackVerdict
from evaluation.storage import SQLiteRepository


class TestFeedbackService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_feedback.db"
        self.repo = SQLiteRepository(db_path=self.db_path)
        self.service = FeedbackService(repository=self.repo)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_record_valid_correct_feedback(self):
        fb = self.service.record_feedback(
            episode_id="ep_100",
            scenario_id="starter_01",
            agent_name="HYBRID",
            verdict=FeedbackVerdict.CORRECT,
            explanation="Well handled.",
        )
        self.assertIsNotNone(fb.feedback_id)
        self.assertEqual(fb.verdict, FeedbackVerdict.CORRECT)

    def test_record_valid_incorrect_feedback_with_action(self):
        fb = self.service.record_feedback(
            episode_id="ep_101",
            scenario_id="starter_02",
            agent_name="BASELINE",
            verdict=FeedbackVerdict.INCORRECT,
            correct_action={"action_type": "REPLY_EMAIL", "email_id": "starter_02", "body": "Accepted."},
            explanation="Customer requested receipt confirmation.",
        )
        self.assertEqual(fb.verdict, FeedbackVerdict.INCORRECT)
        self.assertIsNotNone(fb.correct_action)

    def test_reject_incorrect_feedback_missing_action(self):
        with self.assertRaises(FeedbackValidationError):
            self.service.record_feedback(
                episode_id="ep_102",
                scenario_id="starter_03",
                agent_name="BASELINE",
                verdict=FeedbackVerdict.INCORRECT,
                correct_action=None,
            )

    def test_reject_invalid_action_schema(self):
        with self.assertRaises(FeedbackValidationError):
            self.service.record_feedback(
                episode_id="ep_103",
                scenario_id="starter_04",
                agent_name="BASELINE",
                verdict=FeedbackVerdict.INCORRECT,
                correct_action={"action_type": "INVALID_EXPLODE_ACTION", "email_id": "starter_04"},
            )


if __name__ == "__main__":
    unittest.main()
