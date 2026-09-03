"""
tests/test_learning_loop.py

End-to-end integration tests for Phase 2 Human Feedback & Validated Learning lifecycle.
Verifies complete flow: Before -> Feedback -> Candidate -> Validation -> Promotion -> After.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.api import (
    generate_candidate_lessons,
    record_feedback,
    retrieve_lessons,
    run_harmful_lesson_demo,
    run_learning_cycle_demo,
    validate_lesson,
)
from evaluation.models import FeedbackVerdict, LessonStatus
from evaluation.storage import SQLiteRepository


class TestLearningLifecycle(unittest.TestCase):

    def test_run_successful_learning_cycle(self):
        result = run_learning_cycle_demo()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["validation_report"]["verdict"], "PROMOTED")
        self.assertEqual(result["validation_report"]["safety_violations"], 0)
        self.assertGreater(result["retrieved_promoted_count"], 0)

    def test_run_harmful_lesson_rejection(self):
        result = run_harmful_lesson_demo()
        self.assertEqual(result["status"], "REJECTED_AS_EXPECTED")
        self.assertEqual(result["validation_report"]["verdict"], "REJECTED")
        self.assertFalse(result["promoted"])
        self.assertEqual(result["retrieved_count"], 0)


if __name__ == "__main__":
    unittest.main()
