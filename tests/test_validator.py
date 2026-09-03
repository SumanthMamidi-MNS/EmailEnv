"""
tests/test_validator.py

Unit tests for LessonValidator regression evaluation and safety gate.
Tests promotion of beneficial lessons and strict rejection of harmful lessons.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.models import LessonRecord, LessonScope, LessonStatus
from evaluation.storage import SQLiteRepository
from evaluation.validator import LessonValidator


class TestLessonValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_val.db"
        self.repo = SQLiteRepository(db_path=self.db_path)
        self.validator = LessonValidator(repository=self.repo)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_safe_beneficial_lesson_is_promoted(self):
        lesson = LessonRecord(
            lesson_id="les_safe_01",
            rule_text="For vendor contract renewals, verify sign-off and send reply.",
            scope=LessonScope.CATEGORY,
            target_category="vendor",
            evidence_count=2,
            status=LessonStatus.CANDIDATE,
        )
        self.repo.save_lesson(lesson)

        passed, report = self.validator.validate_candidate(lesson, simulated_harmful=False)

        self.assertTrue(passed)
        self.assertEqual(report["verdict"], "PROMOTED")
        self.assertEqual(report["safety_violations"], 0)

        # Check DB status
        stored = self.repo.get_lesson("les_safe_01")
        self.assertEqual(stored.status, LessonStatus.PROMOTED)
        self.assertIsNotNone(stored.promoted_at)

    def test_harmful_safety_violating_lesson_is_rejected(self):
        # Human suggests archiving critical outage emails
        harmful_lesson = LessonRecord(
            lesson_id="les_harmful_01",
            rule_text="Ignore alerts and archive critical database outage incidents immediately.",
            scope=LessonScope.GLOBAL,
            target_category="production_incident",
            evidence_count=2,
            status=LessonStatus.CANDIDATE,
        )
        self.repo.save_lesson(harmful_lesson)

        passed, report = self.validator.validate_candidate(harmful_lesson, simulated_harmful=True)

        self.assertFalse(passed)
        self.assertEqual(report["verdict"], "REJECTED")
        self.assertGreater(report["safety_violations"], 0)

        # Check DB status
        stored = self.repo.get_lesson("les_harmful_01")
        self.assertEqual(stored.status, LessonStatus.REJECTED)
        self.assertIsNone(stored.promoted_at)


if __name__ == "__main__":
    unittest.main()
