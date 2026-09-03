"""
tests/test_retrieval.py

Unit tests for LessonRetriever filtering and prompt formatting.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.models import LessonRecord, LessonScope, LessonStatus
from evaluation.retrieval import LessonRetriever
from evaluation.storage import SQLiteRepository


class TestLessonRetriever(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_retrieval.db"
        self.repo = SQLiteRepository(db_path=self.db_path)
        self.retriever = LessonRetriever(repository=self.repo)

        # Seed lessons with different statuses
        self.promoted_lesson = LessonRecord(
            lesson_id="les_p1",
            rule_text="Vendor contract renewal requires reply.",
            scope=LessonScope.CATEGORY,
            target_category="vendor",
            status=LessonStatus.PROMOTED,
        )
        self.candidate_lesson = LessonRecord(
            lesson_id="les_c1",
            rule_text="Unvalidated candidate rule.",
            scope=LessonScope.CATEGORY,
            target_category="vendor",
            status=LessonStatus.CANDIDATE,
        )
        self.rejected_lesson = LessonRecord(
            lesson_id="les_r1",
            rule_text="Harmful rejected rule.",
            scope=LessonScope.CATEGORY,
            target_category="vendor",
            status=LessonStatus.REJECTED,
        )

        self.repo.save_lesson(self.promoted_lesson)
        self.repo.save_lesson(self.candidate_lesson)
        self.repo.save_lesson(self.rejected_lesson)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_only_promoted_lessons_are_retrieved(self):
        results = self.retriever.get_relevant_lessons(
            subject="Vendor Contract", sender="rep@vendor.com", body="Review terms.", category="vendor"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].lesson_id, "les_p1")
        self.assertEqual(results[0].status, LessonStatus.PROMOTED)

    def test_prompt_formatting(self):
        formatted = self.retriever.format_lessons_for_prompt([self.promoted_lesson])
        self.assertIn("<trusted_lessons>", formatted)
        self.assertIn("Vendor contract renewal requires reply.", formatted)
        self.assertIn("</trusted_lessons>", formatted)


if __name__ == "__main__":
    unittest.main()
