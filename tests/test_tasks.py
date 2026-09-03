"""
tests/test_tasks.py

Tests for ScenarioLoader, verifying schema validation, corrupt record rejection,
and duplicate ID prevention.
"""

import tempfile
import unittest
from pathlib import Path

from envs.email_env.server.tasks import (
    ScenarioLoader,
    ScenarioValidationError,
)


class TestScenarioLoader(unittest.TestCase):

    def setUp(self):
        self.loader = ScenarioLoader()

    def test_load_all_standard_difficulties(self):
        starter = self.loader.load_difficulty("STARTER")
        self.assertEqual(len(starter), 5)

        medium = self.loader.load_difficulty("MEDIUM")
        self.assertEqual(len(medium), 10)

        advanced = self.loader.load_difficulty("ADVANCED")
        self.assertEqual(len(advanced), 15)

        adversarial = self.loader.load_difficulty("ADVERSARIAL")
        self.assertEqual(len(adversarial), 5)

    def test_rejection_of_missing_fields(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
            f.write('{"id": "bad_01", "subject": "Missing category"}\n')
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ScenarioValidationError):
                self.loader.load_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_rejection_of_duplicate_ids(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
            valid_record = (
                '{"id": "dup_01", "category": "spam", "difficulty": "STARTER", '
                '"subject": "A", "sender": "a@b.com", "body": "C", '
                '"ground_truth_action": "ARCHIVE_EMAIL", "ground_truth_urgency": "LOW"}\n'
            )
            f.write(valid_record)
            f.write(valid_record)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ScenarioValidationError):
                self.loader.load_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_rejection_of_invalid_action_or_urgency(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
            bad_record = (
                '{"id": "bad_02", "category": "spam", "difficulty": "STARTER", '
                '"subject": "A", "sender": "a@b.com", "body": "C", '
                '"ground_truth_action": "INVALID_ACTION", "ground_truth_urgency": "LOW"}\n'
            )
            f.write(bad_record)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ScenarioValidationError):
                self.loader.load_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
