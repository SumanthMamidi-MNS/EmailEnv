"""
tests/test_rubric.py

Tests for EmailRubric reward calculation, severity penalties,
and dangerous action detection.
"""

import unittest
from envs.email_env.models import ClassLabel, EmailStatus, GroundTruthEmail
from envs.email_env.server.rubric import EmailRubric, RubricWeights


class TestEmailRubric(unittest.TestCase):

    def setUp(self):
        self.spam_email = GroundTruthEmail(
            id="spam_01",
            category="spam",
            difficulty="STARTER",
            subject="Win $1M",
            sender="spam@bad.com",
            body="Click here.",
            ground_truth_action="ARCHIVE_EMAIL",
            ground_truth_urgency="LOW",
            status=EmailStatus.READ,
        )

        self.critical_email = GroundTruthEmail(
            id="outage_01",
            category="production_incident",
            difficulty="STARTER",
            subject="Database down",
            sender="ops@company.com",
            body="Production is offline.",
            ground_truth_action="ESCALATE_EMAIL",
            ground_truth_urgency="CRITICAL",
            status=EmailStatus.READ,
        )

    def test_correct_read(self):
        unread = self.spam_email.model_copy(update={"status": EmailStatus.UNREAD})
        reward, feedback, is_valid = EmailRubric.evaluate_read(unread)
        self.assertEqual(reward, RubricWeights.READ_EMAIL)
        self.assertTrue(is_valid)

    def test_correct_vs_catastrophic_classification(self):
        # Correct classification
        reward, _, is_valid = EmailRubric.evaluate_classify(self.spam_email, ClassLabel.SPAM)
        self.assertEqual(reward, RubricWeights.CLASSIFY_CORRECT)
        self.assertTrue(is_valid)

        # Catastrophic classification: labeling critical outage as SPAM
        reward, _, is_valid = EmailRubric.evaluate_classify(self.critical_email, ClassLabel.SPAM)
        self.assertEqual(reward, RubricWeights.CLASSIFY_CATASTROPHIC)
        self.assertLess(reward, 0.0)
        self.assertTrue(is_valid)

    def test_dangerous_archive_penalty(self):
        # Archiving a critical outage should trigger heavy safety penalty
        reward, feedback, is_valid = EmailRubric.evaluate_archive(self.critical_email)
        self.assertEqual(reward, RubricWeights.ARCHIVE_DANGEROUS)
        self.assertIn("SAFETY VIOLATION", feedback)
        self.assertTrue(is_valid)

    def test_correct_escalation(self):
        reward, feedback, is_valid = EmailRubric.evaluate_escalate(
            self.critical_email, reason="Production database is offline."
        )
        self.assertEqual(reward, RubricWeights.ESCALATE_CORRECT)
        self.assertIn("Successfully escalated", feedback)
        self.assertTrue(is_valid)

    def test_unnecessary_escalation(self):
        reward, feedback, is_valid = EmailRubric.evaluate_escalate(
            self.spam_email, reason="Escalating spam to VP."
        )
        self.assertEqual(reward, RubricWeights.ESCALATE_UNNECESSARY)
        self.assertLess(reward, 0.0)
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
