"""
tests/test_fallback.py

Tests for LocalFallbackAgent verifying deterministic decision logic,
zero external API dependencies, and valid action generation.
"""

import unittest
from envs.email_env.models import ActionType, ClassLabel
from agents.fallback import LocalFallbackAgent


class TestLocalFallbackAgent(unittest.TestCase):

    def setUp(self):
        self.fallback = LocalFallbackAgent()

    def test_spam_detection(self):
        decision = self.fallback.decide(
            email_id="e1",
            subject="You won the crypto lottery prize!",
            sender="spam@crypto.org",
            body="Claim your 100x bitcoin reward now.",
        )
        self.assertEqual(decision.classification, ClassLabel.SPAM)
        self.assertEqual(decision.action.action_type, ActionType.ARCHIVE_EMAIL)
        self.assertEqual(decision.source, "fallback")

    def test_urgent_outage_detection(self):
        decision = self.fallback.decide(
            email_id="e2",
            subject="P0 Alert: Database is down and all services offline",
            sender="ops@company.com",
            body="Primary postgres cluster unreachable. Escalate immediately.",
        )
        self.assertEqual(decision.classification, ClassLabel.URGENT)
        self.assertEqual(decision.action.action_type, ActionType.ESCALATE_EMAIL)
        self.assertIsNotNone(decision.action.reason)

    def test_action_required_reply(self):
        decision = self.fallback.decide(
            email_id="e3",
            subject="Please submit your Q1 timesheet by Friday",
            sender="hr@company.com",
            body="Kindly submit your hours before the deadline.",
        )
        self.assertEqual(decision.classification, ClassLabel.ACTION_REQUIRED)
        self.assertEqual(decision.action.action_type, ActionType.REPLY_EMAIL)
        self.assertIsNotNone(decision.action.body)


if __name__ == "__main__":
    unittest.main()
