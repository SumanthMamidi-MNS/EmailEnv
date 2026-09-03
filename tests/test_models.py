"""
tests/test_models.py

Tests for Pydantic Action, Observation, and State models,
verifying serialization and ground truth isolation.
"""

import unittest
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    EmailAction,
    EmailObservation,
    EmailState,
    EmailStatus,
    GroundTruthEmail,
    InboxItem,
    SentItem,
)


class TestEmailModels(unittest.TestCase):

    def test_email_action_factories(self):
        act_read = EmailAction.read("email_1")
        self.assertEqual(act_read.action_type, ActionType.READ_EMAIL)
        self.assertEqual(act_read.email_id, "email_1")

        act_classify = EmailAction.classify("email_1", ClassLabel.URGENT)
        self.assertEqual(act_classify.action_type, ActionType.CLASSIFY_EMAIL)
        self.assertEqual(act_classify.label, ClassLabel.URGENT)

        act_reply = EmailAction.reply("email_1", "Acknowledged.")
        self.assertEqual(act_reply.action_type, ActionType.REPLY_EMAIL)
        self.assertEqual(act_reply.body, "Acknowledged.")

        act_escalate = EmailAction.escalate("email_1", "Outage detected.")
        self.assertEqual(act_escalate.action_type, ActionType.ESCALATE_EMAIL)
        self.assertEqual(act_escalate.reason, "Outage detected.")

        act_archive = EmailAction.archive("email_1")
        self.assertEqual(act_archive.action_type, ActionType.ARCHIVE_EMAIL)

        act_noop = EmailAction.no_op()
        self.assertEqual(act_noop.action_type, ActionType.NO_OP)

    def test_ground_truth_isolation(self):
        gt_email = GroundTruthEmail(
            id="test_01",
            category="production_incident",
            difficulty="STARTER",
            subject="Database down",
            sender="ops@company.com",
            body="Primary node failed.",
            ground_truth_action="ESCALATE_EMAIL",
            ground_truth_urgency="CRITICAL",
        )

        view = gt_email.to_view()
        # View must contain public fields but NOT ground_truth_action or ground_truth_urgency
        self.assertEqual(view.id, "test_01")
        self.assertEqual(view.subject, "Database down")
        self.assertFalse(hasattr(view, "ground_truth_action"))
        self.assertFalse(hasattr(view, "ground_truth_urgency"))

        inbox_item = gt_email.to_inbox_item()
        self.assertEqual(inbox_item.id, "test_01")
        self.assertFalse(hasattr(inbox_item, "body"))
        self.assertFalse(hasattr(inbox_item, "ground_truth_action"))

    def test_observation_structure(self):
        obs = EmailObservation(
            inbox_summary=[
                InboxItem(id="e1", sender="a@b.com", subject="Test", status=EmailStatus.UNREAD)
            ],
            step=1,
            steps_remaining=10,
        )
        self.assertEqual(len(obs.inbox_summary), 1)
        self.assertEqual(obs.step, 1)
        self.assertIn("READ_EMAIL", obs.available_actions)


if __name__ == "__main__":
    unittest.main()
