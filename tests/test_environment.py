"""
tests/test_environment.py

Tests for EmailEnvironment lifecycle, action validation, state transitions,
and episode termination.
"""

import unittest
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    DifficultyLevel,
    EmailAction,
    EmailStatus,
)
from envs.email_env.server.environment import EmailEnvironment


class TestEmailEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = EmailEnvironment()

    def test_reset_lifecycle(self):
        obs = self.env.reset(seed=42, difficulty=DifficultyLevel.STARTER)
        self.assertIsNotNone(obs)
        self.assertEqual(len(obs.inbox_summary), 5)
        self.assertEqual(obs.step, 0)
        self.assertFalse(obs.done)
        self.assertIsNone(obs.current_email)

    def test_read_email_transition(self):
        obs = self.env.reset(seed=42, difficulty="STARTER")
        first_id = obs.inbox_summary[0].id

        act = EmailAction.read(first_id)
        obs, reward, done, info = self.env.step(act)

        self.assertTrue(obs.last_action_valid)
        self.assertGreater(reward, 0.0)
        self.assertIsNotNone(obs.current_email)
        self.assertEqual(obs.current_email.id, first_id)
        self.assertEqual(obs.current_email.status, EmailStatus.READ)

    def test_classify_and_archive(self):
        obs = self.env.reset(seed=42, difficulty="STARTER")
        first_id = obs.inbox_summary[0].id

        # 1. Read
        self.env.step(EmailAction.read(first_id))

        # 2. Classify
        obs, reward, done, info = self.env.step(
            EmailAction.classify(first_id, ClassLabel.SPAM)
        )
        self.assertTrue(obs.last_action_valid)
        self.assertEqual(obs.current_email.assigned_label, ClassLabel.SPAM)

        # 3. Archive
        obs, reward, done, info = self.env.step(EmailAction.archive(first_id))
        self.assertTrue(obs.last_action_valid)
        self.assertEqual(obs.current_email.status, EmailStatus.ARCHIVED)

    def test_invalid_action_handling(self):
        self.env.reset(seed=42, difficulty="STARTER")

        # Acting on non-existent email
        obs, reward, done, info = self.env.step(EmailAction.read("non_existent_id"))
        self.assertFalse(obs.last_action_valid)
        self.assertLess(reward, 0.0)

        # Reply without reading
        obs, reward, done, info = self.env.step(
            EmailAction.reply("starter_001", "Blind reply.")
        )
        self.assertFalse(obs.last_action_valid)
        self.assertLess(reward, 0.0)

    def test_state_snapshot(self):
        self.env.reset(seed=42, difficulty="STARTER")
        state = self.env.state()
        self.assertEqual(state.difficulty, "STARTER")
        self.assertEqual(len(state.emails), 5)
        self.assertFalse(state.done)


if __name__ == "__main__":
    unittest.main()