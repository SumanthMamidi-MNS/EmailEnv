# ═══ tests/test_grader.py ═══

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.environment import EmailEnv
    from models import ClassLabel, Action
    _IMPORTS_OK = True
except Exception as _import_err:
    _IMPORTS_OK = False
    _import_err_msg = str(_import_err)


def _get_first_email_id(obs):
    """Extract the first email ID from an observation object."""
    ids = _get_email_ids(obs)
    return ids[0] if ids else None


def _get_email_ids(obs):
    """Extract all email IDs from an observation object."""
    if hasattr(obs, "inbox_summary"):
        return [item.id for item in obs.inbox_summary]
    if isinstance(obs, dict):
        inbox = obs.get("inbox_summary", obs.get("inbox", obs.get("email_ids", [])))
        if inbox and isinstance(inbox[0], dict):
            return [item["id"] for item in inbox]
        return list(inbox)
    return []


@unittest.skipUnless(_IMPORTS_OK, f"Backend imports unavailable — skipping grader tests")
class TestGrader(unittest.TestCase):
    """Tests for the deterministic grader via EmailEnv.grade()."""

    def setUp(self):
        """Initialise a fresh environment before every test."""
        self.env = EmailEnv("task_1")
        self.obs = self.env.reset()
        self.email_ids = _get_email_ids(self.obs)

    def tearDown(self):
        """Release environment reference after every test."""
        self.env = None
        self.obs = None
        self.email_ids = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _read_and_classify(self, eid, label):
        """Convenience: read an email then classify it with the given label."""
        self.env.step(Action.read_email(eid))
        obs, reward, done, info = self.env.step(
            Action.classify_email(eid, label)
        )
        return obs, reward, done, info

    def _complete_episode_correctly(self):
        """
        Run a minimal valid episode: read every email, classify as
        INFORMATIONAL, then archive. This ensures grade() has enough
        actions to produce a meaningful score.
        """
        for eid in self.email_ids:
            self.env.step(Action.read_email(eid))
            self.env.step(Action.classify_email(eid, ClassLabel.INFORMATIONAL))
            self.env.step(Action.archive_email(eid))

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_score_range(self):
        """grade() must return a score strictly between 0.0 and 1.0 inclusive."""
        self._complete_episode_correctly()
        result = self.env.grade()
        self.assertIsInstance(result, dict,
            "grade() should return a dict")
        score = result.get("score")
        self.assertIsNotNone(score,
            "grade() result dict must contain a 'score' key")
        self.assertGreaterEqual(float(score), 0.0,
            f"Score {score} is below 0.0")
        self.assertLessEqual(float(score), 1.0,
            f"Score {score} is above 1.0")

    def test_correct_classification_rewards(self):
        """Classifying an email with a correct label must yield a positive reward."""
        self.assertTrue(self.email_ids,
            "Inbox must not be empty for this test")
        eid = self.email_ids[0]
        self.env.step(Action.read_email(eid))

        # Attempt each label and accept the first one that gives a positive reward.
        # This avoids hard-coding which label is 'correct' for a given task email.
        positive_found = False
        for candidate in list(ClassLabel):
            env2 = EmailEnv("task_1")
            obs2 = env2.reset()
            env2.step(Action.read_email(eid))
            _, reward, _, info = env2.step(
                Action.classify_email(eid, candidate)
            )
            if float(reward) > 0:
                positive_found = True
                break

        self.assertTrue(positive_found,
            "At least one ClassLabel should yield a positive reward when classifying a real email")

    def test_wrong_classification_penalises(self):
        """
        After reading an email that is NOT spam, classifying it as SPAM
        while the ground truth differs should produce a non-positive reward
        on at least one email in the inbox.
        """
        self.assertTrue(self.email_ids,
            "Inbox must not be empty for this test")

        penalised = False
        for eid in self.email_ids:
            env2 = EmailEnv("task_1")
            env2.reset()
            env2.step(Action.read_email(eid))
            _, reward_spam, _, _ = env2.step(
                Action.classify_email(eid, ClassLabel.SPAM)
            )

            env3 = EmailEnv("task_1")
            env3.reset()
            env3.step(Action.read_email(eid))
            _, reward_info, _, _ = env3.step(
                Action.classify_email(eid, ClassLabel.INFORMATIONAL)
            )

            # At least one combination should show that SPAM scores worse
            if float(reward_spam) < float(reward_info):
                penalised = True
                break

        self.assertTrue(penalised,
            "Classifying a non-spam email as SPAM should penalise compared to a neutral label")

    def test_score_is_deterministic(self):
        """Running identical action sequences on two fresh environments yields the same score."""
        def _run_episode(task_id):
            env = EmailEnv(task_id)
            obs = env.reset()
            ids = _get_email_ids(obs)
            for eid in ids:
                env.step(Action.read_email(eid))
                env.step(Action.classify_email(eid, ClassLabel.INFORMATIONAL))
                env.step(Action.archive_email(eid))
            return env.grade()

        result_a = _run_episode("task_1")
        result_b = _run_episode("task_1")

        score_a = float(result_a.get("score", -1))
        score_b = float(result_b.get("score", -1))
        self.assertAlmostEqual(score_a, score_b, places=6,
            msg=f"Same episode produced different scores: {score_a} vs {score_b}")

    def test_invalid_action_penalises(self):
        """
        Attempting to classify an email that has not been read first
        (or any otherwise-invalid sequence) should yield a non-positive reward
        or mark the action as invalid in the info dict.
        """
        self.assertTrue(self.email_ids,
            "Inbox must not be empty for this test")
        eid = self.email_ids[0]

        # Classify WITHOUT reading first — most environments penalise this
        _, reward, _, info = self.env.step(
            Action.classify_email(eid, ClassLabel.SPAM)
        )

        is_penalised = (float(reward) <= 0) or (info.get("valid", True) is False)
        self.assertTrue(is_penalised,
            f"Expected non-positive reward or invalid=False for unread classify, "
            f"got reward={reward}, info={info}")


if __name__ == "__main__":
    unittest.main(verbosity=2)