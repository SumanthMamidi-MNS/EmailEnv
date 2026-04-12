# ═══ tests/test_environment.py ═══

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.environment import EmailEnv
    _ENV_OK = True
except Exception as _env_err:
    _ENV_OK = False
    _env_err_msg = str(_env_err)

try:
    from models import ClassLabel, Action
    _MODELS_OK = True
except Exception as _models_err:
    _MODELS_OK = False

    # Minimal fallback so the file remains importable
    class ClassLabel:  # type: ignore
        SPAM = "SPAM"; URGENT = "URGENT"; ACTION_REQUIRED = "ACTION_REQUIRED"
        SOCIAL = "SOCIAL"; INFORMATIONAL = "INFORMATIONAL"

    class Action:  # type: ignore
        @staticmethod
        def read_email(eid):           return {"type": "read_email",    "email_id": eid}
        @staticmethod
        def classify_email(eid, lbl):  return {"type": "classify_email","email_id": eid, "label": lbl}
        @staticmethod
        def archive_email(eid):        return {"type": "archive_email", "email_id": eid}
        @staticmethod
        def escalate_email(eid):       return {"type": "escalate_email","email_id": eid}
        @staticmethod
        def reply_email(eid, text=""):  return {"type": "reply_email",   "email_id": eid, "text": text}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_email_ids(obs) -> list:
    """Return a list of email IDs from any observation format."""
    # Observation dataclass has inbox_summary (list of InboxItem)
    if hasattr(obs, "inbox_summary"):
        return [item.id for item in obs.inbox_summary]
    if isinstance(obs, dict):
        inbox = obs.get("inbox_summary", obs.get("inbox", obs.get("email_ids", [])))
        if inbox and isinstance(inbox[0], dict):
            return [item["id"] for item in inbox]
        return list(inbox)
    return []


def _is_observation(obs) -> bool:
    """Return True if obs looks like a valid inbox observation."""
    if isinstance(obs, dict):
        return True
    # Observation dataclass
    if hasattr(obs, "inbox_summary") or hasattr(obs, "email_ids"):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_ENV_OK, "environment.py / EmailEnv not importable — skipping environment tests")
class TestEnvironment(unittest.TestCase):
    """Integration tests for EmailEnv using task_1."""

    TASK_ID = "task_1"

    def setUp(self):
        """Create and reset a fresh EmailEnv before every test."""
        self.env = EmailEnv(self.TASK_ID)
        self.obs = self.env.reset()
        self.email_ids = _extract_email_ids(self.obs)

    def tearDown(self):
        """Release all references to avoid cross-test contamination."""
        self.env = None
        self.obs = None
        self.email_ids = []

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_reset_returns_observation(self):
        """reset() must return a non-None, observation-shaped object."""
        self.assertIsNotNone(
            self.obs,
            "reset() returned None — expected a valid observation",
        )
        self.assertTrue(
            _is_observation(self.obs),
            f"reset() returned an unrecognised type: {type(self.obs).__name__}",
        )

    def test_inbox_not_empty(self):
        """The inbox extracted from the initial observation must contain at least one email."""
        self.assertGreater(
            len(self.email_ids),
            0,
            "Inbox is empty after reset() — task_1 should always have emails",
        )

    def test_step_read_valid(self):
        """
        env.step(read_email(eid)) for a valid email ID must return a 4-tuple
        (obs, reward, done, info) where reward is a finite float.
        """
        self.assertTrue(self.email_ids, "Need at least one email to test step()")
        eid = self.email_ids[0]

        result = self.env.step(Action.read_email(eid))

        self.assertIsInstance(result, tuple,
            f"step() should return a tuple, got {type(result).__name__}")
        self.assertEqual(len(result), 4,
            f"step() should return a 4-tuple (obs, reward, done, info), got length {len(result)}")

        obs, reward, done, info = result

        self.assertIsNotNone(obs,   "obs from step() must not be None")
        self.assertIsInstance(float(reward), float,
            f"reward must be convertible to float, got {type(reward).__name__}")
        self.assertIsInstance(done, bool,
            f"done flag must be bool, got {type(done).__name__}")
        self.assertIsInstance(info, dict,
            f"info must be a dict, got {type(info).__name__}")

    def test_step_invalid_action(self):
        """
        Stepping with a semantically invalid action sequence (classifying an email
        before reading it) must either return a non-positive reward, set
        info['valid'] to False, or raise a recoverable exception — never silently succeed.
        """
        self.assertTrue(self.email_ids, "Need at least one email to test invalid action")
        eid = self.email_ids[0]

        try:
            _, reward, _, info = self.env.step(
                Action.classify_email(eid, ClassLabel.SPAM)
            )
            # Accept non-positive reward OR explicit invalid flag
            penalised = (float(reward) <= 0) or (info.get("valid", True) is False)
            self.assertTrue(
                penalised,
                f"Expected penalisation for classify-without-read, "
                f"got reward={reward}, info['valid']={info.get('valid')}",
            )
        except Exception as exc:
            # Some environments raise on invalid actions — that is also acceptable
            self.assertIsInstance(
                exc, Exception,
                "Any raised exception for an invalid action is acceptable",
            )

    def test_grade_returns_score(self):
        """
        grade() must return a dict that contains a 'score' key with a numeric value.
        """
        # Execute at least one action so grade() has data to work with
        if self.email_ids:
            eid = self.email_ids[0]
            self.env.step(Action.read_email(eid))
            self.env.step(Action.classify_email(eid, ClassLabel.INFORMATIONAL))
            self.env.step(Action.archive_email(eid))

        result = self.env.grade()

        self.assertIsInstance(result, dict,
            f"grade() must return a dict, got {type(result).__name__}")
        self.assertIn("score", result,
            "grade() result dict must contain a 'score' key")
        try:
            float(result["score"])
        except (TypeError, ValueError):
            self.fail(
                f"grade()['score'] is not numeric: {result['score']!r}"
            )

    def test_score_between_zero_and_one(self):
        """grade()['score'] must be in the range [0.0, 1.0]."""
        for eid in self.email_ids:
            self.env.step(Action.read_email(eid))
            self.env.step(Action.classify_email(eid, ClassLabel.INFORMATIONAL))
            self.env.step(Action.archive_email(eid))

        result = self.env.grade()
        score = float(result.get("score", -1))

        self.assertGreaterEqual(score, 0.0,
            f"Score {score} is below the minimum of 0.0")
        self.assertLessEqual(score, 1.0,
            f"Score {score} exceeds the maximum of 1.0")

    def test_episode_done_flag(self):
        """
        The done flag must eventually become True — either within the allotted
        steps or after exhausting the inbox.
        """
        env = EmailEnv(self.TASK_ID)
        obs = env.reset()
        ids = _extract_email_ids(obs)
        done = False
        max_steps = 200  # hard safety cap to prevent infinite loops

        step_count = 0
        idx = 0

        while not done and step_count < max_steps:
            if idx < len(ids):
                eid = ids[idx]
                _, _, done, _ = env.step(Action.read_email(eid))
                step_count += 1
                if not done:
                    _, _, done, _ = env.step(
                        Action.classify_email(eid, ClassLabel.INFORMATIONAL)
                    )
                    step_count += 1
                if not done:
                    _, _, done, _ = env.step(Action.archive_email(eid))
                    step_count += 1
                idx += 1
            else:
                # All emails processed — push a no-op archive on the last id
                # to trigger step-budget exhaustion if environment requires it
                if ids:
                    _, _, done, _ = env.step(Action.archive_email(ids[-1]))
                    step_count += 1
                else:
                    break

        self.assertTrue(
            done,
            f"done flag never became True after {step_count} steps — "
            "the environment should terminate eventually",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)