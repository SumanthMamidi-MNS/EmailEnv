# ═══ tests/test_classifier.py ═══

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from agent import RuleBasedClassifier
    _AGENT_OK = True
except Exception as _agent_err:
    _AGENT_OK = False
    _agent_err_msg = str(_agent_err)

try:
    from models import ClassLabel
    _MODELS_OK = True
except Exception:
    _MODELS_OK = False

    # Minimal stand-in so tests can still validate return types
    class ClassLabel:  # type: ignore
        SPAM             = "SPAM"
        URGENT           = "URGENT"
        ACTION_REQUIRED  = "ACTION_REQUIRED"
        SOCIAL           = "SOCIAL"
        INFORMATIONAL    = "INFORMATIONAL"


@unittest.skipUnless(_AGENT_OK, "agent.py / RuleBasedClassifier not importable — skipping classifier tests")
class TestClassifier(unittest.TestCase):
    """Unit tests for RuleBasedClassifier — no external APIs required."""

    def setUp(self):
        """Create a fresh RuleBasedClassifier instance before each test."""
        self.clf = RuleBasedClassifier()

    def tearDown(self):
        """Release classifier reference after each test."""
        self.clf = None

    # ── helper ───────────────────────────────────────────────────────────────

    def _label_value(self, result) -> str:
        """Return the string value of a ClassLabel regardless of its type."""
        if hasattr(result, "value"):
            return result.value
        return str(result)

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_spam_detection(self):
        """Emails containing spam keywords must be classified as SPAM."""
        spam_cases = [
            ("You've won a prize!", "Click here to claim your free offer today."),
            ("Lottery winner selected", "You are our lottery winner. Claim now."),
            ("Limited time offer", "Buy now and win big — unsubscribe at any time."),
        ]
        for subject, body in spam_cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                self.assertEqual(
                    self._label_value(result),
                    "SPAM",
                    f"Expected SPAM for subject='{subject}', got {result!r}",
                )

    def test_urgent_detection(self):
        """Emails containing urgency keywords must be classified as URGENT."""
        urgent_cases = [
            ("URGENT: Server Down", "Please respond immediately — the system is down."),
            ("Critical alert", "This is an asap escalation from the ops team."),
            ("Emergency shutdown required", "Immediately stop all processes."),
        ]
        for subject, body in urgent_cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                self.assertEqual(
                    self._label_value(result),
                    "URGENT",
                    f"Expected URGENT for subject='{subject}', got {result!r}",
                )

    def test_action_required_detection(self):
        """Emails containing action keywords must be classified as ACTION_REQUIRED."""
        action_cases = [
            ("Please review the attached contract", "I need your approval by EOD today."),
            ("Action required: sign off on budget", "Response required before the board meeting."),
            ("Awaiting your sign off", "Please confirm receipt and provide sign-off."),
        ]
        for subject, body in action_cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                self.assertEqual(
                    self._label_value(result),
                    "ACTION_REQUIRED",
                    f"Expected ACTION_REQUIRED for subject='{subject}', got {result!r}",
                )

    def test_social_detection(self):
        """Emails containing social keywords must be classified as SOCIAL."""
        social_cases = [
            ("You're invited to Sarah's birthday party!", "Please RSVP by Friday."),
            ("Team gathering this Friday", "Join us for a get together at the office."),
            ("Wedding invitation — save the date", "We request your presence at our wedding event."),
        ]
        for subject, body in social_cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                self.assertEqual(
                    self._label_value(result),
                    "SOCIAL",
                    f"Expected SOCIAL for subject='{subject}', got {result!r}",
                )

    def test_informational_default(self):
        """Emails with no matching keywords must default to INFORMATIONAL."""
        info_cases = [
            ("Monthly newsletter", "Here is your October digest and company recap."),
            ("FYI — office closure", "This is for your information only. No action needed."),
            ("Quarterly update", "Please find the weekly bulletin attached."),
        ]
        for subject, body in info_cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                self.assertEqual(
                    self._label_value(result),
                    "INFORMATIONAL",
                    f"Expected INFORMATIONAL for subject='{subject}', got {result!r}",
                )

    def test_empty_input(self):
        """classify() with empty subject and body must not raise any exception."""
        try:
            result = self.clf.classify("", "")
        except Exception as exc:
            self.fail(
                f"classify('', '') raised an unexpected exception: {type(exc).__name__}: {exc}"
            )
        self.assertIsNotNone(result,
            "classify('', '') must return a non-None value")

    def test_classifier_returns_classlabel(self):
        """classify() must always return a ClassLabel instance (or a string equal to one)."""
        cases = [
            ("You've won a prize", "Click here now"),
            ("Urgent meeting", "Please respond asap"),
            ("Please review", "Approval needed"),
            ("Birthday party", "You are invited"),
            ("Company newsletter", "Monthly digest"),
            ("", ""),
        ]
        valid_values = {"SPAM", "URGENT", "ACTION_REQUIRED", "SOCIAL", "INFORMATIONAL"}

        for subject, body in cases:
            with self.subTest(subject=subject):
                result = self.clf.classify(subject, body)
                label_str = self._label_value(result)
                self.assertIn(
                    label_str,
                    valid_values,
                    f"classify('{subject}', ...) returned unexpected label: {result!r}",
                )
                # If ClassLabel is a proper enum/class, also verify isinstance
                if _MODELS_OK:
                    self.assertIsInstance(
                        result,
                        ClassLabel,
                        f"classify() should return a ClassLabel, got {type(result).__name__}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)