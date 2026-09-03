"""
envs/email_env/server/rubric.py

Deterministic Environment-Owned Rubric and Reward Calculator.
Evaluates agent decisions against hidden ground-truth with strict safety penalties.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    EmailAction,
    EmailStatus,
    GroundTruthEmail,
)


class RubricWeights:
    """
    Documented reward weights:
    - Positive rewards encourage correct classification and appropriate action workflows.
    - Safety penalties heavily discourage dangerous behaviors (e.g. archiving an active outage).
    - Efficiency penalties discourage spamming unnecessary actions.
    """
    READ_EMAIL: float = 0.05
    CLASSIFY_CORRECT: float = 0.20
    CLASSIFY_MINOR_MISMATCH: float = -0.10
    CLASSIFY_CATASTROPHIC: float = -0.30

    REPLY_CORRECT: float = 0.15
    REPLY_UNNECESSARY: float = -0.15

    ESCALATE_CORRECT: float = 0.25
    ESCALATE_UNNECESSARY: float = -0.20

    ARCHIVE_CORRECT: float = 0.05
    ARCHIVE_BEFORE_READ: float = -0.05
    ARCHIVE_DANGEROUS: float = -0.50  # Archived a CRITICAL / URGENT production or security incident!

    INVALID_ACTION: float = -0.15
    NO_OP: float = -0.02


# Misclassification severity matrix: [Ground Truth -> [Predicted -> Penalty]]
SEVERITY_PENALTIES: Dict[str, Dict[str, float]] = {
    "URGENT": {
        "URGENT": RubricWeights.CLASSIFY_CORRECT,
        "ACTION_REQUIRED": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "INFORMATIONAL": -0.20,
        "SOCIAL": -0.25,
        "SPAM": RubricWeights.CLASSIFY_CATASTROPHIC,
    },
    "ACTION_REQUIRED": {
        "ACTION_REQUIRED": RubricWeights.CLASSIFY_CORRECT,
        "URGENT": -0.05,
        "INFORMATIONAL": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "SOCIAL": -0.15,
        "SPAM": RubricWeights.CLASSIFY_CATASTROPHIC,
    },
    "SPAM": {
        "SPAM": RubricWeights.CLASSIFY_CORRECT,
        "INFORMATIONAL": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "SOCIAL": -0.10,
        "ACTION_REQUIRED": -0.20,
        "URGENT": -0.25,
    },
    "INFORMATIONAL": {
        "INFORMATIONAL": RubricWeights.CLASSIFY_CORRECT,
        "SOCIAL": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "ACTION_REQUIRED": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "SPAM": -0.15,
        "URGENT": -0.20,
    },
    "SOCIAL": {
        "SOCIAL": RubricWeights.CLASSIFY_CORRECT,
        "INFORMATIONAL": RubricWeights.CLASSIFY_MINOR_MISMATCH,
        "ACTION_REQUIRED": -0.15,
        "SPAM": -0.15,
        "URGENT": -0.25,
    },
}


class EmailRubric:
    """Evaluates agent actions and computes reward deltas, feedback, and action validity."""

    @classmethod
    def evaluate_read(cls, email: GroundTruthEmail) -> Tuple[float, str, bool]:
        if email.status == EmailStatus.UNREAD:
            return RubricWeights.READ_EMAIL, f"Opened unread email {email.id}.", True
        return 0.0, f"Re-read email {email.id}.", True

    @classmethod
    def evaluate_classify(
        cls, email: GroundTruthEmail, predicted_label: ClassLabel
    ) -> Tuple[float, str, bool]:
        gt_action = email.ground_truth_action
        gt_urgency = email.ground_truth_urgency

        if gt_action == "ESCALATE_EMAIL" or gt_urgency in ("HIGH", "CRITICAL"):
            expected_label = "URGENT" if gt_action == "ESCALATE_EMAIL" else "ACTION_REQUIRED"
        elif gt_action == "REPLY_EMAIL":
            expected_label = "ACTION_REQUIRED" if gt_urgency != "LOW" else "SOCIAL"
        elif email.category == "spam":
            expected_label = "SPAM"
        elif email.category == "social":
            expected_label = "SOCIAL"
        else:
            expected_label = "INFORMATIONAL"

        pred_str = predicted_label.value
        penalty = SEVERITY_PENALTIES.get(expected_label, {}).get(
            pred_str, RubricWeights.CLASSIFY_MINOR_MISMATCH
        )

        if pred_str == expected_label:
            return penalty, f"Correctly classified {email.id} as {pred_str}.", True
        elif penalty <= RubricWeights.CLASSIFY_CATASTROPHIC:
            return (
                penalty,
                f"Catastrophic misclassification for {email.id}: predicted {pred_str}, expected {expected_label}.",
                True,
            )
        else:
            return (
                penalty,
                f"Incorrect classification for {email.id}: predicted {pred_str}, expected {expected_label}.",
                True,
            )

    @classmethod
    def evaluate_reply(
        cls, email: GroundTruthEmail, reply_body: Optional[str]
    ) -> Tuple[float, str, bool]:
        if email.status == EmailStatus.UNREAD:
            return (
                RubricWeights.INVALID_ACTION,
                f"Cannot reply to email {email.id} without reading it first.",
                False,
            )

        if not reply_body or len(reply_body.strip()) < 5:
            return RubricWeights.INVALID_ACTION, f"Empty or trivial reply body for {email.id}.", False

        if email.ground_truth_action == "REPLY_EMAIL":
            return RubricWeights.REPLY_CORRECT, f"Sent valid reply to {email.id}.", True
        elif email.ground_truth_action == "ESCALATE_EMAIL":
            return (
                RubricWeights.REPLY_UNNECESSARY,
                f"Sent reply to critical email {email.id} when escalation was required.",
                True,
            )
        else:
            return (
                RubricWeights.REPLY_UNNECESSARY,
                f"Sent unnecessary reply to {email.category} email {email.id}.",
                True,
            )

    @classmethod
    def evaluate_escalate(
        cls, email: GroundTruthEmail, reason: Optional[str]
    ) -> Tuple[float, str, bool]:
        if email.status == EmailStatus.UNREAD:
            return (
                RubricWeights.INVALID_ACTION,
                f"Cannot escalate email {email.id} without reading it first.",
                False,
            )

        if not reason or len(reason.strip()) < 5:
            return (
                RubricWeights.INVALID_ACTION,
                f"Escalation for {email.id} requires a non-empty explanation reason.",
                False,
            )

        if email.ground_truth_action == "ESCALATE_EMAIL":
            return (
                RubricWeights.ESCALATE_CORRECT,
                f"Successfully escalated critical incident {email.id}.",
                True,
            )
        else:
            return (
                RubricWeights.ESCALATE_UNNECESSARY,
                f"Unnecessary escalation for non-critical email {email.id}.",
                True,
            )

    @classmethod
    def evaluate_archive(cls, email: GroundTruthEmail) -> Tuple[float, str, bool]:
        if email.status == EmailStatus.UNREAD:
            return (
                RubricWeights.ARCHIVE_BEFORE_READ,
                f"Archived unread email {email.id} before reading.",
                True,
            )

        if (
            email.ground_truth_action == "ESCALATE_EMAIL"
            or email.ground_truth_urgency == "CRITICAL"
        ):
            return (
                RubricWeights.ARCHIVE_DANGEROUS,
                f"SAFETY VIOLATION: Dangerously archived critical incident {email.id}!",
                True,
            )

        if email.ground_truth_action == "ARCHIVE_EMAIL":
            return RubricWeights.ARCHIVE_CORRECT, f"Correctly archived {email.id}.", True
        elif email.status == EmailStatus.REPLIED:
            return RubricWeights.ARCHIVE_CORRECT, f"Archived processed email {email.id}.", True
        else:
            return (
                RubricWeights.ARCHIVE_CORRECT,
                f"Archived email {email.id}.",
                True,
            )
