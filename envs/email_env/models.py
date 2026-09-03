"""
envs/email_env/models.py

Pydantic models for OpenEnv Action, Observation, and State.
Follows OpenEnv contract with strict ground-truth isolation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ClassLabel(str, Enum):
    SPAM = "SPAM"
    URGENT = "URGENT"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    INFORMATIONAL = "INFORMATIONAL"
    SOCIAL = "SOCIAL"


class EmailStatus(str, Enum):
    UNREAD = "UNREAD"
    READ = "READ"
    ARCHIVED = "ARCHIVED"
    ESCALATED = "ESCALATED"
    REPLIED = "REPLIED"


class ActionType(str, Enum):
    READ_EMAIL = "READ_EMAIL"
    CLASSIFY_EMAIL = "CLASSIFY_EMAIL"
    REPLY_EMAIL = "REPLY_EMAIL"
    ARCHIVE_EMAIL = "ARCHIVE_EMAIL"
    ESCALATE_EMAIL = "ESCALATE_EMAIL"
    NO_OP = "NO_OP"


class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DifficultyLevel(str, Enum):
    STARTER = "STARTER"
    MEDIUM = "MEDIUM"
    ADVANCED = "ADVANCED"
    ADVERSARIAL = "ADVERSARIAL"
    HELD_OUT = "HELD_OUT"
    REGRESSION = "REGRESSION"


# ─────────────────────────────────────────────────────────────────────────────
# Action Model (Agent -> Environment)
# ─────────────────────────────────────────────────────────────────────────────

class EmailAction(BaseModel):
    """Action submitted by an agent to the EmailEnvironment."""
    model_config = ConfigDict(extra="ignore")

    action_type: ActionType
    email_id: Optional[str] = None
    label: Optional[ClassLabel] = None
    body: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def read(cls, email_id: str) -> EmailAction:
        return cls(action_type=ActionType.READ_EMAIL, email_id=email_id)

    @classmethod
    def classify(cls, email_id: str, label: ClassLabel) -> EmailAction:
        return cls(action_type=ActionType.CLASSIFY_EMAIL, email_id=email_id, label=label)

    @classmethod
    def reply(cls, email_id: str, body: str) -> EmailAction:
        return cls(action_type=ActionType.REPLY_EMAIL, email_id=email_id, body=body)

    @classmethod
    def archive(cls, email_id: str) -> EmailAction:
        return cls(action_type=ActionType.ARCHIVE_EMAIL, email_id=email_id)

    @classmethod
    def escalate(cls, email_id: str, reason: str) -> EmailAction:
        return cls(action_type=ActionType.ESCALATE_EMAIL, email_id=email_id, reason=reason)

    @classmethod
    def no_op(cls) -> EmailAction:
        return cls(action_type=ActionType.NO_OP)


# ─────────────────────────────────────────────────────────────────────────────
# Observation Projections (Environment -> Agent)
# Ground-truth labels and evaluation metrics are strictly omitted.
# ─────────────────────────────────────────────────────────────────────────────

class EmailView(BaseModel):
    """Sanitized view of an email that the agent has opened/read."""
    model_config = ConfigDict(extra="ignore")

    id: str
    sender: str
    subject: str
    body: str
    status: EmailStatus
    assigned_label: Optional[ClassLabel] = None
    assigned_action: Optional[str] = None


class InboxItem(BaseModel):
    """Public inbox item visible to agent without reading full body."""
    model_config = ConfigDict(extra="ignore")

    id: str
    sender: str
    subject: str
    status: EmailStatus


class SentItem(BaseModel):
    """Summary of a sent reply."""
    model_config = ConfigDict(extra="ignore")

    in_reply_to: str
    body_preview: str


class EmailObservation(BaseModel):
    """Observation object returned by environment to agent on reset/step."""
    model_config = ConfigDict(extra="ignore")

    current_email: Optional[EmailView] = None
    inbox_summary: List[InboxItem] = Field(default_factory=list)
    sent_summary: List[SentItem] = Field(default_factory=list)
    step: int = 0
    steps_remaining: int = 0
    available_actions: List[str] = Field(
        default_factory=lambda: [
            ActionType.READ_EMAIL.value,
            ActionType.CLASSIFY_EMAIL.value,
            ActionType.REPLY_EMAIL.value,
            ActionType.ARCHIVE_EMAIL.value,
            ActionType.ESCALATE_EMAIL.value,
            ActionType.NO_OP.value,
        ]
    )
    last_action_valid: bool = True
    last_action_feedback: str = ""
    done: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Environment Internal State (Never exposed to Agent)
# ─────────────────────────────────────────────────────────────────────────────

class GroundTruthEmail(BaseModel):
    """Complete ground-truth email record maintained only by the environment."""
    model_config = ConfigDict(extra="ignore")

    id: str
    category: str
    difficulty: str
    subject: str
    sender: str
    body: str
    ground_truth_action: str
    ground_truth_urgency: str
    status: EmailStatus = EmailStatus.UNREAD
    assigned_label: Optional[ClassLabel] = None
    assigned_action: Optional[str] = None
    assigned_reason: Optional[str] = None
    reply_body: Optional[str] = None

    def to_view(self) -> EmailView:
        return EmailView(
            id=self.id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            status=self.status,
            assigned_label=self.assigned_label,
            assigned_action=self.assigned_action,
        )

    def to_inbox_item(self) -> InboxItem:
        return InboxItem(
            id=self.id,
            sender=self.sender,
            subject=self.subject,
            status=self.status,
        )


class EmailState(BaseModel):
    """Complete environment state snapshot."""
    model_config = ConfigDict(extra="ignore")

    episode_id: str
    difficulty: str
    step_count: int = 0
    max_steps: int = 30
    emails: List[GroundTruthEmail] = Field(default_factory=list)
    sent_box: List[Dict[str, Any]] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    cumulative_reward: float = 0.0

    def get_email(self, email_id: str) -> Optional[GroundTruthEmail]:
        for email in self.emails:
            if email.id == email_id:
                return email
        return None
