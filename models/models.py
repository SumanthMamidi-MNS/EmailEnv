"""
models.py — AI Email Assistant OpenEnv
Data models for Action, Observation, and State.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ClassLabel(str, Enum):
    SPAM            = "SPAM"
    URGENT          = "URGENT"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    INFORMATIONAL   = "INFORMATIONAL"
    SOCIAL          = "SOCIAL"


class EmailStatus(str, Enum):
    UNREAD    = "unread"
    READ      = "read"
    ARCHIVED  = "archived"
    ESCALATED = "escalated"


class ActionType(str, Enum):
    READ_EMAIL      = "read_email"
    CLASSIFY_EMAIL  = "classify_email"
    REPLY_EMAIL     = "reply_email"
    ARCHIVE_EMAIL   = "archive_email"
    ESCALATE_EMAIL  = "escalate_email"
    WRITE_MEMORY    = "write_memory"
    NO_OP           = "no_op"


# ─────────────────────────────────────────────
# Email Core
# ─────────────────────────────────────────────

@dataclass
class Email:
    """Ground-truth email stored in State. Not exposed directly to agent."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str                          # ISO 8601
    priority: ClassLabel                    # ground truth — never in Observation
    status: EmailStatus = EmailStatus.UNREAD
    thread_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    assigned_label: Optional[ClassLabel] = None   # agent's classification

    def to_view(self) -> "EmailView":
        """Returns a read-only Observation projection (hides ground-truth priority)."""
        return EmailView(
            id=self.id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            timestamp=self.timestamp,
            thread_id=self.thread_id,
            attachments=list(self.attachments),
            status=self.status,
        )

    def to_inbox_item(self) -> "InboxItem":
        return InboxItem(
            id=self.id,
            sender=self.sender,
            subject=self.subject,
            timestamp=self.timestamp,
            status=self.status,
        )


@dataclass
class SentEmail:
    """A reply that the agent has sent."""
    in_reply_to: str        # email.id
    body: str
    timestamp: str          # ISO 8601


# ─────────────────────────────────────────────
# Observation projections (agent-visible)
# ─────────────────────────────────────────────

@dataclass
class EmailView:
    """Read-only projection of Email for agent Observation. No ground-truth labels."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    status: EmailStatus
    thread_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)


@dataclass
class InboxItem:
    """Lightweight summary for inbox listing."""
    id: str
    sender: str
    subject: str
    timestamp: str
    status: EmailStatus


@dataclass
class SentItem:
    """Summary of a sent email (body preview only)."""
    in_reply_to: str
    timestamp: str
    body_preview: str       # first 100 characters


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

@dataclass
class State:
    """
    Complete ground-truth environment state.
    The agent never accesses this object directly.
    """
    inbox: List[Email] = field(default_factory=list)
    sent_box: List[SentEmail] = field(default_factory=list)
    agent_memory: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    done: bool = False
    max_steps: int = 30

    def get_email(self, email_id: str) -> Optional[Email]:
        for e in self.inbox:
            if e.id == email_id:
                return e
        return None

    def email_exists(self, email_id: str) -> bool:
        return self.get_email(email_id) is not None


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

@dataclass
class Observation:
    """
    What the agent receives each step.
    Derived from State — contains no ground-truth labels.
    """
    current_email: Optional[EmailView]
    inbox_summary: List[InboxItem]
    sent_summary: List[SentItem]
    step: int
    steps_remaining: int
    memory: Dict[str, Any]
    last_action_valid: bool = True
    last_action_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict for agent consumption."""
        return {
            "current_email": (
                {
                    "id": self.current_email.id,
                    "sender": self.current_email.sender,
                    "subject": self.current_email.subject,
                    "body": self.current_email.body,
                    "timestamp": self.current_email.timestamp,
                    "thread_id": self.current_email.thread_id,
                    "attachments": self.current_email.attachments,
                    "status": self.current_email.status.value,
                }
                if self.current_email else None
            ),
            "inbox_summary": [
                {
                    "id": item.id,
                    "sender": item.sender,
                    "subject": item.subject,
                    "timestamp": item.timestamp,
                    "status": item.status.value,
                }
                for item in self.inbox_summary
            ],
            "sent_summary": [
                {
                    "in_reply_to": s.in_reply_to,
                    "timestamp": s.timestamp,
                    "body_preview": s.body_preview,
                }
                for s in self.sent_summary
            ],
            "step": self.step,
            "steps_remaining": self.steps_remaining,
            "memory": self.memory,
            "last_action_valid": self.last_action_valid,
            "last_action_feedback": self.last_action_feedback,
        }


# ─────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────

@dataclass
class Action:
    """Base action container. Use factory methods or direct construction."""
    action_type: ActionType
    email_id: Optional[str] = None
    label: Optional[ClassLabel] = None
    body: Optional[str] = None
    reason: Optional[str] = None
    memory_key: Optional[str] = None
    memory_value: Optional[Any] = None

    # ── Factories ──────────────────────────────

    @staticmethod
    def read_email(email_id: str) -> "Action":
        return Action(action_type=ActionType.READ_EMAIL, email_id=email_id)

    @staticmethod
    def classify_email(email_id: str, label: ClassLabel) -> "Action":
        return Action(action_type=ActionType.CLASSIFY_EMAIL,
                      email_id=email_id, label=label)

    @staticmethod
    def reply_email(email_id: str, body: str) -> "Action":
        return Action(action_type=ActionType.REPLY_EMAIL,
                      email_id=email_id, body=body)

    @staticmethod
    def archive_email(email_id: str) -> "Action":
        return Action(action_type=ActionType.ARCHIVE_EMAIL, email_id=email_id)

    @staticmethod
    def escalate_email(email_id: str, reason: str) -> "Action":
        return Action(action_type=ActionType.ESCALATE_EMAIL,
                      email_id=email_id, reason=reason)

    @staticmethod
    def write_memory(key: str, value: Any) -> "Action":
        return Action(action_type=ActionType.WRITE_MEMORY,
                      memory_key=key, memory_value=value)

    @staticmethod
    def no_op() -> "Action":
        return Action(action_type=ActionType.NO_OP)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "email_id": self.email_id,
            "label": self.label.value if self.label else None,
            "body": self.body,
            "reason": self.reason,
            "memory_key": self.memory_key,
            "memory_value": self.memory_value,
        }


# ─────────────────────────────────────────────
# Action Result (internal, returned by executor)
# ─────────────────────────────────────────────

@dataclass
class ActionResult:
    valid: bool
    feedback: str
    reward_delta: float = 0.0
    opened_email: Optional[Email] = None