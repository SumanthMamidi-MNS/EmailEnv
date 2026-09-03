"""
agents/baseline_agent.py

Deterministic rule-based baseline agent.
Operates using local keyword heuristics, state machines, and zero external API dependencies.
"""

from __future__ import annotations

from typing import Optional
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    EmailAction,
    EmailObservation,
    EmailStatus,
)
from agents.decision import AgentDecision
from agents.fallback import LocalFallbackAgent


class BaselineAgent:
    """
    Deterministic rule-based baseline agent.
    Iterates over inbox items: reads -> classifies -> executes target action -> archives (if non-critical).
    """

    def __init__(self):
        self.fallback = LocalFallbackAgent()

    def act(self, obs: EmailObservation) -> EmailAction:
        """Determines the next environment action from the current observation."""
        # 1. If currently inspecting an email
        if obs.current_email:
            curr = obs.current_email

            # If this email is already fully triaged (archived or escalated), move to next unread
            if curr.status in (EmailStatus.ARCHIVED, EmailStatus.ESCALATED):
                for item in obs.inbox_summary:
                    if item.status == EmailStatus.UNREAD:
                        return EmailAction.read(email_id=item.id)
                return EmailAction.no_op()

            # Decide classification and target action using deterministic rules
            decision = self.fallback.decide(
                email_id=curr.id,
                subject=curr.subject,
                sender=curr.sender,
                body=curr.body,
            )

            # Step A: Classify if not already assigned
            if curr.assigned_label is None:
                return EmailAction.classify(email_id=curr.id, label=decision.classification)

            # Step B: Perform primary action if not already executed
            target_act_type = decision.action.action_type

            if target_act_type == ActionType.ESCALATE_EMAIL:
                if curr.status != EmailStatus.ESCALATED:
                    return decision.action
                # Escalated emails are never archived; move to next
                for item in obs.inbox_summary:
                    if item.status == EmailStatus.UNREAD:
                        return EmailAction.read(email_id=item.id)
                return EmailAction.no_op()

            if target_act_type == ActionType.REPLY_EMAIL and curr.status != EmailStatus.REPLIED:
                return decision.action

            # Step C: Archive non-escalated emails to complete processing
            if curr.status != EmailStatus.ARCHIVED:
                return EmailAction.archive(email_id=curr.id)

        # 2. If no email is currently open, find the next unread email
        for item in obs.inbox_summary:
            if item.status == EmailStatus.UNREAD:
                return EmailAction.read(email_id=item.id)

        # 3. If all handled
        return EmailAction.no_op()
