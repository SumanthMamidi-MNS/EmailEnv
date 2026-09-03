"""
agents/llm_agent.py

Pure LLM-driven autonomous agent (Tier-3).

Runtime Execution Model:
-----------------------
- When LLM is available (valid API key / provider active):
  Executes full LLM reasoning via LLMClient (Gemini 2.5 Flash), enforcing prompt injection
  defense boundaries (<untrusted_email_content>) and validated lesson retrieval.

- When LLM is unavailable (no API key / offline mode):
  Explicitly operates in documented Degraded Fallback Mode, utilizing the Tier-2
  HybridLocalEngine and tagging decisions with source="degraded_fallback".
  This ensures zero runtime crashes while transparently reporting offline execution.
"""

from __future__ import annotations

from typing import Any, Optional
from envs.email_env.models import (
    ActionType,
    EmailAction,
    EmailObservation,
    EmailStatus,
)
from agents.decision import AgentDecision
from agents.heuristic_tiers import HybridLocalEngine
from agents.llm import LLMClient


class LLMAgent:
    """
    Autonomous LLM Agent (Tier-3).
    Orchestrates email triage driven by LLM evaluation and validated lessons.
    Runs in documented degraded fallback mode when LLM is unavailable.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Any] = None,
    ):
        self.llm = llm_client or LLMClient()
        self._fallback_engine = HybridLocalEngine()
        self.retriever = retriever
        self.llm_call_count = 0

    def _evaluate_degraded(self, email_id: str, subject: str, sender: str, body: str) -> AgentDecision:
        """Degraded fallback decision when LLM API is unavailable."""
        decision = self._fallback_engine.decide(email_id, subject, sender, body)
        # Explicitly tag source as degraded fallback
        return AgentDecision(
            action=decision.action,
            classification=decision.classification,
            urgency=decision.urgency,
            confidence=decision.confidence,
            reasoning_summary=f"[Degraded Mode] {decision.reasoning_summary}",
            source="degraded_fallback",
        )

    def act(self, obs: EmailObservation) -> EmailAction:
        """Processes current observation using LLM intelligence (or documented fallback)."""
        # 1. If currently inspecting an email
        if obs.current_email:
            curr = obs.current_email

            if curr.status in (EmailStatus.ARCHIVED, EmailStatus.ESCALATED):
                for item in obs.inbox_summary:
                    if item.status == EmailStatus.UNREAD:
                        return EmailAction.read(email_id=item.id)
                return EmailAction.no_op()

            # Retrieve any active validated lessons
            lessons_text = []
            if self.retriever:
                try:
                    active_lessons = self.retriever.get_relevant_lessons(
                        subject=curr.subject, sender=curr.sender, body=curr.body
                    )
                    lessons_text = [l.rule_text for l in active_lessons]
                except Exception:
                    lessons_text = []

            # Try LLM first; if unavailable, use documented degraded fallback
            if self.llm.is_available():
                self.llm_call_count += 1
                decision: AgentDecision = self.llm.evaluate_email(
                    email_id=curr.id,
                    subject=curr.subject,
                    sender=curr.sender,
                    body=curr.body,
                    lessons=lessons_text if lessons_text else None,
                )
            else:
                decision = self._evaluate_degraded(
                    curr.id, curr.subject, curr.sender, curr.body
                )

            # Step A: Classify
            if curr.assigned_label is None:
                return EmailAction.classify(email_id=curr.id, label=decision.classification)

            # Step B: Execute action
            target_act_type = decision.action.action_type

            if target_act_type == ActionType.ESCALATE_EMAIL:
                if curr.status != EmailStatus.ESCALATED:
                    return decision.action
                for item in obs.inbox_summary:
                    if item.status == EmailStatus.UNREAD:
                        return EmailAction.read(email_id=item.id)
                return EmailAction.no_op()

            elif target_act_type == ActionType.REPLY_EMAIL:
                if curr.status != EmailStatus.REPLIED:
                    return decision.action
                return EmailAction.archive(email_id=curr.id)

            else:
                return EmailAction.archive(email_id=curr.id)

        # 2. If no email is currently opened, open first UNREAD email
        for item in obs.inbox_summary:
            if item.status == EmailStatus.UNREAD:
                return EmailAction.read(email_id=item.id)

        return EmailAction.no_op()
