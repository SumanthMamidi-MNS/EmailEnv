"""
agents/hybrid_agent.py

Intelligent Tiered Hybrid Agent.
Combines Tier-2 local weighted heuristics for high-confidence simple cases
with LLM intelligence and validated lesson retrieval for ambiguous, subtle, or critical emails.

Tier-2 Local Path: HybridLocalEngine (multi-signal weighted scoring)
Tier-3 LLM Path: LLMClient (Gemini 2.5 Flash or equivalent)

The Hybrid's offline fallback (HybridLocalEngine) is intentionally different from
the Baseline's offline fallback (LocalFallbackAgent / first-match scanning) so that
the two agents produce measurably different results, especially on adversarial and
multi-intent emails. This is required by the project's correctness spec.
"""

from __future__ import annotations

from typing import Any, Optional
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    EmailAction,
    EmailObservation,
    EmailStatus,
)
from agents.decision import AgentDecision
from agents.heuristic_tiers import HybridLocalEngine
from agents.llm import LLMClient


class HybridAgent:
    """
    Hybrid Agent optimizing API cost and decision accuracy.

    Decision policy:
      1. Run Tier-2 weighted scoring (HybridLocalEngine) on every email.
      2. If confidence is high AND email is simple → return local decision (no API call).
      3. If email is complex, ambiguous, or confidence is low → delegate to LLM tier.
      4. If LLM is unavailable → return local Tier-2 decision (better than baseline Tier-1).
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        fallback_agent: Optional[Any] = None,  # kept for API compatibility; unused
        retriever: Optional[Any] = None,
        confidence_threshold: float = 0.88,
    ):
        self.llm = llm_client or LLMClient()
        # Tier-2 local engine — intentionally different from LocalFallbackAgent (Tier-1)
        self._local_engine = HybridLocalEngine()
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.llm_call_count = 0
        self.local_call_count = 0

    def evaluate(self, email_id: str, subject: str, sender: str, body: str) -> AgentDecision:
        """Determines decision using local Tier-2 heuristics or LLM escalation with lessons."""
        local_decision = self._local_engine.decide(email_id, subject, sender, body)

        # High-confidence clear spam or routine social can be safely resolved locally
        is_simple_spam = (
            local_decision.classification == ClassLabel.SPAM
            and local_decision.confidence >= self.confidence_threshold
        )
        is_routine_info = (
            local_decision.classification == ClassLabel.INFORMATIONAL
            and "newsletter" in subject.lower()
            and local_decision.confidence >= self.confidence_threshold
        )

        if is_simple_spam or is_routine_info or not self.llm.is_available():
            self.local_call_count += 1
            return local_decision

        # For critical, subtle, or ambiguous emails: delegate to LLM
        self.llm_call_count += 1

        lessons_text = []
        if self.retriever:
            try:
                active_lessons = self.retriever.get_relevant_lessons(
                    subject=subject, sender=sender, body=body
                )
                lessons_text = [l.rule_text for l in active_lessons]
            except Exception:
                lessons_text = []

        return self.llm.evaluate_email(
            email_id,
            subject,
            sender,
            body,
            lessons=lessons_text if lessons_text else None,
        )

    def act(self, obs: EmailObservation) -> EmailAction:
        """Acts upon observation using tiered intelligence."""
        if obs.current_email:
            curr = obs.current_email

            if curr.status in (EmailStatus.ARCHIVED, EmailStatus.ESCALATED):
                for item in obs.inbox_summary:
                    if item.status == EmailStatus.UNREAD:
                        return EmailAction.read(email_id=item.id)
                return EmailAction.no_op()

            decision = self.evaluate(curr.id, curr.subject, curr.sender, curr.body)

            # Step A: Classify
            if curr.assigned_label is None:
                return EmailAction.classify(email_id=curr.id, label=decision.classification)

            # Step B: Act
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

        # Open next unread email
        for item in obs.inbox_summary:
            if item.status == EmailStatus.UNREAD:
                return EmailAction.read(email_id=item.id)

        return EmailAction.no_op()
