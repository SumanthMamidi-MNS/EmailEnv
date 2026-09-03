"""
agents/fallback.py

Local, zero-dependency, deterministic heuristic fallback agent.
Used as the reliable single fallback path whenever LLMs or external services are unavailable.
"""

from __future__ import annotations

from envs.email_env.models import ActionType, ClassLabel, EmailAction
from agents.decision import AgentDecision


SPAM_SIGNALS = [
    "lottery", "winner", "prize", "claim now", "claim your", "click here",
    "100x", "crypto", "bitcoin", "investment opportunity", "presale", "free gift",
    "wire transfer", "inheritance", "unsolicited", "discount", "coupon",
]

URGENT_SIGNALS = [
    "urgent", "critical", "emergency", "immediate", "asap", "p0", "p1", "outage",
    "breach", "security alert", "database is down", "services offline",
    "root certificate", "leaked", "sla breach", "ransomware", "unauthorized access",
    "legal notice", "formal complaint", "harassment", "lawsuit",
]

ACTION_SIGNALS = [
    "please submit", "please confirm", "please review", "please approve",
    "please respond", "due by", "due this", "deadline", "timesheet",
    "contract renewal", "terms update", "questionnaire", "checklist",
    "action required", "action needed", "feedback by",
]

SOCIAL_SIGNALS = [
    "lunch", "happy hour", "party", "birthday", "celebration", "invite",
    "invitation", "rsvp", "5k run", "charity", "gardening club", "team event",
]


class LocalFallbackAgent:
    """Deterministic, rule-based fallback decision engine."""

    def decide(self, email_id: str, subject: str, sender: str, body: str) -> AgentDecision:
        text = f"{subject} {body}".lower()

        # 1. Check for Blatant Spam First
        for signal in SPAM_SIGNALS:
            if signal in text:
                return AgentDecision(
                    action=EmailAction.archive(email_id=email_id),
                    classification=ClassLabel.SPAM,
                    urgency="LOW",
                    confidence=0.92,
                    reasoning_summary=f"Spam filter matched promotional signal '{signal}'.",
                    source="fallback",
                )

        # 2. Check for Critical / Urgent Escalation
        for signal in URGENT_SIGNALS:
            if signal in text:
                return AgentDecision(
                    action=EmailAction.escalate(
                        email_id=email_id,
                        reason=f"Detected critical alert signal '{signal}' in email content.",
                    ),
                    classification=ClassLabel.URGENT,
                    urgency="CRITICAL",
                    confidence=0.88,
                    reasoning_summary=f"Rule-based escalation triggered by urgent pattern '{signal}'.",
                    source="fallback",
                )

        # 3. Check for Action Required / Reply
        for signal in ACTION_SIGNALS:
            if signal in text:
                return AgentDecision(
                    action=EmailAction.reply(
                        email_id=email_id,
                        body=(
                            "Hello, thank you for contacting us. We have received your request "
                            "and are taking the necessary actions by the requested deadline."
                        ),
                    ),
                    classification=ClassLabel.ACTION_REQUIRED,
                    urgency="MEDIUM",
                    confidence=0.82,
                    reasoning_summary=f"Action requested by sender pattern '{signal}'.",
                    source="fallback",
                )

        # 4. Check for Social Invitation
        for signal in SOCIAL_SIGNALS:
            if signal in text:
                if "rsvp" in text or "are you in" in text:
                    return AgentDecision(
                        action=EmailAction.reply(
                            email_id=email_id,
                            body="Thanks for the invitation! I will be glad to join.",
                        ),
                        classification=ClassLabel.SOCIAL,
                        urgency="LOW",
                        confidence=0.85,
                        reasoning_summary="Social invitation with RSVP detected.",
                        source="fallback",
                    )
                return AgentDecision(
                    action=EmailAction.archive(email_id=email_id),
                    classification=ClassLabel.SOCIAL,
                    urgency="LOW",
                    confidence=0.85,
                    reasoning_summary="Social event notification acknowledged and archived.",
                    source="fallback",
                )

        # 5. Default Informational
        return AgentDecision(
            action=EmailAction.archive(email_id=email_id),
            classification=ClassLabel.INFORMATIONAL,
            urgency="LOW",
            confidence=0.75,
            reasoning_summary="Informational update or newsletter archived.",
            source="fallback",
        )
