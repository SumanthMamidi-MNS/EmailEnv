"""
agents/heuristic_tiers.py

Tier-2 Offline Heuristic Decision Engine: HybridLocalEngine.

This engine implements a Weighted Multi-Signal Scoring Algorithm, fundamentally
different from Tier-1 (LocalFallbackAgent's first-match keyword scanning tree).

Key Architectural Differences from Tier-1:
------------------------------------------
1. Multi-Signal Additive Vector: Rather than stopping at the first matched keyword,
   HybridLocalEngine scans all signal categories (Spam, Urgent, Action, Social)
   simultaneously and calculates continuous normalized scores in [0, 1].
2. Urgency Safety Dominance: Evaluates urgency vs spam/action conflicts using
   explicit priority ranking to prevent dangerous misclassifications.
3. Soft Confidence Thresholds: Provides nuanced confidence estimates based on
   aggregate signal mass.
"""

from __future__ import annotations

from typing import Dict, Tuple

from envs.email_env.models import ActionType, ClassLabel, EmailAction
from agents.decision import AgentDecision

# ─────────────────────────────────────────────────────────────────────────────
# Reference Signal Corpus with Calibrated Weights
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
_SPAM: Dict[str, float] = {
    "lottery": 1.0, "winner": 1.0, "prize": 0.95, "claim now": 1.0,
    "claim your": 0.95, "click here": 0.85, "100x": 0.90, "crypto": 0.75,
    "bitcoin": 0.80, "investment opportunity": 0.90, "presale": 0.85,
    "free gift": 0.90, "wire transfer": 0.90, "inheritance": 0.95,
    "unsolicited": 0.80, "discount": 0.60, "coupon": 0.65,
    "unsubscribe": 0.55, "exclusive offer": 0.75, "limited time": 0.65,
}

_URGENT: Dict[str, float] = {
    "outage": 0.95, "p0": 1.0, "p1": 0.90, "critical": 0.85,
    "emergency": 0.90, "breach": 0.95, "security alert": 0.95,
    "database is down": 1.0, "services offline": 0.95, "root certificate": 0.90,
    "leaked": 0.90, "sla breach": 0.90, "ransomware": 1.0,
    "unauthorized access": 0.95, "legal notice": 0.80, "formal complaint": 0.80,
    "harassment": 0.85, "lawsuit": 0.90, "incident": 0.75,
    "production down": 1.0, "data loss": 0.90, "kubernetes": 0.55,
    "k8s": 0.60, "server down": 0.90, "api down": 0.85, "asap": 0.70,
    "immediate action": 0.85, "git secret": 0.95, "secret key": 0.90,
    "credential": 0.80, "password reset": 0.75, "2fa": 0.55,
}

_ACTION: Dict[str, float] = {
    "please submit": 0.90, "please confirm": 0.85, "please review": 0.80,
    "please approve": 0.85, "please respond": 0.80, "due by": 0.85,
    "due this": 0.80, "deadline": 0.80, "timesheet": 0.85,
    "contract renewal": 0.80, "terms update": 0.75, "questionnaire": 0.75,
    "checklist": 0.70, "action required": 0.90, "action needed": 0.90,
    "feedback by": 0.80, "response requested": 0.85, "sign by": 0.80,
    "approval needed": 0.85, "authorize": 0.75, "invoice": 0.70,
    "proposal": 0.65, "nda": 0.75, "agreement": 0.65,
}

_SOCIAL: Dict[str, float] = {
    "lunch": 0.85, "happy hour": 0.90, "party": 0.90, "birthday": 0.90,
    "celebration": 0.85, "invite": 0.80, "invitation": 0.85, "rsvp": 0.95,
    "5k run": 0.90, "charity": 0.70, "gardening club": 0.90,
    "team event": 0.80, "are you in": 0.90, "game night": 0.90,
    "movie night": 0.90, "team outing": 0.85, "office party": 0.90,
    "potluck": 0.85, "book club": 0.85, "happy birthday": 0.95,
}
# fmt: on

_REPLY_TEMPLATES = {
    ClassLabel.SPAM: None,
    ClassLabel.URGENT: (
        "We have received your urgent notification and are treating it with top priority. "
        "Our incident response team has been alerted and is taking immediate action."
    ),
    ClassLabel.ACTION_REQUIRED: (
        "Thank you for reaching out. We have received your request and are processing "
        "it by the stated deadline. We will follow up with confirmation."
    ),
    ClassLabel.SOCIAL: (
        "Thank you for the invitation. I would be happy to attend — looking forward to it!"
    ),
    ClassLabel.INFORMATIONAL: None,
}


def _score_text(text: str, signal_weights: Dict[str, float]) -> float:
    """Returns combined additive score for any matched signals in text."""
    score = 0.0
    for signal, weight in signal_weights.items():
        if signal in text:
            score += weight
    return score


class HybridLocalEngine:
    """
    Tier-2 local heuristic engine.
    Uses weighted additive multi-signal scoring to classify emails and select actions.
    """

    _SPAM_THRESHOLD = 0.70
    _URGENT_THRESHOLD = 0.75
    _ACTION_THRESHOLD = 0.80
    _SOCIAL_THRESHOLD = 0.80

    def decide(self, email_id: str, subject: str, sender: str, body: str) -> AgentDecision:
        text = f"{subject} {sender} {body}".lower()

        spam_score = _score_text(text, _SPAM)
        urgent_score = _score_text(text, _URGENT)
        action_score = _score_text(text, _ACTION)
        social_score = _score_text(text, _SOCIAL)

        # Normalize score to [0, 1] relative to a reference saturation point (2.5)
        _norm = lambda s: min(1.0, s / 2.5)
        spam_n = _norm(spam_score)
        urgent_n = _norm(urgent_score)
        action_n = _norm(action_score)
        social_n = _norm(social_score)

        # Urgency is prioritized over spam when both match (critical safety constraint)
        if urgent_n >= self._URGENT_THRESHOLD and urgent_n >= spam_n:
            label = ClassLabel.URGENT
            confidence = min(0.97, 0.78 + urgent_n * 0.20)
            reason = "Weighted urgency scoring exceeded safety threshold; escalation required."
            return AgentDecision(
                action=EmailAction.escalate(
                    email_id=email_id,
                    reason="Hybrid-tier detected critical pattern via multi-signal urgency analysis.",
                ),
                classification=label,
                urgency="CRITICAL",
                confidence=confidence,
                reasoning_summary=reason,
                source="hybrid_local",
            )

        if spam_n >= self._SPAM_THRESHOLD:
            confidence = min(0.98, 0.72 + spam_n * 0.25)
            return AgentDecision(
                action=EmailAction.archive(email_id=email_id),
                classification=ClassLabel.SPAM,
                urgency="LOW",
                confidence=confidence,
                reasoning_summary="Multi-signal spam scoring aggregate exceeded archive threshold.",
                source="hybrid_local",
            )

        if action_n >= self._ACTION_THRESHOLD:
            confidence = min(0.95, 0.68 + action_n * 0.25)
            reply_body = _REPLY_TEMPLATES[ClassLabel.ACTION_REQUIRED]
            return AgentDecision(
                action=EmailAction.reply(email_id=email_id, body=reply_body),
                classification=ClassLabel.ACTION_REQUIRED,
                urgency="MEDIUM",
                confidence=confidence,
                reasoning_summary="Multi-signal action scoring determined reply is required.",
                source="hybrid_local",
            )

        if social_n >= self._SOCIAL_THRESHOLD:
            confidence = min(0.94, 0.70 + social_n * 0.22)
            needs_reply = "rsvp" in text or "are you in" in text or "invite" in text
            if needs_reply:
                return AgentDecision(
                    action=EmailAction.reply(email_id=email_id, body=_REPLY_TEMPLATES[ClassLabel.SOCIAL]),
                    classification=ClassLabel.SOCIAL,
                    urgency="LOW",
                    confidence=confidence,
                    reasoning_summary="Social event with RSVP request detected via weighted scoring.",
                    source="hybrid_local",
                )
            return AgentDecision(
                action=EmailAction.archive(email_id=email_id),
                classification=ClassLabel.SOCIAL,
                urgency="LOW",
                confidence=confidence,
                reasoning_summary="Social notification archived (no RSVP required).",
                source="hybrid_local",
            )

        # Soft thresholds for moderate signals
        if urgent_n > 0.30:
            return AgentDecision(
                action=EmailAction.escalate(
                    email_id=email_id,
                    reason="Moderate urgency signal detected; escalating for safety review.",
                ),
                classification=ClassLabel.URGENT,
                urgency="HIGH",
                confidence=0.62,
                reasoning_summary="Below-threshold urgency signal triggered precautionary escalation.",
                source="hybrid_local",
            )

        if action_n > 0.25:
            return AgentDecision(
                action=EmailAction.reply(email_id=email_id, body=_REPLY_TEMPLATES[ClassLabel.ACTION_REQUIRED]),
                classification=ClassLabel.ACTION_REQUIRED,
                urgency="LOW",
                confidence=0.58,
                reasoning_summary="Weak action signal detected; attempted informational reply.",
                source="hybrid_local",
            )

        # Default: informational / archive
        return AgentDecision(
            action=EmailAction.archive(email_id=email_id),
            classification=ClassLabel.INFORMATIONAL,
            urgency="LOW",
            confidence=0.72,
            reasoning_summary="No significant signals detected; treated as informational.",
            source="hybrid_local",
        )
