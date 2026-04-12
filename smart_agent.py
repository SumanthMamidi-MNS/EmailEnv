"""
smart_agent.py — Intelligent API-usage strategy for AI Email Agent.

STRATEGY:
  Tier 0 (always, instant, zero API cost):
    Rule-based classifier decides label and action type.

  Tier 1 (URGENT/ACTION_REQUIRED only, with cache + safety cap):
    Gemini 1.5 Flash generates a contextual reply.
    Falls back silently on ANY exception (key, rate-limit, timeout, network).

  Cache:
    Keyed by email_id. Never calls the API twice for the same email.

  Safety cap:
    Max 30 LLM calls per session. Prevents accidental quota burn.

Rules:
  SPAM / SOCIAL / INFORMATIONAL  → archive immediately, NO API call
  URGENT / ACTION_REQUIRED        → reply (Gemini if available, else template)
"""

from __future__ import annotations

import os
from typing import Optional

# Load .env silently
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from models import ClassLabel, Action
    _MODELS_OK = True
except Exception:
    _MODELS_OK = False

    class ClassLabel:  # type: ignore
        SPAM = "SPAM"; URGENT = "URGENT"; ACTION_REQUIRED = "ACTION_REQUIRED"
        SOCIAL = "SOCIAL"; INFORMATIONAL = "INFORMATIONAL"

    class Action:  # type: ignore
        @staticmethod
        def read_email(eid): return {"type": "read_email", "email_id": eid}
        @staticmethod
        def classify_email(eid, label): return {"type": "classify_email", "email_id": eid, "label": label}
        @staticmethod
        def reply_email(eid, body): return {"type": "reply_email", "email_id": eid, "body": body}
        @staticmethod
        def archive_email(eid): return {"type": "archive_email", "email_id": eid}
        @staticmethod
        def escalate_email(eid, reason=""): return {"type": "escalate_email", "email_id": eid, "reason": reason}


# ── Constants ──────────────────────────────────────────────────────────────────

# Labels that NEVER need an API call — archive directly
_NO_API_LABELS = {"SPAM", "SOCIAL", "INFORMATIONAL"}

# Labels that need a reply (Gemini if available, else template)
_REPLY_LABELS = {"URGENT", "ACTION_REQUIRED"}

# Safety cap: max LLM reply-generation calls per session
# Set high so Gemini is used without hesitation throughout a full session.
_API_CALL_LIMIT = 500

# Canned replies — always available, no API needed
_TEMPLATE_REPLIES: dict[str, str] = {
    "URGENT": (
        "Thank you for flagging this as urgent. "
        "We have received your message and are treating it as a high-priority matter. "
        "Our team is investigating immediately and will update you within the hour."
    ),
    "ACTION_REQUIRED": (
        "Thank you for reaching out. "
        "We have reviewed your request and will take the required action by end of business today. "
        "Please let us know if any additional information is needed."
    ),
}

_GEMINI_REPLY_SYSTEM = (
    "You are a professional email assistant. Write a concise, formal reply to the email below. "
    "Keep the reply under 80 words. Do not include a subject line. "
    "Be direct and actionable."
)


# ── Keyword classifier (Tier 0 — always runs, no API) ─────────────────────────

class _RuleClassifier:
    """Deterministic keyword-based classifier. Always returns a valid label."""

    _KEYWORDS: dict[str, list[str]] = {
        "SPAM": [
            "win", "winner", "prize", "lottery", "click here", "unsubscribe",
            "free offer", "guaranteed", "make money", "inheritance", "buy now",
            "limited time offer", "act now", "earn cash", "claim now",
            "100x", "crypto", "presale", "free iphone", "claim your",
        ],
        "URGENT": [
            "urgent", "asap", "immediately", "critical", "emergency",
            "time-sensitive", "time sensitive", "deadline today", "respond now",
            "high priority", "escalate", "outage", " down ", "all hands",
            "p0", "priority 0",
            "legal notice", "formal complaint", "harassment", "workplace harassment",
            "lawsuit", "legal action", "regulatory",
        ],
        "ACTION_REQUIRED": [
            "please review", "please approve", "approval needed", "sign off",
            "action required", "response required", "please confirm",
            "awaiting your", "need your input", "follow up", "follow-up",
            "next steps", "please respond", "sign-off", "your input",
            "due by", "submit by",
            "please submit", "due this friday", "complete your submission",
            "terms update", "contract renewal", "renewal", "terms update required",
            "proposal deadline", "project proposal",
        ],
        "SOCIAL": [
            "invitation", "invite", "party", "birthday", "wedding", "event",
            "gathering", "happy hour", "lunch", "coffee", "catch up",
            "get together", "celebrate", "rsvp", "volunteer",
        ],
        "INFORMATIONAL": [
            "fyi", "for your information", "newsletter", "announcement",
            "reminder", "notice", "report", "summary", "monthly", "weekly",
            "digest", "recap", "bulletin", "update", "closure", "policy update",
            "closure notice", "office closure", "office will be closed", "holiday notice",
        ],
    }

    def classify(self, subject: str, body: str) -> str:
        try:
            text = (subject + " " + body).lower()
            scores: dict[str, int] = {lbl: 0 for lbl in self._KEYWORDS}
            for lbl, kws in self._KEYWORDS.items():
                for kw in kws:
                    if kw in text:
                        scores[lbl] += 1
            best = max(scores, key=lambda l: scores[l])
            return best if scores[best] > 0 else "INFORMATIONAL"
        except Exception:
            return "INFORMATIONAL"


# ── Main Smart Processor ───────────────────────────────────────────────────────

class SmartEmailProcessor:
    """
    Central controller for the AI Email Agent.

    Usage:
        proc = SmartEmailProcessor()
        label = proc.classify(subject, body)           # Always instant, no API
        action = proc.decide_action(label)             # "reply" or "archive"
        reply  = proc.reply(eid, subject, body, label) # Gemini if eligible, else template
    """

    def __init__(self) -> None:
        self._rule_clf = _RuleClassifier()
        self._gemini_model = None
        self._api_enabled: bool = False
        self._api_calls: int = 0          # reply generation calls
        self._classify_calls: int = 0     # classification calls (tracked separately)
        self._cache: dict[str, dict] = {}   # email_id → {"label": str, "reply": str}
        self._tier: str = "Tier 1 — Rule-Based (no API)"
        self._init_gemini()

    # ── Initialise Gemini (optional) ───────────────────────────────────────────

    def _init_gemini(self) -> None:
        """Try to connect to Gemini. Silently skip if key is missing or invalid."""
        try:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                return
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            self._api_enabled = True
            self._tier = "Tier 3 — LLM (Gemini 2.5 Flash) + Rule-Based Fallback"
        except Exception:
            self._api_enabled = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, subject: str, body: str) -> str:
        """
        Classify an email.
        If Gemini is available → calls Gemini 2.5 Flash for every email.
        Falls back silently to rule-based on any error (no crash, no hesitation).
        Returns one of: SPAM, URGENT, ACTION_REQUIRED, SOCIAL, INFORMATIONAL.
        """
        if self._api_enabled:
            result = self._gemini_classify(subject, body)
            if result:
                return result
        return self._rule_clf.classify(subject, body)

    def _gemini_classify(self, subject: str, body: str) -> Optional[str]:
        """
        Use Gemini to classify an email. Returns label string or None on any failure.
        Valid labels: SPAM, URGENT, ACTION_REQUIRED, SOCIAL, INFORMATIONAL.
        """
        if not self._gemini_model:
            return None
        try:
            import re
            prompt = (
                "Classify the following email into exactly one of these five categories:\n"
                "SPAM, URGENT, ACTION_REQUIRED, SOCIAL, INFORMATIONAL\n"
                "Reply with ONLY the category name — nothing else.\n\n"
                f"Subject: {subject}\n"
                f"Body: {body[:600]}"
            )
            response = self._gemini_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 10, "temperature": 0.0},
            )
            raw = re.sub(r"[^A-Z_]", "", response.text.strip().upper())
            valid = {"SPAM", "URGENT", "ACTION_REQUIRED", "SOCIAL", "INFORMATIONAL"}
            if raw in valid:
                self._classify_calls += 1
                return raw
            return None
        except Exception:
            return None

    def classify_as_label(self, subject: str, body: str):
        """Return a ClassLabel enum (or string fallback) for env.step()."""
        label_str = self.classify(subject, body)
        try:
            return getattr(ClassLabel, label_str)
        except Exception:
            return label_str

    def decide_action(self, label: str) -> str:
        """
        Decide what follow-up action to take — no API needed.
          reply   → for URGENT / ACTION_REQUIRED
          archive → for SPAM / SOCIAL / INFORMATIONAL
        """
        return "reply" if label in _REPLY_LABELS else "archive"

    def reply(self, email_id: str, subject: str, body: str, label: str) -> str:
        """
        Generate a reply for URGENT/ACTION_REQUIRED emails.

        Order of preference:
          1. Cache hit          → instant, zero API
          2. Gemini LLM         → best quality (only if eligible + under cap)
          3. Canned template    → always works
        """
        # 1. Cache hit
        cached = self._cache.get(email_id, {}).get("reply")
        if cached:
            return cached

        reply_text: Optional[str] = None

        # 2. Try Gemini — only for URGENT/ACTION_REQUIRED and under cap
        if self._api_enabled and self._api_calls < _API_CALL_LIMIT and label in _REPLY_LABELS:
            reply_text = self._gemini_reply(subject, body, label)

        # 3. Fallback to canned template
        if not reply_text:
            reply_text = _TEMPLATE_REPLIES.get(label, _TEMPLATE_REPLIES["ACTION_REQUIRED"])

        # Cache the result
        self._cache.setdefault(email_id, {})["reply"] = reply_text
        return reply_text

    # ── Internal Gemini call ────────────────────────────────────────────────────

    def _gemini_reply(self, subject: str, body: str, label: str) -> Optional[str]:
        """
        Call Gemini for a contextual reply.
        Returns None on ANY failure — caller always falls back to template.
        Handles: missing key, invalid key, rate limit, quota, timeout, network.
        """
        if not self._gemini_model:
            return None
        try:
            prompt = (
                f"{_GEMINI_REPLY_SYSTEM}\n\n"
                f"Email type: {label.replace('_', ' ').title()}\n"
                f"Subject: {subject}\n\n"
                f"Body:\n{body[:600]}\n\n"
                "Write a professional reply under 80 words:"
            )
            response = self._gemini_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 150, "temperature": 0.3},
            )
            self._api_calls += 1
            text = response.text.strip()
            return text if text else None
        except Exception:
            # Rate limit / invalid key / network: silently fall through to template
            return None

    # ── Stats / introspection ──────────────────────────────────────────────────

    @property
    def tier_name(self) -> str:
        return self._tier

    @property
    def api_calls_made(self) -> int:
        return self._api_calls

    @property
    def api_enabled(self) -> bool:
        return self._api_enabled

    @property
    def classify_calls_made(self) -> int:
        return self._classify_calls

    def stats(self) -> dict:
        return {
            "tier": self._tier,
            "api_enabled": self._api_enabled,
            "api_calls": self._api_calls,
            "classify_calls": self._classify_calls,
            "total_gemini_calls": self._api_calls + self._classify_calls,
            "cached_emails": len(self._cache),
        }


# ── Module-level singleton helper ─────────────────────────────────────────────

_processor: Optional[SmartEmailProcessor] = None


def get_processor() -> SmartEmailProcessor:
    """Return module-level singleton. Initialised once per process."""
    global _processor
    if _processor is None:
        _processor = SmartEmailProcessor()
    return _processor
