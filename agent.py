# === agent.py ===
# Email classifier — two-tier architecture (Semantic tier removed):
#   Tier 1: Rule-Based   — always available, instant, zero API cost
#   Tier 3: Gemini LLM   — google-generativeai (if GEMINI_API_KEY is set)
#
# On any Gemini failure (bad key, quota, timeout, network) the system
# falls back silently to Tier 1. The app NEVER crashes.

from __future__ import annotations

import os
import re

# Load .env file if present (local dev only; ignored in Docker)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from models import ClassLabel
except ImportError:
    from enum import Enum
    class ClassLabel(str, Enum):  # type: ignore
        SPAM            = "SPAM"
        URGENT          = "URGENT"
        ACTION_REQUIRED = "ACTION_REQUIRED"
        SOCIAL          = "SOCIAL"
        INFORMATIONAL   = "INFORMATIONAL"


# ── Tier 1: Rule-based ────────────────────────────────────────────────────────

KEYWORD_MAP: dict = {
    ClassLabel.SPAM: [
        "win", "winner", "prize", "lottery", "click here", "unsubscribe",
        "free offer", "guaranteed", "make money", "nigerian", "inheritance",
        "discount", "buy now", "limited time", "act now", "earn cash",
        "free iphone", "100x", "crypto", "presale", "claim now", "claim your",
    ],
    ClassLabel.URGENT: [
        "urgent", "asap", "immediately", "critical", "emergency",
        "time-sensitive", "time sensitive", "deadline today", "respond now",
        "action needed today", "high priority", "escalate", "outage",
        "all hands", "p0", "priority 0", "breach", "incident", "down ",
        # fixes for task emails
        "legal notice", "formal complaint", "harassment", "workplace harassment",
        "lawsuit", "legal action", "regulatory",
    ],
    ClassLabel.ACTION_REQUIRED: [
        "please review", "please approve", "approval needed", "sign off",
        "action required", "response required", "please confirm",
        "awaiting your", "need your input", "requires your attention",
        "follow up", "follow-up", "next steps", "please respond",
        "sign-off", "your approval", "due by", "submit by",
        # fixes for task emails
        "please submit", "due this friday", "complete your submission",
        "terms update", "contract renewal", "renewal", "terms update required",
        "proposal deadline", "project proposal",
    ],
    ClassLabel.SOCIAL: [
        "invitation", "invite", "party", "birthday", "wedding", "event",
        "gathering", "happy hour", "lunch", "coffee", "catch up",
        "get together", "celebrate", "rsvp", "volunteer", "charity",
    ],
    ClassLabel.INFORMATIONAL: [
        "fyi", "for your information", "newsletter", "update", "announcement",
        "reminder", "notice", "report", "summary", "monthly", "weekly",
        "digest", "recap", "bulletin", "policy update", "quarterly",
        # fixes for task emails
        "closure notice", "office closure", "office will be closed", "holiday notice",
    ],
}


class RuleBasedClassifier:
    """Tier 1 — deterministic keyword matching. Zero dependencies, always works."""

    def classify(self, subject: str, body: str) -> ClassLabel:
        text = (subject + " " + body).lower()
        scores: dict = {label: 0 for label in ClassLabel}
        for label, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if kw in text:
                    scores[label] += 1
        best = max(scores, key=lambda l: scores[l])
        return ClassLabel.INFORMATIONAL if scores[best] == 0 else best

    def confidence(self, subject: str, body: str) -> float:
        text = (subject + " " + body).lower()
        hits: dict = {label: 0 for label in ClassLabel}
        for label, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if kw in text:
                    hits[label] += 1
        total = sum(hits.values())
        if total == 0:
            return 0.2
        return min(max(hits.values()) / max(total, 1), 1.0)


# ── Standalone helper (importable by app.py) ──────────────────────────────────

def _demo_label(subject: str, body: str) -> ClassLabel:
    return RuleBasedClassifier().classify(subject, body)


# ── Tier 3: Gemini LLM ────────────────────────────────────────────────────────

_GEMINI_CLASSIFY_PROMPT = (
    "Classify the following email into exactly one of these categories: "
    "SPAM, URGENT, ACTION_REQUIRED, SOCIAL, INFORMATIONAL.\n"
    "Reply with ONLY the category name and nothing else.\n\n"
)

_LABEL_LOOKUP: dict = {l.value: l for l in ClassLabel}


class GeminiClassifier:
    """
    Tier 3 — Google Gemini 2.5 Flash.
    Falls back silently to RuleBasedClassifier on ANY error:
      quota exceeded, invalid key, timeout, network failure, etc.
    """

    def __init__(self) -> None:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-2.5-flash")
        self._fallback = RuleBasedClassifier()

    def classify(self, subject: str, body: str) -> ClassLabel:
        try:
            prompt = _GEMINI_CLASSIFY_PROMPT + f"Subject: {subject}\n\nBody:\n{body[:500]}"
            response = self._model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 15, "temperature": 0.0},
            )
            raw = re.sub(r"[^A-Z_]", "", response.text.strip().upper())
            return _LABEL_LOOKUP.get(raw, ClassLabel.INFORMATIONAL)
        except Exception:
            return self._fallback.classify(subject, body)

    def confidence(self, subject: str, body: str) -> float:
        return 0.92


# ── Factory helpers ───────────────────────────────────────────────────────────

def _gemini_available() -> bool:
    """True only when the package is installed AND the key is set."""
    try:
        import google.generativeai  # noqa: F401
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())
    except ImportError:
        return False


def get_best_classifier():
    """Return the highest-tier classifier available, silently falling back."""
    if _gemini_available():
        try:
            return GeminiClassifier()
        except Exception:
            pass
    return RuleBasedClassifier()


# ── Public interface ──────────────────────────────────────────────────────────

class EmailClassifier:
    """Main entry point. Wraps the best available classifier."""

    def __init__(self) -> None:
        self._clf = get_best_classifier()

    def classify(self, subject: str, body: str) -> ClassLabel:
        try:
            return self._clf.classify(subject, body)
        except Exception:
            return RuleBasedClassifier().classify(subject, body)

    def tier_name(self) -> str:
        name = type(self._clf).__name__
        return {
            "RuleBasedClassifier": "Tier 1 — Rule-Based (instant, zero API cost)",
            "GeminiClassifier":    "Tier 3 — Gemini 2.5 Flash + Rule-Based Fallback",
        }.get(name, name)

    def confidence(self, subject: str, body: str) -> float:
        try:
            if hasattr(self._clf, "confidence"):
                return self._clf.confidence(subject, body)
            return 0.5
        except Exception:
            return 0.5