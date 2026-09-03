"""
agents/cache.py

Simple, reliable in-memory cache for agent decision reuse.
Caches decisions keyed by email content hash to avoid duplicate processing.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple
from agents.decision import AgentDecision


class AgentCache:
    """In-memory key-value cache for agent decisions."""

    def __init__(self):
        self._store: Dict[str, AgentDecision] = {}
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def compute_key(subject: str, sender: str, body: str) -> str:
        """Generates a deterministic SHA256 key from email content."""
        raw = f"{sender.strip().lower()}|{subject.strip().lower()}|{body.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, subject: str, sender: str, body: str) -> Optional[AgentDecision]:
        """Retrieves a cached decision if available."""
        key = self.compute_key(subject, sender, body)
        if key in self._store:
            self._hits += 1
            decision = self._store[key].model_copy(deep=True)
            decision.source = "cache"
            return decision
        self._misses += 1
        return None

    def put(self, subject: str, sender: str, body: str, decision: AgentDecision) -> None:
        """Stores a decision in the cache."""
        key = self.compute_key(subject, sender, body)
        self._store[key] = decision.model_copy(deep=True)

    def clear(self) -> None:
        """Clears all cached entries and resets metrics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
        }
