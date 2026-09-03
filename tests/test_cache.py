"""
tests/test_cache.py

Tests for AgentCache verifying cache hits, misses, key hashing, and clearing.
"""

import unittest
from envs.email_env.models import ClassLabel, EmailAction
from agents.cache import AgentCache
from agents.decision import AgentDecision


class TestAgentCache(unittest.TestCase):

    def setUp(self):
        self.cache = AgentCache()

    def test_cache_miss_and_hit(self):
        subject = "Invoice Review"
        sender = "billing@vendor.com"
        body = "Please find attached invoice."

        # Initial lookup must miss
        self.assertIsNone(self.cache.get(subject, sender, body))
        self.assertEqual(self.cache.stats()["misses"], 1)

        decision = AgentDecision(
            action=EmailAction.reply("e1", "Approved."),
            classification=ClassLabel.ACTION_REQUIRED,
            urgency="MEDIUM",
            confidence=0.9,
            source="llm",
        )

        # Store in cache
        self.cache.put(subject, sender, body, decision)
        self.assertEqual(self.cache.stats()["size"], 1)

        # Subsequent lookup must hit
        cached_decision = self.cache.get(subject, sender, body)
        self.assertIsNotNone(cached_decision)
        self.assertEqual(cached_decision.classification, ClassLabel.ACTION_REQUIRED)
        self.assertEqual(cached_decision.source, "cache")
        self.assertEqual(self.cache.stats()["hits"], 1)

    def test_cache_clear(self):
        self.cache.put(
            "Subj",
            "a@b.com",
            "Body",
            AgentDecision(
                action=EmailAction.archive("e2"),
                classification=ClassLabel.INFORMATIONAL,
            ),
        )
        self.assertEqual(self.cache.stats()["size"], 1)

        self.cache.clear()
        self.assertEqual(self.cache.stats()["size"], 0)
        self.assertEqual(self.cache.stats()["hits"], 0)


if __name__ == "__main__":
    unittest.main()
