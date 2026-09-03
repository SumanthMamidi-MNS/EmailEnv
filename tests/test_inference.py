"""
tests/test_inference.py

End-to-end integration test running complete headless OpenEnv simulation episodes.
"""

import unittest
from inference import run_episode


class TestInference(unittest.TestCase):

    def test_run_baseline_starter_episode(self):
        result = run_episode(
            agent_type="baseline",
            difficulty="STARTER",
            seed=42,
            verbose=False,
        )
        self.assertTrue(result["done"])
        self.assertGreater(result["steps"], 0)
        self.assertGreater(result["valid_actions"], 0)
        self.assertEqual(result["invalid_actions"], 0)

    def test_run_hybrid_medium_episode(self):
        result = run_episode(
            agent_type="hybrid",
            difficulty="MEDIUM",
            seed=100,
            verbose=False,
        )
        self.assertTrue(result["done"])
        self.assertGreater(result["steps"], 0)
        self.assertGreater(result["valid_actions"], 0)

    def test_run_llm_adversarial_episode(self):
        result = run_episode(
            agent_type="llm",
            difficulty="ADVERSARIAL",
            seed=200,
            verbose=False,
        )
        self.assertTrue(result["done"])
        self.assertGreater(result["steps"], 0)


if __name__ == "__main__":
    unittest.main()
