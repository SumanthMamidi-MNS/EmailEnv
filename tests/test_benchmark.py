"""
tests/test_benchmark.py

Unit and integration tests for BenchmarkEngine and MetricCalculator.
"""

import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark import BenchmarkEngine
from evaluation.metrics import MetricCalculator
from evaluation.models import EpisodeRecord, TrajectoryStep
from evaluation.storage import SQLiteRepository


class TestBenchmarkEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bench.db"
        self.results_dir = Path(self.temp_dir.name) / "results"
        self.repo = SQLiteRepository(db_path=self.db_path)
        self.engine = BenchmarkEngine(repository=self.repo, results_dir=self.results_dir)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_metric_calculator(self):
        steps = [
            TrajectoryStep(
                step_number=1,
                observation_summary={},
                action={"action_type": "READ_EMAIL"},
                action_valid=True,
                reward=0.05,
                feedback="Opened",
            ),
            TrajectoryStep(
                step_number=2,
                observation_summary={},
                action={"action_type": "CLASSIFY_EMAIL"},
                action_valid=True,
                reward=0.20,
                feedback="Correct",
            ),
        ]
        ep = EpisodeRecord(
            episode_id="ep_01",
            scenario_id="s1",
            difficulty="STARTER",
            agent_name="BASELINE",
            steps=steps,
            cumulative_reward=0.25,
            safety_status="SAFE",
            completed=True,
        )

        res = MetricCalculator.calculate("BASELINE", "STARTER", [ep])
        self.assertEqual(res.agent_name, "BASELINE")
        self.assertEqual(res.safety_rate, 1.0)
        self.assertEqual(res.completion_rate, 1.0)
        self.assertEqual(res.invalid_rate, 0.0)
        self.assertGreater(res.accuracy, 0.0)

    def test_run_agent_tier_starter(self):
        result = self.engine.run_agent_tier(agent_name="baseline", difficulty="STARTER", seed=42)
        self.assertEqual(result.agent_name, "BASELINE")
        self.assertEqual(result.difficulty, "STARTER")
        self.assertEqual(result.safety_rate, 1.0)
        self.assertGreater(result.avg_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
