"""
evaluation/metrics.py

Evaluation metrics calculation engine.
Computes empirical accuracy, safety rates, reward averages, and efficiency stats.
"""

from __future__ import annotations

from typing import List
from evaluation.models import BenchmarkResult, EpisodeRecord


class MetricCalculator:
    """Computes benchmark metrics across a collection of recorded episodes."""

    @staticmethod
    def calculate(
        agent_name: str,
        difficulty: str,
        episodes: List[EpisodeRecord],
        api_calls: int = 0,
        cache_hits: int = 0,
        fallback_calls: int = 0,
    ) -> BenchmarkResult:
        if not episodes:
            return BenchmarkResult(
                agent_name=agent_name,
                difficulty=difficulty,
                episodes_count=0,
                accuracy=0.0,
                avg_reward=0.0,
                safety_rate=1.0,
                completion_rate=0.0,
                invalid_rate=0.0,
                avg_steps=0.0,
                api_calls=api_calls,
                cache_hits=cache_hits,
                fallback_calls=fallback_calls,
                dangerous_error_rate=0.0,
            )

        n = len(episodes)
        total_reward = sum(e.cumulative_reward for e in episodes)
        total_steps = sum(len(e.steps) for e in episodes)
        completed_count = sum(1 for e in episodes if e.completed)
        safe_count = sum(1 for e in episodes if e.safety_status == "SAFE")
        dangerous_violations = sum(1 for e in episodes if e.safety_status != "SAFE")

        total_actions = 0
        invalid_actions = 0
        correct_actions = 0

        for ep in episodes:
            for step in ep.steps:
                total_actions += 1
                if not step.action_valid:
                    invalid_actions += 1
                if step.reward > 0.10:  # Correct classify (+0.20), correct reply (+0.15), correct escalate (+0.25)
                    correct_actions += 1

        accuracy = correct_actions / max(total_actions, 1)
        invalid_rate = invalid_actions / max(total_actions, 1)

        return BenchmarkResult(
            agent_name=agent_name,
            difficulty=difficulty,
            episodes_count=n,
            accuracy=round(accuracy, 4),
            avg_reward=round(total_reward / n, 4),
            safety_rate=round(safe_count / n, 4),
            completion_rate=round(completed_count / n, 4),
            invalid_rate=round(invalid_rate, 4),
            avg_steps=round(total_steps / n, 2),
            api_calls=api_calls,
            cache_hits=cache_hits,
            fallback_calls=fallback_calls,
            dangerous_error_rate=round(dangerous_violations / n, 4),
        )
