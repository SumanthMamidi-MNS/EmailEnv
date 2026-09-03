"""
evaluation/benchmark.py

Multi-Agent Benchmark Engine.
Executes fair, reproducible evaluations across Baseline, Hybrid, and LLM agents
over multiple difficulty tiers and held-out scenarios.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from envs.email_env.client import EmailEnvClient
from envs.email_env.models import ActionType, EmailStatus
from evaluation.models import BenchmarkResult, EpisodeRecord, TrajectoryStep
from evaluation.metrics import MetricCalculator
from evaluation.storage import SQLiteRepository
from agents.router import AgentRouter

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


class BenchmarkEngine:
    """Standardized evaluation runner across agents and difficulty sets."""

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        results_dir: Optional[Path] = None,
    ):
        self.repo = repository or SQLiteRepository()
        self.results_dir = results_dir or _RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_agent_tier(
        self,
        agent_name: str,
        difficulty: str = "STARTER",
        seed: int = 42,
        router: Optional[AgentRouter] = None,
    ) -> BenchmarkResult:
        """Runs a single agent on a difficulty tier and records the episode."""
        router = router or AgentRouter()
        agent = router.get_agent(agent_name)
        client = EmailEnvClient()

        start_time = datetime.now(timezone.utc).isoformat()
        obs = client.reset(seed=seed, difficulty=difficulty)

        trajectory: List[TrajectoryStep] = []
        step_num = 0
        safety_status = "SAFE"

        while not obs.done:
            # Record observation summary
            curr_view = obs.current_email.model_dump() if obs.current_email else None
            obs_summary = {
                "step": obs.step,
                "current_email": curr_view,
                "inbox_count": len(obs.inbox_summary),
            }

            action = agent.act(obs)
            obs, reward, done, info = client.step(action)
            step_num += 1

            if "SAFETY VIOLATION" in obs.last_action_feedback:
                safety_status = "VIOLATION"

            trajectory.append(
                TrajectoryStep(
                    step_number=step_num,
                    observation_summary=obs_summary,
                    action=action.model_dump(),
                    action_valid=obs.last_action_valid,
                    reward=reward,
                    feedback=obs.last_action_feedback,
                )
            )

            if done:
                break

        end_time = datetime.now(timezone.utc).isoformat()
        state = client.state()

        episode_record = EpisodeRecord(
            episode_id=state.episode_id,
            scenario_id=f"bench_{difficulty.lower()}_{seed}",
            difficulty=difficulty.upper(),
            agent_name=agent_name.upper(),
            start_time=start_time,
            end_time=end_time,
            steps=trajectory,
            final_action=trajectory[-1].action.get("action_type", "") if trajectory else "",
            cumulative_reward=state.cumulative_reward,
            safety_status=safety_status,
            completed=state.done,
            metadata={"seed": seed, "total_emails": len(state.emails)},
        )

        self.repo.save_episode(episode_record)

        # Compute metric summary
        result = MetricCalculator.calculate(
            agent_name=agent_name.upper(),
            difficulty=difficulty.upper(),
            episodes=[episode_record],
            api_calls=getattr(agent, "llm_call_count", 0),
            cache_hits=router.cache.stats()["hits"],
        )

        return result

    def run_full_benchmark(
        self,
        agents: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Executes full Agent × Difficulty evaluation matrix."""
        target_agents = agents or ["baseline", "hybrid", "llm"]
        target_diffs = difficulties or ["starter", "medium", "advanced", "adversarial"]

        full_results: Dict[str, Dict[str, BenchmarkResult]] = {}

        for ag in target_agents:
            full_results[ag] = {}
            for diff in target_diffs:
                res = self.run_agent_tier(agent_name=ag, difficulty=diff, seed=seed)
                full_results[ag][diff] = res

        # Export structured JSON report
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": {
                ag: {diff: res.model_dump() for diff, res in diff_map.items()}
                for ag, diff_map in full_results.items()
            },
        }

        output_file = self.results_dir / "benchmark_summary.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return full_results
