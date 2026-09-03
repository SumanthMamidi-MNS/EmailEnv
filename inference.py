"""
inference.py

Headless OpenEnv execution runner for AI Email Agent.
Executes complete simulation episodes across Baseline, Hybrid, and LLM agents
without any UI dependencies.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Optional

# Ensure standard UTF-8 output across platforms
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables (.env) safely if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from envs.email_env.client import EmailEnvClient
from envs.email_env.models import ActionType, DifficultyLevel, EmailStatus
from agents.router import AgentRouter


def run_episode(
    agent_type: str = "hybrid",
    difficulty: str = "STARTER",
    seed: Optional[int] = None,
    count: Optional[int] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Executes a single headless simulation episode and returns summary metrics."""
    client = EmailEnvClient(base_url=base_url)
    router = AgentRouter()
    agent = router.get_agent(agent_type)

    if verbose:
        print("=" * 75)
        print("  AI EMAIL AGENT — OpenEnv Headless Simulation")
        print(f"  Agent Policy : {agent_type.upper()}")
        print(f"  Difficulty   : {difficulty.upper()}")
        print(f"  Client Mode  : {'HTTP Server (' + base_url + ')' if base_url else 'In-Process Local'}")
        print("=" * 75)

    obs = client.reset(seed=seed, difficulty=difficulty, count=count)
    step_num = 0
    valid_actions = 0
    invalid_actions = 0

    if verbose:
        print(f"\n  Inbox loaded with {len(obs.inbox_summary)} emails.\n")
        print(f"  {'Step':<6} {'Action':<35} {'Reward':<10} {'Status / Feedback'}")
        print("  " + "-" * 71)

    while not obs.done:
        action = agent.act(obs)
        obs, reward, done, info = client.step(action)
        step_num += 1

        is_valid = obs.last_action_valid
        if is_valid:
            valid_actions += 1
        else:
            invalid_actions += 1

        if verbose:
            act_str = f"{action.action_type.value}({action.email_id or ''})"
            sign = "+" if reward >= 0 else ""
            reward_str = f"{sign}{reward:.3f}"
            status_indicator = "✓" if is_valid else "✗"
            print(f"  [{step_num:2d}] {status_indicator} {act_str:<32} {reward_str:<10} {obs.last_action_feedback[:28]}")

        if done:
            break

    state = client.state()

    if verbose:
        print("  " + "-" * 71)
        print("\n" + "=" * 75)
        print("  EPISODE SUMMARY")
        print("  " + "-" * 71)
        print(f"  Episode ID        : {state.episode_id}")
        print(f"  Total Steps Taken : {state.step_count} / {state.max_steps}")
        print(f"  Valid Actions     : {valid_actions}")
        print(f"  Invalid Actions   : {invalid_actions}")
        print(f"  Cumulative Reward : {state.cumulative_reward:+.4f}")
        print(f"  Sent Replies      : {len(state.sent_box)}")
        print("\n  Final Inbox State:")
        for em in state.emails:
            print(f"    • [{em.id}] {em.status.value:<10} | GT: {em.ground_truth_action:<15} | Subject: {em.subject[:40]}")
        print("=" * 75 + "\n")

    return {
        "episode_id": state.episode_id,
        "steps": state.step_count,
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
        "cumulative_reward": state.cumulative_reward,
        "done": state.done,
    }


def main():
    parser = argparse.ArgumentParser(description="Run headless AI Email Agent OpenEnv simulation.")
    parser.add_argument(
        "--agent",
        type=str,
        default="hybrid",
        choices=["baseline", "hybrid", "llm"],
        help="Agent policy to execute (baseline, hybrid, llm).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="starter",
        choices=["starter", "medium", "advanced", "adversarial", "held_out", "regression"],
        help="Scenario difficulty tier.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for scenario selection.")
    parser.add_argument("--count", type=int, default=None, help="Limit number of scenario emails.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenEnv HTTP server URL.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose step logs.")

    args = parser.parse_args()

    run_episode(
        agent_type=args.agent,
        difficulty=args.difficulty.upper(),
        seed=args.seed,
        count=args.count,
        base_url=args.base_url,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()