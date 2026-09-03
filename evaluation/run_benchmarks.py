"""
evaluation/run_benchmarks.py

CLI entrypoint to execute the full multi-agent benchmark matrix across all difficulty tiers.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure standard UTF-8 output across platforms
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.api import run_benchmark


def main():
    print("===========================================================================")
    print("  AI EMAIL AGENT — Multi-Agent Benchmark Suite (Phase 2)")
    print("===========================================================================")

    agents = ["baseline", "hybrid", "llm"]
    difficulties = ["starter", "medium", "advanced", "adversarial", "held_out"]

    print(f"Agents: {agents}")
    print(f"Difficulties: {difficulties}")
    print("Executing benchmark runs across OpenEnv simulation...\n")

    results = run_benchmark(agents=agents, difficulties=difficulties, seed=42)

    print("\n===========================================================================")
    print("  BENCHMARK MATRIX RESULTS")
    print("===========================================================================")
    header = f"{'Agent':<10} | {'Difficulty':<12} | {'Accuracy':<8} | {'Avg Reward':<10} | {'Safety %':<8} | {'Steps':<6}"
    print(header)
    print("-" * len(header))

    for ag, diff_map in results.items():
        for diff, metrics in diff_map.items():
            print(
                f"{ag.upper():<10} | {diff.upper():<12} | "
                f"{metrics['accuracy']*100:>6.1f}% | "
                f"{metrics['avg_reward']:>+10.2f} | "
                f"{metrics['safety_rate']*100:>7.0f}% | "
                f"{metrics['avg_steps']:>6.1f}"
            )
        print("-" * len(header))

    print("\nBenchmark summary persisted to: evaluation/results/benchmark_summary.json")


if __name__ == "__main__":
    main()
