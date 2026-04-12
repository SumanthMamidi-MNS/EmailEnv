# ═══ benchmark.py ═══
"""
AI Email Agent — benchmark script.
Runs RandomAgent, RuleBasedAgent, and ZeroShotAgent across all tasks.
Run with: python benchmark.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    from models.environment import EmailEnv
    _ENV_OK = True
except Exception as _e:
    print(f"[ERROR] Could not import EmailEnv: {_e}")
    _ENV_OK = False

try:
    from models import ClassLabel, Action
    _MODELS_OK = True
except Exception as _e:
    print(f"[ERROR] Could not import models: {_e}")
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
        def archive_email(eid): return {"type": "archive_email", "email_id": eid}
        @staticmethod
        def escalate_email(eid): return {"type": "escalate_email", "email_id": eid}
        @staticmethod
        def reply_email(eid, text=""): return {"type": "reply_email", "email_id": eid, "text": text}

try:
    from agent import EmailClassifier
    _zeroshot_clf = EmailClassifier()
    _ZEROSHOT_OK = True
except Exception:
    _ZEROSHOT_OK = False
    _zeroshot_clf = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASK_IDS = ["task_1", "task_2", "task_3"]
TASK_LABELS = {"task_1": "Task 1", "task_2": "Task 2", "task_3": "Task 3"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: get email IDs from observation
# ─────────────────────────────────────────────────────────────────────────────

def _email_ids(obs: Any) -> list:
    if isinstance(obs, dict):
        return obs.get("inbox", obs.get("email_ids", list(obs.keys())))
    if hasattr(obs, "inbox"):
        return list(obs.inbox)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Keyword classifier (shared by RuleBasedAgent)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_label(subject: str, body: str) -> str:
    text = (subject + " " + body).lower()
    if any(w in text for w in ["win", "winner", "prize", "lottery", "click here", "free offer", "unsubscribe"]):
        return "SPAM"
    if any(w in text for w in ["urgent", "asap", "immediately", "critical", "emergency", "time-sensitive"]):
        return "URGENT"
    if any(w in text for w in ["please review", "approval", "sign off", "action required", "response required"]):
        return "ACTION_REQUIRED"
    if any(w in text for w in ["invitation", "invite", "party", "birthday", "rsvp", "event"]):
        return "SOCIAL"
    return "INFORMATIONAL"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: RandomAgent
# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """Picks a random action at each step from the available action set."""

    name = "RandomAgent"

    _LABELS = ["SPAM", "URGENT", "ACTION_REQUIRED", "SOCIAL", "INFORMATIONAL"]

    def run_episode(self, task_id: str) -> float:
        env = EmailEnv(task_id)
        obs = env.reset()
        done = False
        email_ids = _email_ids(obs)

        for eid in email_ids:
            if done:
                break
            if random.random() > 0.3:
                _, _, done, _ = env.step(Action.read_email(eid))
                if done:
                    break

            label_str = random.choice(self._LABELS)
            label_obj = getattr(ClassLabel, label_str, label_str)
            _, _, done, _ = env.step(Action.classify_email(eid, label_obj))
            if done:
                break

            follow = random.choice(["archive", "reply", "escalate"])
            if follow == "archive":
                _, _, done, _ = env.step(Action.archive_email(eid))
            elif follow == "reply":
                _, _, done, _ = env.step(Action.reply_email(eid, "Random reply."))
            else:
                _, _, done, _ = env.step(Action.escalate_email(eid))

        result = env.grade()
        return result.get("score", 0.0) if isinstance(result, dict) else float(result)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: RuleBasedAgent
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedAgent:
    """Uses keyword matching to classify, then takes the canonical action."""

    name = "RuleBasedAgent"

    def run_episode(self, task_id: str) -> float:
        env = EmailEnv(task_id)
        obs = env.reset()
        done = False
        email_ids = _email_ids(obs)

        for eid in email_ids:
            if done:
                break

            obs, _, done, info = env.step(Action.read_email(eid))
            if done:
                break

            subject = info.get("subject", "")
            body = info.get("body", "")
            label_str = _keyword_label(subject, body)
            label_obj = getattr(ClassLabel, label_str, label_str)

            obs, _, done, info = env.step(Action.classify_email(eid, label_obj))
            if done:
                break

            if label_str in ("URGENT", "ACTION_REQUIRED"):
                obs, _, done, _ = env.step(
                    Action.reply_email(eid, "Thank you — we are addressing this promptly.")
                )
            else:
                obs, _, done, _ = env.step(Action.archive_email(eid))

        result = env.grade()
        return result.get("score", 0.0) if isinstance(result, dict) else float(result)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: ZeroShotAgent
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotAgent:
    """Uses EmailClassifier (best available tier) for classification."""

    name = "ZeroShotAgent"

    def _classify(self, subject: str, body: str) -> str:
        if _ZEROSHOT_OK and _zeroshot_clf is not None:
            try:
                label = _zeroshot_clf.classify(subject, body)
                return label.value if hasattr(label, "value") else str(label)
            except Exception:
                pass
        return _keyword_label(subject, body)

    def run_episode(self, task_id: str) -> float:
        env = EmailEnv(task_id)
        obs = env.reset()
        done = False
        email_ids = _email_ids(obs)

        for eid in email_ids:
            if done:
                break

            obs, _, done, info = env.step(Action.read_email(eid))
            if done:
                break

            subject = info.get("subject", "")
            body = info.get("body", "")
            label_str = self._classify(subject, body)
            label_obj = getattr(ClassLabel, label_str, label_str)

            obs, _, done, info = env.step(Action.classify_email(eid, label_obj))
            if done:
                break

            if label_str in ("URGENT", "ACTION_REQUIRED"):
                obs, _, done, _ = env.step(
                    Action.reply_email(eid, "Thank you — we are handling this promptly.")
                )
            else:
                obs, _, done, _ = env.step(Action.archive_email(eid))

        result = env.grade()
        return result.get("score", 0.0) if isinstance(result, dict) else float(result)


# ─────────────────────────────────────────────────────────────────────────────
# Table formatter
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(results: dict[str, dict[str, float]]) -> None:
    agents = list(results.keys())
    col_w = 12

    header = f"  {'Agent':<20}" + "".join(
        f"  {TASK_LABELS.get(t, t):>{col_w}}" for t in TASK_IDS
    ) + f"  {'Average':>{col_w}}"
    divider = "  " + "─" * (len(header) - 2)

    print(divider)
    print(header)
    print(divider)

    for agent_name in agents:
        scores = results[agent_name]
        row = f"  {agent_name:<20}"
        values = []
        for tid in TASK_IDS:
            s = scores.get(tid)
            cell = f"{s:.2f}" if s is not None else " N/A"
            row += f"  {cell:>{col_w}}"
            if s is not None:
                values.append(s)
        avg = sum(values) / len(values) if values else 0.0
        # FIX: `f"  {avg:.2f:>{col_w}}"` is an invalid format spec — Python
        # does not support chaining two format specs with a single colon.
        # The format spec `.2f:>{col_w}` raises ValueError at runtime.
        # Fix: pre-format avg to a string then apply the width alignment.
        avg_str = f"{avg:.2f}"
        row += f"  {avg_str:>{col_w}}"
        print(row)

    print(divider)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("═" * 72)
    print("  AI EMAIL AGENT — Benchmark")
    print("  Agents : RandomAgent | RuleBasedAgent | ZeroShotAgent")
    print(f"  Tasks  : {', '.join(TASK_IDS)}")
    print("═" * 72)
    print()

    if not _ENV_OK:
        print("[FATAL] EmailEnv is unavailable. Ensure environment.py is present.")
        sys.exit(1)

    agents = [RandomAgent(), RuleBasedAgent(), ZeroShotAgent()]

    results: dict[str, dict[str, float]] = {a.name: {} for a in agents}

    for agent in agents:
        for task_id in TASK_IDS:
            label = TASK_LABELS.get(task_id, task_id)
            print(f"  Running  {agent.name:<20}  on  {label} ...", end="", flush=True)
            try:
                score = agent.run_episode(task_id)
                results[agent.name][task_id] = round(score, 4)
                print(f"  score = {score:.4f}")
            except Exception as exc:
                results[agent.name][task_id] = 0.0
                print(f"  ERROR: {exc}")

    print()
    print("  RESULTS TABLE")
    _print_table(results)

    averages: dict[str, float] = {}
    for agent_name, scores in results.items():
        vals = [v for v in scores.values() if v is not None]
        averages[agent_name] = round(sum(vals) / len(vals), 4) if vals else 0.0

    payload = {
        "agents": list(results.keys()),
        "tasks": TASK_IDS,
        "scores": results,
        "averages": averages,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "benchmark_results.json"

    try:
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  Saved → {output_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save results: {exc}")

    print()
    print("═" * 72)
    print("  Developed by Sumanth Mamidi")
    print("═" * 72)
    print()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] Unexpected error: {exc}")
        traceback.print_exc()
        sys.exit(1)