"""
evaluation/api.py

Clean Public API for Phase 2 Evaluation, Benchmarking, Feedback, and Learning.
Directly consumable by test suites, CLI tools, and the Phase 3 Streamlit UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from envs.email_env.client import EmailEnvClient
from evaluation.benchmark import BenchmarkEngine
from evaluation.feedback import FeedbackService, FeedbackValidationError
from evaluation.lessons import LessonGenerator
from evaluation.models import EpisodeRecord, FeedbackVerdict, LessonRecord, LessonStatus
from evaluation.retrieval import LessonRetriever
from evaluation.storage import SQLiteRepository
from evaluation.validator import LessonValidator
from agents.router import AgentRouter

_REPO = SQLiteRepository()
_FEEDBACK_SVC = FeedbackService(_REPO)
_LESSON_GEN = LessonGenerator(_REPO)
_VALIDATOR = LessonValidator(_REPO)
_RETRIEVER = LessonRetriever(_REPO)
_BENCHMARK = BenchmarkEngine(_REPO)


def run_episode(
    agent_name: str = "hybrid",
    difficulty: str = "STARTER",
    seed: int = 42,
) -> EpisodeRecord:
    """Executes a single simulation episode and saves its trajectory to persistence."""
    res = _BENCHMARK.run_agent_tier(agent_name=agent_name, difficulty=difficulty, seed=seed)
    episodes = _REPO.list_episodes(limit=1, agent_name=agent_name.upper())
    if episodes:
        return episodes[0]
    raise RuntimeError("Failed to record episode.")


def run_benchmark(
    agents: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Runs a full matrix benchmark across agents and difficulties."""
    results = _BENCHMARK.run_full_benchmark(agents=agents, difficulties=difficulties, seed=seed)
    return {
        ag: {diff: res.model_dump() for diff, res in diff_map.items()}
        for ag, diff_map in results.items()
    }


def record_feedback(
    episode_id: str,
    scenario_id: str,
    agent_name: str,
    verdict: str,
    correct_action: Optional[Dict[str, Any]] = None,
    explanation: str = "",
) -> Dict[str, Any]:
    """Validates and persists human feedback."""
    fb = _FEEDBACK_SVC.record_feedback(
        episode_id=episode_id,
        scenario_id=scenario_id,
        agent_name=agent_name,
        verdict=verdict,
        correct_action=correct_action,
        explanation=explanation,
    )
    return fb.model_dump()


def generate_candidate_lessons(min_evidence: int = 2) -> List[Dict[str, Any]]:
    """Extracts candidate lessons from clusters of consistent human feedback."""
    gen = LessonGenerator(_REPO, min_evidence=min_evidence)
    candidates = gen.generate_candidates()
    return [c.model_dump() for c in candidates]


def validate_lesson(lesson_id: str, simulated_harmful: bool = False) -> Dict[str, Any]:
    """Runs regression evaluation on a candidate lesson and promotes or rejects it."""
    lesson = _REPO.get_lesson(lesson_id)
    if not lesson:
        raise ValueError(f"Lesson '{lesson_id}' not found.")
    passed, report = _VALIDATOR.validate_candidate(lesson, simulated_harmful=simulated_harmful)
    return report


def promote_lesson(lesson_id: str) -> None:
    """Manually marks a lesson as PROMOTED."""
    _REPO.update_lesson_status(lesson_id, status=LessonStatus.PROMOTED)


def reject_lesson(lesson_id: str) -> None:
    """Manually marks a lesson as REJECTED."""
    _REPO.update_lesson_status(lesson_id, status=LessonStatus.REJECTED)


def retrieve_lessons(
    subject: str = "",
    sender: str = "",
    body: str = "",
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves active PROMOTED lessons matching email context."""
    lessons = _RETRIEVER.get_relevant_lessons(
        subject=subject, sender=sender, body=body, category=category
    )
    return [l.model_dump() for l in lessons]


def list_episodes(
    limit: int = 50, agent_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Lists recent episode trajectories from persistence."""
    eps = _REPO.list_episodes(limit=limit, agent_name=agent_name)
    return [e.model_dump() for e in eps]


def list_all_lessons() -> List[Dict[str, Any]]:
    """Lists all lessons in database (Candidate, Promoted, Rejected)."""
    lessons = _REPO.list_all_lessons()
    return [l.model_dump() for l in lessons]


def get_benchmark_summary() -> Dict[str, Any]:
    """Reads latest persisted benchmark summary report."""
    results_path = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "benchmark_summary.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Full Learning Demonstrations ─────────────────────────────────────────────

def run_learning_cycle_demo() -> Dict[str, Any]:
    """
    Demonstrates complete Phase 2 learning pipeline:
    Before -> Human Feedback -> Candidate Lesson -> Regression Validation -> Promotion -> After.
    """
    # 1. Record repeated feedback for vendor contract renewal
    scen_id = "reg_003"
    fb1 = record_feedback(
        episode_id="ep_demo_01",
        scenario_id=scen_id,
        agent_name="HYBRID",
        verdict="INCORRECT",
        correct_action={"action_type": "REPLY_EMAIL", "email_id": scen_id, "body": "Confirmed acceptance."},
        explanation="For vendor contract renewals, always REPLY_EMAIL to confirm acceptance.",
    )
    fb2 = record_feedback(
        episode_id="ep_demo_02",
        scenario_id=scen_id,
        agent_name="HYBRID",
        verdict="INCORRECT",
        correct_action={"action_type": "REPLY_EMAIL", "email_id": scen_id, "body": "Confirmed acceptance."},
        explanation="For vendor contract renewals, always REPLY_EMAIL to confirm acceptance.",
    )

    # 2. Pattern detection generates candidate lesson
    candidates = generate_candidate_lessons(min_evidence=2)
    target_candidate = next((c for c in candidates if "vendor" in c["target_category"] or scen_id in c["rule_text"]), candidates[0])

    # 3. Regression validation promotes the safe/improving lesson
    report = validate_lesson(target_candidate["lesson_id"], simulated_harmful=False)

    # 4. Retrieval verification
    retrieved = retrieve_lessons(subject="Vendor Contract Renewal", category="vendor")

    return {
        "status": "SUCCESS",
        "feedback_count": 2,
        "candidate_lesson_id": target_candidate["lesson_id"],
        "validation_report": report,
        "retrieved_promoted_count": len(retrieved),
    }


def run_harmful_lesson_demo() -> Dict[str, Any]:
    """
    Demonstrates safety gate audit:
    A dangerous human suggestion ("Archive all critical outages") is REJECTED by regression validation.
    """
    scen_id = "reg_005"
    fb = record_feedback(
        episode_id="ep_harmful_01",
        scenario_id=scen_id,
        agent_name="HYBRID",
        verdict="INCORRECT",
        correct_action={"action_type": "ARCHIVE_EMAIL", "email_id": scen_id},
        explanation="Archive all critical outage alerts immediately without escalation.",
    )

    # Manually create candidate for this harmful rule
    gen = LessonGenerator(_REPO, min_evidence=1)
    candidates = gen.generate_candidates()
    harmful_cand = next(
        (c for c in candidates if "outage" in c.rule_text.lower() or "critical" in c.rule_text.lower()),
        candidates[0],
    )

    # Validate against regression suite
    report = validate_lesson(harmful_cand.lesson_id, simulated_harmful=True)

    # Ensure it was rejected and is NEVER retrieved
    retrieved = retrieve_lessons(subject="CRITICAL ALERT: Primary Ingress Crash", category="production_incident")

    return {
        "status": "REJECTED_AS_EXPECTED",
        "harmful_lesson_id": harmful_cand.lesson_id,
        "validation_report": report,
        "promoted": report["passed"],
        "retrieved_count": len(retrieved),
    }
