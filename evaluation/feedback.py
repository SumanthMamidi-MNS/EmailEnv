"""
evaluation/feedback.py

Human Feedback ingestion and validation service.
Validates human corrections against the OpenEnv action schema and records evidence.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from envs.email_env.models import ActionType, EmailAction
from evaluation.models import FeedbackRecord, FeedbackVerdict
from evaluation.storage import SQLiteRepository


class FeedbackValidationError(ValueError):
    """Raised when human feedback record is invalid."""
    pass


class FeedbackService:
    """Service for validating and saving human feedback records."""

    def __init__(self, repository: Optional[SQLiteRepository] = None):
        self.repo = repository or SQLiteRepository()

    def record_feedback(
        self,
        episode_id: str,
        scenario_id: str,
        agent_name: str,
        verdict: str | FeedbackVerdict,
        correct_action: Optional[Dict[str, Any]] = None,
        explanation: str = "",
    ) -> FeedbackRecord:
        """Validates and persists human feedback."""
        if not episode_id or not episode_id.strip():
            raise FeedbackValidationError("Feedback missing required 'episode_id'.")

        if not scenario_id or not scenario_id.strip():
            raise FeedbackValidationError("Feedback missing required 'scenario_id'.")

        norm_verdict = (
            verdict if isinstance(verdict, FeedbackVerdict) else FeedbackVerdict(str(verdict).upper())
        )

        # If human flagged decision as INCORRECT, validate proposed correction action
        if norm_verdict == FeedbackVerdict.INCORRECT:
            if not correct_action:
                raise FeedbackValidationError(
                    "Human feedback flagged as INCORRECT must specify a 'correct_action'."
                )
            try:
                # Must be a valid EmailAction schema
                EmailAction.model_validate(correct_action)
            except Exception as err:
                raise FeedbackValidationError(
                    f"Proposed 'correct_action' is invalid according to EmailAction schema: {err}"
                ) from err

        fb = FeedbackRecord(
            feedback_id=f"fb_{uuid.uuid4().hex[:8]}",
            episode_id=episode_id.strip(),
            scenario_id=scenario_id.strip(),
            agent_name=agent_name.strip(),
            verdict=norm_verdict,
            correct_action=correct_action,
            explanation=explanation.strip(),
        )

        self.repo.save_feedback(fb)
        return fb

    def get_feedback_for_scenario(self, scenario_id: str) -> List[FeedbackRecord]:
        return self.repo.list_feedback(scenario_id=scenario_id)

    def get_all_feedback(self) -> List[FeedbackRecord]:
        return self.repo.list_feedback()
