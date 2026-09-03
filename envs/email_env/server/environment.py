"""
envs/email_env/server/environment.py

Core OpenEnv Environment for Autonomous Email Agent simulation.
Maintains state, executes validated actions, computes rubric rewards,
and returns sanitized observations.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from envs.email_env.models import (
    ActionType,
    DifficultyLevel,
    EmailAction,
    EmailObservation,
    EmailState,
    EmailStatus,
    EmailView,
    GroundTruthEmail,
    InboxItem,
    SentItem,
)
from envs.email_env.server.rubric import EmailRubric, RubricWeights
from envs.email_env.server.tasks import ScenarioLoader


class EmailEnvironment:
    """
    Autonomous Email Agent Simulation Environment.
    Adheres to the OpenEnv client/server contract.
    """

    def __init__(self, loader: Optional[ScenarioLoader] = None, max_steps_per_email: int = 4):
        self.loader = loader or ScenarioLoader()
        self.max_steps_per_email = max_steps_per_email
        self._state: Optional[EmailState] = None
        self._current_opened_email: Optional[GroundTruthEmail] = None

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str | DifficultyLevel = DifficultyLevel.STARTER,
        count: Optional[int] = None,
    ) -> EmailObservation:
        """Resets the environment for a new episode with a fresh scenario batch."""
        diff_str = (
            difficulty.value.upper()
            if isinstance(difficulty, DifficultyLevel)
            else str(difficulty).upper()
        )
        emails = self.loader.get_episode_emails(difficulty=diff_str, seed=seed, count=count)
        max_steps = max(len(emails) * self.max_steps_per_email, 15)

        self._state = EmailState(
            episode_id=str(uuid.uuid4()),
            difficulty=diff_str,
            step_count=0,
            max_steps=max_steps,
            emails=emails,
            sent_box=[],
            action_history=[],
            done=False,
            cumulative_reward=0.0,
        )
        self._current_opened_email = None

        return self._make_observation(
            last_valid=True, feedback="Environment reset. New episode started."
        )

    def step(
        self, action: EmailAction | Dict[str, Any]
    ) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:
        """
        Executes an agent action, updates internal state, calculates rubric reward,
        and returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        if self._state.done:
            obs = self._make_observation(
                last_valid=False, feedback="Episode already terminated."
            )
            return obs, 0.0, True, {"error": "Episode finished"}

        # Normalize action input
        if isinstance(action, dict):
            action = EmailAction.model_validate(action)

        self._state.step_count += 1
        reward_delta, feedback, is_valid = self._execute_action(action)

        self._state.cumulative_reward += reward_delta

        # Record action in history
        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "action": action.model_dump(),
                "reward": reward_delta,
                "valid": is_valid,
                "feedback": feedback,
            }
        )

        # Check termination condition
        all_handled = all(
            email.status in (EmailStatus.ARCHIVED, EmailStatus.ESCALATED, EmailStatus.REPLIED)
            for email in self._state.emails
        )
        time_limit_reached = self._state.step_count >= self._state.max_steps

        if all_handled or time_limit_reached:
            self._state.done = True

        obs = self._make_observation(last_valid=is_valid, feedback=feedback)
        info = {
            "step": self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "all_handled": all_handled,
            "time_limit": time_limit_reached,
            "feedback": feedback,
        }

        return obs, reward_delta, self._state.done, info

    def state(self) -> EmailState:
        """Returns internal state snapshot for server/inspection."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized.")
        return self._state

    # ── Action Execution Logic ────────────────────────────────────────────────

    def _execute_action(self, action: EmailAction) -> Tuple[float, str, bool]:
        at = action.action_type

        if at == ActionType.NO_OP:
            return RubricWeights.NO_OP, "No-op action taken.", True

        # For actions targeting an email, validate email exists
        if not action.email_id:
            return (
                RubricWeights.INVALID_ACTION,
                f"Action {at.value} missing required target 'email_id'.",
                False,
            )

        email = self._state.get_email(action.email_id)
        if not email:
            return (
                RubricWeights.INVALID_ACTION,
                f"Target email '{action.email_id}' does not exist in inbox.",
                False,
            )

        if at == ActionType.READ_EMAIL:
            reward, feedback, is_valid = EmailRubric.evaluate_read(email)
            if is_valid:
                email.status = EmailStatus.READ
                self._current_opened_email = email
            return reward, feedback, is_valid

        elif at == ActionType.CLASSIFY_EMAIL:
            if not action.label:
                return (
                    RubricWeights.INVALID_ACTION,
                    f"CLASSIFY_EMAIL missing required 'label' parameter for {email.id}.",
                    False,
                )
            reward, feedback, is_valid = EmailRubric.evaluate_classify(email, action.label)
            if is_valid:
                email.assigned_label = action.label
            return reward, feedback, is_valid

        elif at == ActionType.REPLY_EMAIL:
            reward, feedback, is_valid = EmailRubric.evaluate_reply(email, action.body)
            if is_valid:
                email.status = EmailStatus.REPLIED
                email.reply_body = action.body
                email.assigned_action = "REPLY"
                self._state.sent_box.append(
                    {
                        "in_reply_to": email.id,
                        "body": action.body,
                        "body_preview": (action.body[:100] + "...")
                        if len(action.body or "") > 100
                        else action.body,
                    }
                )
            return reward, feedback, is_valid

        elif at == ActionType.ESCALATE_EMAIL:
            reward, feedback, is_valid = EmailRubric.evaluate_escalate(email, action.reason)
            if is_valid:
                email.status = EmailStatus.ESCALATED
                email.assigned_action = "ESCALATE"
                email.assigned_reason = action.reason
            return reward, feedback, is_valid

        elif at == ActionType.ARCHIVE_EMAIL:
            reward, feedback, is_valid = EmailRubric.evaluate_archive(email)
            if is_valid:
                email.status = EmailStatus.ARCHIVED
                email.assigned_action = "ARCHIVE"
            return reward, feedback, is_valid

        return (
            RubricWeights.INVALID_ACTION,
            f"Unsupported action type: {at.value}",
            False,
        )

    # ── Observation Construction ─────────────────────────────────────────────

    def _make_observation(self, last_valid: bool, feedback: str) -> EmailObservation:
        """Constructs an observation containing only agent-permitted data."""
        inbox_summary = [e.to_inbox_item() for e in self._state.emails]
        sent_summary = [
            SentItem(in_reply_to=s["in_reply_to"], body_preview=s["body_preview"])
            for s in self._state.sent_box
        ]

        current_view: Optional[EmailView] = None
        if self._current_opened_email:
            current_view = self._current_opened_email.to_view()

        steps_remaining = max(0, self._state.max_steps - self._state.step_count)

        return EmailObservation(
            current_email=current_view,
            inbox_summary=inbox_summary,
            sent_summary=sent_summary,
            step=self._state.step_count,
            steps_remaining=steps_remaining,
            last_action_valid=last_valid,
            last_action_feedback=feedback,
            done=self._state.done,
        )
