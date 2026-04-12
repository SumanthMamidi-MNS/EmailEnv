"""
environment.py — AI Email Assistant OpenEnv
Core environment implementing reset(), step(), and state management.
Follows the OpenEnv interface contract.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from models.grader import CompositeGrader, EpisodeData
from models import (
    Action,
    ActionResult,
    ActionType,
    ClassLabel,
    Email,
    EmailStatus,
    InboxItem,
    Observation,
    SentEmail,
    SentItem,
    State,
)
from models.tasks import TaskConfig, get_task


# ─────────────────────────────────────────────
# Reward constants
# ─────────────────────────────────────────────

class RewardConfig:
    READ_BEFORE_ACT:              float =  0.05
    CLASSIFY_CORRECT:             float =  0.20
    CLASSIFY_WRONG:               float = -0.10
    REPLY_NEEDED_SENT:            float =  0.10
    REPLY_NOT_NEEDED_NOT_SENT:    float =  0.10
    REPLY_UNNECESSARY:            float = -0.15
    REPLY_QUALITY_MULTIPLIER:     float =  0.30   # × quality score
    ESCALATE_CORRECT:             float =  0.25
    ESCALATE_WRONG:               float = -0.20
    ARCHIVE_CORRECT:              float =  0.05
    ARCHIVE_BEFORE_READ:          float = -0.05
    NO_OP_PENALTY:                float = -0.02
    INVALID_ACTION_PENALTY:       float = -0.10
    THREAD_CONTEXT_BONUS:         float =  0.15
    BUDGET_COMPLETION_BONUS:      float =  0.10
    EXCESS_STEP_PENALTY:          float = -0.05


R = RewardConfig()


# ─────────────────────────────────────────────
# Helper: current UTC timestamp
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Action Executor
# ─────────────────────────────────────────────

class ActionExecutor:
    """
    Validates and applies actions to State.
    Returns an ActionResult with reward delta and feedback.
    """

    def __init__(self, task: TaskConfig):
        self._task = task

    def execute(self, action: Action, state: State) -> ActionResult:
        at = action.action_type

        if at == ActionType.READ_EMAIL:
            return self._read_email(action, state)
        elif at == ActionType.CLASSIFY_EMAIL:
            return self._classify_email(action, state)
        elif at == ActionType.REPLY_EMAIL:
            return self._reply_email(action, state)
        elif at == ActionType.ARCHIVE_EMAIL:
            return self._archive_email(action, state)
        elif at == ActionType.ESCALATE_EMAIL:
            return self._escalate_email(action, state)
        elif at == ActionType.WRITE_MEMORY:
            return self._write_memory(action, state)
        elif at == ActionType.NO_OP:
            return ActionResult(
                valid=True,
                feedback="No-op taken.",
                reward_delta=R.NO_OP_PENALTY,
            )
        else:
            return ActionResult(
                valid=False,
                feedback=f"Unknown action type: {at}",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )

    # ── Individual action handlers ───────────────────────

    def _read_email(self, action: Action, state: State) -> ActionResult:
        if not action.email_id:
            return ActionResult(
                valid=False,
                feedback="read_email requires email_id.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        email = state.get_email(action.email_id)
        if email is None:
            return ActionResult(
                valid=False,
                feedback=f"Email '{action.email_id}' does not exist.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.UNREAD:
            email.status = EmailStatus.READ
        return ActionResult(
            valid=True,
            feedback=f"Email '{action.email_id}' read successfully.",
            reward_delta=R.READ_BEFORE_ACT,
            opened_email=email,
        )

    def _classify_email(self, action: Action, state: State) -> ActionResult:
        if not action.email_id or action.label is None:
            return ActionResult(
                valid=False,
                feedback="classify_email requires email_id and label.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        email = state.get_email(action.email_id)
        if email is None:
            return ActionResult(
                valid=False,
                feedback=f"Email '{action.email_id}' does not exist.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.UNREAD:
            return ActionResult(
                valid=False,
                feedback=f"Must read email '{action.email_id}' before classifying.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )

        # Check if already classified (re-classify is allowed but penalised below)
        was_already_classified = email.assigned_label is not None
        email.assigned_label = action.label

        if action.label == email.priority:
            delta = R.CLASSIFY_CORRECT
            feedback = f"Correct classification: {action.label.value}."
        else:
            delta = R.CLASSIFY_WRONG
            if was_already_classified:
                delta += R.CLASSIFY_WRONG  # extra penalty for flip-flopping
            feedback = (
                f"Wrong classification: predicted {action.label.value}. "
                f"Hint: re-read the email carefully."
            )

        return ActionResult(valid=True, feedback=feedback, reward_delta=delta)

    def _reply_email(self, action: Action, state: State) -> ActionResult:
        if not action.email_id or not action.body:
            return ActionResult(
                valid=False,
                feedback="reply_email requires email_id and body.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if len(action.body.strip()) < 10:
            return ActionResult(
                valid=False,
                feedback="Reply body too short (minimum 10 characters).",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        email = state.get_email(action.email_id)
        if email is None:
            return ActionResult(
                valid=False,
                feedback=f"Email '{action.email_id}' does not exist.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.UNREAD:
            return ActionResult(
                valid=False,
                feedback=f"Must read email '{action.email_id}' before replying.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.ESCALATED:
            return ActionResult(
                valid=False,
                feedback="Cannot reply to an escalated email — it is under human review.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )

        # Check if already replied
        already_replied = any(s.in_reply_to == action.email_id for s in state.sent_box)
        if already_replied:
            return ActionResult(
                valid=False,
                feedback=f"Already replied to '{action.email_id}'. Duplicate reply rejected.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )

        state.sent_box.append(SentEmail(
            in_reply_to=action.email_id,
            body=action.body,
            timestamp=_now_iso(),
        ))

        needed = action.email_id in self._task.emails_requiring_reply
        delta = R.REPLY_NEEDED_SENT if needed else R.REPLY_UNNECESSARY
        feedback = (
            "Reply sent successfully."
            if needed
            else "Reply sent to an email that did not require a response."
        )
        return ActionResult(valid=True, feedback=feedback, reward_delta=delta)

    def _archive_email(self, action: Action, state: State) -> ActionResult:
        if not action.email_id:
            return ActionResult(
                valid=False,
                feedback="archive_email requires email_id.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        email = state.get_email(action.email_id)
        if email is None:
            return ActionResult(
                valid=False,
                feedback=f"Email '{action.email_id}' does not exist.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.ESCALATED:
            return ActionResult(
                valid=False,
                feedback="Cannot archive an escalated email — conflict.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.UNREAD:
            email.status = EmailStatus.ARCHIVED
            return ActionResult(
                valid=True,
                feedback="Email archived (without reading).",
                reward_delta=R.ARCHIVE_BEFORE_READ,
            )
        email.status = EmailStatus.ARCHIVED
        return ActionResult(
            valid=True,
            feedback=f"Email '{action.email_id}' archived.",
            reward_delta=R.ARCHIVE_CORRECT,
        )

    def _escalate_email(self, action: Action, state: State) -> ActionResult:
        if not action.email_id:
            return ActionResult(
                valid=False,
                feedback="escalate_email requires email_id.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if not action.reason or len(action.reason.strip()) < 5:
            return ActionResult(
                valid=False,
                feedback="escalate_email requires a non-empty reason (min 5 chars).",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        email = state.get_email(action.email_id)
        if email is None:
            return ActionResult(
                valid=False,
                feedback=f"Email '{action.email_id}' does not exist.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        if email.status == EmailStatus.ARCHIVED:
            return ActionResult(
                valid=False,
                feedback="Cannot escalate an archived email — conflict.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )

        email.status = EmailStatus.ESCALATED
        needed = action.email_id in self._task.emails_requiring_escalation
        delta = R.ESCALATE_CORRECT if needed else R.ESCALATE_WRONG
        feedback = (
            "Email escalated for human review."
            if needed
            else f"Unnecessary escalation of '{action.email_id}'."
        )
        return ActionResult(valid=True, feedback=feedback, reward_delta=delta)

    def _write_memory(self, action: Action, state: State) -> ActionResult:
        if not action.memory_key:
            return ActionResult(
                valid=False,
                feedback="write_memory requires memory_key.",
                reward_delta=R.INVALID_ACTION_PENALTY,
            )
        state.agent_memory[action.memory_key] = action.memory_value
        return ActionResult(
            valid=True,
            feedback=f"Memory key '{action.memory_key}' stored.",
            reward_delta=0.0,
        )


# ─────────────────────────────────────────────
# Observation Builder
# ─────────────────────────────────────────────

class ObservationBuilder:
    """Derives an Observation from the current State."""

    @staticmethod
    def build(
        state: State,
        last_result: Optional[ActionResult] = None,
    ) -> Observation:
        current_email = (
            last_result.opened_email.to_view()
            if last_result and last_result.opened_email
            else None
        )

        inbox_summary = [e.to_inbox_item() for e in state.inbox]

        sent_summary = [
            SentItem(
                in_reply_to=s.in_reply_to,
                timestamp=s.timestamp,
                body_preview=s.body[:100],
            )
            for s in state.sent_box
        ]

        return Observation(
            current_email=current_email,
            inbox_summary=inbox_summary,
            sent_summary=sent_summary,
            step=state.step,
            steps_remaining=state.max_steps - state.step,
            memory=copy.deepcopy(state.agent_memory),
            last_action_valid=last_result.valid if last_result else True,
            last_action_feedback=last_result.feedback if last_result else "Episode started.",
        )


# ─────────────────────────────────────────────
# Episode Tracker (for grader at episode end)
# ─────────────────────────────────────────────

class EpisodeTracker:
    """Collects data during an episode for end-of-episode grading."""

    def __init__(self):
        self.action_log: List[Action] = []
        self.invalid_action_count: int = 0
        self.replies: Dict[str, str] = {}             # email_id → reply body
        self.escalated_ids: Set[str] = set()
        self.escalation_reasons: Dict[str, str] = {}
        self.thread_context_used: Set[str] = set()

    def record_action(self, action: Action, result: ActionResult, thread_pairs: List[tuple]):
        self.action_log.append(action)
        if not result.valid:
            self.invalid_action_count += 1

        if action.action_type == ActionType.REPLY_EMAIL and result.valid and action.body:
            self.replies[action.email_id] = action.body
            # Detect thread context usage: does the reply mention content from parent email?
            for parent_id, child_id in thread_pairs:
                if action.email_id == child_id:
                    # Heuristic: if memory contains parent reference or body mentions
                    # keywords from typical thread context (contract ID, case ID etc.)
                    # We mark it used if the reply is non-trivial (>50 words) — a
                    # richer check is done by ReplyQualityGrader via required_phrases
                    if len(action.body.split()) > 50:
                        self.thread_context_used.add(child_id)

        if action.action_type == ActionType.ESCALATE_EMAIL and result.valid:
            self.escalated_ids.add(action.email_id)
            self.escalation_reasons[action.email_id] = action.reason or ""


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────

class EmailEnv:
    """
    AI Email Assistant environment.

    Usage:
        env = EmailEnv("task_1")
        obs = env.reset()
        while not env.done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        result = env.grade()
    """

    def __init__(self, task_id: str):
        self._task_id = task_id
        self._task: TaskConfig = get_task(task_id)
        self._state: State = State()
        self._executor: ActionExecutor = ActionExecutor(self._task)
        self._tracker: EpisodeTracker = EpisodeTracker()
        self._grader: CompositeGrader = CompositeGrader()
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._step_count: int = 0

    # ── Public interface ─────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._state = State(
            inbox=copy.deepcopy(self._task.emails),
            sent_box=[],
            agent_memory={},
            step=0,
            done=False,
            max_steps=self._task.max_steps,
        )
        self._tracker = EpisodeTracker()
        self._cumulative_reward = 0.0
        self._done = False
        self._step_count = 0
        return ObservationBuilder.build(self._state, last_result=None)

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply an action to the environment.

        Returns:
            observation  — new Observation after the action
            reward       — step-level reward delta (float)
            done         — whether the episode has ended
            info         — diagnostic dict
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        self._state.step += 1
        self._step_count += 1

        # Execute action
        result = self._executor.execute(action, self._state)

        # Track
        self._tracker.record_action(action, result, self._task.thread_pairs)

        # Compute step reward
        step_reward = result.reward_delta

        # Thread context bonus for reply
        if (
            action.action_type == ActionType.REPLY_EMAIL
            and result.valid
            and action.email_id in self._tracker.thread_context_used
        ):
            step_reward += R.THREAD_CONTEXT_BONUS

        # Reply omission penalty (informational email not replied to is good — handled at end)
        # No-reply-needed check is incorporated in the cumulative end-reward via the grader.

        self._cumulative_reward += step_reward

        # Check done conditions
        budget_exceeded = self._step_count >= self._task.max_steps
        all_handled = self._all_emails_handled()

        if budget_exceeded:
            # Penalise for exceeding budget
            if self._step_count > self._task.max_steps:
                self._cumulative_reward += R.EXCESS_STEP_PENALTY
            self._done = True
            self._state.done = True
        elif all_handled:
            # Bonus for finishing within budget
            self._cumulative_reward += R.BUDGET_COMPLETION_BONUS
            self._done = True
            self._state.done = True

        obs = ObservationBuilder.build(self._state, last_result=result)

        info = {
            "step": self._step_count,
            "step_reward": round(step_reward, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "done": self._done,
            "action_valid": result.valid,
            "valid": result.valid,                   # alias for backward compatibility
            "action_feedback": result.feedback,
            "steps_remaining": self._task.max_steps - self._step_count,
        }

        return obs, round(step_reward, 4), self._done, info

    def grade(self) -> Dict[str, Any]:
        """
        Run the CompositeGrader on the completed episode.
        Returns a dict with 'score' (0.0–1.0) and detailed breakdown.

        Should be called after the episode is done.
        """
        predicted_labels = {
            e.id: e.assigned_label for e in self._state.inbox
        }

        episode_data = EpisodeData(
            task=self._task,
            action_log=self._tracker.action_log,
            invalid_action_count=self._tracker.invalid_action_count,
            steps_used=self._step_count,
            predicted_labels=predicted_labels,
            replies=self._tracker.replies,
            escalated_ids=self._tracker.escalated_ids,
            escalation_reasons=self._tracker.escalation_reasons,
            thread_context_used=self._tracker.thread_context_used,
        )

        grader_result = self._grader.grade(episode_data)

        return {
            "score": grader_result.score,
            "cumulative_step_reward": round(self._cumulative_reward, 4),
            "breakdown": grader_result.breakdown,
            "notes": grader_result.notes,
            "steps_used": self._step_count,
            "steps_budget": self._task.max_steps,
            "task_id": self._task.task_id,
        }

    # ── Properties ───────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done

    @property
    def state(self) -> State:
        """Direct state access — intended for testing/debugging only."""
        return self._state

    @property
    def task(self) -> TaskConfig:
        return self._task

    @property
    def cumulative_reward(self) -> float:
        return round(self._cumulative_reward, 4)

    # ── Internals ────────────────────────────────────────

    def _all_emails_handled(self) -> bool:
        """
        Episode can end early if every email has been fully acted upon.
        "Fully handled" = classified AND replied (if required) AND
        placed in a terminal state (ARCHIVED or ESCALATED).
        READ alone is not terminal — the agent must explicitly archive or escalate.
        """
        for email in self._state.inbox:
            # Must be classified
            if email.assigned_label is None:
                return False
            # Must have received a reply if required
            if email.id in self._task.emails_requiring_reply:
                replied = any(s.in_reply_to == email.id for s in self._state.sent_box)
                if not replied:
                    return False
            # Must be in a terminal state
            if email.status not in (EmailStatus.ARCHIVED, EmailStatus.ESCALATED):
                return False
        return True

    def render(self) -> str:
        """Human-readable state snapshot for debugging."""
        lines = [
            f"=== EmailEnv | Task: {self._task.task_id} | Step: {self._step_count}/{self._task.max_steps} ===",
            f"Cumulative reward: {self._cumulative_reward:.4f}",
            "",
            "INBOX:",
        ]
        for email in self._state.inbox:
            label_str = email.assigned_label.value if email.assigned_label else "?"
            lines.append(
                f"  [{email.status.value:10s}] {email.id} | "
                f"From: {email.sender[:30]:30s} | "
                f"Label: {label_str:16s} | "
                f"Subject: {email.subject[:50]}"
            )
        lines.append("")
        lines.append(f"SENT ({len(self._state.sent_box)} emails):")
        for sent in self._state.sent_box:
            lines.append(
                f"  → {sent.in_reply_to} | {sent.timestamp[:19]} | "
                f"{sent.body[:60]}..."
            )
        lines.append(f"\nMEMORY: {json.dumps(self._state.agent_memory, indent=2)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────

def make_env(task_id: str) -> EmailEnv:
    """Factory function — preferred way to instantiate the environment."""
    return EmailEnv(task_id)