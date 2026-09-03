"""
evaluation/models.py

Pydantic data models for Episode recording, Trajectories, Human Feedback,
Validated Lessons, and Benchmark reporting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FeedbackVerdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"


class LessonStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class LessonScope(str, Enum):
    SCENARIO = "scenario"
    CATEGORY = "category"
    GLOBAL = "global"


class TrajectoryStep(BaseModel):
    """Single step taken during an episode simulation."""
    model_config = ConfigDict(extra="ignore")

    step_number: int
    observation_summary: Dict[str, Any]
    action: Dict[str, Any]
    action_valid: bool
    reward: float
    feedback: str


class EpisodeRecord(BaseModel):
    """Complete trajectory and metadata of a finished episode."""
    model_config = ConfigDict(extra="ignore")

    episode_id: str
    scenario_id: str
    difficulty: str
    agent_name: str
    start_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: List[TrajectoryStep] = Field(default_factory=list)
    final_action: str = ""
    cumulative_reward: float = 0.0
    safety_status: str = "SAFE"  # "SAFE" | "VIOLATION"
    completed: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRecord(BaseModel):
    """Human feedback or expert correction on an agent's episode decision."""
    model_config = ConfigDict(extra="ignore")

    feedback_id: str
    episode_id: str
    scenario_id: str
    agent_name: str
    verdict: FeedbackVerdict
    correct_action: Optional[Dict[str, Any]] = None
    explanation: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LessonRecord(BaseModel):
    """A structured lesson generated from repeated human corrections."""
    model_config = ConfigDict(extra="ignore")

    lesson_id: str
    rule_text: str
    scope: LessonScope = LessonScope.CATEGORY
    target_category: str = "general"
    source_feedback_ids: List[str] = Field(default_factory=list)
    evidence_count: int = 1
    contradiction_count: int = 0
    status: LessonStatus = LessonStatus.CANDIDATE
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: Optional[str] = None
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Aggregated benchmark metrics for an agent on a specific difficulty tier."""
    model_config = ConfigDict(extra="ignore")

    agent_name: str
    difficulty: str
    episodes_count: int
    accuracy: float
    avg_reward: float
    safety_rate: float
    completion_rate: float
    invalid_rate: float
    avg_steps: float
    api_calls: int = 0
    cache_hits: int = 0
    fallback_calls: int = 0
    dangerous_error_rate: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
