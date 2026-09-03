"""
agents/decision.py

Standardized Agent Decision Schema across all agent architectures
(Baseline, Fallback, LLM, and Hybrid).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from envs.email_env.models import ClassLabel, EmailAction


class AgentDecision(BaseModel):
    """
    Standardized decision container output by all agents.
    Guarantees consistent reporting and downstream environment consumption.
    """
    model_config = ConfigDict(extra="ignore")

    action: EmailAction
    classification: ClassLabel
    urgency: str = Field(default="LOW", description="LOW | MEDIUM | HIGH | CRITICAL")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="")
    source: str = Field(default="baseline", description="baseline | llm | fallback | cache")
