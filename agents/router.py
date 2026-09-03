"""
agents/router.py

Centralized Agent Routing and Factory.
Manages agent selection, cache interception, and unified execution policies.
"""

from __future__ import annotations

from typing import Dict, Optional, Union
from envs.email_env.models import EmailAction, EmailObservation
from agents.baseline_agent import BaselineAgent
from agents.cache import AgentCache
from agents.fallback import LocalFallbackAgent
from agents.hybrid_agent import HybridAgent
from agents.llm import LLMClient
from agents.llm_agent import LLMAgent


class AgentRouter:
    """Central router managing agent selection and cache policy."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        cache: Optional[AgentCache] = None,
    ):
        self.cache = cache or AgentCache()
        self.llm_client = llm_client or LLMClient()
        self.fallback = LocalFallbackAgent()

        self.baseline_agent = BaselineAgent()
        self.llm_agent = LLMAgent(llm_client=self.llm_client)
        self.hybrid_agent = HybridAgent(
            llm_client=self.llm_client, fallback_agent=self.fallback
        )

    def get_agent(self, agent_type: str = "hybrid") -> Union[BaselineAgent, LLMAgent, HybridAgent]:
        """Returns the configured agent instance."""
        norm_type = agent_type.strip().lower()
        if norm_type == "baseline":
            return self.baseline_agent
        elif norm_type == "llm":
            return self.llm_agent
        elif norm_type == "hybrid":
            return self.hybrid_agent
        else:
            raise ValueError(f"Unknown agent type '{agent_type}'. Supported: baseline, hybrid, llm")
