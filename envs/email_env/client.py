"""
envs/email_env/client.py

Client interface for connecting to OpenEnv Email Environment.
Supports both remote HTTP server connection and direct local in-process execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import requests

from envs.email_env.models import (
    DifficultyLevel,
    EmailAction,
    EmailObservation,
    EmailState,
)
from envs.email_env.server.environment import EmailEnvironment


class EmailEnvClient:
    """
    OpenEnv Client for EmailEnvironment.
    If base_url is provided, sends requests to the OpenEnv HTTP server.
    If base_url is None, runs directly against a local in-process EmailEnvironment.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self._local_env: Optional[EmailEnvironment] = None
        if not self.base_url:
            self._local_env = EmailEnvironment()

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str | DifficultyLevel = DifficultyLevel.STARTER,
        count: Optional[int] = None,
    ) -> EmailObservation:
        diff_str = (
            difficulty.value.upper()
            if isinstance(difficulty, DifficultyLevel)
            else str(difficulty).upper()
        )

        if self.base_url:
            resp = requests.post(
                f"{self.base_url}/reset",
                json={"seed": seed, "difficulty": diff_str, "count": count},
                timeout=10,
            )
            resp.raise_for_status()
            return EmailObservation.model_validate(resp.json())
        else:
            return self._local_env.reset(seed=seed, difficulty=diff_str, count=count)

    def step(
        self, action: EmailAction
    ) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:
        if self.base_url:
            resp = requests.post(
                f"{self.base_url}/step",
                json=action.model_dump(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            obs = EmailObservation.model_validate(data["observation"])
            return obs, float(data["reward"]), bool(data["done"]), data.get("info", {})
        else:
            return self._local_env.step(action)

    def state(self) -> EmailState:
        if self.base_url:
            resp = requests.get(f"{self.base_url}/state", timeout=10)
            resp.raise_for_status()
            return EmailState.model_validate(resp.json())
        else:
            return self._local_env.state()

    def close(self) -> None:
        pass
