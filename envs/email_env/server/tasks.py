"""
envs/email_env/server/tasks.py

Scenario loading, rigorous validation, and episode scenario provisioning.
Validates ground-truth schema, rejects corrupt data, and guarantees reproducible runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

from envs.email_env.models import DifficultyLevel, GroundTruthEmail


VALID_CATEGORIES = {
    "spam",
    "social",
    "informational",
    "routine_requests",
    "billing",
    "customer",
    "vendor",
    "executive",
    "production_incident",
    "payment_failure",
    "security_incident",
    "outage",
    "prompt_injection",
    "misleading_subject",
    "buried_urgency",
    "keyword_trap",
    "multi_intent",
}

VALID_ACTIONS = {"ARCHIVE_EMAIL", "REPLY_EMAIL", "ESCALATE_EMAIL"}
VALID_URGENCIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_DIFFICULTIES = {"STARTER", "MEDIUM", "ADVANCED", "ADVERSARIAL", "HELD_OUT", "REGRESSION"}


class ScenarioValidationError(ValueError):
    """Raised when a scenario file or record fails validation."""
    pass


class ScenarioLoader:
    """Loads and validates scenario JSONL files from data/scenarios/."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to project root data/scenarios
            self.data_dir = Path(__file__).resolve().parents[3] / "data" / "scenarios"
        else:
            self.data_dir = Path(data_dir)

    def load_file(self, file_path: Path) -> List[GroundTruthEmail]:
        """Loads and validates a single .jsonl scenario file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")

        scenarios: List[GroundTruthEmail] = []
        seen_ids: Set[str] = set()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    raise ScenarioValidationError(
                        f"Malformed JSON on line {line_idx} in {file_path.name}: {err}"
                    ) from err

                self.validate_record(record, file_path.name, line_idx, seen_ids)
                seen_ids.add(record["id"])

                scenarios.append(
                    GroundTruthEmail(
                        id=record["id"],
                        category=record["category"],
                        difficulty=record["difficulty"],
                        subject=record["subject"],
                        sender=record["sender"],
                        body=record["body"],
                        ground_truth_action=record["ground_truth_action"],
                        ground_truth_urgency=record["ground_truth_urgency"],
                    )
                )

        if not scenarios:
            raise ScenarioValidationError(f"No valid scenarios found in {file_path.name}")

        return scenarios

    def validate_record(
        self, record: Dict, filename: str, line_idx: int, seen_ids: Set[str]
    ) -> None:
        """Validates an individual scenario dictionary."""
        required_fields = [
            "id",
            "category",
            "difficulty",
            "subject",
            "sender",
            "body",
            "ground_truth_action",
            "ground_truth_urgency",
        ]

        for field in required_fields:
            if field not in record:
                raise ScenarioValidationError(
                    f"Missing required field '{field}' on line {line_idx} in {filename}"
                )
            if not isinstance(record[field], str) or not record[field].strip():
                raise ScenarioValidationError(
                    f"Field '{field}' must be a non-empty string on line {line_idx} in {filename}"
                )

        if record["id"] in seen_ids:
            raise ScenarioValidationError(
                f"Duplicate scenario ID '{record['id']}' on line {line_idx} in {filename}"
            )

        if record["difficulty"].upper() not in VALID_DIFFICULTIES:
            raise ScenarioValidationError(
                f"Invalid difficulty '{record['difficulty']}' on line {line_idx} in {filename}"
            )

        if record["category"].lower() not in VALID_CATEGORIES:
            raise ScenarioValidationError(
                f"Invalid category '{record['category']}' on line {line_idx} in {filename}"
            )

        if record["ground_truth_action"] not in VALID_ACTIONS:
            raise ScenarioValidationError(
                f"Invalid ground_truth_action '{record['ground_truth_action']}' on line {line_idx} in {filename}"
            )

        if record["ground_truth_urgency"].upper() not in VALID_URGENCIES:
            raise ScenarioValidationError(
                f"Invalid ground_truth_urgency '{record['ground_truth_urgency']}' on line {line_idx} in {filename}"
            )

    def load_difficulty(self, difficulty: DifficultyLevel | str) -> List[GroundTruthEmail]:
        """Loads all scenarios for a given difficulty tier."""
        diff_str = difficulty.value.lower() if isinstance(difficulty, DifficultyLevel) else difficulty.lower()
        file_path = self.data_dir / f"{diff_str}.jsonl"
        return self.load_file(file_path)

    def get_episode_emails(
        self,
        difficulty: DifficultyLevel | str = DifficultyLevel.STARTER,
        seed: Optional[int] = None,
        count: Optional[int] = None,
    ) -> List[GroundTruthEmail]:
        """Returns a seeded batch of emails for an episode."""
        emails = self.load_difficulty(difficulty)
        if count is not None and count < len(emails):
            rng = random.Random(seed)
            emails = rng.sample(emails, count)
        elif seed is not None:
            rng = random.Random(seed)
            emails = emails.copy()
            rng.shuffle(emails)
        return [e.model_copy(deep=True) for e in emails]
