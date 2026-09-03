"""
evaluation/lessons.py

Pattern detection and Candidate Lesson extraction.
Identifies repeated human feedback patterns and compiles candidate lessons for regression testing.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List, Optional

from evaluation.models import (
    FeedbackRecord,
    FeedbackVerdict,
    LessonRecord,
    LessonScope,
    LessonStatus,
)
from evaluation.storage import SQLiteRepository

KNOWN_CATEGORIES = [
    "vendor",
    "customer",
    "billing",
    "production_incident",
    "security_incident",
    "outage",
    "spam",
    "social",
    "informational",
    "routine_requests",
]


class LessonGenerator:
    """Detects recurring human correction patterns and generates candidate lessons."""

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        min_evidence: int = 2,
    ):
        self.repo = repository or SQLiteRepository()
        self.min_evidence = min_evidence

    def generate_candidates(self) -> List[LessonRecord]:
        """
        Analyzes all recorded feedback. When repeated corrections (>= min_evidence)
        are found for a scenario or category, extracts a Candidate Lesson.
        """
        all_feedback = self.repo.list_feedback()

        # Group incorrect corrections by scenario_id and target action
        clusters: Dict[str, List[FeedbackRecord]] = defaultdict(list)

        for fb in all_feedback:
            if fb.verdict == FeedbackVerdict.INCORRECT and fb.correct_action:
                act_type = fb.correct_action.get("action_type", "UNKNOWN")
                key = f"{fb.scenario_id}|{act_type}"
                clusters[key].append(fb)

        generated_candidates: List[LessonRecord] = []

        for key, fb_list in clusters.items():
            if len(fb_list) >= self.min_evidence:
                scenario_id, act_type = key.split("|", 1)
                source_ids = [f.feedback_id for f in fb_list]
                explanations = [f.explanation for f in fb_list if f.explanation]
                summary_rule = explanations[0] if explanations else (
                    f"For scenario {scenario_id}, always execute action {act_type}."
                )

                # Deduce category from explanation keywords or scenario id
                target_cat = "general"
                combined_text = (summary_rule + " " + scenario_id).lower()
                for cat in KNOWN_CATEGORIES:
                    if cat in combined_text:
                        target_cat = cat
                        break

                lesson_id = f"les_{uuid.uuid4().hex[:8]}"
                candidate = LessonRecord(
                    lesson_id=lesson_id,
                    rule_text=summary_rule,
                    scope=LessonScope.CATEGORY,
                    target_category=target_cat,
                    source_feedback_ids=source_ids,
                    evidence_count=len(fb_list),
                    contradiction_count=0,
                    status=LessonStatus.CANDIDATE,
                )

                self.repo.save_lesson(candidate)
                generated_candidates.append(candidate)

        return generated_candidates
