"""
evaluation/retrieval.py

Lesson retrieval engine.
Retrieves ONLY validated, PROMOTED institutional lessons to assist agent decision-making.
Guarantees that candidate and rejected lessons are never activated.
"""

from __future__ import annotations

from typing import List, Optional
from evaluation.models import LessonRecord, LessonStatus
from evaluation.storage import SQLiteRepository


class LessonRetriever:
    """Retrieval interface for validated institutional knowledge."""

    def __init__(self, repository: Optional[SQLiteRepository] = None):
        self.repo = repository or SQLiteRepository()

    def get_relevant_lessons(
        self,
        subject: str = "",
        sender: str = "",
        body: str = "",
        category: Optional[str] = None,
    ) -> List[LessonRecord]:
        """
        Retrieves all active PROMOTED lessons relevant to the given email context.
        Candidate and Rejected lessons are strictly filtered out.
        """
        promoted = self.repo.get_promoted_lessons()

        text = f"{subject} {sender} {body}".lower()
        matched_lessons: List[LessonRecord] = []

        for lesson in promoted:
            # Check if lesson is strictly PROMOTED
            if lesson.status != LessonStatus.PROMOTED:
                continue

            # If global scope, always include
            if lesson.scope.value == "global":
                matched_lessons.append(lesson)
                continue

            # If category matches or rule keywords appear in email text
            target_cat = (lesson.target_category or "").lower()
            if target_cat and (
                target_cat in text
                or target_cat == (category or "").lower()
                or (category and category.lower() in target_cat)
            ):
                matched_lessons.append(lesson)
                continue

            # Check rule text overlap
            rule_words = [w.lower() for w in lesson.rule_text.split() if len(w) > 4]
            if any(w in text for w in rule_words):
                matched_lessons.append(lesson)

        return matched_lessons

    def format_lessons_for_prompt(self, lessons: List[LessonRecord]) -> str:
        """Formats retrieved lessons for injection into trusted prompt context."""
        if not lessons:
            return ""
        lines = ["<trusted_lessons>"]
        for idx, lesson in enumerate(lessons, 1):
            lines.append(f"- Lesson {idx} ({lesson.target_category}): {lesson.rule_text}")
        lines.append("</trusted_lessons>")
        return "\n".join(lines)
