"""
evaluation/__init__.py

Public package exports for Phase 2 Evaluation, Persistence, Benchmarking,
Human Feedback, and Validated Learning.
"""

from evaluation.models import (
    BenchmarkResult,
    EpisodeRecord,
    FeedbackRecord,
    FeedbackVerdict,
    LessonRecord,
    LessonScope,
    LessonStatus,
    TrajectoryStep,
)
from evaluation.storage import SQLiteRepository
from evaluation.metrics import MetricCalculator
from evaluation.retrieval import LessonRetriever
from evaluation.feedback import FeedbackService, FeedbackValidationError
from evaluation.lessons import LessonGenerator
from evaluation.validator import LessonValidator
from evaluation.benchmark import BenchmarkEngine
from evaluation.api import (
    generate_candidate_lessons,
    get_benchmark_summary,
    list_all_lessons,
    list_episodes,
    promote_lesson,
    record_feedback,
    reject_lesson,
    retrieve_lessons,
    run_benchmark,
    run_episode,
    run_harmful_lesson_demo,
    run_learning_cycle_demo,
    validate_lesson,
)

__all__ = [
    "BenchmarkResult",
    "EpisodeRecord",
    "FeedbackRecord",
    "FeedbackVerdict",
    "LessonRecord",
    "LessonScope",
    "LessonStatus",
    "TrajectoryStep",
    "SQLiteRepository",
    "MetricCalculator",
    "LessonRetriever",
    "FeedbackService",
    "FeedbackValidationError",
    "LessonGenerator",
    "LessonValidator",
    "BenchmarkEngine",
    "run_episode",
    "run_benchmark",
    "record_feedback",
    "generate_candidate_lessons",
    "validate_lesson",
    "promote_lesson",
    "reject_lesson",
    "retrieve_lessons",
    "list_episodes",
    "list_all_lessons",
    "get_benchmark_summary",
    "run_learning_cycle_demo",
    "run_harmful_lesson_demo",
]
