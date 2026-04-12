"""
grader.py — AI Email Assistant OpenEnv
Deterministic graders that score agent performance per task.
All scores are in [0.0, 1.0]. No LLM calls — rule-based only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from models import Action, ActionType, ClassLabel, EmailStatus
from models.tasks import ReplyCriteria, TaskConfig


# ─────────────────────────────────────────────
# Severity matrix — misclassification penalties
# Higher = worse mistake (e.g. calling urgent "spam" is very bad)
# ─────────────────────────────────────────────

_SEVERITY_MATRIX: Dict[ClassLabel, Dict[ClassLabel, float]] = {
    ClassLabel.URGENT: {
        ClassLabel.URGENT:          1.0,
        ClassLabel.ACTION_REQUIRED: 0.5,
        ClassLabel.INFORMATIONAL:   0.1,
        ClassLabel.SOCIAL:          0.0,
        ClassLabel.SPAM:            0.0,   # catastrophic mis-label
    },
    ClassLabel.ACTION_REQUIRED: {
        ClassLabel.ACTION_REQUIRED: 1.0,
        ClassLabel.URGENT:          0.7,
        ClassLabel.INFORMATIONAL:   0.4,
        ClassLabel.SOCIAL:          0.2,
        ClassLabel.SPAM:            0.0,
    },
    ClassLabel.SPAM: {
        ClassLabel.SPAM:            1.0,
        ClassLabel.INFORMATIONAL:   0.3,
        ClassLabel.SOCIAL:          0.2,
        ClassLabel.ACTION_REQUIRED: 0.0,
        ClassLabel.URGENT:          0.0,
    },
    ClassLabel.INFORMATIONAL: {
        ClassLabel.INFORMATIONAL:   1.0,
        ClassLabel.ACTION_REQUIRED: 0.6,
        ClassLabel.SOCIAL:          0.5,
        ClassLabel.SPAM:            0.2,
        ClassLabel.URGENT:          0.2,
    },
    ClassLabel.SOCIAL: {
        ClassLabel.SOCIAL:          1.0,
        ClassLabel.INFORMATIONAL:   0.7,
        ClassLabel.ACTION_REQUIRED: 0.4,
        ClassLabel.SPAM:            0.2,
        ClassLabel.URGENT:          0.1,
    },
}


def _severity_score(ground_truth: ClassLabel, predicted: ClassLabel) -> float:
    """Returns a per-email classification score in [0.0, 1.0]."""
    row = _SEVERITY_MATRIX.get(ground_truth, {})
    return row.get(predicted, 0.0)


# ─────────────────────────────────────────────
# Sub-grader result carriers
# ─────────────────────────────────────────────

@dataclass
class GraderResult:
    score: float                            # 0.0 – 1.0
    breakdown: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# 1. ClassificationGrader
# ─────────────────────────────────────────────

class ClassificationGrader:
    """
    Scores classification accuracy with severity-weighted mistakes.
    Penalises skipped emails (no label assigned).
    """

    def grade(
        self,
        ground_truth_map: Dict[str, ClassLabel],    # email_id → true label
        predicted_map: Dict[str, Optional[ClassLabel]],  # email_id → agent label
    ) -> GraderResult:

        total = len(ground_truth_map)
        if total == 0:
            return GraderResult(score=1.0, notes=["No emails to classify."])

        per_email: Dict[str, float] = {}
        notes: List[str] = []

        for email_id, gt_label in ground_truth_map.items():
            pred = predicted_map.get(email_id)
            if pred is None:
                per_email[email_id] = 0.0
                notes.append(f"{email_id}: not classified (0.0)")
            else:
                s = _severity_score(gt_label, pred)
                per_email[email_id] = s
                mark = "✓" if s == 1.0 else "✗"
                notes.append(
                    f"{email_id}: {mark} gt={gt_label.value} pred={pred.value} score={s:.2f}"
                )

        score = sum(per_email.values()) / total

        return GraderResult(
            score=round(score, 4),
            breakdown={"per_email": per_email, "total_emails": total},
            notes=notes,
        )


# ─────────────────────────────────────────────
# 2. ReplyQualityGrader
# ─────────────────────────────────────────────

_FORMAL_INDICATORS = [
    "dear", "sincerely", "regards", "thank you", "please", "kindly",
    "would you", "could you", "i am writing", "we would", "per our",
    "as discussed", "further to", "i confirm", "i approve", "acknowledged",
]

_INFORMAL_INDICATORS = [
    "hey!", "lol", "omg", "gonna", "wanna", "yeah", "nah", "sup",
    "thx", "u r", "ur ", "btw", "idk", "tbh", "asap lol",
]


def _count_words(text: str) -> int:
    return len(text.strip().split())


def _contains_any(text: str, terms: List[str]) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in terms)


def _contains_all(text: str, terms: List[str]) -> bool:
    lower = text.lower()
    return all(t.lower() in lower for t in terms)


class ReplyQualityGrader:
    """
    Scores reply quality across 5 dimensions (equal weight):
      1. Relevance      — at least one required keyword present
      2. Completeness   — all required phrases present
      3. Tone           — formal/informal match
      4. Length         — within word count bounds
      5. Thread context — if thread email, references original (checked separately)
    """

    def grade(
        self,
        reply_body: str,
        criteria: ReplyCriteria,
        references_thread_context: bool = False,
        is_thread_email: bool = False,
    ) -> GraderResult:

        if not reply_body or not reply_body.strip():
            return GraderResult(
                score=0.0,
                notes=["Empty reply body."],
            )

        scores: Dict[str, float] = {}
        notes: List[str] = []

        # 1. Relevance: at least one keyword present
        has_keyword = _contains_any(reply_body, criteria.required_keywords)
        scores["relevance"] = 1.0 if has_keyword else 0.0
        notes.append(
            f"Relevance: {'✓' if has_keyword else '✗'} "
            f"(keywords: {criteria.required_keywords})"
        )

        # 2. Completeness: ALL required phrases present
        phrase_hits = [
            p for p in criteria.required_phrases
            if p.lower() in reply_body.lower()
        ]
        completeness = (
            len(phrase_hits) / len(criteria.required_phrases)
            if criteria.required_phrases else 1.0
        )
        scores["completeness"] = round(completeness, 4)
        notes.append(
            f"Completeness: {len(phrase_hits)}/{len(criteria.required_phrases)} phrases "
            f"({phrase_hits})"
        )

        # 3. Tone
        formal_count = sum(
            1 for t in _FORMAL_INDICATORS if t in reply_body.lower()
        )
        informal_count = sum(
            1 for t in _INFORMAL_INDICATORS if t in reply_body.lower()
        )
        if criteria.formal_tone:
            # Formal expected: reward formal signals, penalise informal
            tone_score = min(1.0, formal_count * 0.25) if informal_count == 0 else max(
                0.0, min(1.0, formal_count * 0.25) - informal_count * 0.3
            )
        else:
            # Informal expected: any natural text passes
            tone_score = 1.0 if informal_count == 0 or formal_count > 0 else 0.5
        scores["tone"] = round(min(1.0, max(0.0, tone_score)), 4)
        notes.append(
            f"Tone: formal_signals={formal_count} informal_signals={informal_count} "
            f"score={scores['tone']}"
        )

        # 4. Length
        wc = _count_words(reply_body)
        if wc < criteria.min_words:
            length_score = wc / criteria.min_words  # partial for too short
        elif wc > criteria.max_words:
            overage = wc - criteria.max_words
            length_score = max(0.3, 1.0 - (overage / criteria.max_words))
        else:
            length_score = 1.0
        scores["length"] = round(length_score, 4)
        notes.append(
            f"Length: {wc} words (range {criteria.min_words}–{criteria.max_words}) "
            f"score={scores['length']}"
        )

        # 5. Thread context (only relevant for thread emails)
        if is_thread_email:
            scores["thread_context"] = 1.0 if references_thread_context else 0.0
            notes.append(
                f"Thread context: {'✓' if references_thread_context else '✗'}"
            )
        else:
            scores["thread_context"] = 1.0   # N/A → full marks
            notes.append("Thread context: N/A (not a thread email) → 1.0")

        final = sum(scores.values()) / len(scores)

        return GraderResult(
            score=round(final, 4),
            breakdown=scores,
            notes=notes,
        )


# ─────────────────────────────────────────────
# 3. EscalationGrader
# ─────────────────────────────────────────────

class EscalationGrader:
    """
    F1-based accuracy for escalation decisions, plus reason quality.
    score = 0.7 × F1 + 0.3 × reason_quality
    """

    def grade(
        self,
        escalated_ids: Set[str],                          # agent's escalations
        ground_truth_ids: Set[str],                       # should be escalated
        reasons: Dict[str, str],                          # email_id → reason text
        expected_keywords: Dict[str, List[str]],          # email_id → expected kws
    ) -> GraderResult:

        notes: List[str] = []

        # F1
        true_positives = escalated_ids & ground_truth_ids
        false_positives = escalated_ids - ground_truth_ids
        false_negatives = ground_truth_ids - escalated_ids

        precision = (
            len(true_positives) / len(escalated_ids)
            if escalated_ids else (1.0 if not ground_truth_ids else 0.0)
        )
        recall = (
            len(true_positives) / len(ground_truth_ids)
            if ground_truth_ids else 1.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        notes.append(
            f"Escalation F1: {f1:.4f} "
            f"(TP={len(true_positives)} FP={len(false_positives)} FN={len(false_negatives)})"
        )

        # Reason quality per correctly escalated email
        reason_scores: List[float] = []
        for email_id in true_positives:
            reason_text = reasons.get(email_id, "")
            kws = expected_keywords.get(email_id, [])
            if not kws:
                reason_scores.append(1.0)
                continue
            hits = [k for k in kws if k.lower() in reason_text.lower()]
            q = len(hits) / len(kws)
            reason_scores.append(q)
            notes.append(
                f"Reason quality for {email_id}: {len(hits)}/{len(kws)} keywords → {q:.2f}"
            )

        reason_quality = (
            sum(reason_scores) / len(reason_scores) if reason_scores else
            (0.0 if ground_truth_ids else 1.0)
        )

        final = round(0.7 * f1 + 0.3 * reason_quality, 4)

        return GraderResult(
            score=final,
            breakdown={
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "reason_quality": round(reason_quality, 4),
            },
            notes=notes,
        )


# ─────────────────────────────────────────────
# 4. WorkflowGrader
# ─────────────────────────────────────────────

class WorkflowGrader:
    """
    Scores agent workflow quality:
      1. read_before_act_ratio  — emails read before classify/reply
      2. priority_order_score   — urgent emails handled before low-priority
      3. no_invalid_actions     — fraction of valid actions
      4. budget_efficiency      — completed within step budget
    """

    def grade(
        self,
        action_log: List[Action],
        total_emails: int,
        invalid_action_count: int,
        steps_used: int,
        steps_budget: int,
        urgent_email_ids: Set[str],
        low_priority_ids: Set[str],
    ) -> GraderResult:

        notes: List[str] = []
        scores: Dict[str, float] = {}

        # 1. Read-before-act ratio
        read_set: Set[str] = set()
        acted_without_read = 0
        acted_with_read = 0

        for action in action_log:
            if action.action_type == ActionType.READ_EMAIL and action.email_id:
                read_set.add(action.email_id)
            elif action.action_type in (
                ActionType.CLASSIFY_EMAIL,
                ActionType.REPLY_EMAIL,
                ActionType.ARCHIVE_EMAIL,
                ActionType.ESCALATE_EMAIL,
            ) and action.email_id:
                if action.email_id in read_set:
                    acted_with_read += 1
                else:
                    acted_without_read += 1

        total_acts = acted_with_read + acted_without_read
        read_ratio = (acted_with_read / total_acts) if total_acts > 0 else 1.0
        scores["read_before_act"] = round(read_ratio, 4)
        notes.append(f"Read-before-act: {acted_with_read}/{total_acts} → {read_ratio:.4f}")

        # 2. Priority ordering: urgent emails should appear before low-priority in action log
        urgent_first_step: Optional[int] = None
        low_first_step: Optional[int] = None

        for i, action in enumerate(action_log):
            if action.email_id in urgent_email_ids and urgent_first_step is None:
                urgent_first_step = i
            if action.email_id in low_priority_ids and low_first_step is None:
                low_first_step = i

        if urgent_email_ids and low_priority_ids:
            if urgent_first_step is None:
                priority_score = 0.0  # never touched urgent
            elif low_first_step is None:
                priority_score = 1.0  # handled urgent, ignored low (acceptable)
            else:
                priority_score = 1.0 if urgent_first_step < low_first_step else 0.2
        else:
            priority_score = 1.0  # no ordering requirement

        scores["priority_order"] = round(priority_score, 4)
        notes.append(
            f"Priority order: urgent_step={urgent_first_step} "
            f"low_step={low_first_step} → {priority_score:.4f}"
        )

        # 3. No invalid actions
        total_actions = len(action_log)
        valid_ratio = (
            1.0 - (invalid_action_count / total_actions)
            if total_actions > 0 else 1.0
        )
        scores["no_invalid_actions"] = round(max(0.0, valid_ratio), 4)
        notes.append(
            f"Invalid actions: {invalid_action_count}/{total_actions} "
            f"→ {scores['no_invalid_actions']:.4f}"
        )

        # 4. Budget efficiency
        if steps_used <= steps_budget:
            budget_score = 1.0
        else:
            overage = steps_used - steps_budget
            budget_score = max(0.0, 1.0 - (overage / steps_budget))
        scores["budget_efficiency"] = round(budget_score, 4)
        notes.append(
            f"Budget: {steps_used}/{steps_budget} steps → {budget_score:.4f}"
        )

        final = sum(scores.values()) / len(scores)

        return GraderResult(
            score=round(final, 4),
            breakdown=scores,
            notes=notes,
        )


# ─────────────────────────────────────────────
# 5. CompositeGrader
# ─────────────────────────────────────────────

@dataclass
class EpisodeData:
    """All data needed to grade a completed episode."""
    task: TaskConfig
    action_log: List[Action]
    invalid_action_count: int
    steps_used: int

    # Classification
    predicted_labels: Dict[str, Optional[ClassLabel]]   # email_id → agent label

    # Replies
    replies: Dict[str, str]                              # email_id → reply body

    # Escalations
    escalated_ids: Set[str]
    escalation_reasons: Dict[str, str]                   # email_id → reason

    # Thread context: did agent reference the parent email in the thread reply?
    thread_context_used: Set[str] = field(default_factory=set)   # email_ids where context used


class CompositeGrader:
    """
    Orchestrates all sub-graders and computes a final weighted score.

    Task 1: 0.80 classification + 0.20 workflow
    Task 2: 0.35 classification + 0.45 reply + 0.20 workflow
    Task 3: 0.25 classification + 0.30 reply + 0.25 escalation + 0.20 workflow
    """

    WEIGHTS: Dict[str, Dict[str, float]] = {
        "task_1_classification": {
            "classification": 0.80,
            "reply":          0.00,
            "escalation":     0.00,
            "workflow":       0.20,
        },
        "task_2_reply": {
            "classification": 0.35,
            "reply":          0.45,
            "escalation":     0.00,
            "workflow":       0.20,
        },
        "task_3_multistep": {
            "classification": 0.25,
            "reply":          0.30,
            "escalation":     0.25,
            "workflow":       0.20,
        },
    }

    def __init__(self):
        self._cls_grader   = ClassificationGrader()
        self._reply_grader = ReplyQualityGrader()
        self._esc_grader   = EscalationGrader()
        self._wf_grader    = WorkflowGrader()

    def grade(self, data: EpisodeData) -> GraderResult:
        task = data.task
        weights = self.WEIGHTS.get(
            task.task_id,
            {"classification": 0.4, "reply": 0.3, "escalation": 0.1, "workflow": 0.2},
        )

        sub_results: Dict[str, GraderResult] = {}
        all_notes: List[str] = [f"=== Grading: {task.task_id} ==="]

        # ── Ground-truth maps ────────────────────────────
        ground_truth_labels = {e.id: e.priority for e in task.emails}

        urgent_ids = {
            e.id for e in task.emails
            if e.priority == ClassLabel.URGENT
        }
        low_ids = {
            e.id for e in task.emails
            if e.priority in (ClassLabel.SPAM, ClassLabel.SOCIAL, ClassLabel.INFORMATIONAL)
        }

        # ── 1. Classification ────────────────────────────
        cls_result = self._cls_grader.grade(ground_truth_labels, data.predicted_labels)
        sub_results["classification"] = cls_result
        all_notes += ["--- ClassificationGrader ---"] + cls_result.notes

        # ── 2. Reply Quality ─────────────────────────────
        if weights["reply"] > 0 and task.reply_criteria:
            criteria_map = {c.email_id: c for c in task.reply_criteria}
            thread_email_ids = {child for _, child in task.thread_pairs}

            reply_scores: List[float] = []
            reply_notes: List[str] = []

            # Score replies that were sent to emails that required one
            for email_id in task.emails_requiring_reply:
                criteria = criteria_map.get(email_id)
                if criteria is None:
                    continue
                reply_body = data.replies.get(email_id, "")
                is_thread = email_id in thread_email_ids
                used_ctx = email_id in data.thread_context_used

                if not reply_body:
                    reply_scores.append(0.0)
                    reply_notes.append(f"{email_id}: no reply sent → 0.0")
                    continue

                r = self._reply_grader.grade(
                    reply_body=reply_body,
                    criteria=criteria,
                    references_thread_context=used_ctx,
                    is_thread_email=is_thread,
                )
                reply_scores.append(r.score)
                reply_notes += [f"{email_id} reply:"] + r.notes

            # Penalise replies sent to emails that didn't need one
            for email_id, body in data.replies.items():
                if email_id not in task.emails_requiring_reply and body:
                    reply_scores.append(max(0.0, (reply_scores[-1] if reply_scores else 0.5) - 0.3))
                    reply_notes.append(f"{email_id}: unnecessary reply sent → penalty")

            avg_reply = sum(reply_scores) / len(reply_scores) if reply_scores else 1.0
            reply_result = GraderResult(
                score=round(avg_reply, 4),
                breakdown={"per_email_scores": dict(zip(task.emails_requiring_reply, reply_scores))},
                notes=reply_notes,
            )
            sub_results["reply"] = reply_result
            all_notes += ["--- ReplyQualityGrader ---"] + reply_notes

        else:
            sub_results["reply"] = GraderResult(score=1.0, notes=["Reply grading not applicable."])

        # ── 3. Escalation ────────────────────────────────
        if weights["escalation"] > 0 and task.emails_requiring_escalation:
            esc_result = self._esc_grader.grade(
                escalated_ids=data.escalated_ids,
                ground_truth_ids=task.emails_requiring_escalation,
                reasons=data.escalation_reasons,
                expected_keywords=task.escalation_keywords,
            )
            sub_results["escalation"] = esc_result
            all_notes += ["--- EscalationGrader ---"] + esc_result.notes
        else:
            sub_results["escalation"] = GraderResult(score=1.0, notes=["Escalation not applicable."])

        # ── 4. Workflow ──────────────────────────────────
        wf_result = self._wf_grader.grade(
            action_log=data.action_log,
            total_emails=len(task.emails),
            invalid_action_count=data.invalid_action_count,
            steps_used=data.steps_used,
            steps_budget=task.max_steps,
            urgent_email_ids=urgent_ids,
            low_priority_ids=low_ids,
        )
        sub_results["workflow"] = wf_result
        all_notes += ["--- WorkflowGrader ---"] + wf_result.notes

        # ── Final weighted score ─────────────────────────
        final = sum(
            weights[k] * sub_results[k].score
            for k in ("classification", "reply", "escalation", "workflow")
        )
        final = round(max(0.0, min(1.0, final)), 4)

        all_notes.append(
            f"=== FINAL SCORE: {final:.4f} | "
            + " | ".join(
                f"{k}={sub_results[k].score:.4f}×{weights[k]}"
                for k in ("classification", "reply", "escalation", "workflow")
            )
            + " ==="
        )

        return GraderResult(
            score=final,
            breakdown={
                "weights": weights,
                "sub_scores": {k: v.score for k, v in sub_results.items()},
                "sub_breakdowns": {k: v.breakdown for k, v in sub_results.items()},
            },
            notes=all_notes,
        )