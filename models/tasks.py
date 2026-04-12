"""
tasks.py — AI Email Assistant OpenEnv
Defines Task 1 (Easy/5 emails), Task 2 (Medium/10 emails), Task 3 (Advanced/15 emails)
with synthetic email data, expected actions, and grader criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from models import ClassLabel, Email, EmailStatus


# ─────────────────────────────────────────────
# Reply Criteria — deterministic grader rules
# ─────────────────────────────────────────────

@dataclass
class ReplyCriteria:
    """Expected content rules for a reply. Used by ReplyQualityGrader."""
    email_id: str
    required_keywords: List[str]     # at least one must appear (case-insensitive)
    required_phrases: List[str]      # all must appear (case-insensitive)
    min_words: int = 20
    max_words: int = 300
    formal_tone: bool = True         # True = formal expected


# ─────────────────────────────────────────────
# Task Config
# ─────────────────────────────────────────────

@dataclass
class TaskConfig:
    task_id: str
    description: str
    max_steps: int
    emails: List[Email]
    emails_requiring_reply: Set[str]         # email IDs that need a reply
    emails_requiring_escalation: Set[str]    # email IDs that need escalation
    reply_criteria: List[ReplyCriteria]
    escalation_keywords: Dict[str, List[str]]  # email_id → expected reason keywords
    expected_order: Optional[List[str]] = None  # priority-ordered email IDs (Hard task)
    thread_pairs: List[tuple] = field(default_factory=list)  # [(parent_id, child_id)]


# ─────────────────────────────────────────────
# Synthetic Email Factory
# ─────────────────────────────────────────────

def _make_email(
    id: str,
    sender: str,
    subject: str,
    body: str,
    priority: ClassLabel,
    timestamp: str,
    thread_id: Optional[str] = None,
    attachments: Optional[List[str]] = None,
) -> Email:
    return Email(
        id=id,
        sender=sender,
        subject=subject,
        body=body,
        priority=priority,
        timestamp=timestamp,
        thread_id=thread_id,
        attachments=attachments or [],
        status=EmailStatus.UNREAD,
    )


# ─────────────────────────────────────────────
# TASK 1 — Easy: Classification (5 emails)
# ─────────────────────────────────────────────

def build_task1() -> TaskConfig:
    emails = [
        _make_email(
            id="t1_e1",
            sender="noreply@lottery-win.biz",
            subject="CONGRATULATIONS! You won $1,000,000 — claim now!",
            body=(
                "Dear Lucky Winner, You have been selected to receive $1,000,000. "
                "Click the link below to claim your prize immediately. "
                "This offer expires in 24 hours. Provide your bank details to proceed."
            ),
            priority=ClassLabel.SPAM,
            timestamp="2024-03-01T08:00:00Z",
        ),
        _make_email(
            id="t1_e2",
            sender="cto@company.com",
            subject="URGENT: Production database is down — all hands needed",
            body=(
                "Team, our production database (db-prod-01) crashed 10 minutes ago. "
                "All services are impacted. I need everyone on a call immediately. "
                "Join the war room: https://meet.company.com/warroom — this is P0."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T08:05:00Z",
        ),
        _make_email(
            id="t1_e3",
            sender="hr@company.com",
            subject="Please submit your Q1 timesheet by Friday",
            body=(
                "Hi team, a reminder that Q1 timesheets are due this Friday by 5 PM. "
                "Please log in to the HR portal and complete your submission. "
                "Late submissions will delay payroll processing. Thank you."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T09:00:00Z",
        ),
        _make_email(
            id="t1_e4",
            sender="newsletter@techdigest.io",
            subject="Tech Digest Weekly: AI trends, cloud costs, and more",
            body=(
                "This week in tech: OpenAI releases new model benchmarks. "
                "AWS announces 15% cost reduction on EC2 instances. "
                "GitHub Copilot now supports 30 new languages. Read more below."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T09:30:00Z",
        ),
        _make_email(
            id="t1_e5",
            sender="alice@company.com",
            subject="Team lunch this Friday — are you in?",
            body=(
                "Hey! We're planning a team lunch at the Italian place on Friday at 1 PM. "
                "Let me know if you can make it so I can book the right table size. "
                "It'll be a nice break from the sprint grind!"
            ),
            priority=ClassLabel.SOCIAL,
            timestamp="2024-03-01T10:00:00Z",
        ),
    ]

    return TaskConfig(
        task_id="task_1",
        description=(
            "Classify 5 emails into the correct category. "
            "Read each email before classifying it."
        ),
        max_steps=15,
        emails=emails,
        emails_requiring_reply=set(),
        emails_requiring_escalation=set(),
        reply_criteria=[],
        escalation_keywords={},
    )


# ─────────────────────────────────────────────
# TASK 2 — Medium: Reply Generation (10 emails)
# ─────────────────────────────────────────────

def build_task2() -> TaskConfig:
    emails = [
        # ── Original 3 emails ──────────────────────────────────────────────────
        _make_email(
            id="t2_e1",
            sender="client@bigcorp.com",
            subject="Follow-up: Project proposal deadline",
            body=(
                "Hi, I'm following up on the project proposal we discussed last week. "
                "Our board meeting is on March 15th and we need the final proposal by March 12th. "
                "Could you confirm whether your team can meet this deadline? "
                "Also please clarify the pricing breakdown for the Phase 2 deliverables. "
                "This is time-sensitive for us."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T08:00:00Z",
        ),
        _make_email(
            id="t2_e2",
            sender="alerts@monitoring.company.com",
            subject="INFO: Weekly system health report — all systems nominal",
            body=(
                "Weekly System Health Summary for 2024-W09:\n"
                "- API Gateway: 99.98% uptime\n"
                "- Database cluster: 99.95% uptime\n"
                "- CDN: 100% uptime\n"
                "- Average response time: 142ms\n"
                "No incidents recorded this week. Next report: 2024-03-08."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T08:30:00Z",
        ),
        _make_email(
            id="t2_e3",
            sender="manager@company.com",
            subject="URGENT: Budget approval needed before EOD",
            body=(
                "Hi, I need your written approval for the $45,000 software licensing budget "
                "for Q2 before end of business today. Finance has a hard cutoff at 6 PM. "
                "Without this approval the licenses will lapse and the engineering team "
                "will lose access to critical tools starting Monday. Please reply ASAP."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T09:00:00Z",
        ),
        # ── 7 new emails to reach 10 total ─────────────────────────────────────
        _make_email(
            id="t2_e4",
            sender="noreply@promo-deals247.com",
            subject="You've been selected for a FREE iPhone 15 — claim in 4 hours!",
            body=(
                "Congratulations! You've been randomly selected to receive a FREE iPhone 15 Pro. "
                "Click the link within 4 hours to claim your prize before someone else does. "
                "No purchase necessary — just fill in your shipping details at our secure site. "
                "This offer expires at midnight!"
            ),
            priority=ClassLabel.SPAM,
            timestamp="2024-03-01T09:45:00Z",
        ),
        _make_email(
            id="t2_e5",
            sender="hr@company.com",
            subject="Happy Hour this Thursday — join the team!",
            body=(
                "Hi team! We're hosting a company happy hour this Thursday at 5:30 PM "
                "at The Blue Sparrow bar (2 blocks from the office). "
                "Drinks are on the company. Please RSVP by Wednesday "
                "so we can reserve the right space. Hope to see everyone there!"
            ),
            priority=ClassLabel.SOCIAL,
            timestamp="2024-03-01T10:00:00Z",
        ),
        _make_email(
            id="t2_e6",
            sender="legal@company.com",
            subject="Action required: NDA review for new vendor partnership",
            body=(
                "Hi, I need you to review the attached NDA for our new vendor partnership "
                "with DataSync Inc. Legal has flagged Section 7 (data retention clauses) "
                "as potentially problematic. Please review and confirm your approval "
                "or send back your comments by end of week. Response required."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T10:30:00Z",
        ),
        _make_email(
            id="t2_e7",
            sender="internal@company.com",
            subject="Company newsletter — March 2024",
            body=(
                "March Newsletter: Welcome to our 3 new hires this month! "
                "Engineering shipped the v2.4 product update on schedule. "
                "Don't forget the all-hands meeting on March 15th at 10 AM. "
                "Office closure notice: March 29th for Good Friday. "
                "Read the full update on the intranet portal."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T11:00:00Z",
        ),
        _make_email(
            id="t2_e8",
            sender="compliance@company.com",
            subject="URGENT: GDPR data subject access request — deadline March 3rd",
            body=(
                "We have received a GDPR data subject access request from a customer "
                "(Ref: DSAR-2024-0392). Under GDPR Article 15, we must respond within "
                "30 days; the deadline is March 3rd. I need you to pull and verify the "
                "customer's data immediately. Please confirm receipt and expected completion "
                "time ASAP — this is a legal obligation."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T11:15:00Z",
        ),
        _make_email(
            id="t2_e9",
            sender="mike@company.com",
            subject="Sarah's birthday lunch Tuesday — you in?",
            body=(
                "Hey! We're organising a surprise birthday lunch for Sarah on Tuesday at 12:30 PM. "
                "We're going to Trattoria Romano — her favourite spot. "
                "Can you make it? Let me know by tomorrow so we can book the table. "
                "Keep it quiet so it stays a surprise!"
            ),
            priority=ClassLabel.SOCIAL,
            timestamp="2024-03-01T11:45:00Z",
        ),
        _make_email(
            id="t2_e10",
            sender="hr@company.com",
            subject="Annual performance review — self-assessment form due March 15th",
            body=(
                "Hi, as part of our annual review cycle, please complete your self-assessment form "
                "and submit it via the HR portal by March 15th. "
                "This includes ratings on all 6 competency areas plus a 200-word development "
                "goals section. Managers will begin reviews on March 18th — late submissions "
                "may affect your review timeline. Action required."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T13:00:00Z",
        ),
    ]

    reply_criteria = [
        ReplyCriteria(
            email_id="t2_e1",
            required_keywords=["deadline", "march", "confirm", "proposal", "phase"],
            required_phrases=["march 12", "phase 2"],
            min_words=30,
            max_words=200,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t2_e3",
            required_keywords=["approve", "approval", "budget", "confirm", "authorized"],
            required_phrases=["45,000", "q2"],
            min_words=15,
            max_words=150,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t2_e6",
            required_keywords=["nda", "review", "section", "approve", "confirm", "datasync"],
            required_phrases=["section 7", "datasync"],
            min_words=20,
            max_words=150,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t2_e8",
            required_keywords=["gdpr", "dsar", "data", "request", "deadline", "march"],
            required_phrases=["march 3", "dsar-2024"],
            min_words=20,
            max_words=150,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t2_e10",
            required_keywords=["performance", "review", "submit", "form", "march", "portal"],
            required_phrases=["march 15", "self-assessment"],
            min_words=15,
            max_words=100,
            formal_tone=True,
        ),
    ]

    return TaskConfig(
        task_id="task_2",
        description=(
            "Handle a 10-email inbox. Classify each email, write replies to those that need one "
            "(urgent, action-required), and archive the rest. Do not reply to spam or informational emails."
        ),
        max_steps=40,
        emails=emails,
        emails_requiring_reply={"t2_e1", "t2_e3", "t2_e6", "t2_e8", "t2_e10"},
        emails_requiring_escalation=set(),
        reply_criteria=reply_criteria,
        escalation_keywords={},
    )


# ─────────────────────────────────────────────
# TASK 3 — Advanced: Full Inbox (15 emails)
# ─────────────────────────────────────────────

def build_task3() -> TaskConfig:
    emails = [
        # ── Original 8 emails ──────────────────────────────────────────────────
        _make_email(
            id="t3_e1",
            sender="vendor@supplierco.com",
            subject="Contract renewal — terms update required",
            body=(
                "Dear team, our annual contract (CN-2024-089) is due for renewal on April 1st. "
                "We have updated the service terms in Section 4.2 (liability cap now $500K). "
                "Please review and confirm acceptance, or request a call to negotiate. "
                "We need a response by March 20th to process the renewal paperwork."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T07:00:00Z",
            thread_id="thread_vendor_contract",
        ),
        _make_email(
            id="t3_e2",
            sender="vendor@supplierco.com",
            subject="RE: Contract renewal — terms update required",
            body=(
                "Following up on my previous email regarding contract CN-2024-089. "
                "I noticed we haven't received a response yet. Our deadline is firm on March 20th. "
                "If we don't hear back, we'll assume the new terms are accepted by default. "
                "Please confirm either way. Section 4 changes are non-negotiable."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-03T09:00:00Z",
            thread_id="thread_vendor_contract",
        ),
        _make_email(
            id="t3_e3",
            sender="security@company.com",
            subject="CRITICAL: Unauthorized access attempt detected on admin account",
            body=(
                "Security alert: We detected 47 failed login attempts on admin@company.com "
                "from IP 185.220.101.47 (Tor exit node) between 02:00-03:15 UTC today. "
                "The account has been temporarily locked. Immediate action required: "
                "1) Verify no breach occurred, 2) Enable MFA if not done, 3) Review audit logs. "
                "Contact security@company.com or call ext. 911 immediately."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T04:00:00Z",
        ),
        _make_email(
            id="t3_e4",
            sender="external@lawfirm.com",
            subject="Legal Notice: Formal complaint — workplace harassment claim",
            body=(
                "Dear HR Department, this letter constitutes formal notice of a workplace "
                "harassment complaint filed by our client against a current employee of your "
                "organization. Under applicable employment law, you are required to initiate "
                "an internal investigation within 5 business days. Failure to do so may result "
                "in regulatory penalties. Please contact our office to discuss resolution. "
                "Reference case: WH-2024-0087."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T08:00:00Z",
        ),
        _make_email(
            id="t3_e5",
            sender="deals@crypto-moon.net",
            subject="100x your investment with CryptoMoon token — limited presale!",
            body=(
                "Don't miss the biggest crypto opportunity of 2024! CryptoMoon (CMT) is launching "
                "its presale in 48 hours. Early investors get 100x returns GUARANTEED. "
                "Send 0.5 ETH to our wallet and receive 50,000 CMT tokens. Act now — "
                "only 500 spots available! Visit crypto-moon.net/invest to claim yours."
            ),
            priority=ClassLabel.SPAM,
            timestamp="2024-03-01T08:15:00Z",
        ),
        _make_email(
            id="t3_e6",
            sender="events@company.com",
            subject="You're invited: Annual company picnic — Saturday April 6th",
            body=(
                "Hi everyone! Our annual company picnic is happening on April 6th at Riverside Park. "
                "Food, games, and family-friendly fun from 11 AM to 4 PM. "
                "RSVP by March 25th via the HR portal so we can plan catering. "
                "Looking forward to seeing you there!"
            ),
            priority=ClassLabel.SOCIAL,
            timestamp="2024-03-01T09:00:00Z",
        ),
        _make_email(
            id="t3_e7",
            sender="facilities@company.com",
            subject="Office closure notice: March 29th (Good Friday)",
            body=(
                "Please note that our offices will be closed on Friday, March 29th for Good Friday. "
                "Remote work is permitted but not required. The building will be unstaffed. "
                "If you have an urgent issue, contact the on-call manager via Slack. "
                "Normal operations resume Monday, April 1st."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T09:30:00Z",
        ),
        _make_email(
            id="t3_e8",
            sender="ops@company.com",
            subject="URGENT: Payment gateway down — customer transactions failing",
            body=(
                "Our payment gateway has been returning 502 errors for the past 20 minutes. "
                "Approximately 340 customer transactions have failed since 10:40 AM. "
                "Revenue impact: ~$18,000/hour. The on-call engineer is investigating. "
                "We need sign-off to activate the backup payment processor (Stripe failover). "
                "Please confirm authorization immediately."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T11:00:00Z",
        ),
        # ── 7 new emails to reach 15 total ─────────────────────────────────────
        _make_email(
            id="t3_e9",
            sender="invest@crypto-gains-fast.io",
            subject="500% ROI guaranteed — exclusive investment club invite",
            body=(
                "This exclusive investment opportunity is available for 48 hours only. "
                "Our proprietary trading algorithm guarantees 500% ROI in 30 days. "
                "Minimum investment: $500. Send funds to our secure wallet today. "
                "Join 10,000 successful investors who already claimed their returns! "
                "Act now — claim your spot before it's gone."
            ),
            priority=ClassLabel.SPAM,
            timestamp="2024-03-01T11:30:00Z",
        ),
        _make_email(
            id="t3_e10",
            sender="communications@company.com",
            subject="Company-wide policy update: Remote work guidelines effective April 1",
            body=(
                "Effective April 1st, the remote work policy is updated as follows: "
                "1) Core hours are 10 AM - 3 PM local time Monday-Thursday. "
                "2) Maximum 3 remote days per week for individual contributors. "
                "3) All-hands and team meetings require in-person attendance. "
                "Full policy document available on the intranet. Questions? Contact HR."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T11:45:00Z",
        ),
        _make_email(
            id="t3_e11",
            sender="finance@supplier-corp.com",
            subject="Invoice #INV-2024-0847 due — please approve payment by March 20th",
            body=(
                "Dear team, please find attached Invoice #INV-2024-0847 for $12,450 "
                "for software licenses delivered in February (PO ref: PO-2024-0231). "
                "Payment terms are Net 30 and the due date is March 20th. "
                "Please approve the payment in the finance portal and send us confirmation. "
                "Late payment will incur a 2% monthly penalty."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T12:00:00Z",
        ),
        _make_email(
            id="t3_e12",
            sender="csr@company.com",
            subject="Volunteer day — City Food Bank charity event this Saturday",
            body=(
                "Hi everyone! We're partnering with the City Food Bank for a volunteer day "
                "this Saturday. Meet at the office at 9 AM and we'll carpool together. "
                "It's a great team-building opportunity and the company will match your "
                "donated hours with a cash donation. Sign up on the intranet by Thursday."
            ),
            priority=ClassLabel.SOCIAL,
            timestamp="2024-03-01T12:30:00Z",
        ),
        _make_email(
            id="t3_e13",
            sender="ciso@company.com",
            subject="CRITICAL SECURITY ALERT: Potential data breach — immediate response required",
            body=(
                "Security incident detected: Our intrusion detection system flagged unusual "
                "data exfiltration from the customer database at 02:47 UTC. Approximately "
                "15,000 customer records may be affected. Under GDPR/CCPA we have 72 hours "
                "to notify authorities. I need the incident response team assembled immediately. "
                "Call emergency line ext. 911. This is Priority 0."
            ),
            priority=ClassLabel.URGENT,
            timestamp="2024-03-01T13:00:00Z",
        ),
        _make_email(
            id="t3_e14",
            sender="finance@company.com",
            subject="Q4 2023 financial results summary — board presentation March 10th",
            body=(
                "Q4 2023 Financial Summary: Revenue $24.7M (up 18% YoY). "
                "Gross margin: 72%. Operating expenses: $18.2M. "
                "Net income: $2.1M. Cash position: $44M. "
                "Full report available on the investor relations portal. "
                "Board presentation is scheduled for March 10th at 9 AM."
            ),
            priority=ClassLabel.INFORMATIONAL,
            timestamp="2024-03-01T13:30:00Z",
        ),
        _make_email(
            id="t3_e15",
            sender="hr@company.com",
            subject="New hire onboarding checklist — action required before Friday",
            body=(
                "Hi, as the hiring manager for the new Software Engineer starting Monday, "
                "please complete the onboarding checklist before Friday: "
                "1) Submit system access requests in the IT portal. "
                "2) Schedule a 1:1 for their first day. "
                "3) Assign a buddy from your team. "
                "4) Confirm their equipment has been ordered. "
                "Response required — please confirm completion."
            ),
            priority=ClassLabel.ACTION_REQUIRED,
            timestamp="2024-03-01T14:00:00Z",
        ),
    ]

    reply_criteria = [
        ReplyCriteria(
            email_id="t3_e2",
            required_keywords=["contract", "renewal", "cn-2024-089", "terms", "review"],
            required_phrases=["march 20", "section 4"],
            min_words=30,
            max_words=250,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t3_e3",
            required_keywords=["security", "mfa", "audit", "breach", "locked"],
            required_phrases=["admin account", "audit log"],
            min_words=25,
            max_words=200,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t3_e8",
            required_keywords=["authorize", "approve", "stripe", "backup", "payment"],
            required_phrases=["stripe failover", "authorize"],
            min_words=15,
            max_words=150,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t3_e11",
            required_keywords=["invoice", "payment", "approve", "confirm", "march"],
            required_phrases=["inv-2024-0847", "march 20"],
            min_words=20,
            max_words=150,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t3_e13",
            required_keywords=["security", "breach", "incident", "data", "response"],
            required_phrases=["incident response", "priority 0"],
            min_words=20,
            max_words=200,
            formal_tone=True,
        ),
        ReplyCriteria(
            email_id="t3_e15",
            required_keywords=["onboarding", "checklist", "it", "access", "equipment", "friday"],
            required_phrases=["system access", "onboarding checklist"],
            min_words=20,
            max_words=150,
            formal_tone=True,
        ),
    ]

    return TaskConfig(
        task_id="task_3",
        description=(
            "Handle a full inbox of 15 emails. Triage, classify, reply to urgent/action emails, "
            "escalate the legal notice, archive spam and informational emails, handle vendor threads "
            "with context, and prioritize urgent emails first."
        ),
        max_steps=60,
        emails=emails,
        emails_requiring_reply={"t3_e2", "t3_e3", "t3_e8", "t3_e11", "t3_e13", "t3_e15"},
        emails_requiring_escalation={"t3_e4"},
        reply_criteria=reply_criteria,
        escalation_keywords={
            "t3_e4": ["legal", "harassment", "complaint", "investigation", "law"]
        },
        expected_order=[
            "t3_e3", "t3_e8", "t3_e13", "t3_e4",
            "t3_e1", "t3_e2", "t3_e11", "t3_e15",
            "t3_e6", "t3_e12", "t3_e7", "t3_e10", "t3_e14",
            "t3_e5", "t3_e9",
        ],
        thread_pairs=[("t3_e1", "t3_e2")],
    )


# ─────────────────────────────────────────────
# Public registry — LAZY factories
# ─────────────────────────────────────────────

_TASK_BUILDERS: Dict[str, Any] = {
    "task_1": build_task1,
    "task_2": build_task2,
    "task_3": build_task3,
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in _TASK_BUILDERS:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(_TASK_BUILDERS.keys())}"
        )
    return _TASK_BUILDERS[task_id]()


# ─────────────────────────────────────────────
# Exported TASKS dict — importable by app.py
# ─────────────────────────────────────────────

TASKS: Dict[str, Dict] = {
    "task_1": {"label": "Starter Inbox (5 emails)",   "email_count": 5,  "max_steps": 15},
    "task_2": {"label": "Medium Inbox (10 emails)",   "email_count": 10, "max_steps": 40},
    "task_3": {"label": "Advanced Inbox (15 emails)", "email_count": 15, "max_steps": 60},
}