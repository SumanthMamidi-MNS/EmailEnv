"""
tests/test_agents.py

Tests for BaselineAgent, LLMAgent, HybridAgent, and AgentRouter.
"""

import unittest
from envs.email_env.models import (
    ActionType,
    ClassLabel,
    EmailAction,
    EmailObservation,
    EmailStatus,
    EmailView,
    InboxItem,
)
from agents.baseline_agent import BaselineAgent
from agents.hybrid_agent import HybridAgent
from agents.llm import LLMClient, MockLLMProvider
from agents.llm_agent import LLMAgent
from agents.router import AgentRouter


class TestAgents(unittest.TestCase):

    def setUp(self):
        self.mock_provider = MockLLMProvider(mode="success")
        self.mock_llm = LLMClient(provider=self.mock_provider)

        self.sample_inbox = [
            InboxItem(id="e1", sender="hr@company.com", subject="Submit Timesheet", status=EmailStatus.UNREAD),
            InboxItem(id="e2", sender="ops@company.com", subject="Database down", status=EmailStatus.UNREAD),
        ]

    def test_baseline_agent_unread_read_action(self):
        agent = BaselineAgent()
        obs = EmailObservation(inbox_summary=self.sample_inbox)

        action = agent.act(obs)
        self.assertEqual(action.action_type, ActionType.READ_EMAIL)
        self.assertEqual(action.email_id, "e1")

    def test_baseline_agent_classify_and_reply_flow(self):
        agent = BaselineAgent()
        opened_view = EmailView(
            id="e1",
            sender="hr@company.com",
            subject="Please submit your Q1 timesheet by Friday",
            body="Submit your timesheet.",
            status=EmailStatus.READ,
            assigned_label=None,
        )
        obs = EmailObservation(
            current_email=opened_view,
            inbox_summary=self.sample_inbox,
        )

        # First action must classify
        act1 = agent.act(obs)
        self.assertEqual(act1.action_type, ActionType.CLASSIFY_EMAIL)
        self.assertEqual(act1.label, ClassLabel.ACTION_REQUIRED)

        # Update view to simulate classification done
        opened_view.assigned_label = ClassLabel.ACTION_REQUIRED
        act2 = agent.act(obs)
        self.assertEqual(act2.action_type, ActionType.REPLY_EMAIL)

    def test_llm_agent_flow(self):
        agent = LLMAgent(llm_client=self.mock_llm)
        obs = EmailObservation(inbox_summary=self.sample_inbox)

        action = agent.act(obs)
        self.assertEqual(action.action_type, ActionType.READ_EMAIL)

    def test_hybrid_agent_routing(self):
        agent = HybridAgent(llm_client=self.mock_llm)

        # Simple spam should be evaluated locally without invoking LLM
        dec_spam = agent.evaluate(
            email_id="s1",
            subject="You won $1,000,000 crypto lottery prize!",
            sender="lottery@scam.com",
            body="Claim your bitcoin reward immediately.",
        )
        self.assertEqual(dec_spam.classification, ClassLabel.SPAM)
        self.assertEqual(agent.local_call_count, 1)
        self.assertEqual(agent.llm_call_count, 0)

        # Complex / ambiguous request should call LLM
        dec_complex = agent.evaluate(
            email_id="c1",
            subject="Contract Renewal terms update needed",
            sender="partner@corp.com",
            body="Review agreement amendments.",
        )
        self.assertEqual(agent.llm_call_count, 1)

    def test_agent_router(self):
        router = AgentRouter(llm_client=self.mock_llm)
        b = router.get_agent("baseline")
        self.assertIsInstance(b, BaselineAgent)

        l = router.get_agent("llm")
        self.assertIsInstance(l, LLMAgent)

        h = router.get_agent("hybrid")
        self.assertIsInstance(h, HybridAgent)


if __name__ == "__main__":
    unittest.main()
