"""
tests/test_environment.py — Local tests for the Email Triage environment.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import EmailTriageAction
from server.environment import (
    EmailTriageEnvironment,
    grade_classify,
    grade_prioritize,
    grade_route_and_respond,
)


# ─── Grader unit tests ───────────────────────

class TestClassifyGrader:
    def test_correct_spam(self):
        action = EmailTriageAction(category="spam")
        reward, _ = grade_classify(action, {"category": "spam"})
        assert reward == 1.0

    def test_wrong_category(self):
        action = EmailTriageAction(category="general")
        reward, _ = grade_classify(action, {"category": "urgent"})
        assert reward == 0.0

    def test_invalid_category(self):
        action = EmailTriageAction(category="random")
        reward, _ = grade_classify(action, {"category": "spam"})
        assert reward < 0.1

    def test_no_category(self):
        action = EmailTriageAction()
        reward, _ = grade_classify(action, {"category": "spam"})
        assert reward == 0.0

    def test_close_category_partial_credit(self):
        action = EmailTriageAction(category="support")
        reward, _ = grade_classify(action, {"category": "urgent"})
        assert 0.0 < reward < 1.0


class TestPrioritizeGrader:
    CORRECT = ["p003", "p001", "p005", "p004", "p002"]

    def test_perfect_order(self):
        action = EmailTriageAction(priority=self.CORRECT)
        reward, _ = grade_prioritize(action, {"priority": self.CORRECT})
        assert reward == 1.0

    def test_reversed_order(self):
        action = EmailTriageAction(priority=list(reversed(self.CORRECT)))
        reward, _ = grade_prioritize(action, {"priority": self.CORRECT})
        assert reward < 0.5

    def test_missing_ids(self):
        action = EmailTriageAction(priority=["p001", "p002"])
        reward, _ = grade_prioritize(action, {"priority": self.CORRECT})
        assert reward <= 0.1

    def test_no_priority(self):
        action = EmailTriageAction()
        reward, _ = grade_prioritize(action, {"priority": self.CORRECT})
        assert reward == 0.0


class TestRouteAndRespondGrader:
    EXPECTED = {
        "department": "billing",
        "reply_keywords": ["received", "billing", "review", "sorry", "apologize", "look into", "investigate"],
    }

    def test_correct_dept_and_good_reply(self):
        action = EmailTriageAction(
            department="billing",
            reply_draft="Thank you, we have received your email and our billing team will look into this.",
        )
        reward, _ = grade_route_and_respond(action, self.EXPECTED)
        assert reward >= 0.5

    def test_correct_dept_bad_reply(self):
        action = EmailTriageAction(department="billing", reply_draft="ok")
        reward, _ = grade_route_and_respond(action, self.EXPECTED)
        assert reward == 0.5  # Only dept score

    def test_wrong_dept_good_reply(self):
        action = EmailTriageAction(
            department="support",
            reply_draft="We received your billing concern and will investigate.",
        )
        reward, _ = grade_route_and_respond(action, self.EXPECTED)
        assert reward < 0.5

    def test_no_action(self):
        action = EmailTriageAction()
        reward, _ = grade_route_and_respond(action, self.EXPECTED)
        assert reward == 0.0


# ─── Environment lifecycle tests ─────────────

class TestEnvironmentLifecycle:
    def test_reset_returns_observation(self):
        env = EmailTriageEnvironment(task="classify_email")
        obs = env.reset()
        assert obs.task == "classify_email"
        assert len(obs.emails) > 0
        assert obs.done is False

    def test_step_classify(self):
        env = EmailTriageEnvironment(task="classify_email")
        env.reset()
        action = EmailTriageAction(category="spam")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert isinstance(done, bool)
        assert "step" in info

    def test_step_prioritize(self):
        env = EmailTriageEnvironment(task="prioritize_inbox")
        env.reset()
        action = EmailTriageAction(priority=["p003", "p001", "p005", "p004", "p002"])
        obs, reward, done, info = env.step(action)
        assert 0.0 <= reward <= 1.0

    def test_step_route(self):
        env = EmailTriageEnvironment(task="route_and_respond")
        env.reset()
        action = EmailTriageAction(
            department="billing",
            reply_draft="We have received your billing inquiry and will review it shortly."
        )
        obs, reward, done, info = env.step(action)
        assert 0.0 <= reward <= 1.0

    def test_full_classify_episode(self):
        env = EmailTriageEnvironment(task="classify_email")
        obs = env.reset()
        total_reward = 0.0
        for _ in range(10):
            if obs.done:
                break
            action = EmailTriageAction(category="general")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        assert total_reward >= 0.0

    def test_state_returns_state(self):
        env = EmailTriageEnvironment(task="classify_email")
        env.reset()
        state = env.state()
        assert state.task == "classify_email"
        assert state.step == 0

    def test_reward_in_valid_range(self):
        """All tasks must produce rewards in [0, 1]."""
        for task in ["classify_email", "prioritize_inbox", "route_and_respond"]:
            env = EmailTriageEnvironment(task=task)
            env.reset()
            if task == "classify_email":
                action = EmailTriageAction(category="urgent")
            elif task == "prioritize_inbox":
                action = EmailTriageAction(priority=["p003", "p001", "p005", "p004", "p002"])
            else:
                action = EmailTriageAction(
                    department="billing",
                    reply_draft="Thank you, our billing team will investigate this."
                )
            _, reward, _, _ = env.step(action)
            assert 0.0 <= reward <= 1.0, f"Reward out of range for task {task}: {reward}"
