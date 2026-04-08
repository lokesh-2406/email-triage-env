"""
environment.py — Email Triage Environment

Three tasks:
  1. classify_email   (easy)   — classify a single email into one of 5 categories
  2. prioritize_inbox (medium) — rank 5 emails by urgency
  3. route_and_respond (hard)  — route to department AND draft a short reply

Each task runs for MAX_STEPS steps; the grader scores each step [0.0, 1.0].
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from models import Email, EmailTriageAction, EmailTriageObservation, EmailTriageState

# ─────────────────────────────────────────────
# Static email dataset
# ─────────────────────────────────────────────

CLASSIFY_EMAILS: List[Dict[str, Any]] = [
    {
        "email": Email(
            id="e001",
            sender="boss@company.com",
            subject="URGENT: Server down in production",
            body="Our main API server is returning 500 errors. All customers affected. Fix ASAP.",
            timestamp="2024-01-15T09:00:00Z",
        ),
        "answer": {"category": "urgent"},
    },
    {
        "email": Email(
            id="e002",
            sender="noreply@lottery.com",
            subject="You've won $1,000,000!",
            body="Congratulations! Click here to claim your prize. Limited time offer.",
            timestamp="2024-01-15T09:05:00Z",
        ),
        "answer": {"category": "spam"},
    },
    {
        "email": Email(
            id="e003",
            sender="billing@stripe.com",
            subject="Invoice #4521 - Payment due",
            body="Your monthly invoice of $299 is due on January 31. Please update your payment method.",
            timestamp="2024-01-15T09:10:00Z",
        ),
        "answer": {"category": "billing"},
    },
    {
        "email": Email(
            id="e004",
            sender="user@example.com",
            subject="Can't log in to my account",
            body="I've been trying to reset my password for 2 days but never receive the reset email.",
            timestamp="2024-01-15T09:15:00Z",
        ),
        "answer": {"category": "support"},
    },
    {
        "email": Email(
            id="e005",
            sender="newsletter@techblog.io",
            subject="Weekly tech digest - AI trends",
            body="This week in AI: GPT updates, open source releases, and more...",
            timestamp="2024-01-15T09:20:00Z",
        ),
        "answer": {"category": "general"},
    },
]

PRIORITIZE_INBOX_DATASET: List[Dict[str, Any]] = [
    {
        "emails": [
            Email(id="p001", sender="ceo@company.com", subject="Board meeting moved to TODAY 2PM",
                  body="Emergency board meeting at 2PM. Attendance mandatory.", timestamp="2024-01-15T08:00:00Z"),
            Email(id="p002", sender="no-reply@newsletter.com", subject="Top 10 productivity tips",
                  body="Read our latest newsletter on productivity.", timestamp="2024-01-15T07:30:00Z"),
            Email(id="p003", sender="security@company.com", subject="Suspicious login detected on your account",
                  body="A login from an unknown IP was detected. Please verify immediately.", timestamp="2024-01-15T08:30:00Z"),
            Email(id="p004", sender="hr@company.com", subject="Reminder: Submit timesheets by Friday",
                  body="Please remember to submit your timesheet by end of business Friday.", timestamp="2024-01-15T08:15:00Z"),
            Email(id="p005", sender="client@bigcorp.com", subject="Contract renewal - decision needed by EOD",
                  body="We need your signature on the renewal contract by 5PM today or the deal lapses.", timestamp="2024-01-15T08:45:00Z"),
        ],
        # Correct order: p003, p001, p005, p004, p002
        "answer": {"priority": ["p003", "p001", "p005", "p004", "p002"]},
    },
]

ROUTE_AND_RESPOND_DATASET: List[Dict[str, Any]] = [
    {
        "email": Email(
            id="r001",
            sender="john@client.com",
            subject="Overcharged on last invoice",
            body="Hi, I was charged $500 but my contract says $350/month. Please investigate and refund the difference.",
            timestamp="2024-01-15T10:00:00Z",
        ),
        "answer": {
            "department": "billing",
            "reply_keywords": ["received", "billing", "review", "sorry", "apologize", "look into", "investigate"],
        },
    },
    {
        "email": Email(
            id="r002",
            sender="startup@newco.com",
            subject="Interested in enterprise plan",
            body="We're a 200-person startup evaluating your enterprise plan. Can someone reach out to discuss pricing?",
            timestamp="2024-01-15T10:05:00Z",
        ),
        "answer": {
            "department": "sales",
            "reply_keywords": ["received", "sales", "team", "reach out", "contact", "touch", "discuss"],
        },
    },
    {
        "email": Email(
            id="r003",
            sender="developer@client.com",
            subject="API rate limit keeps resetting unexpectedly",
            body="Our integration keeps hitting rate limits even though we're under quota. Seems like a bug on your end.",
            timestamp="2024-01-15T10:10:00Z",
        ),
        "answer": {
            "department": "engineering",
            "reply_keywords": ["received", "engineer", "technical", "investigate", "look into", "bug", "team"],
        },
    },
]

VALID_CATEGORIES = {"spam", "urgent", "billing", "support", "general"}
VALID_DEPARTMENTS = {"sales", "engineering", "billing", "hr", "support"}
MAX_STEPS = 3  # Each episode processes up to 3 emails


# ─────────────────────────────────────────────
# Graders
# ─────────────────────────────────────────────

def grade_classify(action: EmailTriageAction, expected: Dict[str, Any]) -> Tuple[float, str]:
    """Grade the classify_email task. Returns (reward, feedback)."""
    if action.category is None:
        return 0.0, "No category provided. Please set the 'category' field."

    cat = action.category.lower().strip()

    if cat not in VALID_CATEGORIES:
        return 0.05, f"Invalid category '{cat}'. Must be one of: {sorted(VALID_CATEGORIES)}."

    if cat == expected["category"]:
        return 1.0, f"Correct! '{cat}' is the right category."
    else:
        # Partial credit: close categories
        close = {
            ("urgent", "support"): 0.3,
            ("support", "urgent"): 0.3,
            ("billing", "support"): 0.2,
            ("support", "billing"): 0.2,
        }
        partial = close.get((cat, expected["category"]), 0.0)
        feedback = f"Incorrect. Got '{cat}', expected '{expected['category']}'."
        if partial > 0:
            feedback += f" Partial credit awarded ({partial})."
        return partial, feedback


def grade_prioritize(action: EmailTriageAction, expected: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade the prioritize_inbox task using Kendall-tau style scoring.
    Full credit if order matches; partial for partially correct orderings.
    """
    if action.priority is None or len(action.priority) == 0:
        return 0.0, "No priority list provided. Please set the 'priority' field with email IDs in order."

    predicted = [p.strip() for p in action.priority]
    correct = expected["priority"]

    if len(predicted) != len(correct):
        return 0.1, f"Expected {len(correct)} emails in priority list, got {len(predicted)}."

    # Exact match
    if predicted == correct:
        return 1.0, "Perfect prioritization!"

    # Kendall-tau distance (normalized)
    n = len(correct)
    correct_rank = {e: i for i, e in enumerate(correct)}
    pred_rank = {e: i for i, e in enumerate(predicted) if e in correct_rank}

    if len(pred_rank) < n:
        missing = set(correct) - set(predicted)
        return 0.1, f"Some email IDs are missing or wrong: {missing}"

    concordant = 0
    total_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            e_i, e_j = correct[i], correct[j]
            if (pred_rank[e_i] < pred_rank[e_j]):
                concordant += 1

    tau = concordant / total_pairs
    feedback = f"Kendall-tau concordance: {tau:.2f}. "
    if tau >= 0.8:
        feedback += "Excellent prioritization!"
    elif tau >= 0.6:
        feedback += "Good, but a couple of emails are out of order."
    else:
        feedback += f"Needs improvement. Correct order: {correct}"

    return round(tau, 2), feedback


def grade_route_and_respond(action: EmailTriageAction, expected: Dict[str, Any]) -> Tuple[float, str]:
    """
    Grade route_and_respond. 50% for correct department, 50% for reply quality.
    """
    dept_score = 0.0
    reply_score = 0.0
    feedback_parts = []

    # Department routing (50%)
    if action.department is None:
        feedback_parts.append("No department provided.")
    else:
        dept = action.department.lower().strip()
        if dept not in VALID_DEPARTMENTS:
            feedback_parts.append(f"Invalid department '{dept}'. Must be one of: {sorted(VALID_DEPARTMENTS)}.")
        elif dept == expected["department"]:
            dept_score = 0.5
            feedback_parts.append(f"✓ Correct department: '{dept}'.")
        else:
            feedback_parts.append(f"✗ Wrong department. Got '{dept}', expected '{expected['department']}'.")

    # Reply draft quality (50%)
    if action.reply_draft is None or len(action.reply_draft.strip()) < 20:
        feedback_parts.append("Reply draft is missing or too short.")
    else:
        reply_lower = action.reply_draft.lower()
        keywords = expected["reply_keywords"]
        matched = [kw for kw in keywords if kw in reply_lower]
        kw_ratio = len(matched) / len(keywords)

        # Length check (should be 1-3 sentences, roughly 20-300 chars)
        length = len(action.reply_draft)
        if length > 600:
            length_penalty = 0.5
            feedback_parts.append("Reply is too long (keep to 1-3 sentences).")
        else:
            length_penalty = 1.0

        reply_score = round(min(kw_ratio, 1.0) * 0.5 * length_penalty, 2)

        if kw_ratio >= 0.4:
            feedback_parts.append(f"✓ Reply looks appropriate. Matched keywords: {matched}")
        else:
            feedback_parts.append(f"Reply is missing key elements. Expected to reference: {keywords[:3]}")

    total = round(dept_score + reply_score, 2)
    return total, " ".join(feedback_parts)


# ─────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    Email Triage environment implementing the OpenEnv interface.
    Supports three tasks with increasing difficulty.
    """

    TASK_NAMES = ["classify_email", "prioritize_inbox", "route_and_respond"]

    def __init__(self, task: Optional[str] = None, seed: Optional[int] = None):
        self._task = task or "classify_email"
        self._seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[EmailTriageState] = None
        self._dataset_index = 0

    # ─── OpenEnv interface ───────────────────

    def reset(self) -> EmailTriageObservation:
        self._dataset_index = 0
        emails, expected = self._sample_emails()
        self._state = EmailTriageState(
            task=self._task,
            step=0,
            emails=emails,
            expected=expected,
            cumulative_reward=0.0,
            done=False,
        )
        return self._make_observation(reward=0.0, feedback="Episode started. Process the email(s).")

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Grade the action
        reward, feedback = self._grade(action)
        self._state.cumulative_reward += reward
        self._state.step += 1

        # Advance dataset for multi-step tasks
        self._dataset_index += 1
        total_steps = self._total_steps()
        done = self._state.step >= total_steps

        self._state.done = done

        # Prepare next emails if not done
        if not done:
            next_emails, next_expected = self._sample_emails()
            self._state.emails = next_emails
            self._state.expected = next_expected

        obs = self._make_observation(reward=reward, feedback=feedback)
        info: Dict[str, Any] = {
            "step": self._state.step,
            "cumulative_reward": self._state.cumulative_reward,
        }
        return obs, reward, done, info

    def state(self) -> EmailTriageState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # ─── Internal helpers ────────────────────

    def _total_steps(self) -> int:
        if self._task == "classify_email":
            return len(CLASSIFY_EMAILS)
        elif self._task == "prioritize_inbox":
            return len(PRIORITIZE_INBOX_DATASET)
        else:  # route_and_respond
            return len(ROUTE_AND_RESPOND_DATASET)

    def _sample_emails(self) -> Tuple[List[Email], Dict[str, Any]]:
        if self._task == "classify_email":
            idx = self._dataset_index % len(CLASSIFY_EMAILS)
            item = CLASSIFY_EMAILS[idx]
            return [item["email"]], item["answer"]
        elif self._task == "prioritize_inbox":
            idx = self._dataset_index % len(PRIORITIZE_INBOX_DATASET)
            item = PRIORITIZE_INBOX_DATASET[idx]
            return item["emails"], item["answer"]
        else:  # route_and_respond
            idx = self._dataset_index % len(ROUTE_AND_RESPOND_DATASET)
            item = ROUTE_AND_RESPOND_DATASET[idx]
            return [item["email"]], item["answer"]

    def _grade(self, action: EmailTriageAction) -> Tuple[float, str]:
        if self._task == "classify_email":
            return grade_classify(action, self._state.expected)
        elif self._task == "prioritize_inbox":
            return grade_prioritize(action, self._state.expected)
        else:
            return grade_route_and_respond(action, self._state.expected)

    def _get_instructions(self) -> str:
        if self._task == "classify_email":
            return (
                "Classify the email into exactly one category: spam | urgent | billing | support | general. "
                "Set the 'category' field in your action."
            )
        elif self._task == "prioritize_inbox":
            ids = [e.id for e in self._state.emails] if self._state else []
            return (
                f"Rank the {len(ids)} emails from highest to lowest urgency. "
                f"Set 'priority' to a list of email IDs. Email IDs: {ids}"
            )
        else:
            return (
                "Route the email to the correct department (sales | engineering | billing | hr | support) "
                "AND write a brief 1-3 sentence acknowledgment reply. "
                "Set both 'department' and 'reply_draft' in your action."
            )

    def _make_observation(self, reward: float, feedback: str) -> EmailTriageObservation:
        return EmailTriageObservation(
            task=self._state.task,
            emails=self._state.emails,
            instructions=self._get_instructions(),
            reward=reward,
            feedback=feedback,
            done=self._state.done,
        )
