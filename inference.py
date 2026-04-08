"""
inference.py — Email Triage OpenEnv inference script
======================================================

Required environment variables:
  API_BASE_URL   The API endpoint for the LLM  (default: Groq)
  MODEL_NAME     The model identifier           (default: llama-3.1-8b-instant)
  HF_TOKEN       Your Hugging Face / API key   (can also be a Groq key)

Stdout format (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─── Configuration ───────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required.", file=sys.stderr)
    sys.exit(1)

ENV_BASE_URL = os.getenv("EMAIL_TRIAGE_URL", "http://localhost:7860")
MAX_STEPS    = 10
TEMPERATURE  = 0.2
MAX_TOKENS   = 400
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS = [
    "classify_email",
    "prioritize_inbox",
    "route_and_respond",
]

# ─── Logging helpers ─────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string to single line
    action_safe = action.replace("\n", " ").replace("\r", " ")[:200]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── HTTP helpers ────────────────────────────

def env_reset(task: str) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

# ─── Prompt builders ─────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant. You process emails and output structured JSON actions.

ALWAYS respond with valid JSON only. No markdown, no code blocks, no explanation.
Just a raw JSON object with the required fields.

Available action fields:
- "category": one of [spam, urgent, billing, support, general]
- "priority": ordered list of email IDs (highest to lowest urgency)
- "department": one of [sales, engineering, billing, hr, support]
- "reply_draft": a 1-3 sentence acknowledgment reply

Use only the fields relevant to the current task.
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    task = obs.get("task", "")
    instructions = obs.get("instructions", "")
    emails = obs.get("emails", [])
    feedback = obs.get("feedback", "")

    email_block = ""
    for e in emails:
        email_block += f"\n--- Email ID: {e['id']} ---\n"
        email_block += f"From: {e['sender']}\n"
        email_block += f"Subject: {e['subject']}\n"
        email_block += f"Body: {e['body']}\n"

    prompt = f"Task: {task}\nInstructions: {instructions}\n"
    if feedback and "Episode started" not in feedback:
        prompt += f"Previous feedback: {feedback}\n"
    prompt += f"\nEmail(s) to process:{email_block}\n"
    prompt += "\nRespond with a JSON action object only."
    return prompt


def get_action_from_model(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        # Strip markdown fences if any
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        return action
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw: {raw[:200]}", file=sys.stderr, flush=True)
        return _fallback_action(obs)
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", file=sys.stderr, flush=True)
        return _fallback_action(obs)


def _fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Safe fallback if model fails."""
    task = obs.get("task", "")
    if task == "classify_email":
        return {"category": "general"}
    elif task == "prioritize_inbox":
        ids = [e["id"] for e in obs.get("emails", [])]
        return {"priority": ids}
    else:
        return {"department": "support", "reply_draft": "Thank you for contacting us. We have received your email and will respond shortly."}


# ─── Single task run ─────────────────────────

def run_task(client: OpenAI, task: str) -> Dict[str, Any]:
    """Run one full episode for a task. Returns summary dict."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task, env="email_triage_env", model=MODEL_NAME)

    try:
        obs = env_reset(task)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action = get_action_from_model(client, obs)

            try:
                result = env_step(action)
            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action=str(action), reward=0.0, done=True, error=error_msg)
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", {})

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=done, error=None)

            if done:
                break

            # Small delay to avoid rate limits
            time.sleep(0.5)

        # Score = average reward across steps
        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Task error: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task, "score": score, "success": success, "steps": steps_taken, "rewards": rewards}


# ─── Main ────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"[INFO] Running Email Triage inference with model={MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] Environment URL: {ENV_BASE_URL}", file=sys.stderr, flush=True)

    all_results = []
    for task in TASKS:
        print(f"\n[INFO] ═══ Running task: {task} ═══", file=sys.stderr, flush=True)
        result = run_task(client, task)
        all_results.append(result)
        time.sleep(1)

    # Summary
    print("\n[INFO] ═══ Final Summary ═══", file=sys.stderr, flush=True)
    for r in all_results:
        print(f"[INFO] {r['task']}: score={r['score']:.3f} success={r['success']}", file=sys.stderr, flush=True)

    overall = sum(r["score"] for r in all_results) / len(all_results)
    print(f"[INFO] Overall average score: {overall:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
