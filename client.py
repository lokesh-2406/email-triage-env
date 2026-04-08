"""
client.py — HTTP client for the Email Triage OpenEnv environment.

Usage:
    from client import EmailTriageClient

    client = EmailTriageClient("http://localhost:7860")
    obs = client.reset(task="classify_email")
    result = client.step({"category": "urgent"})
"""

import httpx
from typing import Any, Dict, Optional


class EmailTriageClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if task:
            payload["task"] = task
        if seed is not None:
            payload["seed"] = seed
        r = httpx.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = httpx.post(f"{self.base_url}/step", json={"action": action}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
