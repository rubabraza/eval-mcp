"""The LLM judge layer and the pluggable provider interface.

Providers are swappable behind a tiny ``Provider`` protocol so the harness can
run against a local Ollama model, a free API tier like Gemini, Anthropic, or a
fully offline ``mock`` provider (the default) for tests and demos.

API keys are read from environment variables and must never be committed.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Protocol, runtime_checkable

import httpx

from .graders import parse_output
from .models import JudgeResult


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a prompt string into a completion string."""

    name: str

    def complete(self, prompt: str) -> str: ...


class OllamaProvider:
    """Local, free inference via an Ollama server (https://ollama.com)."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{self.model}"

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


class GeminiProvider:
    """Google Gemini via REST. Free tier: https://aistudio.google.com/app/apikey."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.name = f"gemini:{self.model}"
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    def complete(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        resp = httpx.post(
            url,
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


class AnthropicProvider:
    """Anthropic Claude. Install with ``uv add anthropic``."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.name = f"anthropic:{self.model}"
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def complete(self, prompt: str) -> str:
        from anthropic import Anthropic  # imported lazily; optional dependency

        client = Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class MockProvider:
    """Offline, deterministic provider so the harness runs with no LLM at all.

    For classification prompts it keyword-matches the ticket text (the part
    after the final ``Ticket:`` marker). For judge prompts it returns a
    well-formed verdict. It is intentionally simple: it keeps the repo
    clone-and-run, and powers the test suite without network access.
    """

    name = "mock"

    _CATEGORY_KEYWORDS = {
        "billing": [
            "charge", "charged", "invoice", "refund", "payment", "billed",
            "currency", "receipt", "subscription", "discount", "overcharged",
        ],
        "technical": [
            "crash", "error", "bug", "not working", "500", "down", "sync",
            "loading", "broken", "connection", "timezone",
        ],
        "account": [
            "login", "password", "access", "2fa", "admin", "email address",
            "ownership", "profile", "hacked", "unauthorized", "locked out",
        ],
        "shipping": [
            "package", "delivery", "deliver", "shipped", "ship", "tracking",
            "order", "address", "arrived", "damaged",
        ],
        "feedback": [
            "love", "please add", "nice to have", "feature", "confusing",
            "helpful", "thank", "great", "experience",
        ],
    }
    _HIGH = [
        "twice", "double", "down", "hacked", "unauthorized", "500", "lost",
        "never arrived", "overcharged", "locked out", "former employee",
    ]
    _LOW = [
        "how do i", "where can i", "when is", "when will", "can i", "please add",
        "nice to have", "love", "thank", "great", "discount", "helpful",
    ]

    def complete(self, prompt: str) -> str:
        low = prompt.lower()

        # Judge prompt branch: return a structured verdict.
        if "rubric" in low and "score" in low:
            return json.dumps(
                {"score": 1.0, "pass": True, "reason": "mock judge: output looks correct"}
            )

        # Classification branch: isolate the ticket text if a marker exists.
        marker = low.rfind("ticket:")
        text = (prompt[marker + len("ticket:") :] if marker != -1 else prompt).lower()

        category, best = "feedback", 0
        for cat, kws in self._CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in text)
            if hits > best:
                best, category = hits, cat

        if any(kw in text for kw in self._HIGH):
            priority = "high"
        elif any(kw in text for kw in self._LOW):
            priority = "low"
        else:
            priority = "medium"

        return json.dumps({"category": category, "priority": priority})


_PROVIDERS = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def get_provider(name: Optional[str] = None) -> Provider:
    """Resolve a provider by name, falling back to ``EVAL_PROVIDER`` then mock."""
    name = (name or os.environ.get("EVAL_PROVIDER", "mock")).lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Choose from {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name]()


_JUDGE_TEMPLATE = """You are grading a support-ticket triage answer.

Ticket: {input}
Expected label (JSON): {expected}
Model answer: {output}

Rubric:
{rubric}

Reply with ONLY a JSON object and nothing else:
{{"score": <float 0..1>, "pass": <true|false>, "reason": "<one short sentence>"}}
"""


def judge_output(
    input_text: str,
    output: str,
    rubric: str,
    expected: str = "",
    provider: Optional[Provider] = None,
) -> JudgeResult:
    """Grade a single answer with the LLM judge and return a structured result."""
    provider = provider or get_provider()
    prompt = _JUDGE_TEMPLATE.format(
        input=input_text, expected=expected, output=output, rubric=rubric
    )
    raw = provider.complete(prompt)
    parsed = parse_output(raw) or {}

    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    passed = bool(parsed.get("pass", score >= 0.7))
    reason = str(parsed.get("reason", "no reason provided"))

    return JudgeResult(score=score, passed=passed, reason=reason)
