"""Pydantic models shared across the eval harness."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The support-ticket categories a prompt must choose from."""

    billing = "billing"
    technical = "technical"
    account = "account"
    shipping = "shipping"
    feedback = "feedback"


class Priority(str, Enum):
    """The priority levels a prompt must choose from."""

    low = "low"
    medium = "medium"
    high = "high"


class Expected(BaseModel):
    """The ground-truth label for a single eval case."""

    category: Category
    priority: Priority


class EvalCase(BaseModel):
    """One row of a dataset: an input and its expected label."""

    id: str
    input: str
    expected: Expected


class Prediction(BaseModel):
    """The structured answer a prompt is expected to produce for each ticket.

    Its JSON schema (``Prediction.model_json_schema()``) is what the
    deterministic grader validates model output against.
    """

    category: Category
    priority: Priority


class JudgeResult(BaseModel):
    """The structured verdict returned by the LLM judge layer."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str


class CaseResult(BaseModel):
    """The full grading record for a single case within a run."""

    case_id: str
    input: str
    expected: Expected
    output: str
    parsed: Optional[dict] = None
    schema_valid: bool = False
    category_match: bool = False
    priority_match: bool = False
    deterministic_score: float = 0.0
    judge: Optional[JudgeResult] = None
    score: float = 0.0
    passed: bool = False


class RunMeta(BaseModel):
    """Summary header written as the first line of a run file."""

    kind: str = "run"
    run_id: str
    dataset: str
    prompt_version: str
    provider: str
    total: int
    passed: int
    failed: int
    score: float
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Comparison(BaseModel):
    """The diff between two runs produced by ``compare_versions``."""

    run_a: str
    run_b: str
    score_a: float
    score_b: float
    score_delta: float
    newly_passing: list[str]
    newly_failing: list[str]
    still_passing: list[str]
    still_failing: list[str]
