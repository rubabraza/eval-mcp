"""Deterministic grading: JSON parsing, schema validation, exact-match scoring.

This layer is free, fast, and requires no LLM. It is the first (and often
sufficient) grading layer; the LLM judge in ``judge.py`` is an optional second.
"""

from __future__ import annotations

import json
from typing import Optional

from jsonschema import Draft202012Validator

from .models import CaseResult, EvalCase, Prediction

# Build the JSON schema once from the Pydantic model. The deterministic
# "JSON-schema check" validates model output against this.
_PREDICTION_SCHEMA = Prediction.model_json_schema()
_VALIDATOR = Draft202012Validator(_PREDICTION_SCHEMA)


def parse_output(output: Optional[str]) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw model output.

    Handles prose, code fences, and trailing text by grabbing the substring
    between the first ``{`` and the last ``}``.
    """
    if not output:
        return None
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def schema_errors(parsed: Optional[dict]) -> list[str]:
    """Return a list of JSON-schema validation errors (empty means valid)."""
    if parsed is None:
        return ["output is not a valid JSON object"]
    return [e.message for e in _VALIDATOR.iter_errors(parsed)]


def grade_case(case: EvalCase, output: str) -> CaseResult:
    """Grade a single model output against a case, deterministically."""
    parsed = parse_output(output)
    schema_valid = not schema_errors(parsed)

    category_match = False
    priority_match = False
    if schema_valid and parsed is not None:
        category_match = str(parsed.get("category")) == case.expected.category.value
        priority_match = str(parsed.get("priority")) == case.expected.priority.value

    # Category is primary, priority secondary; both require valid schema.
    deterministic_score = 0.0
    if schema_valid:
        deterministic_score = 0.5 * category_match + 0.5 * priority_match

    passed = schema_valid and category_match and priority_match

    return CaseResult(
        case_id=case.id,
        input=case.input,
        expected=case.expected,
        output=output,
        parsed=parsed,
        schema_valid=schema_valid,
        category_match=category_match,
        priority_match=priority_match,
        deterministic_score=round(deterministic_score, 4),
        score=round(deterministic_score, 4),
        passed=passed,
    )
