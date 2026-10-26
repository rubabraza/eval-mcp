from evalcore.graders import grade_case, parse_output, schema_errors
from evalcore.models import EvalCase

CASE = EvalCase.model_validate(
    {
        "id": "x",
        "input": "I was charged twice",
        "expected": {"category": "billing", "priority": "high"},
    }
)


def test_parse_plain_json():
    assert parse_output('{"category":"billing","priority":"high"}')["category"] == "billing"


def test_parse_with_prose_and_code_fence():
    out = 'Sure!\n```json\n{"category":"billing","priority":"high"}\n```'
    assert parse_output(out)["priority"] == "high"


def test_parse_invalid_returns_none():
    assert parse_output("no json here") is None
    assert parse_output("") is None
    assert parse_output(None) is None


def test_schema_errors_on_bad_enum():
    assert schema_errors({"category": "nonsense", "priority": "high"})
    assert not schema_errors({"category": "billing", "priority": "high"})


def test_grade_correct():
    r = grade_case(CASE, '{"category":"billing","priority":"high"}')
    assert r.passed
    assert r.schema_valid and r.category_match and r.priority_match
    assert r.score == 1.0


def test_grade_wrong_category_half_score():
    r = grade_case(CASE, '{"category":"technical","priority":"high"}')
    assert not r.passed
    assert r.category_match is False
    assert r.priority_match is True
    assert r.score == 0.5


def test_grade_bad_schema_zero_score():
    r = grade_case(CASE, '{"category":"nonsense","priority":"high"}')
    assert not r.schema_valid
    assert r.score == 0.0
