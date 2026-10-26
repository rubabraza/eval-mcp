from evalcore.graders import parse_output
from evalcore.judge import MockProvider, get_provider, judge_output
from evalcore.models import JudgeResult


def test_mock_classifies_billing():
    out = MockProvider().complete("instructions...\nTicket:\nI was charged twice for my invoice")
    parsed = parse_output(out)
    assert parsed["category"] == "billing"
    assert parsed["priority"] == "high"


def test_mock_judge_branch_returns_verdict():
    out = MockProvider().complete("Rubric:\nbe strict\nReply with score pass reason")
    parsed = parse_output(out)
    assert "score" in parsed and "pass" in parsed


def test_get_provider_defaults_to_mock():
    assert get_provider("mock").name == "mock"


def test_get_provider_unknown_raises():
    try:
        get_provider("not-a-provider")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_judge_output_with_fake_provider():
    class FakeJudge:
        name = "fake"

        def complete(self, prompt: str) -> str:
            return '{"score": 0.9, "pass": true, "reason": "looks right"}'

    result = judge_output(
        "some ticket",
        '{"category":"billing","priority":"high"}',
        "rubric",
        provider=FakeJudge(),
    )
    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert result.score == 0.9
