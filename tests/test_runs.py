import json

from evalcore.datasets import load_dataset
from evalcore.judge import MockProvider
from evalcore.runs import compare_versions, load_run, run_eval


class OracleProvider:
    """Returns the correct label for every case -> a perfect run."""

    name = "oracle"

    def __init__(self, cases):
        self.cases = cases

    def complete(self, prompt: str) -> str:
        # Look only at the actual injected ticket (after the final "Ticket:"),
        # so example tickets inside the prompt template don't cause false matches.
        marker = prompt.rfind("Ticket:")
        tail = prompt[marker:] if marker != -1 else prompt
        for c in self.cases:
            if c.input in tail:
                return json.dumps(
                    {
                        "category": c.expected.category.value,
                        "priority": c.expected.priority.value,
                    }
                )
        return "{}"


class WrongProvider:
    """Always answers feedback/low -> only genuinely feedback/low cases pass."""

    name = "wrong"

    def complete(self, prompt: str) -> str:
        return json.dumps({"category": "feedback", "priority": "low"})


def test_run_eval_writes_file_and_summary(tmp_path):
    meta = run_eval(
        dataset="tickets",
        prompt_version="v1",
        provider=MockProvider(),
        runs_dir=tmp_path,
    )
    assert meta.total >= 30
    assert meta.provider == "mock"
    # The run file exists and round-trips.
    reloaded_meta, cases = load_run(meta.run_id, runs_dir=tmp_path)
    assert reloaded_meta.run_id == meta.run_id
    assert len(cases) == meta.total


def test_run_eval_with_judge_blends_score(tmp_path):
    meta = run_eval(
        dataset="tickets",
        prompt_version="v1",
        provider=MockProvider(),
        use_judge=True,
        runs_dir=tmp_path,
    )
    _, cases = load_run(meta.run_id, runs_dir=tmp_path)
    assert all(c.judge is not None for c in cases)


def test_oracle_run_is_perfect(tmp_path):
    cases = load_dataset("tickets")
    meta = run_eval(
        dataset="tickets",
        prompt_version="v2",
        provider=OracleProvider(cases),
        runs_dir=tmp_path,
    )
    assert meta.passed == meta.total
    assert meta.score == 1.0


def test_compare_versions_detects_improvement(tmp_path):
    cases = load_dataset("tickets")
    run_a = run_eval(
        dataset="tickets",
        prompt_version="v1",
        provider=WrongProvider(),
        runs_dir=tmp_path,
    )
    run_b = run_eval(
        dataset="tickets",
        prompt_version="v2",
        provider=OracleProvider(cases),
        runs_dir=tmp_path,
    )

    cmp = compare_versions(run_a.run_id, run_b.run_id, runs_dir=tmp_path)
    assert cmp.score_delta > 0
    assert len(cmp.newly_passing) > 0
    assert cmp.newly_failing == []
