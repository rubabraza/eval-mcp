"""Running evals, persisting results, and comparing two runs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .datasets import load_dataset
from .graders import grade_case
from .judge import Provider, get_provider, judge_output
from .models import CaseResult, Comparison, RunMeta

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Placeholder in prompt files that gets replaced with each ticket's input.
INPUT_PLACEHOLDER = "{{INPUT}}"

DEFAULT_RUBRIC = (
    "The answer must pick the correct category and a reasonable priority, "
    "returned as a single JSON object."
)


def load_prompt(prompt_version: str, prompts_dir: Optional[Path] = None) -> str:
    """Load a prompt template by version name (e.g. 'v1' or 'v1.txt')."""
    directory = prompts_dir or PROMPTS_DIR
    stem = prompt_version[:-4] if prompt_version.endswith(".txt") else prompt_version
    path = directory / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _new_run_id(dataset: str, prompt_version: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{dataset}_{prompt_version}_{ts}_{uuid.uuid4().hex[:6]}"


def run_eval(
    dataset: str,
    prompt_version: str,
    provider: Optional[Provider] = None,
    use_judge: bool = False,
    rubric: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    prompts_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
) -> RunMeta:
    """Run every case through a prompt version, grade it, and save results.

    Results are written to ``runs/<run_id>.jsonl``: the first line is a
    ``RunMeta`` summary, each following line is a ``CaseResult``.
    """
    provider = provider or get_provider()
    cases = load_dataset(dataset, datasets_dir=datasets_dir)
    template = load_prompt(prompt_version, prompts_dir=prompts_dir)

    out_dir = runs_dir or RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem = dataset[:-6] if dataset.endswith(".jsonl") else dataset
    prompt_stem = prompt_version[:-4] if prompt_version.endswith(".txt") else prompt_version
    run_id = _new_run_id(dataset_stem, prompt_stem)

    results: list[CaseResult] = []
    for case in cases:
        prompt = template.replace(INPUT_PLACEHOLDER, case.input)
        output = provider.complete(prompt)
        result = grade_case(case, output)

        if use_judge:
            verdict = judge_output(
                input_text=case.input,
                output=output,
                rubric=rubric or DEFAULT_RUBRIC,
                expected=case.expected.model_dump_json(),
                provider=provider,
            )
            result.judge = verdict
            # Blend deterministic and judge scores equally.
            result.score = round((result.deterministic_score + verdict.score) / 2, 4)

        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.score for r in results) / total, 4) if total else 0.0

    meta = RunMeta(
        run_id=run_id,
        dataset=dataset_stem,
        prompt_version=prompt_stem,
        provider=getattr(provider, "name", "unknown"),
        total=total,
        passed=passed,
        failed=total - passed,
        score=avg_score,
    )

    path = out_dir / f"{run_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(meta.model_dump_json() + "\n")
        for r in results:
            f.write(r.model_dump_json() + "\n")

    return meta


def load_run(
    run_id: str, runs_dir: Optional[Path] = None
) -> tuple[RunMeta, list[CaseResult]]:
    """Load a run file into its summary and list of case results."""
    directory = runs_dir or RUNS_DIR
    stem = run_id[:-6] if run_id.endswith(".jsonl") else run_id
    path = directory / f"{stem}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {path}")

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"Run file is empty: {path}")
    meta = RunMeta.model_validate_json(lines[0])
    cases = [CaseResult.model_validate_json(l) for l in lines[1:]]
    return meta, cases


def list_runs(runs_dir: Optional[Path] = None) -> list[dict]:
    """List all saved runs (summary metadata only)."""
    directory = runs_dir or RUNS_DIR
    if not directory.exists():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            meta, _ = load_run(path.stem, runs_dir=directory)
            out.append(meta.model_dump())
        except Exception:  # noqa: BLE001 - skip malformed files
            continue
    return out


def compare_versions(
    run_a: str, run_b: str, runs_dir: Optional[Path] = None
) -> Comparison:
    """Compare two runs: newly passing/failing cases and the score delta."""
    meta_a, cases_a = load_run(run_a, runs_dir=runs_dir)
    meta_b, cases_b = load_run(run_b, runs_dir=runs_dir)

    pass_a = {c.case_id: c.passed for c in cases_a}
    pass_b = {c.case_id: c.passed for c in cases_b}
    shared = set(pass_a) & set(pass_b)

    newly_passing = sorted(c for c in shared if pass_b[c] and not pass_a[c])
    newly_failing = sorted(c for c in shared if pass_a[c] and not pass_b[c])
    still_passing = sorted(c for c in shared if pass_a[c] and pass_b[c])
    still_failing = sorted(c for c in shared if not pass_a[c] and not pass_b[c])

    return Comparison(
        run_a=meta_a.run_id,
        run_b=meta_b.run_id,
        score_a=meta_a.score,
        score_b=meta_b.score,
        score_delta=round(meta_b.score - meta_a.score, 4),
        newly_passing=newly_passing,
        newly_failing=newly_failing,
        still_passing=still_passing,
        still_failing=still_failing,
    )
