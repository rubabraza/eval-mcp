"""MCP server for the eval harness.

This is a thin renderer over ``evalcore``: every tool here just validates
inputs and delegates to the plain-Python logic. All the real work lives in the
``evalcore/`` package so it can be tested without MCP.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from evalcore import datasets as datasets_mod
from evalcore import runs as runs_mod
from evalcore.datasets import load_dataset
from evalcore.judge import get_provider, judge_output

mcp = FastMCP("EvalMCP", log_level="ERROR")


@mcp.tool(
    name="list_datasets",
    description="List the datasets in datasets/ with a case count for each.",
)
def list_datasets() -> list[dict]:
    return datasets_mod.list_datasets()


@mcp.tool(
    name="run_eval",
    description="Run each case in a dataset through a prompt version, grade it, "
    "and save results to runs/<run_id>.jsonl. Returns the run summary.",
)
def run_eval(
    dataset: str = Field(description="Dataset name, e.g. 'tickets'"),
    prompt_version: str = Field(description="Prompt file name, e.g. 'v1' or 'v2'"),
    provider: Optional[str] = Field(
        default=None, description="Provider: mock, ollama, gemini, anthropic"
    ),
    use_judge: bool = Field(
        default=False, description="Also run the optional LLM judge layer"
    ),
) -> dict:
    meta = runs_mod.run_eval(
        dataset=dataset,
        prompt_version=prompt_version,
        provider=get_provider(provider),
        use_judge=use_judge,
    )
    return meta.model_dump()


@mcp.tool(
    name="judge",
    description="Grade a single answer with the LLM judge and return a result "
    "with score, pass/fail, and reason.",
)
def judge(
    case_id: str = Field(description="Case id from the dataset, e.g. 't001'"),
    output: str = Field(description="The model answer to grade"),
    rubric: str = Field(description="The grading rubric"),
    dataset: str = Field(default="tickets", description="Dataset containing the case"),
    provider: Optional[str] = Field(
        default=None, description="Provider for the judge: mock, ollama, gemini, anthropic"
    ),
) -> dict:
    cases = {c.id: c for c in load_dataset(dataset)}
    if case_id not in cases:
        raise ValueError(f"Unknown case_id {case_id!r} in dataset {dataset!r}")
    case = cases[case_id]
    result = judge_output(
        input_text=case.input,
        output=output,
        rubric=rubric,
        expected=case.expected.model_dump_json(),
        provider=get_provider(provider),
    )
    return result.model_dump()


@mcp.tool(
    name="compare_versions",
    description="Compare two runs: list cases that newly pass, newly fail, and "
    "the score difference.",
)
def compare_versions(
    run_a: str = Field(description="First run id"),
    run_b: str = Field(description="Second run id"),
) -> dict:
    return runs_mod.compare_versions(run_a, run_b).model_dump()


if __name__ == "__main__":
    mcp.run(transport="stdio")
