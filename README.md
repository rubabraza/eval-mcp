# eval-mcp

A small eval harness any MCP client can drive. You keep a set of test cases, run
a prompt against them, grade the answers, and compare two prompt versions.

The example task is **support-ticket triage**: each ticket should be classified
into a `category` and a `priority`. The dataset is hand-written and synthetic —
no real or proprietary data.

```json
{"id": "t001", "input": "I was charged twice for my October invoice", "expected": {"category": "billing", "priority": "high"}}
```

## Why it's built this way

All real logic lives in `evalcore/` as plain, testable Python. `server.py` is a
**thin MCP layer** on top — each tool just validates inputs and delegates. This
is the "thin renderer, fat service" pattern: the harness is fully usable (and
unit-tested) without MCP, and the MCP server is trivial to reason about.

```
eval-mcp/
  server.py            # MCP tools: thin wrappers only
  evalcore/            # real logic, plain Python, testable without MCP
    models.py          # Pydantic models (cases, predictions, results, runs)
    datasets.py        # load/list datasets from datasets/
    graders.py         # deterministic grading: JSON-schema + exact match
    judge.py           # LLM judge + pluggable provider interface
    runs.py            # run_eval, persistence, compare_versions
  datasets/tickets.jsonl
  prompts/v1.txt  prompts/v2.txt
  runs/                # generated run_<id>.jsonl files (git-ignored)
  tests/
```

## The four tools

| Tool | What it does |
| --- | --- |
| `list_datasets()` | Lists datasets in `datasets/` with a case count for each. |
| `run_eval(dataset, prompt_version)` | Runs each case through a prompt file (e.g. `prompts/v1.txt`), grades it, and saves results to `runs/<run_id>.jsonl`. |
| `judge(case_id, output, rubric)` | Grades a single answer with the LLM judge and returns a Pydantic result: `score`, `pass`/`fail`, `reason`. |
| `compare_versions(run_a, run_b)` | Lists what changed: cases that newly pass, newly fail, and the score difference. |

## Grading, in two layers

1. **Deterministic (default, free, fast).** The output is parsed as JSON,
   validated against the `Prediction` JSON schema, then exact-matched on
   `category` and `priority`. This alone is enough to make the repo useful.
2. **LLM judge (optional).** Behind a small `Provider` interface so you can run
   it on whatever you have:
   - **mock** — offline, deterministic. The default, so the repo runs with zero
     setup and powers the test suite.
   - **ollama** — local and free ([ollama.com](https://ollama.com)).
   - **gemini** — Google's free API tier
     ([get a key](https://aistudio.google.com/app/apikey); check current limits).
   - **anthropic** — Claude models (SDK included as a dependency).

Select a provider with the `EVAL_PROVIDER` environment variable (or the
`provider` argument). **API keys are read from environment variables and are
never committed** — copy `.env.example` to `.env` and fill in what you use.

## Quick start

```bash
uv sync

# Run the test suite (uses the offline mock provider)
uv run pytest -q

# Inspect the MCP server interactively
uv run mcp dev server.py
```

### Try it from Python (offline, no keys needed)

```python
from evalcore.runs import run_eval, compare_versions

a = run_eval("tickets", "v1")   # uses the mock provider by default
b = run_eval("tickets", "v2")
print(compare_versions(a.run_id, b.run_id).model_dump())
```

### Use a real model

```bash
cp .env.example .env
# then either:
#   EVAL_PROVIDER=ollama     (after: ollama pull llama3.2)
#   EVAL_PROVIDER=gemini     (set GEMINI_API_KEY)
#   EVAL_PROVIDER=anthropic  (set ANTHROPIC_API_KEY)
```

## Run file format

Each `runs/<run_id>.jsonl` is JSON Lines: the **first line** is the run summary
(`RunMeta`), and **each following line** is a per-case result (`CaseResult`).

## Connect from an MCP client

Point any MCP client at the stdio server:

```json
{
  "mcpServers": {
    "eval-mcp": { "command": "uv", "args": ["run", "server.py"] }
  }
}
```

## License

MIT
