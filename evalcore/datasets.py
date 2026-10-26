"""Loading and listing datasets stored as JSONL under ``datasets/``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import EvalCase

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _stem(name: str) -> str:
    return name[:-6] if name.endswith(".jsonl") else name


def list_datasets(datasets_dir: Optional[Path] = None) -> list[dict]:
    """Return every dataset in the datasets dir with its case count."""
    directory = datasets_dir or DATASETS_DIR
    out: list[dict] = []
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.jsonl")):
        count = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        out.append({"name": path.stem, "cases": count})
    return out


def load_dataset(name: str, datasets_dir: Optional[Path] = None) -> list[EvalCase]:
    """Load a dataset by name (with or without the .jsonl suffix)."""
    directory = datasets_dir or DATASETS_DIR
    path = directory / f"{_stem(name)}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[EvalCase] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise ValueError(f"{path.name} line {i}: invalid case: {exc}") from exc
    return cases
