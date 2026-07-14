"""Load and validate prompt datasets from YAML into typed Prompt objects.

Loading through Pydantic means a malformed or mis-categorized prompt raises on load
instead of silently corrupting a run. The runner loads the seed set plus any generated
set and concatenates them.
"""

from __future__ import annotations

import os

import yaml

from evals.schemas import Prompt

SEED_PATH = os.path.join(os.path.dirname(__file__), "datasets", "seed_prompts.yaml")
GENERATED_PATH = os.path.join(os.path.dirname(__file__), "datasets", "generated_prompts.yaml")


def load_prompts(path: str) -> list[Prompt]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    return [Prompt(**item) for item in raw]


def load_all(include_generated: bool = True) -> list[Prompt]:
    prompts = load_prompts(SEED_PATH)
    if include_generated:
        prompts += load_prompts(GENERATED_PATH)
    _check_unique_ids(prompts)
    return prompts


def _check_unique_ids(prompts: list[Prompt]) -> None:
    seen: set[str] = set()
    for p in prompts:
        if p.id in seen:
            raise ValueError(f"Duplicate prompt id: {p.id!r}")
        seen.add(p.id)
