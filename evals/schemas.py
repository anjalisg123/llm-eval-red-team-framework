"""Typed contracts that flow through the whole pipeline.

Everything is a Pydantic model so that (a) the dataset files are validated on load —
a malformed prompt fails loudly instead of silently skewing results — and (b) the
judge's output is parsed into a known shape instead of trusting free-form LLM text.
These schemas are the single source of truth for what a "prompt", a "response", and
a "judgment" are.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The six failure categories the framework probes for."""

    HALLUCINATION = "hallucination"
    REFUSAL = "refusal"                 # over-refusal AND under-refusal (see categories.py)
    INCONSISTENCY = "inconsistency"
    DRIFT = "drift"                     # off-topic drift across multi-turn context
    INJECTION = "injection"             # direct + indirect prompt injection
    LEAKAGE = "leakage"                 # system-prompt / raw-context extraction


class Turn(BaseModel):
    """One message in a multi-turn conversation sent to the target."""

    role: str  # "user" | "assistant"
    content: str


class Prompt(BaseModel):
    """A single adversarial test case.

    `expected_behavior` is a plain-language description of what a *correct* system
    should do. It is handed to the judge as the grading target — the judge scores the
    actual answer against this, not against a golden string, because for most of these
    categories there is no single right answer, only a right behavior.
    """

    id: str
    category: Category
    query: str
    # Optional multi-turn lead-up (used by the drift category). The final `query` is
    # sent as the last user turn.
    history: list[Turn] = Field(default_factory=list)
    expected_behavior: str
    # Free-form notes: which corpus fact this targets, why it's adversarial, etc.
    notes: str = ""
    # For inconsistency: prompts sharing a group_id are paraphrases of one question.
    group_id: str | None = None
    source: str = "seed"  # "seed" (hand-written) or "generated" (LLM-expanded)


class Response(BaseModel):
    """What the target returned for a prompt (the black-box view)."""

    prompt_id: str
    answer: str
    contexts: list[dict] = Field(default_factory=list)  # {text, source, distance}
    latency_ms: float = 0.0
    error: str | None = None


class Judgment(BaseModel):
    """The judge's structured verdict for one response.

    `passed` is the headline: did the system behave correctly for this category?
    `score` is a 0.0–1.0 severity-aware grade. `reasoning` is kept for auditing and is
    what the human-validation step compares against.
    """

    prompt_id: str
    category: Category
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # Category-specific evidence the judge is asked to extract, e.g. the unsupported
    # claim it found, or the injected instruction it detected being followed.
    evidence: str = ""


class Result(BaseModel):
    """A prompt + its response + its judgment, i.e. one fully-evaluated row."""

    prompt: Prompt
    response: Response
    judgment: Judgment | None = None
