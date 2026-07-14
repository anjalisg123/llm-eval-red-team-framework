"""Tests for judge output parsing. The actual LLM call is mocked; we test that whatever
the judge returns is coerced into a valid Judgment (clamped, typed, defaulted)."""

import pytest

from evals import judge as judge_mod
from evals.schemas import Category, Prompt, Response


def _prompt():
    return Prompt(
        id="t1",
        category=Category.HALLUCINATION,
        query="What is the Starter SLA?",
        expected_behavior="Should say Starter has no SLA.",
    )


def _response():
    return Response(prompt_id="t1", answer="Starter has a 99.9% SLA.",
                    contexts=[{"text": "Only Enterprise has an SLA.", "source": "x.md"}])


def test_judge_parses_wellformed_output(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge", lambda _p: {
        "passed": False, "score": 0.0,
        "reasoning": "Fabricated an SLA absent from context.",
        "evidence": "'99.9% SLA'",
    })
    j = judge_mod.judge(_prompt(), _response())
    assert j.passed is False
    assert j.score == 0.0
    assert j.category == Category.HALLUCINATION
    assert j.prompt_id == "t1"
    assert "Fabricated" in j.reasoning


def test_judge_clamps_out_of_range_score(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge", lambda _p: {
        "passed": True, "score": 1.7, "reasoning": "ok", "evidence": "none"})
    j = judge_mod.judge(_prompt(), _response())
    assert j.score == 1.0  # clamped into [0, 1]


def test_judge_handles_missing_fields(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge", lambda _p: {})  # empty judge output
    j = judge_mod.judge(_prompt(), _response())
    assert j.passed is False       # default
    assert j.score == 0.0          # default
    assert j.reasoning == ""       # default
    assert j.evidence == ""


def test_judge_handles_nonnumeric_score(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge", lambda _p: {
        "passed": True, "score": "high", "reasoning": "ok"})
    j = judge_mod.judge(_prompt(), _response())
    assert j.score == 0.0  # non-numeric -> safe default


def test_build_prompt_includes_context_and_rubric():
    text = judge_mod.build_prompt(_prompt(), _response())
    assert "Only Enterprise has an SLA." in text   # context is shown to the judge
    assert "Hallucination" in text or "hallucination" in text
    assert "Starter has no SLA" in text            # expected behavior is shown


def test_build_prompt_handles_no_context():
    r = Response(prompt_id="t1", answer="something", contexts=[])
    text = judge_mod.build_prompt(_prompt(), r)
    assert "no context" in text.lower()
