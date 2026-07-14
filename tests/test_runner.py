"""Tests for the runner's orchestration logic: summary aggregation and inconsistency
grouping. LLM/HTTP boundaries are not exercised here — only the pure reduction logic."""

import pytest

from evals import judge as judge_mod
from evals import metrics, runner
from evals.schemas import Category, Judgment, Prompt, Response, Result


def _result(cat, passed, score):
    p = Prompt(id=f"{cat.value}-{passed}-{score}", category=cat, query="q",
               expected_behavior="e")
    r = Response(prompt_id=p.id, answer="a")
    j = Judgment(prompt_id=p.id, category=cat, passed=passed, score=score, reasoning="r")
    return Result(prompt=p, response=r, judgment=j)


def test_summarize_counts_and_rates():
    results = [
        _result(Category.HALLUCINATION, True, 1.0),
        _result(Category.HALLUCINATION, False, 0.0),
        _result(Category.INJECTION, True, 1.0),
    ]
    s = runner._summarize(results)
    assert s["total"] == 3
    assert s["passed"] == 2
    assert s["overall_pass_rate"] == pytest.approx(2 / 3, abs=1e-3)  # summary rounds to 4 dp
    assert s["per_category"]["hallucination"]["pass_rate"] == pytest.approx(0.5)
    assert s["per_category"]["injection"]["pass_rate"] == pytest.approx(1.0)


def test_summarize_empty():
    s = runner._summarize([])
    assert s["total"] == 0
    assert s["overall_pass_rate"] is None


def test_inconsistency_grouping_shares_group_verdict(monkeypatch):
    # The judge decides the verdict; cosine is only reported. Force both.
    monkeypatch.setattr(judge_mod, "judge_consistency",
                        lambda question, answers: {"passed": False,
                                                   "conflicting_pair": "Answer 1 vs Answer 2",
                                                   "reasoning": "They disagree."})
    monkeypatch.setattr(metrics, "pairwise_consistency", lambda answers: 0.42)
    prompts = [
        Prompt(id="c1", category=Category.INCONSISTENCY, query="q1",
               expected_behavior="e", group_id="g"),
        Prompt(id="c2", category=Category.INCONSISTENCY, query="q2",
               expected_behavior="e", group_id="g"),
    ]
    responses = {
        "c1": Response(prompt_id="c1", answer="ans one"),
        "c2": Response(prompt_id="c2", answer="ans two"),
    }
    out = runner._score_inconsistency(prompts, responses)
    assert len(out) == 2
    # Both members carry the same (group) verdict from the judge, not from cosine.
    assert all(r.judgment.passed is False for r in out)
    assert all(r.judgment.score == pytest.approx(0.0) for r in out)   # fail -> 0.0
    assert all("Answer 1 vs Answer 2" in r.judgment.evidence for r in out)
    assert all("0.42" in r.judgment.reasoning for r in out)           # cosine reported
    assert all(r.judgment.category == Category.INCONSISTENCY for r in out)


def test_inconsistency_pass_verdict_scores_one(monkeypatch):
    monkeypatch.setattr(judge_mod, "judge_consistency",
                        lambda question, answers: {"passed": True,
                                                   "conflicting_pair": "none",
                                                   "reasoning": "Consistent."})
    monkeypatch.setattr(metrics, "pairwise_consistency", lambda answers: 0.9)
    prompts = [
        Prompt(id="a1", category=Category.INCONSISTENCY, query="q", expected_behavior="e",
               group_id="ga"),
        Prompt(id="b1", category=Category.INCONSISTENCY, query="q", expected_behavior="e",
               group_id="gb"),
    ]
    responses = {"a1": Response(prompt_id="a1", answer="x"),
                 "b1": Response(prompt_id="b1", answer="y")}
    out = runner._score_inconsistency(prompts, responses)
    assert {r.prompt.id for r in out} == {"a1", "b1"}
    assert all(r.judgment.passed is True and r.judgment.score == pytest.approx(1.0)
               for r in out)
