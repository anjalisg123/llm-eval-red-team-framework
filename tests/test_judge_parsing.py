"""Tests for judge output parsing. The actual LLM call is mocked; we test that whatever
the judge returns is coerced into a valid Judgment (clamped, typed, defaulted)."""


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


# --- judge_consistency (the inconsistency-group verdict; LLM call mocked) ---------------

def test_judge_consistency_parses_verdict(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge_consistency", lambda _p: {
        "consistent": False,
        "conflicting_pair": "Answer 1 says 99.95% but Answer 2 says 99.0%",
        "reasoning": "Numeric contradiction.",
    })
    out = judge_mod.judge_consistency("What uptime?", ["99.95%", "99.0%"])
    assert out["passed"] is False
    assert "99.0%" in out["conflicting_pair"]


def test_judge_consistency_defaults_on_empty_output(monkeypatch):
    monkeypatch.setattr(judge_mod, "_call_judge_consistency", lambda _p: {})
    out = judge_mod.judge_consistency("q", ["a", "b"])
    assert out["passed"] is False          # missing "consistent" -> default False
    assert out["conflicting_pair"] == "none"


def test_judge_consistency_short_circuits_without_llm(monkeypatch):
    # Fewer than two non-empty answers: must NOT call the LLM.
    def _boom(_p):
        raise AssertionError("LLM should not be called for < 2 answers")
    monkeypatch.setattr(judge_mod, "_call_judge_consistency", _boom)
    out = judge_mod.judge_consistency("q", ["only one", "   "])
    assert out["passed"] is True
