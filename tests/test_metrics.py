"""Tests for the metrics we compute ourselves. No LLM calls here — pure math.

These are the tests that matter most in an interview: they prove the scoring logic is
correct independent of any model's behavior.
"""

import math

import numpy as np
import pytest

from evals import metrics


# --- Cohen's kappa ---------------------------------------------------------------------

def test_kappa_perfect_agreement():
    assert metrics.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_kappa_total_disagreement():
    # Judge and human never agree -> kappa should be strongly negative (worse than chance).
    assert metrics.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_kappa_chance_level_is_zero():
    # Known worked example: p_o = 0.5, p_e = 0.5 -> kappa = 0.
    judge = [1, 1, 0, 0]
    human = [1, 0, 1, 0]
    assert metrics.cohen_kappa(judge, human) == pytest.approx(0.0)


def test_kappa_constant_labels():
    # Both raters always say "pass" -> perfect but degenerate agreement.
    assert metrics.cohen_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.cohen_kappa([1, 0], [1])


def test_agreement_summary_shape():
    s = metrics.agreement_summary([1, 0, 1], [1, 0, 0])
    assert s["n"] == 3
    assert 0.0 <= s["raw_agreement"] <= 1.0
    assert s["judge_pass_rate"] == pytest.approx(2 / 3, abs=1e-3)  # summary rounds to 4 dp


# --- pairwise similarity ---------------------------------------------------------------

def test_pairwise_identical_vectors_is_one():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert metrics.mean_pairwise_similarity(v) == pytest.approx(1.0)


def test_pairwise_orthogonal_vectors_is_zero():
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.mean_pairwise_similarity(v) == pytest.approx(0.0)


def test_pairwise_single_vector_is_trivially_consistent():
    assert metrics.mean_pairwise_similarity(np.array([[1.0, 0.0]])) == 1.0


# --- consistency_score threshold logic (embed() is monkeypatched) ----------------------

def _fake_embed(mapping):
    """Return an embed() replacement that maps known strings to fixed unit vectors."""
    def _embed(texts):
        return np.array([mapping[t] for t in texts], dtype=np.float64)
    return _embed


def test_consistency_agreeing_and_correct_passes(monkeypatch):
    vecs = {
        "A": [1.0, 0.0], "A2": [1.0, 0.0],   # identical answers
        "exp": [1.0, 0.0],                    # expected == answers
    }
    monkeypatch.setattr(metrics, "embed", _fake_embed(vecs))
    out = metrics.consistency_score(["A", "A2"], expected="exp")
    assert out["passed"] is True
    assert out["pairwise"] == pytest.approx(1.0)


def test_consistency_agreeing_but_wrong_fails(monkeypatch):
    # Answers agree with each other but are far from the expected answer.
    vecs = {
        "W": [0.0, 1.0], "W2": [0.0, 1.0],   # identical to each other
        "exp": [1.0, 0.0],                    # orthogonal to the answers
    }
    monkeypatch.setattr(metrics, "embed", _fake_embed(vecs))
    out = metrics.consistency_score(["W", "W2"], expected="exp")
    assert out["pairwise"] == pytest.approx(1.0)   # mutually consistent
    assert out["vs_expected"] == pytest.approx(0.0)
    assert out["passed"] is False                  # ...but wrong -> fail


def test_consistency_contradictory_answers_fail(monkeypatch):
    vecs = {"A": [1.0, 0.0], "B": [0.0, 1.0], "exp": [1.0, 0.0]}
    monkeypatch.setattr(metrics, "embed", _fake_embed(vecs))
    out = metrics.consistency_score(["A", "B"], expected="exp")
    assert out["pairwise"] == pytest.approx(0.0)
    assert out["passed"] is False


def test_consistency_fewer_than_two_answers():
    out = metrics.consistency_score(["only one"], expected="x")
    assert out["passed"] is True
    assert not math.isnan(out["pairwise"])
