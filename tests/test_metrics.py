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


# --- pairwise_consistency: an INFORMATIONAL signal only (embed() is monkeypatched) ------
#
# Note the deliberate absence of any pass/fail assertions here: pairwise_consistency no
# longer decides consistency. Empirically, cosine cannot separate same-policy paraphrases
# from same-wording contradictions, so the verdict moved to judge.judge_consistency. These
# tests only pin the similarity math.

def _fake_embed(mapping):
    """Return an embed() replacement that maps known strings to fixed unit vectors."""
    def _embed(texts):
        return np.array([mapping[t] for t in texts], dtype=np.float64)
    return _embed


def test_pairwise_identical_answers_is_one(monkeypatch):
    monkeypatch.setattr(metrics, "embed", _fake_embed({"A": [1.0, 0.0], "A2": [1.0, 0.0]}))
    assert metrics.pairwise_consistency(["A", "A2"]) == pytest.approx(1.0)


def test_pairwise_orthogonal_answers_is_zero(monkeypatch):
    monkeypatch.setattr(metrics, "embed", _fake_embed({"A": [1.0, 0.0], "B": [0.0, 1.0]}))
    assert metrics.pairwise_consistency(["A", "B"]) == pytest.approx(0.0)


def test_pairwise_ignores_empty_answers(monkeypatch):
    # Blank answers are dropped before scoring; two identical non-empty ones -> 1.0.
    monkeypatch.setattr(metrics, "embed", _fake_embed({"A": [1.0, 0.0], "A2": [1.0, 0.0]}))
    assert metrics.pairwise_consistency(["A", "  ", "A2"]) == pytest.approx(1.0)


def test_pairwise_fewer_than_two_answers_is_trivially_one():
    # No embed() call needed; a single answer is trivially self-consistent.
    out = metrics.pairwise_consistency(["only one"])
    assert out == 1.0
    assert not math.isnan(out)
