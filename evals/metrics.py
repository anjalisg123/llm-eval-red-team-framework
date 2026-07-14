"""Metrics computed by us, not by the judge.

Two things live here:

1. **Consistency scoring** for the inconsistency category. We embed the answers to a
   group of paraphrased questions and measure how tightly they cluster. The score is the
   mean pairwise cosine similarity across the group's answers. High = the system gives the
   same answer regardless of phrasing (good); low = it contradicts itself. This is a
   *measurement*, not a judgment, which is exactly why the LLM judge is not involved.

   We also embed a canonical statement of the expected answer and check every answer is
   close to it — this catches the case where the paraphrase answers agree with EACH OTHER
   but are consistently wrong.

2. **Judge validation** — Cohen's kappa between the judge's pass/fail labels and a human's
   labels on a subset. Raw agreement can look high just because most cases pass; kappa
   corrects for agreement expected by chance. This is how we earn the right to trust the
   judge instead of asserting it.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from target.config import settings

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# --------------------------------------------------------------------------------------
# Embeddings + similarity
# --------------------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, d) array of L2-normalized embeddings."""
    resp = _openai().embeddings.create(model=settings.embedding_model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def mean_pairwise_similarity(vectors: np.ndarray) -> float:
    """Mean cosine similarity over all unordered pairs. Vectors must be normalized."""
    n = len(vectors)
    if n < 2:
        return 1.0  # a single answer is trivially self-consistent
    sim = vectors @ vectors.T          # cosine sims (normalized inputs)
    iu = np.triu_indices(n, k=1)       # upper triangle, exclude the diagonal
    return float(sim[iu].mean())


def pairwise_consistency(answers: list[str]) -> float:
    """Mean pairwise cosine similarity across a group's answers — an INFORMATIONAL signal
    only, no pass/fail.

    We deliberately do NOT threshold this to decide consistency. Empirically, embedding
    cosine cannot separate "same policy, different words" from "opposite policy, same
    words": a one-token contradiction ("99.95%" vs "99.0%") stays near 1.0 while a genuine
    paraphrase in different vocabulary can fall below 0.72. Cosine measures lexical/topical
    overlap, not logical agreement. The pass/fail verdict is therefore made by the judge
    (see judge.judge_consistency); this number is reported alongside it as a cheap sanity
    signal. Returns 1.0 for fewer than two non-empty answers (trivially self-consistent).
    """
    non_empty = [a for a in answers if a and a.strip()]
    if len(non_empty) < 2:
        return 1.0
    return round(mean_pairwise_similarity(embed(non_empty)), 4)


# --------------------------------------------------------------------------------------
# Judge validation: Cohen's kappa vs. human labels
# --------------------------------------------------------------------------------------

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Cohen's kappa for two raters over binary pass(1)/fail(0) labels.

    Implemented directly (no sklearn dependency needed at runtime) so the stat is legible
    and testable. kappa = (p_o - p_e) / (1 - p_e).
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("label lists must be the same length")
    n = len(judge_labels)
    if n == 0:
        return float("nan")

    a = np.array(judge_labels)
    b = np.array(human_labels)
    p_o = float(np.mean(a == b))  # observed agreement

    # Expected agreement by chance, from each rater's marginal class frequencies.
    p_e = 0.0
    for cls in (0, 1):
        p_e += (np.mean(a == cls)) * (np.mean(b == cls))

    if p_e == 1.0:  # both raters gave a single constant label
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def agreement_summary(judge_labels: list[int], human_labels: list[int]) -> dict:
    a = np.array(judge_labels)
    b = np.array(human_labels)
    return {
        "n": len(judge_labels),
        "raw_agreement": round(float(np.mean(a == b)), 4) if len(a) else None,
        "cohen_kappa": round(cohen_kappa(judge_labels, human_labels), 4) if len(a) else None,
        "judge_pass_rate": round(float(np.mean(a)), 4) if len(a) else None,
        "human_pass_rate": round(float(np.mean(b)), 4) if len(b) else None,
    }
