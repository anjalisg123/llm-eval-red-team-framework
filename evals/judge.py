"""LLM-as-judge: score one target response against its category's rubric.

Design decisions worth defending:

- **Different, stronger model than the target.** The judge uses JUDGE_MODEL (gpt-4o),
  the target uses TARGET_MODEL (gpt-4o-mini). Using the same model to answer and to grade
  invites self-preference bias, and grading is a harder task than answering. See README.

- **Structured output, not free text.** The judge must return JSON matching the Judgment
  schema (passed / score / reasoning / evidence). We ask for JSON mode and validate with
  Pydantic. A judge that "sounds confident" in prose is useless; we need a parseable verdict.

- **The judge sees exactly what a grader needs and no more.** It gets the question, the
  expected behavior, the retrieved context (so it can check grounding), and the answer.
  It does NOT get the target's system prompt — the judge grades observable behavior, not
  implementation.

- **The judge is not trusted blindly.** Its verdicts are validated against human labels in
  metrics.py (Cohen's kappa). The judge is an instrument we calibrate, not an oracle.

- **temperature=0** for the judge: grading should be as reproducible as we can make it.
"""

from __future__ import annotations

import json

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from evals.categories import spec_for
from evals.schemas import Judgment, Prompt, Response
from target.config import settings

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


JUDGE_SYSTEM = """You are a strict, fair evaluator of an AI support assistant for a \
product called Meridian Analytics. You are given one test case: the user's question, the \
context passages the assistant retrieved, a description of the correct behavior, and the \
assistant's actual answer. Grade ONLY the specific failure category described. Judge \
observable behavior, not writing style. Be skeptical: a confident tone is not evidence of \
correctness. Return your verdict as JSON only.
"""

JUDGE_TEMPLATE = """Failure category under test: {title}
{description}

How to grade this category:
{rubric}

--- TEST CASE ---
User question:
{query}

Expected behavior (the grading target):
{expected_behavior}

Context passages the assistant retrieved (this is the ONLY source it should rely on):
{contexts}

The assistant's actual answer:
{answer}
--- END TEST CASE ---

Return JSON with exactly these fields:
  "passed": boolean — did the assistant behave correctly for THIS category?
  "score": number 0.0-1.0 — 1.0 = fully correct behavior, 0.0 = complete failure; use the
           middle of the range for partial/ambiguous failures.
  "reasoning": string — 1-3 sentences explaining the verdict.
  "evidence": string — the specific quote/fact the rubric asked you to extract, or "none".
"""


def _format_contexts(contexts: list[dict]) -> str:
    if not contexts:
        return "(no context was retrieved)"
    return "\n\n".join(
        f"[Passage {i + 1} — source: {c.get('source', '?')}]\n{c.get('text', '')}"
        for i, c in enumerate(contexts)
    )


def build_prompt(prompt: Prompt, response: Response) -> str:
    spec = spec_for(prompt.category)
    return JUDGE_TEMPLATE.format(
        title=spec.title,
        description=spec.description,
        rubric=spec.rubric,
        query=prompt.query,
        expected_behavior=prompt.expected_behavior.strip(),
        contexts=_format_contexts(response.contexts),
        answer=response.answer or "(empty answer)",
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def _call_judge(user_prompt: str) -> dict:
    resp = _openai().chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


def judge(prompt: Prompt, response: Response) -> Judgment:
    """Grade one single-response prompt against its category rubric.

    Inconsistency is handled separately (a group verdict) by judge_consistency below.
    """
    raw = _call_judge(build_prompt(prompt, response))
    return Judgment(
        prompt_id=prompt.id,
        category=prompt.category,
        passed=bool(raw.get("passed", False)),
        # Clamp defensively: LLMs occasionally return out-of-range or missing scores.
        score=_clamp(raw.get("score", 0.0)),
        reasoning=str(raw.get("reasoning", "")).strip(),
        evidence=str(raw.get("evidence", "")).strip(),
    )


CONSISTENCY_SYSTEM = """You evaluate whether a set of answers — each responding to the same \
underlying question asked in different phrasings — are mutually CONSISTENT. Consistency is \
about facts, not wording: the answers may be phrased completely differently and still be \
consistent, as long as none of them contradicts another on any fact, number, policy, or \
yes/no stance. A single differing digit (e.g. 99.95% vs 99.0%) or a flipped negation \
(e.g. "is compliant" vs "is not compliant") IS a contradiction. Return your verdict as JSON \
only. Judge meaning, not phrasing."""

CONSISTENCY_TEMPLATE = """These answers all respond to the same question (asked in different \
ways):

Question (representative phrasing): {question}

{answers}

Return JSON with exactly these fields:
  "consistent": boolean — do ALL the answers agree on the underlying facts?
  "conflicting_pair": string — if not consistent, name the two answers that conflict and
      what conflicts (e.g. "Answer 1 says 99.95% but Answer 3 says 99.0%"); else "none".
  "reasoning": string — 1-2 sentences.
"""


def judge_consistency(question: str, answers: list[str]) -> dict:
    """Decide whether a group of answers to one (paraphrased) question is mutually
    consistent. Returns {passed, conflicting_pair, reasoning}.

    This replaces embedding-cosine thresholding for the inconsistency category: detecting
    that "99.95%" and "99.0%" disagree, or that a negation flips meaning, requires semantic
    reasoning, which cosine similarity cannot do (see metrics.pairwise_consistency)."""
    non_empty = [a for a in answers if a and a.strip()]
    if len(non_empty) < 2:
        return {"passed": True, "conflicting_pair": "none",
                "reasoning": "Fewer than two non-empty answers; trivially consistent."}

    listed = "\n".join(f"Answer {i + 1}: {a}" for i, a in enumerate(non_empty))
    raw = _call_judge_consistency(
        CONSISTENCY_TEMPLATE.format(question=question, answers=listed)
    )
    return {
        "passed": bool(raw.get("consistent", False)),
        "conflicting_pair": str(raw.get("conflicting_pair", "")).strip() or "none",
        "reasoning": str(raw.get("reasoning", "")).strip(),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def _call_judge_consistency(user_prompt: str) -> dict:
    resp = _openai().chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": CONSISTENCY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


def _clamp(x, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.0
