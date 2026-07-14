"""Orchestrate a full evaluation run.

Flow:
  load prompts  ->  query the target (HTTP)  ->  score  ->  write results JSON

Scoring is routed by category:
  - inconsistency  -> grouped; the judge decides if the group's answers agree, with
                      pairwise cosine reported alongside as an informational signal
  - everything else -> scored one-by-one by the LLM judge

The run is deterministic-by-construction where it can be (judge temperature 0, target
temperature 0.2, a frozen prompt set) so two runs against the same target are comparable.
Output is a single timestamped JSON file plus a summary, consumed by the dashboard and the
validation step.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from evals import judge as judge_mod
from evals import metrics
from evals.client import query_target
from evals.dataset import load_all
from evals.schemas import Category, Judgment, Prompt, Response, Result

RESULTS_DIR = "results"


def run(include_generated: bool = True, limit: int | None = None) -> dict:
    prompts = load_all(include_generated=include_generated)
    if limit:
        prompts = prompts[:limit]

    print(f"Loaded {len(prompts)} prompts. Querying target at the configured URL...")

    # 1. Query the target for every prompt (black box over HTTP).
    responses: dict[str, Response] = {}
    for i, p in enumerate(prompts, 1):
        responses[p.id] = query_target(p)
        status = "ERR" if responses[p.id].error else "ok"
        print(f"  [{i}/{len(prompts)}] {p.id:28s} {status}")

    # 2. Score. Inconsistency is grouped; all other categories go to the judge.
    results: list[Result] = []
    incon = [p for p in prompts if p.category == Category.INCONSISTENCY]
    judged = [p for p in prompts if p.category != Category.INCONSISTENCY]

    print(f"\nJudging {len(judged)} single-response prompts with the LLM judge...")
    for i, p in enumerate(judged, 1):
        resp = responses[p.id]
        if resp.error:
            j = Judgment(prompt_id=p.id, category=p.category, passed=False, score=0.0,
                         reasoning=f"Target error: {resp.error}", evidence="none")
        else:
            j = judge_mod.judge(p, resp)
        results.append(Result(prompt=p, response=resp, judgment=j))
        print(f"  [{i}/{len(judged)}] {p.id:28s} {'PASS' if j.passed else 'FAIL'} ({j.score:.2f})")

    if incon:
        print(f"\nScoring {len(incon)} inconsistency prompts by judge-based group agreement...")
        results.extend(_score_inconsistency(incon, responses))

    # 3. Persist.
    summary = _summarize(results)
    out = _write(results, summary, include_generated)
    print(f"\nWrote {len(results)} results to {out}")
    _print_summary(summary)
    return {"results": results, "summary": summary, "path": out}


def _score_inconsistency(prompts: list[Prompt], responses: dict[str, Response]) -> list[Result]:
    """Group paraphrases by group_id, score each group, attach the group verdict to every member."""
    groups: dict[str, list[Prompt]] = defaultdict(list)
    for p in prompts:
        groups[p.group_id or p.id].append(p)

    out: list[Result] = []
    for gid, members in groups.items():
        answers = [responses[m.id].answer for m in members]
        # Verdict comes from the judge (handles negation/numeric contradictions); the
        # cosine similarity is reported alongside as an informational sanity signal only.
        verdict = judge_mod.judge_consistency(members[0].query, answers)
        pairwise = metrics.pairwise_consistency(answers)
        for m in members:
            j = Judgment(
                prompt_id=m.id,
                category=Category.INCONSISTENCY,
                passed=verdict["passed"],
                score=1.0 if verdict["passed"] else 0.0,
                reasoning=(
                    f"Group '{gid}' ({len(members)} paraphrases): {verdict['reasoning']} "
                    f"(pairwise cosine={pairwise}, informational only)."
                ),
                evidence=verdict["conflicting_pair"],
            )
            out.append(Result(prompt=m, response=responses[m.id], judgment=j))
    return out


def _summarize(results: list[Result]) -> dict:
    by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0, "scores": []})
    for r in results:
        j = r.judgment
        c = by_cat[r.prompt.category.value]
        c["total"] += 1
        c["passed"] += int(bool(j and j.passed))
        if j:
            c["scores"].append(j.score)

    per_category = {}
    for cat, c in by_cat.items():
        per_category[cat] = {
            "total": c["total"],
            "passed": c["passed"],
            "failed": c["total"] - c["passed"],
            "pass_rate": round(c["passed"] / c["total"], 4) if c["total"] else None,
            "mean_score": round(sum(c["scores"]) / len(c["scores"]), 4) if c["scores"] else None,
        }
    total = len(results)
    passed = sum(1 for r in results if r.judgment and r.judgment.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "overall_pass_rate": round(passed / total, 4) if total else None,
        "per_category": per_category,
    }


def _write(results: list[Result], summary: dict, include_generated: bool) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"run_{ts}.json")
    payload = {
        "run_id": ts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "include_generated": include_generated,
        "summary": summary,
        "results": [json.loads(r.model_dump_json()) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Also update a stable "latest" pointer for the dashboard.
    with open(os.path.join(RESULTS_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _print_summary(summary: dict) -> None:
    print("\n=== SUMMARY ===")
    print(f"Overall: {summary['passed']}/{summary['total']} passed "
          f"({summary['overall_pass_rate']})")
    for cat, s in sorted(summary["per_category"].items()):
        print(f"  {cat:15s} {s['passed']}/{s['total']} pass_rate={s['pass_rate']} "
              f"mean_score={s['mean_score']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the LLM eval / red-team suite.")
    ap.add_argument("--no-generated", action="store_true",
                    help="Use only hand-written seed prompts.")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of prompts (debug).")
    args = ap.parse_args()
    run(include_generated=not args.no_generated, limit=args.limit)
