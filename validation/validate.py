"""Compare the judge's pass/fail labels to human labels on a subset, and report agreement.

This is the step that keeps the framework honest: an LLM judge is only trustworthy insofar
as it agrees with a careful human on the cases a human actually checked. We report raw
agreement (intuitive but inflated when most cases pass) and Cohen's kappa (chance-corrected,
the number to actually trust).

Usage:
    python -m validation.validate                       # uses results/latest.json
    python -m validation.validate path/to/run.json
"""

from __future__ import annotations

import csv
import json
import os
import sys

from evals.metrics import agreement_summary

LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels.csv")


def load_human_labels(path: str = LABELS_PATH) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["prompt_id"]] = int(row["human_passed"])
    return labels


def load_judge_labels(run_path: str) -> dict[str, int]:
    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        r["judgment"]["prompt_id"]: int(bool(r["judgment"]["passed"]))
        for r in data["results"]
        if r.get("judgment")
    }


def compute(run_path: str) -> dict:
    human = load_human_labels()
    judge = load_judge_labels(run_path)

    # Align on the intersection: only prompts a human actually labeled.
    ids = [pid for pid in human if pid in judge]
    judge_labels = [judge[pid] for pid in ids]
    human_labels = [human[pid] for pid in ids]

    summary = agreement_summary(judge_labels, human_labels)
    disagreements = [pid for pid in ids if judge[pid] != human[pid]]
    summary["disagreements"] = disagreements
    summary["labeled_ids"] = ids
    return summary


def _interpret_kappa(k: float | None) -> str:
    if k is None:
        return "n/a"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> None:
    run_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "latest.json")
    if not os.path.exists(run_path):
        run_path = os.path.join("results", "sample_run", "latest.json")
    s = compute(run_path)
    print(f"Validation against human labels ({run_path}):")
    print(f"  labeled subset size : {s['n']}")
    print(f"  raw agreement       : {s['raw_agreement']}")
    print(f"  Cohen's kappa       : {s['cohen_kappa']}  ({_interpret_kappa(s['cohen_kappa'])})")
    print(f"  judge pass rate     : {s['judge_pass_rate']}")
    print(f"  human pass rate     : {s['human_pass_rate']}")
    if s["disagreements"]:
        print(f"  disagreements       : {', '.join(s['disagreements'])}")


if __name__ == "__main__":
    main()
