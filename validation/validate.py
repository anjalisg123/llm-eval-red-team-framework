"""Compare the judge's pass/fail labels to human labels on a subset, and report agreement.

This is the step that keeps the framework honest: an LLM judge is only trustworthy insofar
as it agrees with a careful human on the cases a human actually checked. We report raw
agreement (intuitive but inflated when most cases pass) and Cohen's kappa (chance-corrected,
the number to actually trust).

STALENESS GUARD: each human label is bound to the exact answer it was made against, via a
short hash of that answer (`answer_sha`). Before a label counts toward kappa, we recompute
the hash from the *current* run and require it to match. If the run's answer for a prompt has
changed since it was labeled, the label is flagged STALE and excluded — because a label that
describes a different answer tells you nothing about the current judge. This is exactly the
failure that made a real run's kappa collapse against labels authored for the sample data.

Usage:
    python -m validation.validate                       # uses results/latest.json
    python -m validation.validate path/to/run.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

from evals.metrics import agreement_summary

LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels.csv")


def answer_sha(text: str) -> str:
    """Short, stable fingerprint of an answer. Shared by the label-template generator and
    the validator so a label can be bound to the answer it was made against."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def load_human_labels(path: str = LABELS_PATH) -> dict[str, dict]:
    """pid -> {passed: int|None, answer_sha: str|None}. Blank human_passed => passed=None
    (not yet labeled). Missing answer_sha column => None (unverifiable, old format)."""
    labels: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("human_passed") or "").strip()
            passed = int(raw) if raw in ("0", "1") else None
            sha = (row.get("answer_sha") or "").strip() or None
            labels[row["prompt_id"]] = {"passed": passed, "answer_sha": sha}
    return labels


def load_run(run_path: str) -> dict[str, dict]:
    """pid -> {passed: int, answer: str, sha: str} from a run's results."""
    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for r in data["results"]:
        j = r.get("judgment")
        if not j:
            continue
        ans = r["response"].get("answer", "")
        out[j["prompt_id"]] = {
            "passed": int(bool(j["passed"])),
            "answer": ans,
            "sha": answer_sha(ans),
        }
    return out


def compute(run_path: str, labels_path: str = LABELS_PATH) -> dict:
    human = load_human_labels(labels_path)
    run = load_run(run_path)

    valid_ids: list[str] = []      # labeled AND answer matches the current run
    stale: list[str] = []          # labeled but the run's answer changed since labeling
    unlabeled: list[str] = []      # present in the run but human_passed is blank
    missing: list[str] = []        # labeled but the prompt isn't in this run
    unverifiable: list[str] = []   # labeled but no answer_sha to check against (old format)

    for pid, lab in human.items():
        if pid not in run:
            missing.append(pid)
            continue
        if lab["passed"] is None:
            unlabeled.append(pid)
            continue
        if lab["answer_sha"] is None:
            unverifiable.append(pid)
            valid_ids.append(pid)          # can't prove stale; count it but warn
            continue
        if lab["answer_sha"] != run[pid]["sha"]:
            stale.append(pid)
            continue
        valid_ids.append(pid)

    judge_labels = [run[pid]["passed"] for pid in valid_ids]
    human_labels = [human[pid]["passed"] for pid in valid_ids]

    summary = agreement_summary(judge_labels, human_labels)
    summary["disagreements"] = [pid for pid in valid_ids
                                if run[pid]["passed"] != human[pid]["passed"]]
    summary["labeled_ids"] = valid_ids
    summary["stale"] = stale
    summary["unlabeled"] = unlabeled
    summary["missing"] = missing
    summary["unverifiable"] = unverifiable
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
    print(f"  valid labeled subset : {s['n']}")
    print(f"  raw agreement        : {s['raw_agreement']}")
    print(f"  Cohen's kappa        : {s['cohen_kappa']}  ({_interpret_kappa(s['cohen_kappa'])})")
    print(f"  judge pass rate      : {s['judge_pass_rate']}")
    print(f"  human pass rate      : {s['human_pass_rate']}")
    if s["disagreements"]:
        print(f"  disagreements        : {', '.join(s['disagreements'])}")

    # Loud, actionable warnings about anything excluded from the kappa.
    if s["stale"]:
        print(f"\n  ⚠️  STALE (excluded — the run's answer changed since labeling): "
              f"{', '.join(s['stale'])}")
        print("      Re-label these against the current answers "
              "(regenerate the template with make_label_template.py).")
    if s["unlabeled"]:
        print(f"\n  ▫️  UNLABELED (human_passed blank): {', '.join(s['unlabeled'])}")
    if s["missing"]:
        print(f"\n  ▫️  MISSING (labeled prompt not in this run): {', '.join(s['missing'])}")
    if s["unverifiable"]:
        print(f"\n  ⚠️  UNVERIFIABLE (no answer_sha; counted but unbound): "
              f"{', '.join(s['unverifiable'])}")

    if s["n"] == 0:
        print("\n  No valid labels to score. Fill in human_passed in validation/human_labels.csv "
              "(generate it with: python -m validation.make_label_template).")


if __name__ == "__main__":
    main()
