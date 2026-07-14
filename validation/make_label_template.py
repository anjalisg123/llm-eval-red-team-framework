"""Generate a human-labeling template from a real evaluation run.

Writes one row per judged prompt with everything a human needs to label it — the question,
the expected behavior, and the target's actual answer — plus an `answer_sha` that binds the
label to that exact answer (the validator uses it to detect stale labels later).

BLIND BY DESIGN: the judge's own verdict is deliberately NOT included. If a human sees the
judge's label while labeling, agreement is inflated by anchoring bias and the resulting kappa
overstates how much the judge can be trusted. Label from the answer alone; then run
`python -m validation.validate` to compare.

Usage:
    python -m validation.make_label_template                    # from results/latest.json
    python -m validation.make_label_template path/to/run.json
    python -m validation.make_label_template --force            # overwrite existing labels
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from validation.validate import answer_sha, LABELS_PATH

FIELDS = ["prompt_id", "category", "run_id", "answer_sha",
          "question", "expected_behavior", "answer", "human_passed", "note"]


def build_rows(run_path: str) -> tuple[list[dict], str]:
    with open(run_path, encoding="utf-8") as f:
        data = json.load(f)
    run_id = data.get("run_id", "?")
    rows = []
    for r in data["results"]:
        if not r.get("judgment"):
            continue
        p, resp = r["prompt"], r["response"]
        ans = resp.get("answer", "")
        rows.append({
            "prompt_id": p["id"],
            "category": p["category"],
            "run_id": run_id,
            "answer_sha": answer_sha(ans),
            "question": p.get("query", ""),
            "expected_behavior": " ".join((p.get("expected_behavior") or "").split()),
            "answer": ans,
            "human_passed": "",   # <- YOU fill this in: 1 = behaved correctly, 0 = failed
            "note": "",
        })
    # Sort by category then id so related cases sit together while labeling.
    rows.sort(key=lambda x: (x["category"], x["prompt_id"]))
    return rows, run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_path", nargs="?", default=os.path.join("results", "latest.json"))
    ap.add_argument("--out", default=LABELS_PATH, help="output CSV (default: human_labels.csv)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing labels file")
    args = ap.parse_args()

    if not os.path.exists(args.run_path):
        raise SystemExit(f"Run file not found: {args.run_path}. Do a real `make eval` first.")

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(
            f"{args.out} already exists. Pass --force to overwrite (this will discard any "
            "labels you've already entered), or use --out to write elsewhere."
        )

    rows, run_id = build_rows(args.run_path)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} unlabeled rows from run '{run_id}' to {args.out}")
    print("Now open it and set human_passed to 1 (correct) or 0 (failed) for each row,")
    print("then run:  python -m validation.validate")


if __name__ == "__main__":
    main()
