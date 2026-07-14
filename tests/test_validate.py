"""Tests for the judge-validation staleness guard: a human label only counts toward kappa
when its answer_sha still matches the current run's answer."""

import csv
import json


from validation import validate as v


def _write_run(path, rows):
    """rows: list of (pid, passed, answer)."""
    results = [{"prompt": {"id": pid}, "response": {"answer": ans},
                "judgment": {"prompt_id": pid, "passed": passed}}
               for pid, passed, ans in rows]
    path.write_text(json.dumps({"run_id": "r", "results": results}))


def _write_labels(path, rows):
    """rows: list of dicts with prompt_id, answer_sha, human_passed."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "answer_sha", "human_passed", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({"note": "", **r})


def test_matching_sha_is_scored(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [("p1", 1, "an answer")])
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [{"prompt_id": "p1", "answer_sha": v.answer_sha("an answer"),
                            "human_passed": "1"}])
    s = v.compute(str(run), labels_path=str(labels))
    assert s["labeled_ids"] == ["p1"]
    assert s["stale"] == []
    assert s["n"] == 1


def test_mismatched_sha_is_excluded_as_stale(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [("p1", 1, "the NEW answer")])
    labels = tmp_path / "labels.csv"
    # Label was made against a different (old) answer.
    _write_labels(labels, [{"prompt_id": "p1", "answer_sha": v.answer_sha("the OLD answer"),
                            "human_passed": "0"}])
    s = v.compute(str(run), labels_path=str(labels))
    assert s["stale"] == ["p1"]
    assert s["labeled_ids"] == []
    assert s["n"] == 0                 # nothing valid to score
    assert s["cohen_kappa"] is None


def test_blank_label_is_unlabeled_not_scored(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [("p1", 1, "a")])
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [{"prompt_id": "p1", "answer_sha": v.answer_sha("a"),
                            "human_passed": ""}])
    s = v.compute(str(run), labels_path=str(labels))
    assert s["unlabeled"] == ["p1"]
    assert s["n"] == 0


def test_label_for_absent_prompt_is_missing(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [("p1", 1, "a")])
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [{"prompt_id": "ghost", "answer_sha": "x", "human_passed": "1"}])
    s = v.compute(str(run), labels_path=str(labels))
    assert s["missing"] == ["ghost"]
    assert s["n"] == 0


def test_disagreement_detected_among_valid_labels(tmp_path):
    run = tmp_path / "run.json"
    _write_run(run, [("p1", 1, "a"), ("p2", 0, "b")])
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [
        {"prompt_id": "p1", "answer_sha": v.answer_sha("a"), "human_passed": "1"},  # agree
        {"prompt_id": "p2", "answer_sha": v.answer_sha("b"), "human_passed": "1"},  # disagree
    ])
    s = v.compute(str(run), labels_path=str(labels))
    assert set(s["labeled_ids"]) == {"p1", "p2"}
    assert s["disagreements"] == ["p2"]
