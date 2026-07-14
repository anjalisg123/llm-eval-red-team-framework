"""Streamlit dashboard for an evaluation run.

Reads a results JSON (default: results/latest.json, falling back to the committed
illustrative sample) and renders: headline pass rate, per-category pass/fail, a drill-down
of failures with the judge's reasoning, and the judge-vs-human validation (Cohen's kappa).

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

# Make sibling packages importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.validate import compute as compute_validation  # noqa: E402

# Validated status colors from the dataviz reference palette (fixed, never themed).
PASS_COLOR = "#0ca30c"   # good
FAIL_COLOR = "#d03b3b"   # critical

LATEST = os.path.join("results", "latest.json")
SAMPLE = os.path.join("results", "sample_run", "latest.json")

st.set_page_config(page_title="LLM Eval & Red-Team", layout="wide")


def _find_run_files() -> list[str]:
    files = []
    if os.path.exists(LATEST):
        files.append(LATEST)
    rdir = "results"
    if os.path.isdir(rdir):
        files += sorted(
            (os.path.join(rdir, f) for f in os.listdir(rdir) if f.startswith("run_")),
            reverse=True,
        )
    if os.path.exists(SAMPLE):
        files.append(SAMPLE)
    # De-dup, preserve order.
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out or [SAMPLE]


@st.cache_data(show_spinner=False)
def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def results_df(run: dict) -> pd.DataFrame:
    rows = []
    for r in run["results"]:
        j = r.get("judgment") or {}
        rows.append({
            "prompt_id": r["prompt"]["id"],
            "category": r["prompt"]["category"],
            "passed": bool(j.get("passed")),
            "score": j.get("score"),
            "answer": r["response"].get("answer", ""),
            "reasoning": j.get("reasoning", ""),
            "evidence": j.get("evidence", ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Sidebar: choose a run
# --------------------------------------------------------------------------------------
st.sidebar.title("Evaluation run")
run_files = _find_run_files()
chosen = st.sidebar.selectbox("Results file", run_files, index=0)
run = load_run(chosen)
df = results_df(run)
summary = run["summary"]

st.title("LLM Evaluation & Red-Teaming — Results")

if run.get("illustrative"):
    st.warning(
        "**Illustrative sample data — not a real evaluation run.** "
        "Generate a real run with `make ingest && make serve` then `make eval`, "
        "which writes `results/latest.json`.",
        icon="⚠️",
    )
st.caption(f"Run: `{run.get('run_id', '?')}`  ·  created {run.get('created_at', '?')}  "
           f"·  source: `{chosen}`")

# --------------------------------------------------------------------------------------
# KPI row (stat tiles — the headline numbers, no chart needed)
# --------------------------------------------------------------------------------------
try:
    val = compute_validation(chosen)
except Exception:
    val = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall pass rate", f"{summary['overall_pass_rate']:.0%}")
c2.metric("Prompts evaluated", summary["total"])
c3.metric("Failures", summary["failed"])
c4.metric("Judge κ (vs human)", f"{val['cohen_kappa']:.2f}" if val and val["cohen_kappa"] is not None else "n/a")

st.divider()

# --------------------------------------------------------------------------------------
# Per-category pass/fail — stacked bar, status colors, always count-labeled
# --------------------------------------------------------------------------------------
st.subheader("Pass / fail by failure category")

cat_rows = []
for cat, s in summary["per_category"].items():
    cat_rows.append({"category": cat, "outcome": "passed", "count": s["passed"]})
    cat_rows.append({"category": cat, "outcome": "failed", "count": s["failed"]})
cat_df = pd.DataFrame(cat_rows)

# Order categories by pass rate so the weakest area is visually obvious.
order = sorted(summary["per_category"], key=lambda c: summary["per_category"][c]["pass_rate"])

bar = (
    alt.Chart(cat_df)
    .mark_bar(cornerRadius=4, stroke="white", strokeWidth=2)  # 2px surface gap between segments
    .encode(
        x=alt.X("count:Q", title="Prompts", stack="zero",
                axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("category:N", title=None, sort=order),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(domain=["passed", "failed"], range=[PASS_COLOR, FAIL_COLOR]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        order=alt.Order("outcome:N", sort="descending"),
        tooltip=["category", "outcome", "count"],
    )
    .properties(height=max(180, 42 * len(order)))
)
# Direct count labels so identity/magnitude never rely on color alone.
labels = (
    alt.Chart(cat_df[cat_df["count"] > 0])
    .mark_text(color="white", fontWeight="bold", dx=-10)
    .encode(
        x=alt.X("count:Q", stack="zero", bandPosition=0.5),
        y=alt.Y("category:N", sort=order),
        detail="outcome:N",
        text="count:Q",
    )
)
st.altair_chart(bar + labels, use_container_width=True)

# Category summary table (the required table view; identity never color-only).
tbl = pd.DataFrame(summary["per_category"]).T.reset_index(names="category")
tbl = tbl[["category", "passed", "failed", "total", "pass_rate", "mean_score"]]
st.dataframe(tbl.sort_values("pass_rate"), use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------------------
# Drill-down: failures with the judge's reasoning
# --------------------------------------------------------------------------------------
st.subheader("Failure drill-down")
cats = ["(all)"] + sorted(df["category"].unique())
pick = st.selectbox("Filter by category", cats)
show_only_fail = st.checkbox("Show failures only", value=True)

view = df.copy()
if pick != "(all)":
    view = view[view["category"] == pick]
if show_only_fail:
    view = view[~view["passed"]]

if view.empty:
    st.info("No rows match the current filter.")
else:
    for _, row in view.iterrows():
        icon = "✅" if row["passed"] else "❌"
        score = f"{row['score']:.2f}" if row["score"] is not None else "—"
        with st.expander(f"{icon} `{row['prompt_id']}`  ·  {row['category']}  ·  score {score}"):
            st.markdown(f"**Judge reasoning:** {row['reasoning']}")
            if row["evidence"] and row["evidence"].lower() != "none":
                st.markdown(f"**Evidence:** {row['evidence']}")
            st.markdown("**Target answer:**")
            st.code(row["answer"] or "(empty)", language="text")

st.divider()

# --------------------------------------------------------------------------------------
# Judge validation panel
# --------------------------------------------------------------------------------------
st.subheader("Judge validation (LLM-as-judge vs. human labels)")
if val is None or val.get("n", 0) == 0:
    st.info("No overlapping human labels found in validation/human_labels.csv for this run.")
else:
    v1, v2, v3 = st.columns(3)
    v1.metric("Labeled subset", val["n"])
    v2.metric("Raw agreement", f"{val['raw_agreement']:.0%}")
    v3.metric("Cohen's κ", f"{val['cohen_kappa']:.2f}")
    st.caption(
        "Raw agreement is inflated when most cases pass; Cohen's κ corrects for chance "
        "and is the number to trust. κ ≥ 0.6 is 'substantial', ≥ 0.8 'almost perfect'."
    )
    if val["disagreements"]:
        st.markdown("**Judge/human disagreements:** " +
                    ", ".join(f"`{d}`" for d in val["disagreements"]))
