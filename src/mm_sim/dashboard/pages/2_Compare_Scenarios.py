"""Compare Scenarios page — overlay metrics across scenarios."""
from __future__ import annotations

import streamlit as st

from mm_sim.dashboard import charts, loader

st.set_page_config(page_title="Compare Scenarios", layout="wide")
st.title("Compare Scenarios")

exp_dir_str = st.session_state.get("experiments_dir", "experiments")

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(f"No experiments found under `{exp_dir_str}`.")
    st.stop()

season = st.sidebar.selectbox("season", seasons, index=len(seasons) - 1)
all_scenarios = loader.cached_list_scenarios(exp_dir_str, season)
if not all_scenarios:
    st.warning(f"No scenarios in `{season}`.")
    st.stop()

# Group sweep-point ids by sweep name so we can offer "select all in sweep"
sweeps: dict[str, list[str]] = {}
regular_scenarios: list[str] = []
for s in all_scenarios:
    if "/" in s:
        sweep_name = s.split("/", 1)[0]
        sweeps.setdefault(sweep_name, []).append(s)
    else:
        regular_scenarios.append(s)

if sweeps:
    st.sidebar.caption("Add all points from a sweep:")
    for sweep_name, pts in sweeps.items():
        if st.sidebar.button(
            f"+ {sweep_name} ({len(pts)})", key=f"add_{sweep_name}"
        ):
            current = st.session_state.get("compare_selected", list(regular_scenarios))
            st.session_state["compare_selected"] = list(dict.fromkeys(current + pts))

selected = st.sidebar.multiselect(
    "scenarios",
    all_scenarios,
    default=regular_scenarios,
    key="compare_selected",
)
if not selected:
    st.info("Pick at least one scenario.")
    st.stop()

metric_choice = st.sidebar.selectbox(
    "focus metric",
    [
        "active population",
        "retention",
        "match quality",
        "rating error",
        "blowout share",
    ],
)

st.caption("Version policy: the latest version of each scenario is used.")

runs = []
meta_rows = []
for scen in selected:
    versions = loader.cached_list_versions(exp_dir_str, season, scen)
    if not versions:
        continue
    ver = versions[-1]
    exp = loader.cached_load_run(exp_dir_str, season, scen, ver)
    runs.append((scen, exp.aggregate))
    m = exp.metadata
    meta_rows.append(
        {
            "scenario": scen,
            "version": ver,
            "seed": m.seed,
            "git_sha": (m.git_sha or "")[:8],
            "elapsed_s": m.elapsed_seconds,
        }
    )

if not runs:
    st.warning("Selected scenarios have no versions.")
    st.stop()

metric_fn = {
    "active population": charts.population_over_time,
    "retention": charts.retention_over_time,
    "match quality": charts.match_quality_over_time,
    "rating error": charts.rating_error_over_time,
    "blowout share": charts.blowout_share_over_time,
}[metric_choice]

st.plotly_chart(metric_fn(runs), use_container_width=True, key="focus")

st.subheader("Small multiples")
st.plotly_chart(charts.small_multiples(runs), use_container_width=True, key="sm_grid")

st.subheader("Run metadata")
st.dataframe(meta_rows)
