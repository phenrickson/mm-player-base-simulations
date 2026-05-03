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

st.caption("Version policy: the latest version of each scenario is used.")

runs = []
calib_runs = []  # (label, match_teams) pairs, only for scenarios with match_teams logged
cohort_runs = []  # (label, population) pairs, only for scenarios with population logged
inflow_per_run: dict[str, float] = {}
meta_rows = []
for scen in selected:
    versions = loader.cached_list_versions(exp_dir_str, season, scen)
    if not versions:
        continue
    ver = versions[-1]
    exp = loader.cached_load_run(exp_dir_str, season, scen, ver)
    runs.append((scen, exp.aggregate))
    inflow_per_run[scen] = (
        exp.config.population.daily_new_player_fraction
        * exp.config.population.initial_size
    )
    if exp.match_teams is not None and exp.match_teams.height > 0:
        calib_runs.append((scen, exp.match_teams))
    if exp.population is not None and exp.population.height > 0:
        cohort_runs.append((scen, exp.population))
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

tab_overview, tab_cohorts = st.tabs(["Overview", "Cohorts"])

with tab_overview:
    metric_choice = st.selectbox(
        "focus metric",
        [
            "active population",
            "retention",
            "daily quits",
            "daily quit rate",
            "match quality",
            "rating error",
            "blowout share",
            "MM calibration",
        ],
    )
    if metric_choice == "MM calibration":
        if calib_runs:
            st.plotly_chart(
                charts.mm_calibration_over_time(calib_runs),
                use_container_width=True,
                key="focus",
            )
        else:
            st.info(
                "No selected scenarios have match_teams.parquet. "
                "Re-run scenarios after the per-match-per-team logging commit."
            )
    elif metric_choice == "daily quits":
        st.plotly_chart(
            charts.quit_count_over_time(runs, inflow_per_run),
            use_container_width=True,
            key="focus",
        )
    elif metric_choice == "daily quit rate":
        st.plotly_chart(
            charts.quit_rate_over_time(runs, inflow_per_run),
            use_container_width=True,
            key="focus",
        )
    else:
        metric_fn = {
            "active population": charts.population_over_time,
            "retention": charts.retention_over_time,
            "match quality": charts.match_quality_over_time,
            "rating error": charts.rating_error_over_time,
            "blowout share": charts.blowout_share_over_time,
        }[metric_choice]
        st.plotly_chart(metric_fn(runs), use_container_width=True, key="focus")

    st.subheader("Metric grid")
    st.plotly_chart(
        charts.small_multiples(runs), use_container_width=True, key="sm_grid"
    )

    if calib_runs:
        st.subheader("MM rating calibration")
        st.plotly_chart(
            charts.mm_calibration_over_time(calib_runs),
            use_container_width=True,
            key="calib_bottom",
        )

        st.subheader("Teams extracting per match")
        st.plotly_chart(
            charts.extracts_per_match_over_time(calib_runs),
            use_container_width=True,
            key="extracts_per_match",
        )

    st.subheader("Run metadata")
    st.dataframe(meta_rows)

with tab_cohorts:
    if not cohort_runs:
        st.info(
            "No selected scenarios have population.parquet. "
            "Re-run scenarios with population snapshots to see cohorts."
        )
    else:
        st.subheader("Retention by day-0 skill decile")
        st.caption(
            "Deciles are assigned per scenario from that scenario's own "
            "day-0 true_skill distribution."
        )
        st.plotly_chart(
            charts.retention_by_decile_faceted(cohort_runs),
            use_container_width=True,
            key="cohort_facets",
        )
        st.subheader("Daily churn rate by experience cohort")
        st.plotly_chart(
            charts.churn_rate_by_experience_cohort(cohort_runs),
            use_container_width=True,
            key="cohort_churn",
        )

        st.subheader("Per-player trajectories across scenarios")
        st.caption(
            "Same random sample of players in every scenario. Each player "
            "starts from the same day-0 true_skill; one line per "
            "(player, scenario) shows how each matchmaking system shaped "
            "their season."
        )
        n_players = st.slider(
            "sample size",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
            key="cohort_traj_sample",
        )
        st.plotly_chart(
            charts.player_trajectory_by_scenario(
                cohort_runs, n_players=n_players
            ),
            use_container_width=True,
            key="cohort_traj",
        )
