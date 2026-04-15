"""Single Run page — deep-dive into one experiment."""
from __future__ import annotations

import polars as pl
import streamlit as st

from mm_sim.dashboard import charts, loader

st.set_page_config(page_title="Single Run", layout="wide")
st.title("Single Run")

exp_dir_str = st.session_state.get("experiments_dir", "experiments")

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(f"No experiments found under `{exp_dir_str}`.")
    st.stop()

season = st.sidebar.selectbox("season", seasons, index=len(seasons) - 1)
scenarios = loader.cached_list_scenarios(exp_dir_str, season)
if not scenarios:
    st.warning(f"No scenarios in `{season}`.")
    st.stop()

scenario = st.sidebar.selectbox("scenario", scenarios)
versions = loader.cached_list_versions(exp_dir_str, season, scenario)
if not versions:
    st.warning(f"No versions under `{season}/{scenario}`.")
    st.stop()

version = st.sidebar.selectbox(
    "version", versions, index=len(versions) - 1
)

exp = loader.cached_load_run(exp_dir_str, season, scenario, version)
agg = exp.aggregate
label = f"{scenario}/{version}"

# KPI row
last_row = agg.sort("day").tail(1).row(0, named=True)
day0 = agg.row(0, named=True)["active_count"]
retention = last_row["active_count"] / day0 if day0 else 0.0
col1, col2, col3, col4 = st.columns(4)
col1.metric("final active", f"{last_row['active_count']:,}")
col2.metric("overall retention", f"{retention:.1%}")
col3.metric("mean match quality (\u2193better)", f"{last_row['win_prob_dev_mean']:.3f}")
col4.metric("mean rating error", f"{last_row['rating_error_mean']:.3f}")

# Charts
runs = [(label, agg)]
st.plotly_chart(charts.population_over_time(runs), use_container_width=True)
st.plotly_chart(charts.retention_over_time(runs), use_container_width=True)
st.plotly_chart(charts.match_quality_over_time(runs), use_container_width=True)
st.plotly_chart(charts.rating_error_over_time(runs), use_container_width=True)
st.plotly_chart(charts.blowout_share_over_time(runs), use_container_width=True)

if exp.population is not None:
    max_day = int(exp.population["day"].max())
    day = st.slider("skill-distribution day", 0, max_day, max_day)
    st.plotly_chart(
        charts.skill_distribution(exp.population, day=day),
        use_container_width=True,
    )

with st.expander("config.json", expanded=False):
    st.json(exp.config.model_dump(mode="json"))

if exp.population is not None:
    with st.expander("population preview (first 1000 rows of selected day)"):
        preview_day = st.number_input(
            "day", min_value=0, max_value=max_day, value=max_day, step=1
        )
        st.dataframe(
            exp.population.filter(pl.col("day") == preview_day)
            .head(1000)
            .to_pandas()
        )
