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

version = st.sidebar.selectbox("version", versions, index=len(versions) - 1)

exp = loader.cached_load_run(exp_dir_str, season, scenario, version)
agg = exp.aggregate
label = f"{scenario}/{version}"
runs = [(label, agg)]

# Header KPIs shared across all tabs
last_row = agg.sort("day").tail(1).row(0, named=True)
day0 = agg.row(0, named=True)["active_count"]
retention = last_row["active_count"] / day0 if day0 else 0.0
col1, col2, col3, col4 = st.columns(4)
col1.metric("final active", f"{last_row['active_count']:,}")
col2.metric("overall retention", f"{retention:.1%}")
col3.metric("mean match quality (\u2193better)", f"{last_row['win_prob_dev_mean']:.3f}")
col4.metric("mean rating error", f"{last_row['rating_error_mean']:.3f}")

tab_overview, tab_agg, tab_cohorts, tab_players, tab_matches = st.tabs(
    ["Overview", "Aggregate detail", "Cohorts", "Players", "Matches"]
)

# --- Overview ---------------------------------------------------------
with tab_overview:
    st.plotly_chart(charts.population_over_time(runs), use_container_width=True, key="ov_pop")
    st.plotly_chart(charts.retention_over_time(runs), use_container_width=True, key="ov_ret")
    st.plotly_chart(charts.match_quality_over_time(runs), use_container_width=True, key="ov_mq")
    st.plotly_chart(charts.rating_error_over_time(runs), use_container_width=True, key="ov_re")
    st.plotly_chart(charts.blowout_share_over_time(runs), use_container_width=True, key="ov_bo")

    with st.expander("config.json", expanded=False):
        st.json(exp.config.model_dump(mode="json"))

# --- Aggregate detail -------------------------------------------------
with tab_agg:
    st.plotly_chart(
        charts.skill_percentile_bands(agg),
        use_container_width=True,
        key="agg_bands",
    )
    st.plotly_chart(
        charts.experience_gear_over_time(agg),
        use_container_width=True,
        key="agg_xp_gear",
    )
    st.plotly_chart(
        charts.lobby_range_percentiles(agg),
        use_container_width=True,
        key="agg_lobby",
    )

# --- Cohorts (skill decile @ day 0) -----------------------------------
with tab_cohorts:
    if exp.population is None:
        st.info("No population.parquet saved for this run.")
    else:
        st.caption("Cohorts are defined by each player's true-skill decile on day 0.")
        st.plotly_chart(
            charts.retention_by_skill_decile(exp.population),
            use_container_width=True,
            key="cohort_retention",
        )
        st.plotly_chart(
            charts.cohort_metric_by_skill_decile(exp.population, "observed_skill"),
            use_container_width=True,
            key="cohort_observed",
        )
        st.plotly_chart(
            charts.cohort_metric_by_skill_decile(exp.population, "experience"),
            use_container_width=True,
            key="cohort_experience",
        )
        st.plotly_chart(
            charts.cohort_metric_by_skill_decile(exp.population, "gear"),
            use_container_width=True,
            key="cohort_gear",
        )

# --- Players ----------------------------------------------------------
with tab_players:
    if exp.population is None:
        st.info("No population.parquet saved for this run.")
    else:
        all_ids = exp.population["player_id"].unique().to_list()
        st.caption(f"{len(all_ids):,} total players in this run.")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            sample_size = st.slider(
                "random sample size", 1, 30, 10, key="players_sample_size"
            )
            import random
            seed = st.number_input("sample seed", value=0, step=1, key="players_seed")
            rng = random.Random(seed)
            default_ids = rng.sample(all_ids, min(sample_size, len(all_ids)))
            picked = st.multiselect(
                "players to plot",
                options=all_ids,
                default=default_ids,
                key="players_picked",
            )
        with col_b:
            metric = st.selectbox(
                "metric",
                [
                    "true_skill",
                    "observed_skill",
                    "experience",
                    "gear",
                    "matches_played",
                    "loss_streak",
                ],
                key="players_metric",
            )

        if picked:
            st.plotly_chart(
                charts.player_trajectories(exp.population, picked, metric),
                use_container_width=True,
                key="players_traj",
            )

        max_day = int(exp.population["day"].max())
        slider_day = st.slider("snapshot day", 0, max_day, max_day, key="players_slider_day")

        presets = {
            "experience vs observed_skill": ("experience", "observed_skill"),
            "true_skill vs observed_skill": ("true_skill", "observed_skill"),
            "true_skill vs gear": ("true_skill", "gear"),
            "experience vs gear": ("experience", "gear"),
            "matches_played vs observed_skill": ("matches_played", "observed_skill"),
            "custom…": None,
        }
        preset_choice = st.selectbox(
            "scatter pair", list(presets.keys()), index=0, key="scatter_preset"
        )
        cols_pop = [c for c in exp.population.columns if c not in ("day", "player_id", "party_id", "active")]
        if presets[preset_choice] is None:
            sc_col_a, sc_col_b = st.columns(2)
            with sc_col_a:
                x_col = st.selectbox("x axis", cols_pop, index=cols_pop.index("experience"), key="scatter_x")
            with sc_col_b:
                y_col = st.selectbox("y axis", cols_pop, index=cols_pop.index("observed_skill"), key="scatter_y")
        else:
            x_col, y_col = presets[preset_choice]
        st.plotly_chart(
            charts.player_scatter(exp.population, slider_day, x_col, y_col),
            use_container_width=True,
            key="players_scatter",
        )

        st.plotly_chart(
            charts.skill_distribution(exp.population, day=slider_day),
            use_container_width=True,
            key="players_dist",
        )

        with st.expander("population preview (first 1000 rows of selected day)"):
            st.dataframe(
                exp.population.filter(pl.col("day") == slider_day)
                .head(1000)
                .to_pandas()
            )

# --- Matches ----------------------------------------------------------
with tab_matches:
    if exp.matches is None or exp.matches.height == 0:
        st.info("No matches.parquet saved for this run.")
    else:
        st.caption(f"{exp.matches.height:,} matches logged.")
        metric = st.selectbox(
            "match metric",
            ["lobby_range", "lobby_std", "team_gap", "win_prob_dev"],
            key="matches_metric",
        )
        st.plotly_chart(
            charts.matches_metric_over_time(exp.matches, metric),
            use_container_width=True,
            key="matches_time",
        )
        st.plotly_chart(
            charts.matches_metric_distribution(exp.matches, metric),
            use_container_width=True,
            key="matches_dist",
        )
        blowout_share = (exp.matches["is_blowout"].sum() / exp.matches.height)
        st.metric("overall blowout share", f"{blowout_share:.1%}")
