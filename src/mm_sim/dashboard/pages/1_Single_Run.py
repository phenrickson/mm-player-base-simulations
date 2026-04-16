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

(
    tab_overview,
    tab_agg,
    tab_cohorts,
    tab_players,
    tab_matches,
    tab_player_detail,
) = st.tabs(
    [
        "Overview",
        "Aggregate detail",
        "Cohorts",
        "Players",
        "Matches",
        "Player detail",
    ]
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
            charts.cohort_metric_by_skill_decile(exp.population, "gear"),
            use_container_width=True,
            key="cohort_gear",
        )

# --- Players ----------------------------------------------------------
with tab_players:
    if exp.population is None:
        st.info("No population.parquet saved for this run.")
    else:
        only_day0 = st.checkbox(
            "only players present on day 0", value=False, key="players_only_day0"
        )
        if only_day0:
            all_ids = (
                exp.population.filter(pl.col("day") == 0)["player_id"]
                .unique()
                .to_list()
            )
        else:
            all_ids = exp.population["player_id"].unique().to_list()
        st.caption(
            f"{len(all_ids):,} players in pool"
            + (" (day-0 only)" if only_day0 else " (all, including later joiners)")
        )

        sample_sizes = [1, 5, 10, 25, 50, 100] + list(range(150, 1001, 50))
        col_a, col_b = st.columns([2, 1])
        with col_a:
            sample_size = st.select_slider(
                "random sample size",
                options=sample_sizes,
                value=10,
                key="players_sample_size",
            )
            import random
            rng = random.Random(1999)
            sample_ids = rng.sample(all_ids, min(sample_size, len(all_ids)))
            extra_ids = st.multiselect(
                "add specific player_ids",
                options=[i for i in all_ids if i not in sample_ids],
                default=[],
                key="players_extra",
            )
            picked = sample_ids + extra_ids
        with col_b:
            metric = st.selectbox(
                "metric",
                [
                    "observed_skill",
                    "true_skill",
                    "experience",
                    "gear",
                    "season_progress",
                    "matches_played",
                    "loss_streak",
                ],
                key="players_metric_v2",
            )
            color_by = st.selectbox(
                "color by",
                [
                    "day-0 true_skill",
                    "day-0 talent_ceiling",
                    "day-0 join_day",
                    "final observed_skill",
                    "final gear",
                    "final experience",
                    "final matches_played",
                    "none",
                ],
                index=0,
                key="players_color_by_v2",
            )
            x_axis_choice = st.radio(
                "x axis",
                ["season", "player"],
                horizontal=True,
                help=(
                    "season = calendar day of the season; "
                    "player = days since each player joined"
                ),
                key="players_x_axis",
            )

        st.caption(f"plotting {len(picked)} players")
        if picked:
            st.plotly_chart(
                charts.player_trajectories(
                    exp.population,
                    picked,
                    metric,
                    color_by=None if color_by == "none" else color_by,
                    x_axis=x_axis_choice,
                ),
                use_container_width=True,
                key="players_traj",
            )

        max_day = int(exp.population["day"].max())
        slider_day = st.slider("snapshot day", 0, max_day, max_day, key="players_slider_day")

        presets = {
            "experience \u2192 observed_skill": ("experience", "observed_skill"),
            "true_skill \u2192 observed_skill": ("true_skill", "observed_skill"),
            "true_skill \u2192 gear": ("true_skill", "gear"),
            "experience \u2192 gear": ("experience", "gear"),
            "matches_played \u2192 observed_skill": ("matches_played", "observed_skill"),
            "custom\u2026": None,
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

# --- Player detail ----------------------------------------------------
with tab_player_detail:
    if exp.match_teams is None or exp.match_teams.height == 0:
        st.info(
            "No match_teams.parquet for this run. Re-run the scenario after "
            "the per-match-per-team logging commit."
        )
    elif exp.population is None:
        st.info("No population.parquet saved for this run.")
    else:
        mt = exp.match_teams
        all_ids = sorted(exp.population["player_id"].unique().to_list())

        # Random button must run before the number_input is instantiated so
        # it can seed the widget's session state.
        col_pick, col_rand = st.columns([3, 1])
        with col_rand:
            st.write("")  # spacer
            if st.button("random", key="player_detail_random"):
                import random as _rnd
                st.session_state["player_detail_id"] = int(_rnd.choice(all_ids))
        with col_pick:
            if "player_detail_id" not in st.session_state:
                st.session_state["player_detail_id"] = int(all_ids[0])
            picked_id = st.number_input(
                "player_id",
                min_value=int(min(all_ids)),
                max_value=int(max(all_ids)),
                step=1,
                key="player_detail_id",
            )

        pid = int(picked_id)

        # Player metadata
        day0_row = (
            exp.population.filter(
                (pl.col("player_id") == pid) & (pl.col("day") == 0)
            )
            .head(1)
        )
        if day0_row.height == 0:
            day0_row = (
                exp.population.filter(pl.col("player_id") == pid)
                .sort("day")
                .head(1)
            )
        final_row = (
            exp.population.filter(pl.col("player_id") == pid)
            .sort("day")
            .tail(1)
        )
        if day0_row.height == 0 or final_row.height == 0:
            st.warning(f"player {pid} not found in population.")
        else:
            d0 = day0_row.row(0, named=True)
            df = final_row.row(0, named=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("true_skill", f"{d0.get('true_skill', 0):.3f}")
            c2.metric(
                "observed_skill (final)",
                f"{df.get('observed_skill', 0):.3f}",
                delta=f"{df.get('observed_skill', 0) - d0.get('observed_skill', 0):+.3f}",
            )
            c3.metric("final gear", f"{df.get('gear', 0):.3f}")
            c4.metric(
                "season_progress (final)",
                f"{df.get('season_progress', 0):.1%}",
            )
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("join_day", f"{int(d0.get('join_day', 0))}")
            c6.metric(
                "matches_played (final)",
                f"{int(df.get('matches_played', 0))}",
            )
            c7.metric("active at end", "yes" if df.get("active", False) else "no")
            c8.metric(
                "talent_ceiling",
                f"{d0.get('talent_ceiling', float('nan')):.3f}",
            )

            # Multi-panel trajectories on player-day axis. Only show days
            # where the player was active (drop pre-join and post-churn rows).
            traj = (
                exp.population.filter(
                    (pl.col("player_id") == pid) & pl.col("active")
                )
                .sort("day")
                .with_columns(
                    (pl.col("day") - pl.col("join_day")).alias("player_day")
                )
            )

            # Run-wide bounds so axes stay consistent when switching players.
            active_pop = exp.population.filter(pl.col("active"))

            def _range(col: str, lo_floor: float | None = None,
                       hi_ceil: float | None = None,
                       pad: float = 0.05) -> tuple[float, float]:
                if active_pop.height == 0 or col not in active_pop.columns:
                    return (lo_floor or 0.0, hi_ceil or 1.0)
                lo = float(active_pop[col].min())
                hi = float(active_pop[col].max())
                if lo == hi:
                    hi = lo + 1.0
                span = hi - lo
                lo -= span * pad
                hi += span * pad
                if lo_floor is not None:
                    lo = max(lo, lo_floor)
                if hi_ceil is not None:
                    hi = min(hi, hi_ceil)
                return (lo, hi)

            obs_range = _range("observed_skill")
            true_range = _range("true_skill")
            gear_range = _range("gear", lo_floor=0.0, hi_ceil=1.0)
            sp_range = _range("season_progress", lo_floor=0.0, hi_ceil=1.0)
            mp_range = _range("matches_played", lo_floor=0.0)

            max_player_day = int(active_pop.select(
                (pl.col("day") - pl.col("join_day")).max()
            ).item() or 0)
            pd_range = (0, max(max_player_day, 1))

            # Net wins derived from match_teams: +1 per extract, -1 per death,
            # cumulative. Shows as a single line that bobs up and down.
            mt_player = mt.filter(
                pl.col("player_ids").list.contains(pid)
            ).sort(["day", "match_idx"])
            if mt_player.height > 0:
                cum = (
                    mt_player.select(["day", "extracted"])
                    .with_columns(
                        pl.when(pl.col("extracted"))
                        .then(1)
                        .otherwise(-1)
                        .cum_sum()
                        .alias("net_wins")
                    )
                    .group_by("day")
                    .agg(pl.col("net_wins").last())
                    .sort("day")
                )
                join_day = int(d0.get("join_day", 0))
                cum = cum.with_columns((pl.col("day") - join_day).alias("player_day"))
            else:
                cum = None

            import plotly.graph_objects as _go
            from plotly.subplots import make_subplots as _mks

            panels = [
                ("observed_skill", "observed_skill", traj),
                ("true_skill", "true_skill", traj),
                ("gear", "gear", traj),
                ("season_progress", "season_progress", traj),
            ]
            cols, rows = 2, 3
            fig = _mks(
                rows=rows,
                cols=cols,
                subplot_titles=[
                    "observed_skill", "true_skill",
                    "gear", "season_progress",
                    "net wins (+1 extract, -1 death)", "matches_played",
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.12,
            )
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for (title, col, df), (r, c) in zip(panels, positions):
                fig.add_trace(
                    _go.Scatter(
                        x=df["player_day"].to_list(),
                        y=df[col].to_list(),
                        mode="lines",
                        line=dict(color="#6fa8dc"),
                        showlegend=False,
                    ),
                    row=r, col=c,
                )
            # Net wins (row 3 col 1)
            if cum is not None:
                fig.add_trace(
                    _go.Scatter(
                        x=cum["player_day"].to_list(),
                        y=cum["net_wins"].to_list(),
                        mode="lines",
                        line=dict(color="#6fa8dc"),
                        showlegend=False,
                    ),
                    row=3, col=1,
                )
            # matches_played (row 3 col 2)
            fig.add_trace(
                _go.Scatter(
                    x=traj["player_day"].to_list(),
                    y=traj["matches_played"].to_list(),
                    mode="lines",
                    line=dict(color="#6fa8dc"),
                    showlegend=False,
                ),
                row=3, col=2,
            )
            # Shared x axis (player day) across all panels.
            for r in (1, 2, 3):
                for c in (1, 2):
                    fig.update_xaxes(range=pd_range, row=r, col=c)
            # Per-metric y ranges (stable across players).
            fig.update_yaxes(range=obs_range, row=1, col=1)
            fig.update_yaxes(range=true_range, row=1, col=2)
            fig.update_yaxes(range=gear_range, row=2, col=1)
            fig.update_yaxes(range=sp_range, row=2, col=2)
            fig.update_yaxes(range=mp_range, row=3, col=2)
            # Net wins: symmetric around 0 based on matches_played max
            # (a player could in principle be all-wins or all-losses).
            nw_max = int(mp_range[1])
            fig.update_yaxes(range=(-nw_max, nw_max), row=3, col=1)

            fig.update_xaxes(title_text="player day", row=3, col=1)
            fig.update_xaxes(title_text="player day", row=3, col=2)
            fig.update_layout(
                title=f"Player {pid} trajectories (x = day \u2212 join_day)",
                height=750,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, key="pd_multi")

            # Match history — filter rows whose player_ids list contains pid.
            mt_player = mt.filter(
                pl.col("player_ids").list.contains(pid)
            ).sort(["day", "match_idx"])
            st.subheader(f"Match history ({mt_player.height} matches)")
            if mt_player.height == 0:
                st.info("No matches recorded for this player.")
            else:
                display = mt_player.select(
                    [
                        "day",
                        "match_idx",
                        "team_idx",
                        "mean_true_skill_before",
                        "mean_observed_skill_before",
                        "mean_gear_before",
                        "team_strength",
                        "expected_extract",
                        "extracted",
                        "kills",
                        "killed_by_team",
                    ]
                ).to_pandas()
                st.dataframe(display, use_container_width=True, height=500)

                # Summary stats
                extract_rate = float(mt_player["extracted"].mean())
                total_kills = int(mt_player["kills"].sum())
                mean_expected = float(mt_player["expected_extract"].mean())
                st.caption(
                    f"extract rate: {extract_rate:.1%} \u2022 "
                    f"mean expected_extract: {mean_expected:.2f} \u2022 "
                    f"total kills: {total_kills}"
                )
