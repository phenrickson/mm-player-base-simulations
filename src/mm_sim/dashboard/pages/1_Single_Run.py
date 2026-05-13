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
col3.metric(
    "favorite E[extract] (\u2193better)",
    f"{last_row['favorite_expected_extract_mean']:.3f}",
)
col4.metric("mean rating error", f"{last_row['rating_error_mean']:.3f}")

(
    tab_overview,
    tab_agg,
    tab_cohorts,
    tab_players,
    tab_matches,
    tab_player_detail,
    tab_match_detail,
) = st.tabs(
    [
        "Overview",
        "Aggregate detail",
        "Cohorts",
        "Players",
        "Matches",
        "Player detail",
        "Match detail",
    ]
)

# --- Overview ---------------------------------------------------------
with tab_overview:
    st.plotly_chart(charts.population_over_time(runs), use_container_width=True, key="ov_pop")
    st.plotly_chart(charts.retention_over_time(runs), use_container_width=True, key="ov_ret")
    st.plotly_chart(charts.match_quality_over_time(runs), use_container_width=True, key="ov_mq")
    st.plotly_chart(charts.rating_error_over_time(runs), use_container_width=True, key="ov_re")
    if exp.match_teams is not None and exp.match_teams.height > 0:
        st.plotly_chart(
            charts.mm_calibration_over_time([(label, exp.match_teams)]),
            use_container_width=True,
            key="ov_calib",
        )
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
                value=200,
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
                    "net_wins",
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
            pop_for_chart = exp.population
            if metric == "net_wins":
                if exp.match_teams is None or exp.match_teams.height == 0:
                    st.info(
                        "net_wins needs match_teams.parquet; re-run the "
                        "scenario after the per-team logging commit."
                    )
                    pop_for_chart = None
                else:
                    # Per-player cumulative extracts: explode match_teams to
                    # (day, match_idx, player_id, extracted), cumsum extracts
                    # per player, take the last value per day, then forward-fill
                    # across the population's day grid so chart lines stay
                    # continuous on days a player didn't play.
                    mt_picked = (
                        exp.match_teams
                        .filter(
                            pl.col("player_ids").list.eval(
                                pl.element().is_in(picked)
                            ).list.any()
                        )
                        .select(["day", "match_idx", "player_ids", "extracted"])
                        .explode("player_ids")
                        .rename({"player_ids": "player_id"})
                        .filter(pl.col("player_id").is_in(picked))
                        .sort(["player_id", "day", "match_idx"])
                        .with_columns(
                            pl.when(pl.col("extracted"))
                            .then(1)
                            .otherwise(-1)
                            .cum_sum()
                            .over("player_id")
                            .alias("net_wins")
                        )
                        .group_by(["player_id", "day"])
                        .agg(pl.col("net_wins").last())
                    )
                    pop_for_chart = (
                        exp.population
                        .filter(pl.col("player_id").is_in(picked))
                        .join(mt_picked, on=["player_id", "day"], how="left")
                        .sort(["player_id", "day"])
                        .with_columns(
                            pl.col("net_wins")
                            .forward_fill()
                            .over("player_id")
                            .fill_null(0)
                        )
                    )
            if pop_for_chart is not None:
                st.plotly_chart(
                    charts.player_trajectories(
                        pop_for_chart,
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

            # Pre-compute extraction stats for this player so they can be
            # surfaced in the top-level metric block.
            mt_player_top = mt.filter(
                pl.col("player_ids").list.contains(pid)
            )
            n_matches_mt = int(mt_player_top.height)
            n_extracts = (
                int(mt_player_top["extracted"].sum()) if n_matches_mt > 0 else 0
            )
            extract_rate_top = (
                (n_extracts / n_matches_mt) if n_matches_mt > 0 else float("nan")
            )

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
            c7.metric("extractions", f"{n_extracts}")
            c8.metric(
                "extract rate",
                f"{extract_rate_top:.1%}" if n_matches_mt > 0 else "n/a",
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

            # Gear and season_progress are semantically bounded to [0, 1];
            # other metrics autoscale per-player (None == plotly default).
            gear_range = (0.0, 1.0)
            sp_range = (0.0, 1.0)
            obs_range = None
            true_range = None
            mp_range = None
            nw_range = None

            # Shared x axis across panels, based on this player's span.
            max_player_day = int(traj["player_day"].max() or 0)
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
                horizontal_spacing=0.12,
                vertical_spacing=0.17,
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
            # Dotted y=0 reference on the skill and net-wins panels.
            for (rr, cc) in [(1, 1), (1, 2), (3, 1)]:
                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="#666",
                    row=rr, col=cc,
                )
            # Only apply explicit y ranges for the semantically-bounded
            # metrics; the rest autoscale to each player's data.
            fig.update_yaxes(range=gear_range, row=2, col=1)
            fig.update_yaxes(range=sp_range, row=2, col=2)

            fig.update_xaxes(title_text="player day", row=3, col=1)
            fig.update_xaxes(title_text="player day", row=3, col=2)
            fig.update_layout(
                title=f"Player {pid} trajectories (x = day \u2212 join_day)",
                height=820,
                hovermode="x unified",
                margin=dict(l=60, r=30, t=70, b=50),
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
                # Per-team expected extract under each view: softmax over
                # team strengths, scaled by the scenario's mean_extractors
                # per match. This preserves ranking even when the realized
                # k was 0 or n_teams (where Plackett-Luce marginals collapse
                # to all-zeros or all-ones and lose the ranking signal).
                import numpy as _np

                beta = float(exp.config.outcomes.strength_sensitivity)
                mean_k = float(exp.config.outcomes.mean_extractors_per_match)
                match_keys = mt_player.select(["day", "match_idx"]).unique()
                full_matches = mt.join(
                    match_keys, on=["day", "match_idx"], how="inner"
                ).sort(["day", "match_idx", "team_idx"])

                def _softmax_scaled(strengths_arr: _np.ndarray) -> _np.ndarray:
                    shifted = beta * strengths_arr
                    shifted = shifted - shifted.max()
                    w = _np.exp(shifted)
                    return (w / w.sum()) * mean_k

                def _elo_view(strengths_arr: _np.ndarray) -> _np.ndarray:
                    n = int(strengths_arr.shape[0])
                    if n < 2:
                        return _np.full(n, 0.5, dtype=_np.float64)
                    out = _np.zeros(n, dtype=_np.float64)
                    for a in range(n):
                        opp = _np.concatenate(
                            [strengths_arr[:a], strengths_arr[a + 1 :]]
                        )
                        out[a] = (1.0 / (1.0 + 10.0 ** (opp - strengths_arr[a]))).mean()
                    return out

                exp_rows: list[dict] = []
                for (day_v, match_v), group in full_matches.group_by(
                    ["day", "match_idx"], maintain_order=True
                ):
                    obs_strengths = _np.array(
                        group["mean_observed_skill_before"].to_list(),
                        dtype=_np.float64,
                    )
                    true_strengths = _np.array(
                        group["mean_true_skill_before"].to_list(),
                        dtype=_np.float64,
                    )
                    # MM view uses Elo; truth view uses softmax × mean_k.
                    obs_vals = _elo_view(obs_strengths)
                    true_vals = _softmax_scaled(true_strengths)
                    for team_idx, obs_val, true_val in zip(
                        group["team_idx"].to_list(), obs_vals, true_vals
                    ):
                        exp_rows.append(
                            {
                                "day": int(day_v),
                                "match_idx": int(match_v),
                                "team_idx": int(team_idx),
                                "expected_extract_observed": float(obs_val),
                                "expected_extract_true": float(true_val),
                            }
                        )
                exp_df = pl.DataFrame(exp_rows)
                mt_player_aug = mt_player.join(
                    exp_df,
                    on=["day", "match_idx", "team_idx"],
                    how="left",
                )

                display = mt_player_aug.rename(
                    {
                        "expected_extract_observed": "elo_E_win_obs",
                        "expected_extract_true": "extract_share_true",
                    }
                ).select(
                    [
                        "day",
                        "match_idx",
                        "team_idx",
                        "mean_true_skill_before",
                        "mean_observed_skill_before",
                        "mean_gear_before",
                        "team_strength",
                        "elo_E_win_obs",
                        "extract_share_true",
                        "extracted",
                        "kills",
                        "killed_by_team",
                    ]
                ).to_pandas()
                st.dataframe(display, use_container_width=True, height=500)

                # Summary stats
                extract_rate = float(mt_player["extracted"].mean())
                total_kills = int(mt_player["kills"].sum())
                mean_elo_obs = float(
                    mt_player_aug["expected_extract_observed"].mean()
                )
                mean_share_true = float(
                    mt_player_aug["expected_extract_true"].mean()
                )
                st.caption(
                    f"extract rate: {extract_rate:.1%} \u2022 "
                    f"mean Elo E[win] (obs): {mean_elo_obs:.2f} \u2022 "
                    f"mean extract share (true): {mean_share_true:.2f} \u2022 "
                    f"total kills: {total_kills}   "
                    f"(obs col = pairwise Elo the rating updater sees; "
                    f"true col = softmax \u00d7 mean_k = {mean_k:.2f})"
                )

# --- Match detail ----------------------------------------------------
with tab_match_detail:
    if exp.match_teams is None or exp.match_teams.height == 0:
        st.info("No match_teams.parquet for this run.")
    elif exp.population is None:
        st.info("No population.parquet saved for this run.")
    else:
        mt_all = exp.match_teams
        days_with_matches = sorted(mt_all["day"].unique().to_list())
        if not days_with_matches:
            st.info("No matches logged.")
        else:
            # Lookup row: "day,match_idx" text input + random button. Both
            # must run before the select_sliders so they can seed session state.
            lookup_col, random_col = st.columns([3, 1])
            with lookup_col:
                lookup_raw = st.text_input(
                    "jump to match (day,match_idx)",
                    value="",
                    placeholder="e.g. 42,7",
                    key="match_detail_lookup",
                )
            with random_col:
                st.write("")  # vertical spacer to align with the text input
                if st.button("random match", key="match_detail_random"):
                    import random as _rnd
                    rnd_day = int(_rnd.choice(days_with_matches))
                    rnd_matches = (
                        mt_all.filter(pl.col("day") == rnd_day)["match_idx"]
                        .unique()
                        .to_list()
                    )
                    st.session_state["match_detail_day"] = rnd_day
                    st.session_state["match_detail_match"] = int(
                        _rnd.choice(rnd_matches)
                    )

            # Only apply the lookup when the text changes, so moving the
            # sliders afterwards isn't undone on every rerun.
            lookup_val = lookup_raw.strip()
            last_applied = st.session_state.get("match_detail_lookup_applied", "")
            if lookup_val and lookup_val != last_applied:
                try:
                    d_str, m_str = lookup_val.split(",")
                    d_val = int(d_str.strip())
                    m_val = int(m_str.strip())
                except ValueError:
                    st.warning("Format must be `day,match_idx` (e.g. `42,7`).")
                else:
                    if d_val not in days_with_matches:
                        st.warning(f"No matches on day {d_val}.")
                    else:
                        valid_matches = (
                            mt_all.filter(pl.col("day") == d_val)["match_idx"]
                            .unique()
                            .to_list()
                        )
                        if m_val not in valid_matches:
                            st.warning(
                                f"match_idx {m_val} not found on day {d_val}."
                            )
                        else:
                            st.session_state["match_detail_day"] = d_val
                            st.session_state["match_detail_match"] = m_val
                            st.session_state["match_detail_lookup_applied"] = lookup_val

            md_col_a, md_col_b = st.columns(2)
            with md_col_a:
                sel_day = st.select_slider(
                    "day",
                    options=days_with_matches,
                    value=days_with_matches[0],
                    key="match_detail_day",
                )
            day_matches = sorted(
                mt_all.filter(pl.col("day") == sel_day)["match_idx"]
                .unique()
                .to_list()
            )
            with md_col_b:
                sel_match = st.select_slider(
                    "match_idx",
                    options=day_matches,
                    value=day_matches[0],
                    key="match_detail_match",
                )

            match_rows = (
                mt_all.filter(
                    (pl.col("day") == sel_day)
                    & (pl.col("match_idx") == sel_match)
                )
                .sort("team_idx")
            )

            if match_rows.height == 0:
                st.warning("No data for this match.")
            else:
                import plotly.graph_objects as _go
                day_pop = exp.population.filter(pl.col("day") == sel_day)

                n_teams = match_rows.height
                team_palette = [
                    "#e06666", "#6fa8dc", "#93c47d", "#f6b26b",
                    "#8e7cc3", "#76a5af", "#c27ba0", "#ffd966",
                ]
                team_colors = {i: team_palette[i % len(team_palette)]
                               for i in range(n_teams)}

                rows_list = list(match_rows.iter_rows(named=True))
                team_idxs = [int(r["team_idx"]) for r in rows_list]
                team_labels = [f"Team {i}" for i in team_idxs]
                # team_strength column is outcome-model strength (true_skill +
                # gear), i.e. the hidden truth that drove the dice roll. The
                # MM-view equivalent is mean_observed_skill_before — what the
                # matchmaker actually saw when forming this lobby.
                true_strengths = [float(r["team_strength"]) for r in rows_list]
                obs_strengths = [
                    float(r["mean_observed_skill_before"]) for r in rows_list
                ]
                # Expected extract uses a different formula per view, so that
                # each side shows what its own system actually computes:
                #   - true_skill (outcome)  : softmax(beta * strength) * mean_k
                #                             — the outcome model's winner-draw.
                #   - observed_skill (MM)   : pairwise Elo averaged over lobby
                #                             — the rating updater's view.
                import numpy as _np_md
                _beta_md = float(exp.config.outcomes.strength_sensitivity)
                _mean_k_md = float(
                    exp.config.outcomes.mean_extractors_per_match
                )

                def _expected_via_softmax(skills: list[float]) -> list[float]:
                    arr = _np_md.array(skills, dtype=_np_md.float64)
                    shifted = _beta_md * arr
                    shifted = shifted - shifted.max()
                    w = _np_md.exp(shifted)
                    return ((w / w.sum()) * _mean_k_md).tolist()

                def _expected_via_elo(skills: list[float]) -> list[float]:
                    n = len(skills)
                    if n < 2:
                        return [0.5] * n
                    out: list[float] = []
                    for a in range(n):
                        pairs = [
                            1.0 / (1.0 + 10.0 ** (skills[b] - skills[a]))
                            for b in range(n) if b != a
                        ]
                        out.append(sum(pairs) / len(pairs))
                    return out

                true_expecteds = _expected_via_softmax(true_strengths)
                obs_expecteds = _expected_via_elo(obs_strengths)
                extracted_flags = [bool(r["extracted"]) for r in rows_list]
                actuals = [1.0 if e else 0.0 for e in extracted_flags]
                kills_per_team = [int(r["kills"]) for r in rows_list]
                killed_by = [int(r["killed_by_team"]) for r in rows_list]

                # Lobby summary — "strongest team" uses outcome-model
                # strength since that's what actually decided the match.
                n_extract = sum(extracted_flags)
                strongest_pos = max(range(n_teams), key=lambda i: true_strengths[i])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("teams", n_teams)
                c2.metric("extracted", f"{n_extract}/{n_teams}")
                c3.metric("strongest team", f"#{team_idxs[strongest_pos]}")
                c4.metric(
                    "strongest share",
                    f"{true_expecteds[strongest_pos]:.2f}",
                )

                # Colorblind-safe palette (blue/orange)
                EXTRACT_COLOR = "#1f77b4"
                DIED_COLOR = "#ff7f0e"

                # 1. Team strength + expected extract side-by-side.
                st.subheader("Team strength and extraction odds")
                strength_view = st.radio(
                    "team strength view",
                    ["observed_skill (MM view)", "true_skill (outcome)"],
                    horizontal=True,
                    key="md_strength_view",
                    help=(
                        "observed_skill: what the matchmaker saw when "
                        "forming teams. true_skill: the hidden truth that "
                        "drove the extraction dice roll."
                    ),
                )
                use_observed = strength_view.startswith("observed_skill")
                strengths = obs_strengths if use_observed else true_strengths
                expecteds = obs_expecteds if use_observed else true_expecteds
                match_mean = float(sum(strengths) / len(strengths))
                # Sort by team_idx so Team 0 is on top, Team 1 below, etc.
                # Plotly draws y from bottom up, so reverse the order.
                order = sorted(range(n_teams), key=lambda i: team_idxs[i], reverse=True)
                sorted_labels = [team_labels[i] for i in order]
                sorted_colors = [
                    EXTRACT_COLOR if extracted_flags[i] else DIED_COLOR
                    for i in order
                ]
                sorted_outcomes = [
                    "extracted" if extracted_flags[i] else "died"
                    for i in order
                ]

                from plotly.subplots import make_subplots as _mks2
                combined = _mks2(
                    rows=1,
                    cols=2,
                    shared_yaxes=True,
                    horizontal_spacing=0.15,
                    subplot_titles=("team strength", "expected extract"),
                )
                # Left panel: strength dots on a number line with outcome
                # label. Place label on the opposite side of the match-mean
                # line so it never extends past the axis edge.
                text_positions = [
                    "middle right" if strengths[i] < match_mean
                    else "middle left"
                    for i in order
                ]
                combined.add_trace(
                    _go.Scatter(
                        x=[strengths[i] for i in order],
                        y=sorted_labels,
                        mode="markers+text",
                        marker=dict(
                            size=22,
                            color=sorted_colors,
                            line=dict(color="#222", width=1),
                        ),
                        text=sorted_outcomes,
                        textposition=text_positions,
                        textfont=dict(size=12),
                        hovertemplate="%{y}: strength %{x:.3f}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
                combined.add_vline(
                    x=match_mean,
                    line_dash="dash",
                    line_color="#888",
                    annotation_text=f"match mean {match_mean:.2f}",
                    annotation_position="bottom right",
                    row=1, col=1,
                )
                # Right panel: expected extract bars with outcome labels.
                combined.add_trace(
                    _go.Bar(
                        y=sorted_labels,
                        x=[expecteds[i] for i in order],
                        orientation="h",
                        marker_color=sorted_colors,
                        text=[
                            f"{expecteds[i]:.2f} \u2192 {sorted_outcomes[j]}"
                            for j, i in enumerate(order)
                        ],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{y}: expected %{x:.2f}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1, col=2,
                )
                # Reference line: for the softmax (truth) view, the uniform-
                # strength baseline share is mean_k / n_teams. For the Elo
                # (MM) view, every team's pairwise E[win] equals 0.5 when
                # ratings are identical.
                _ref_x = (
                    _mean_k_md / n_teams if not use_observed else 0.5
                )
                _ref_label = (
                    f"uniform {_ref_x:.2f}" if not use_observed else "0.5"
                )
                combined.add_vline(
                    x=_ref_x,
                    line_dash="dot",
                    line_color="#666",
                    annotation_text=_ref_label,
                    annotation_position="top",
                    row=1, col=2,
                )
                # Pad the left x-axis so outside-left/right labels don't
                # clip at the plot edge.
                strength_min = min(strengths)
                strength_max = max(strengths)
                strength_pad = max(0.05, 0.18 * (strength_max - strength_min))
                combined.update_xaxes(
                    title_text="team_strength",
                    range=[strength_min - strength_pad, strength_max + strength_pad],
                    row=1, col=1,
                )
                if use_observed:
                    _right_title = "Elo E[win vs lobby] (MM view)"
                    _right_range = [0, 1.05]
                else:
                    _right_title = (
                        f"extract share (softmax \u00d7 mean_k = {_mean_k_md:.2f})"
                    )
                    _right_range = [0, max(1.25, _mean_k_md + 0.25)]
                combined.update_xaxes(
                    title_text=_right_title,
                    range=_right_range,
                    row=1, col=2,
                )
                combined.update_layout(
                    height=max(280, 80 * n_teams + 40),
                    margin=dict(l=80, r=60, t=60, b=60),
                    showlegend=False,
                )
                st.plotly_chart(
                    combined, use_container_width=True, key="md_strength"
                )

                # 2. Players in the match
                st.subheader("Players in the match")
                all_pids = []
                all_team = []
                for r in rows_list:
                    for pid in r["player_ids"]:
                        all_pids.append(int(pid))
                        all_team.append(int(r["team_idx"]))
                player_df = day_pop.filter(
                    pl.col("player_id").is_in(all_pids)
                ).select(
                    ["player_id", "true_skill", "observed_skill", "gear"]
                )
                pdict = {
                    int(row["player_id"]): row
                    for row in player_df.iter_rows(named=True)
                }
                metric_choice = st.radio(
                    "x axis",
                    ["true_skill", "observed_skill"],
                    horizontal=True,
                    key="md_player_axis",
                )
                players_fig = _go.Figure()
                # Plotly draws categoryarray bottom-to-top, so reverse the
                # sorted team list to put Team 0 on top and higher indices below.
                team_order_labels = [
                    f"Team {i}" for i in sorted(team_idxs, reverse=True)
                ]
                # Track max x per team so outcome label sits past the rightmost player.
                team_max_x: dict[int, float] = {}
                for t_idx in team_idxs:
                    pids_for_team = [
                        p for p, tt in zip(all_pids, all_team) if tt == t_idx
                    ]
                    xs = [pdict[p][metric_choice] for p in pids_for_team]
                    text = [
                        f"pid {p} \u2022 true={pdict[p]['true_skill']:.2f} "
                        f"\u2022 obs={pdict[p]['observed_skill']:.2f}"
                        for p in pids_for_team
                    ]
                    extracted_this_team = extracted_flags[team_idxs.index(t_idx)]
                    team_max_x[t_idx] = max(xs) if xs else 0.0
                    players_fig.add_trace(
                        _go.Scatter(
                            x=xs,
                            y=[f"Team {t_idx}"] * len(xs),
                            mode="markers",
                            marker=dict(
                                size=14,
                                color=team_colors[t_idx],
                                symbol=(
                                    "circle" if extracted_this_team else "x"
                                ),
                                line=dict(color="#222", width=1),
                            ),
                            name=f"Team {t_idx}",
                            text=text,
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False,
                        )
                    )
                # Outcome labels sit clearly past each team's rightmost player.
                all_xs = [pdict[p][metric_choice] for p in all_pids]
                x_min = min(all_xs)
                x_max = max(all_xs)
                label_offset = max(0.03, 0.04 * (x_max - x_min))
                outcome_xs = [team_max_x[t] + label_offset for t in team_idxs]
                outcome_ys = [f"Team {t}" for t in team_idxs]
                outcome_text = [
                    "extracted" if extracted_flags[team_idxs.index(t)] else "died"
                    for t in team_idxs
                ]
                players_fig.add_trace(
                    _go.Scatter(
                        x=outcome_xs,
                        y=outcome_ys,
                        mode="text",
                        text=outcome_text,
                        textposition="middle right",
                        textfont=dict(size=12, color="#ddd"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Pad x-axis on the right so outcome labels have room.
                x_pad = max(0.15, 0.3 * (x_max - x_min))
                players_fig.update_layout(
                    xaxis_title=metric_choice,
                    xaxis=dict(range=[x_min - 0.05, x_max + x_pad]),
                    yaxis=dict(
                        categoryorder="array",
                        categoryarray=team_order_labels,
                    ),
                    height=max(240, 70 * n_teams + 40),
                    margin=dict(l=60, r=30, t=30, b=50),
                )
                st.plotly_chart(
                    players_fig, use_container_width=True, key="md_players"
                )


                # 4. Per-player rating change
                st.subheader("Per-player rating change from this match")
                has_delta_data = (
                    "observed_skill_before" in match_rows.columns
                    and match_rows["observed_skill_before"][0] is not None
                )
                if not has_delta_data:
                    st.info(
                        "This run predates per-match rating delta logging. "
                        "Re-run the scenario to populate rating changes."
                    )
                else:
                    # Faceted dumbbell: one row per team (single column), so
                    # each team's before/after pattern reads clearly. The
                    # per-team delta goes in the subplot title; per-player
                    # deltas are hidden (they're equal within a team).
                    from plotly.subplots import make_subplots as _mks_d

                    teams_in_order = sorted(
                        {int(r["team_idx"]) for r in rows_list}
                    )
                    # Build per-team player rows keyed by team_idx.
                    per_team: dict[int, list[dict]] = {
                        t: [] for t in teams_in_order
                    }
                    for r in rows_list:
                        t_idx = int(r["team_idx"])
                        pids = list(r["player_ids"])
                        before = list(r["observed_skill_before"])
                        after = list(r["observed_skill_after"])
                        for p, b, a in zip(pids, before, after):
                            per_team[t_idx].append(
                                {
                                    "pid": int(p),
                                    "before": float(b),
                                    "after": float(a),
                                    "delta": float(a) - float(b),
                                }
                            )

                    # Team-level delta = mean of player deltas (they're equal
                    # under the current updater, but take the mean so the
                    # header stays truthful if that ever changes).
                    team_deltas = {
                        t: (
                            sum(d["delta"] for d in per_team[t])
                            / max(1, len(per_team[t]))
                        )
                        for t in teams_in_order
                    }

                    extracted_by_team = {
                        int(r["team_idx"]): bool(r["extracted"])
                        for r in rows_list
                    }
                    subplot_titles = [
                        f"Team {t} \u2022 {'extracted' if extracted_by_team[t] else 'died'} \u2022 \u0394 {team_deltas[t]:+.4f}"
                        for t in teams_in_order
                    ]

                    # Shared x range so teams are visually comparable.
                    all_xs = [
                        v
                        for t in teams_in_order
                        for d in per_team[t]
                        for v in (d["before"], d["after"])
                    ]
                    x_min = min(all_xs)
                    x_max = max(all_xs)
                    x_pad = max(0.05, 0.15 * (x_max - x_min))
                    # Midpoint reference: the match's before-rating center
                    # line, drawn across every facet. Players whose after
                    # marker sits on the same vertical as before = no change.
                    x_mid = 0.5 * (x_min + x_max)

                    delta_fig = _mks_d(
                        rows=len(teams_in_order),
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.12,
                    )
                    for row_i, t in enumerate(teams_in_order, start=1):
                        color = team_colors[t]
                        # Stable y order within the facet: by pid asc,
                        # reversed so the lowest pid renders on top.
                        team_rows = sorted(per_team[t], key=lambda d: d["pid"])
                        y_labels = [f"pid {d['pid']}" for d in team_rows]
                        y_order_local = list(reversed(y_labels))

                        for d in team_rows:
                            label = f"pid {d['pid']}"
                            delta_fig.add_trace(
                                _go.Scatter(
                                    x=[d["before"], d["after"]],
                                    y=[label, label],
                                    mode="lines",
                                    line=dict(color=color, width=2),
                                    hoverinfo="skip",
                                    showlegend=False,
                                ),
                                row=row_i, col=1,
                            )
                            delta_fig.add_trace(
                                _go.Scatter(
                                    x=[d["before"]],
                                    y=[label],
                                    mode="markers",
                                    marker=dict(
                                        size=11,
                                        color="rgba(0,0,0,0)",
                                        line=dict(color=color, width=2),
                                        symbol="circle",
                                    ),
                                    hovertemplate=(
                                        f"pid {d['pid']} \u2022 before="
                                        f"{d['before']:.4f}<extra></extra>"
                                    ),
                                    showlegend=False,
                                ),
                                row=row_i, col=1,
                            )
                            delta_fig.add_trace(
                                _go.Scatter(
                                    x=[d["after"]],
                                    y=[label],
                                    mode="markers",
                                    marker=dict(
                                        size=11,
                                        color=color,
                                        line=dict(color="#222", width=1),
                                        symbol="circle",
                                    ),
                                    hovertemplate=(
                                        f"pid {d['pid']} \u2022 after="
                                        f"{d['after']:.4f}<extra></extra>"
                                    ),
                                    showlegend=False,
                                ),
                                row=row_i, col=1,
                            )
                        delta_fig.update_yaxes(
                            categoryorder="array",
                            categoryarray=y_order_local,
                            row=row_i, col=1,
                        )

                    delta_fig.update_xaxes(
                        range=[x_min - x_pad, x_max + x_pad],
                        title_text="observed_skill (open = before, filled = after)",
                        row=len(teams_in_order), col=1,
                    )
                    delta_fig.update_layout(
                        height=max(220, 130 * len(teams_in_order) + 60),
                        margin=dict(l=80, r=40, t=40, b=50),
                    )
                    st.plotly_chart(
                        delta_fig, use_container_width=True, key="md_deltas"
                    )
