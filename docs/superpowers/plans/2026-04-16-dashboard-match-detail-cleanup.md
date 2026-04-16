# 2026-04-16 — Dashboard cleanup: match detail, match-quality metric, cohort comparisons

Summary of dashboard changes made during a single working session. These
are already implemented; this doc is a record of what changed and why,
not a plan of work to do.

## Context

Working in the Streamlit dashboard (`src/mm_sim/dashboard/`) against
experiments that now use multi-team extraction. Several panels were
either showing NaN (due to two-team-era metrics), visually cluttered,
or failed to answer the question the user was actually asking when
looking at a match.

## Changes

### 1. Match detail — "Team strength and extraction odds" cleanup
File: `src/mm_sim/dashboard/pages/1_Single_Run.py`

- Sort team y-axis so Team 0 is always on top, Team 1 below, etc.
  (fixed order by `team_idx`, not by strength).
- Label outcome ("extracted" / "died") directly on each team in both
  panels instead of a legend. Left-panel label flips left/right of
  each dot based on its position relative to the match-mean line so
  labels don't clip at the axis edge.
- Right panel: pad x-range to `[0, 1.25]`, `cliponaxis=False`, so the
  "0.56 → extracted" outside-labels don't collide with the 0.5 line.
- Move "match mean" annotation to `bottom right` to stop it colliding
  with the subplot title.

### 2. Match detail — true-skill vs observed-skill toggle
File: `src/mm_sim/dashboard/pages/1_Single_Run.py`

Added a radio above the strength/extraction chart:

- **observed_skill (MM view)** — `mean_observed_skill_before` per team;
  expected extract computed from the pairwise Elo formula used by
  `ExtractEloUpdater` (ELO_SCALE=1.0). This is what the rating
  updater grades teams against.
- **true_skill (outcome)** — `team_strength` column (outcome-model
  true-skill strength + gear) and the logged `expected_extract` (the
  value the outcome model actually rolled against).

Rationale: on day 1 observed_skill is ~uniform, so MM-view should be
flat. The outcome strength is always true-skill-driven (physics), so
toggling between views isolates "what the MM knew" vs "the hidden
truth that decided the match".

### 3. Match detail — jump-to-match lookup
File: `src/mm_sim/dashboard/pages/1_Single_Run.py`

Added a `day,match_idx` text input + "random match" button that seed
the existing day/match_idx select_sliders via `st.session_state`.
Lookup only applies when the input changes (tracked via
`match_detail_lookup_applied`), so moving the sliders afterwards
isn't overridden on rerun.

### 4. Match quality metric — use `favorite_expected_extract_mean`
Files:
- `src/mm_sim/dashboard/charts.py`
- `src/mm_sim/dashboard/pages/1_Single_Run.py`
- `src/mm_sim/plots.py`
- `tests/dashboard/test_charts.py`

The old "Match quality (win-prob dev)" panel used `win_prob_dev_mean`,
which is only defined in 2-team mode — NaN under multi-team
extraction, producing an empty panel. Switched to
`favorite_expected_extract_mean` (the extraction-mode analog already
logged in snapshots). Added a reference line at 0.25 (the chance
baseline for 4-team lobbies). Also updated:
- Header KPI in the Single Run page (`col3.metric`).
- `small_multiples` panel label.
- `plots.py:_plot_favorite_win_prob` to auto-detect extraction mode
  (matching the logic already in `compare.py`).
- Test fixture adds `favorite_expected_extract_mean`; test renamed.

### 5. MM rating calibration chart
Files:
- `src/mm_sim/dashboard/charts.py`
- `src/mm_sim/dashboard/pages/1_Single_Run.py`
- `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`

New chart: **MM rating calibration** = per-day mean
`|MM's pairwise-Elo E[extract] − actual E[extract]|` across all teams
in all matches. Shows whether observed_skill ratings are converging
toward the true-skill view over the season.

Added two lines per scenario:
- Solid: actual gap using observed_skill.
- Dotted ("floor"): the irreducible gap from plugging true_skill
  directly into the pairwise-Elo formula — isolates formula mismatch
  (Elo logistic vs outcome-model normal CDF) so you can see how much
  of the total gap is actually MM rating error.

Computed in the dashboard layer from `match_teams.parquet` (columns
`mean_observed_skill_before`, `mean_true_skill_before`,
`expected_extract`) — no re-running needed on existing experiments.

Wired into:
- **Single Run** Overview tab — below rating error.
- **Compare Scenarios** — new "MM calibration" focus-metric option,
  plus an always-on section at the bottom of the Overview tab.

### 6. Compare Scenarios — tabs and cohort views
File: `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`

Split the page into **Overview** and **Cohorts** tabs. Overview keeps
the existing focus-metric chart, metric grid, MM calibration section,
and run metadata table.

Cohorts tab adds:

- **Retention by day-0 skill decile** — 2×5 grid, one subplot per
  decile (subplot titles "bottom 10%", "10–20%", …, "80–90%",
  "top 10%"), one line per scenario per panel. Deciles assigned
  per-scenario against that scenario's own day-0 true_skill.
  Implementation: `charts.retention_by_decile_faceted`.
- **Daily churn rate by experience cohort** — 1×3 grid faceted by
  matches_played bucket (new <20 / casual 20–49 / experienced ≥50),
  one line per scenario. Ported directly from
  `compare.py:_plot_churn_rate_for_cohort` so numbers match the
  static PNG output. Implementation:
  `charts.churn_rate_by_experience_cohort`.

Both charts drop their plotly titles in favor of Streamlit
subheaders to avoid title duplication.

## Files touched

- `src/mm_sim/dashboard/pages/1_Single_Run.py`
- `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`
- `src/mm_sim/dashboard/charts.py`
- `src/mm_sim/plots.py`
- `tests/dashboard/test_charts.py`

## Tests

Full suite (117 tests) passes after each change.

## Observations (from exploring the results)

- **MM calibration in skill_only/v1**: solid line sits at ~0.113,
  floor at ~0.103 after day ~7. Actual rating error is only ~0.01 —
  the MM is essentially fully converged within the first week of a
  90-day season. The "plateau" is overwhelmingly formula mismatch,
  not MM failure.
- **Churn by experience cohort**: new-player daily churn is
  consistently higher than casual/experienced cohorts across all
  scenarios, and grows over the season. random_mm is worst, but
  skill_only also shows elevated new-player churn — suggests a
  cold-start problem where new players at default rating get placed
  against hidden-skill opponents. Not investigated in this session;
  noted for future work.
