# Multi-Team Extraction Match Model — Design

**Status:** Design approved 2026-04-15. Awaiting user review before plan.
**Branch:** `multi-team-extraction`

## Motivation

The current sim resolves matches as 2 teams × 6 players, fully zero-sum: one team wins, one loses, ratings transfer pairwise, gear transfers from loser to winner. Season progression is a flat per-match earn, which produces a deterministic per-player-day climb to the ceiling — survival, not skill or outcome variance, is the only differentiator.

This collapses several research questions we want to study:

1. **Team composition matters but is invisible.** Mixed-skill parties are rated by their composite mean; their internal asymmetry never shows up in match outcomes or matchmaking decisions.
2. **Season progression is orthogonal to outcomes.** Whether you win or lose, you earn the same. So progression doesn't interact with the feedback loop between skill, matchmaking, and churn.
3. **Gear transfer is binary.** Stomping a weak team and upsetting a strong team produce identical gear flows.
4. **Two-team-only obscures the "extraction" dynamic** real shooters have, where survival without winning is a meaningful outcome distinct from "lost the match."

This design extends the sim to **multi-team extraction matches**: 4 teams × 3 players per lobby, where each team independently extracts (survives) or dies, kills are attributed team→team for gear transfer, and season progression is outcome-responsive. The matchmaker becomes two-stage (form teams, then assemble lobbies) with independent policies at each stage — exposing a 2D experimental space we can sweep.

Two-team mode remains supported so existing scenarios and historical experiments stay valid; multi-team is a new mode opted into via config.

---

## Design overview

### Match shape

- **Lobby:** 12 players, 4 teams × 3 players. Configurable via `lobby_size`, `teams_per_lobby`, `team_size` (constrained so size × count = lobby_size).
- **Pre-formed parties:** trio = a complete team; duo = team-with-one-gap to be filled by the matchmaker; solo = ad-hoc team-member to be combined with two other un-teamed players.
- **Per-team outcome:** each team independently either **extracts** or is **wiped**. Multiple teams can extract in the same match (no forced winner; no forced ranking).
- **Kill credits:** when team X is the killer of team Y, we record a directed credit `(X, Y)` for use in gear transfer.

### Outcome resolution

Each team rolls independently:

```text
team_strength = mean over team members of (true_skill + gear_weight × gear)
match_mean    = mean of all teams' strengths
roll          = (team_strength − match_mean) + noise
extract       = (roll > 0)
```

Where `noise ~ Normal(0, σ)` with σ calibrated so that a team exactly at `match_mean` extracts with probability `baseline_extract_prob` (default 0.4 — extraction feels earned; more teams die than survive on average).

The relationship between `baseline_extract_prob` and `σ` is inverse: lower base rate ⇒ higher effective threshold ⇒ less noise needed for the same skill-sensitivity. The design exposes `baseline_extract_prob` (the user-facing knob) and `strength_sensitivity` (how steeply prob scales with strength delta), and derives `σ` internally.

**Kill attribution:** for each dead team Y, the credited killer is the **weakest extracting team whose strength exceeds Y**. If no extractor is stronger than Y (a rare upset case where Y died despite outranking all extractors), the strongest extractor takes credit. If no team extracts at all, no kill credits are issued and the gear/transfer step is skipped entirely (the environment killed everyone; nobody loots).

### Two-stage matchmaker

Replace the current single-stage matchmaker with two independently configurable stages.

**Stage 1 — Team formation.** Operates on un-teamed players: solos and duos with one open slot. Pre-formed trios skip this stage entirely. Combines candidates into complete teams of 3 using its own composite weights and tolerance window.

**Stage 2 — Lobby assembly.** Operates on complete teams (pre-formed trios + stage-1-formed teams). Assembles 4 teams into a lobby using its own composite weights and tolerance window.

Each stage has independent `composite_weights` (skill / experience / gear) and `max_rating_spread` (with `max_rating_spread_growth` for queue-time relaxation). No team is ever rejected — tighter windows simply produce longer queues and eventually accept further-away matches.

This exposes a 2D experimental surface: team-formation policy × lobby-assembly policy. Examples worth sweeping:

- **Tight teams + loose lobbies:** uniform parties stomp/get-stomped — models "premades vs pubs."
- **Loose teams + tight lobbies:** mixed-skill parties play balanced lobbies — models "matchmaker fixes things at the lobby level."
- **Random teams + skill lobbies:** solo-queue with SBMM — models contemporary public matchmaking.
- **Skill teams + random lobbies:** find similar teammates, fight anyone — old-school matchmaking.

### Gear transfer

Replaces the current "loser's gear transfers to winner at flat rate" rule with three components.

**1. Extract growth.** Every player on an extracting team gains a flat `extract_growth` (default 0.003). Dead-team players gain nothing from the match.

**2. Kill-credited transfer with strength bonus.** When team X is credited with killing team Y:

```text
delta = Y_strength − X_strength
rate  = transfer_rate × max(punching_down_floor, 1 + strength_bonus × delta)

for each y in Y.players:
    loss = y.gear × rate
    y.gear -= loss
    pool  += loss

gain_per_x_player = (pool × transfer_efficiency) / team_size
for each x in X.players:
    x.gear = min(max_gear, x.gear + gain_per_x_player)   # excess → void
```

- `delta > 0` (X killed a stronger team — an upset): rate scales up multiplicatively. A 2σ upset produces a transformative gear gain.
- `delta < 0` (X stomped a weaker team): rate floors at `transfer_rate × punching_down_floor` (default 0.2). You still get *some* loot from a kill, but stomping isn't lucrative.
- `transfer_efficiency` (default 0.9) is the fraction of stripped gear that actually reaches the killer — the rest evaporates. Models loot decay / spoilage / "you couldn't carry it all."
- Cap-at-`max_gear` is enforced on the receiver; loser still loses regardless (you can only carry so much).

**3. No-extractor case.** Skip all gear updates. No transfers, no decay, no losses. Modeled as "the environment killed everyone — no one looted anything."

### Season progression

Replaces the current flat per-match earn with a **season-length-invariant**, **outcome-weighted**, **concave** progression model.

**Season-level invariant base rate:** the user configures `base_earn_per_season` — total raw earn-points an average fully-participating player accumulates over one season. The per-match rate is derived:

```text
expected_matches = mean_matches_per_day × season_days
base_per_match   = base_earn_per_season / expected_matches
```

Changing `season_days` from 60 to 120 automatically halves the per-match rate, keeping season-relative behavior consistent.

**Outcome-responsive components.** Three weights split where progress comes from:

```text
earn_raw = base_per_match × normalize(
    participation_weight × 1                    # everyone who played
  + extraction_weight   × (1 if extracted)      # surviving
  + kill_weight         × kills                 # for each kill credit
)
```

Where `normalize` divides each outcome term by its expected frequency, so a player with exactly average outcomes earns exactly `base_per_match` raw on an average match. Players who extract above their expected rate, or get more kills, earn faster; players who die a lot earn only the participation share.

Defaults: `participation=0.3, extraction=0.5, kill=0.2`. These belong in `defaults.toml` so scenarios can override.

**Concave application.** Raw earn is bent toward diminishing returns:

```text
earn_this_match = earn_raw × (1 - current_progress)^concavity
```

At `concavity=1.0`, halfway is half-rate; near the ceiling, gains shrink dramatically. Users can adjust to make late-progress harder (`concavity=2.0`) or near-linear (`concavity=0.5`).

**Target shape:** with default weights, average extract rate, full participation, and `base_earn_per_season=0.8` and `concavity=1.0`: median player reaches ~50% by season end; top performers reach 100%; many players never finish. This makes the season pass an *aspiration* whose completion depends on outcomes, not a guarantee for anyone who logs in.

**Churn pressure** (behind/boredom) keeps the same shape it does today: `behind_weight × max(0, expected − actual)` and `boredom_weight × max(0, actual − expected)` for the early season. The expected curve is `1 − exp(−curve_steepness × day/season_days)` as before.

### Rating update

Use an **expected-vs-actual extract** rule for `observed_skill`:

```text
for each team in lobby:
    actual    = 1 if extracted else 0
    expected  = team.expected_extract     # already computed during outcome resolution
    delta     = k_factor × (actual − expected) / team_size
    for each player in team:
        player.observed_skill += delta
```

The `expected_extract` is the pre-noise probability from the extract roll, computed during outcome resolution. Strong teams "should" extract; failing to do so is a big rating drop. Weak teams "shouldn't" extract; succeeding is a big gain. Equal split across the 3 team members preserves the current team-outcome → per-player-rating model.

**Kills are not factored into rating (v1).** A team that camps and extracts is rated the same as one that fights and extracts. Rating measures survival likelihood; kills show up in gear and season progress instead. If the rating signal proves too coarse in practice (e.g., it can't distinguish dominant teams from cautious ones), upgrade to a score-based rule that includes `kill_bonus × kills` on both expected and actual sides. Flagged as future work.

---

## Configuration changes

New / modified config blocks. Existing blocks not shown here keep their current shape.

```python
class MatchmakerConfig:
    kind: str = "composite"          # or "random"
    lobby_size: int = 12
    teams_per_lobby: int = 4         # 2 (legacy) or 4 (extraction)
    # team_size is derived as lobby_size / teams_per_lobby; lobby_size
    # must be evenly divisible by teams_per_lobby (validated).

    team_formation: StageConfig      # composite_weights + max_rating_spread + growth
    lobby_assembly: StageConfig      # same shape as team_formation


class StageConfig:
    composite_weights: dict[str, float] = {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    max_rating_spread: float = 0.3
    max_rating_spread_growth: float = 0.05


class OutcomeConfig:
    kind: str = "extraction"         # "default" (2-team) or "extraction" (multi-team)
    gear_weight: float = 0.5         # how much gear contributes to team strength

    # Extraction-specific
    baseline_extract_prob: float = 0.4
    strength_sensitivity: float = 1.0


class GearConfig:
    max_gear: float = 1.0
    extract_growth: float = 0.003
    transfer_rate: float = 0.05
    strength_bonus: float = 1.0
    punching_down_floor: float = 0.2
    transfer_efficiency: float = 0.9


class SeasonProgressionConfig:
    enabled: bool = True
    base_earn_per_season: float = 0.8
    concavity: float = 1.0
    participation_weight: float = 0.3
    extraction_weight: float = 0.5
    kill_weight: float = 0.2
    behind_weight: float = 0.02
    boredom_weight: float = 0.01
    boredom_cutoff: float = 0.7
    curve_steepness: float = 3.0


class RatingUpdaterConfig:
    kind: str = "elo_extract"        # new for extraction; "elo" still works for 2-team
    k_factor: float = 32.0
```

`scenarios/defaults.toml` adopts:
- `outcomes.kind = "extraction"`, `outcomes.gear_weight = 0.5`
- All three progression systems on (already done)
- `matchmaker.teams_per_lobby = 4`
- Season progression weights set explicitly (not relying on code defaults)

---

## Component boundaries

The implementation should preserve existing module boundaries and keep new logic isolated.

- **`mm_sim/outcomes/extraction.py`** (new) — implements the extract-roll outcome resolver. Pluggable via `OutcomeConfig.kind`. Existing `default.py` (2-team) stays untouched.
- **`mm_sim/matchmaker/two_stage.py`** (new) — implements team formation + lobby assembly. Existing `composite_mm.py` (single-stage) is the 2-team path.
- **`mm_sim/gear.py`** — extended with the strength-scaled, decay-applied transfer for extraction outcomes. Legacy 2-team transfer path preserved.
- **`mm_sim/season_progression.py`** — extended with outcome-weighted earn and concavity application. Legacy flat-earn path preserved as the "default" outcome type.
- **`mm_sim/rating_updaters/elo_extract.py`** (new) — expected-vs-actual rule. Existing `elo.py` and `kpm.py` stay untouched.
- **`mm_sim/engine.py`** — match loop coordinates the new pieces. The split between resolving outcomes, attributing kills, transferring gear, updating ratings, and updating progression stays as small composable steps (one function each) for testability.

Each new module gets focused unit tests covering: typical case, edge cases (no extractors, ties, single-extract, all-extract), parameter extremes (no decay, full decay, no bonus, max bonus), and config-driven behavior changes.

---

## Out of scope (deferred)

These came up during brainstorming but don't make v1 to keep scope tight:

- **Per-player contribution weighting** in rating updates (currently equal split across team).
- **Score-based Elo** (kills factored into rating). Flagged for future after we see whether v1 rating is too coarse.
- **Auto-calibration** of progression rates from target percentile outcomes (we picked plain config knobs over self-calibrating models).
- **Variable team sizes** within a lobby (e.g. mixed 2-player and 4-player teams). Lobby is a clean grid for v1.
- **Per-archetype team behavior** (aggressive vs cautious play styles affecting extract rolls). We picked noise-driven survival variance instead.
- **Dynamic lobby fill timeout** (queue-time-based aggressiveness ramping). The sim uses growing tolerance windows already; we won't add real-time queue economics in v1.

---

## Open questions for review

1. Confirm the `OutcomeConfig.kind = "extraction"` toggle approach (parallel paths) vs. always using the extraction outcome and degrading to 2-team behavior at `teams_per_lobby=2`. The toggle is more conservative for v1.
2. Should historical 2-team scenarios stay valid by default (current proposal) or migrate to the new outcome model? Migration would mean changing the meaning of existing experiment names — we'd want to bump the season suffix.
3. Is `team_size=3` the only multi-team configuration we want to support in v1, or do we want to also handle e.g. `team_size=2, teams_per_lobby=6`? More flexibility = more test surface.
