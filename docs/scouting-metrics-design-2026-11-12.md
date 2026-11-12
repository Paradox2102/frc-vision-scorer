# Scouting metrics from match analysis — design (2026-11-12)

## Goal

Turn finished match replays into per-team scouting metrics for an event, through a
framework that new metrics plug into. First two metrics:

1. **Auto paths** (any season): each team's autonomous path per match, with
   repeated routines grouped into clusters.
2. **Shift heatmaps** (2026 REBUILT only): where a team's robot spends time in
   each hub-shift category (active / inactive / endgame, plus auto and
   transition), optionally split by driver station 1/2/3.

Later: push results to RoboRecon (roborecon.net, repo Paradox2102/RoboRecon3).
Neither side has that API yet. This design prepares a versioned export payload
and a sink interface for it. The push itself is a future task.

## Inputs (what already exists)

- `replays` table: `event_key`, `match_key`, `status`, `meta_path`, `created_at`.
  Several replays can exist for one match (A/B runs). The loader picks one per match.
- `meta.json`, written by the pipeline:
  - `frames[]`: about 5 Hz. Each frame has `frame_num` (source video frame) and
    `robots[]` with `{slot, alliance, team_number, confirmed, field_pos:[x,y]}`.
    Team labels are the final, window-label-filled ones. About 40% of
    robot-time has no name, so every metric reports its coverage.
  - `summary.periods[]`: `{label: auto|teleop, start_frame, end_frame,
    start_seconds, end_seconds}` from the clock scan. This maps a source frame
    to match time.
  - `summary.autonomous_paths[slot]`: auto samples `{frame_num, field_pos,
    occluded, corner_only}`, filtered and smoothed by the pipeline.
- `matches` and `match_robots`: TBA station order (`position` 1–3) and
  `tba_{red,blue}_hub_auto`. For 2026, the hub auto totals decide which alliance
  goes inactive first.
- Field frame: WPILib meters. Origin is the blue-right corner, +x runs toward
  red, and y=0 is the scoring-table side. The 2026 field is **rotationally**
  symmetric (checked from the tag layout: max error 0.3 mm rotated, 3.2 m
  mirrored).

## 2026 shift schedule (WPILib 2026 game-data reference)

Teleop time is the countdown from 140 to 0. Transition is >130 (both hubs
active). Shift 1 is 130–105, Shift 2 is 105–80, Shift 3 is 80–55, Shift 4 is
55–30. Endgame is ≤30 (both active). The alliance that scored **more fuel in
auto** has its hub **inactive in Shifts 1 and 3** and active in 2 and 4. On a
tie, or with no TBA auto data, shift activity is `unknown`. Those samples go to
an `unknown` bucket and are never guessed.

## Architecture: `web/scouting/` package

| module | job |
|---|---|
| `model.py` | dataclasses: `Sample`, `TeamMatch`, `MatchData` |
| `clock.py` | source frame → match time (`t_match` = seconds since auto start, game clock) + phase |
| `geometry.py` | season field size, symmetry (rotate / mirror, read from the tag layout), alliance-perspective transform, 0..1 normalisation |
| `loader.py` | `load_event(event_key) -> list[MatchData]`: picks a replay per match, attributes team identity, attaches stations and hub-auto totals |
| `games/rebuilt2026.py` | shift schedule, hub-active state per alliance per time |
| `registry.py` | `Metric` base class + `@register`; `available_metrics(year)`; `run_metric(event_key, key, params)` with an mtime-keyed cache |
| `export.py` | versioned export envelope; `Sink` protocol; `JsonFileSink`; `RoboReconSink` stub (raises `NotImplementedError`, documents the future contract) |
| `metrics/auto_paths.py` | metric 1 |
| `metrics/shift_heatmap.py` | metric 2 |

Each metric is a class with `key`, `title`, `version`, `years` (None means
universal), `description`, `default_params`, and
`compute(matches, params) -> dict`. A metric returns plain JSON, keyed by team
number as a string. Adding a metric means one file under `metrics/` plus an import
in `metrics/__init__.py`.

### Identity attribution

- **Per-sample:** a robot sample belongs to `team_number` in that frame. Unlabelled
  samples are dropped and counted toward coverage.
- **Auto path:** a `summary.autonomous_paths[slot]` series is assigned to the
  slot's majority team label over the auto-period frames. The path records
  `label_share`. If the runner-up share is ≥ 0.25, it is flagged
  `identity_conflict`. When the summary series is missing, the path is built
  from frames by team label.

### Alliance perspective

Every metric stores positions in **alliance perspective**: the team's own
alliance wall is at x=0. For a red robot on a rotational field, that means
(x, y) → (L−x, W−y). On a mirrored field it means (L−x, y). This makes red-match
and blue-match paths of the same team comparable. Absolute coordinates stay
available on each sample.

## Metric 1: auto paths (`auto_paths`, universal)

- One path per (team, match): samples within the auto period, as
  `[t_auto_s, x, y]` in alliance perspective.
- Quality fields: `coverage` (fraction of auto seconds with a sample),
  `label_share`, `identity_conflict`, `n_samples`.
- **Similarity** compares paths time-aligned on a 0.25 s grid over the auto
  window. A path is linearly interpolated across gaps ≤ 1.5 s. Distance is the
  mean point distance over the overlap, minimised over a time shift of ±1.0 s
  (an auto that starts late should still match). The pair counts as not
  comparable when the overlap is < 40% of the shorter path's span or < 3 s.
- **Grouping** uses average-linkage agglomerative clustering. A threshold
  (default 0.9 m) cuts the tree, and incomparable pairs count as +inf. Each
  cluster reports `members` (match keys), `count`, `representative` (medoid
  path), and `mean_spread_m`. It also reports `start` (median start point) and
  `end`.
- Clusters are sorted by count, largest first.

## Metric 2: shift heatmap (`shift_heatmap`, 2026)

- Grid: 0.5 m cells over the field, in alliance perspective (34 × 17 for 2026).
- Buckets: `auto`, `transition`, `active`, `inactive`, `endgame`, `unknown`.
  The UI shows active, inactive and endgame by default.
- Each sample adds its frame period in seconds to its cell. Output per team:
  `by_station["all"|"1"|"2"|"3"][bucket] = {seconds, cells:
  sparse [[ix, iy, seconds]]}`. It also outputs `matches_by_station` and
  `coverage` (labelled seconds ÷ expected seconds per bucket).
- Derived summaries per bucket and station: `own_half_frac`,
  `opp_half_frac`, `mean_x`, and `lateral` (mean y). These answer questions like
  "do they play defense in the far half?" directly.

## Export envelope (for RoboRecon later)

```json
{"schema": "frc-vision-scorer.scouting", "schema_version": 1,
 "metric": "auto_paths", "metric_version": 1,
 "event_key": "2026cur", "year": 2026, "generated_at": "ISO-8601",
 "frame": {"units": "m", "field_m": [16.541, 8.069],
           "perspective": "alliance",  "symmetry": "rotational",
           "note": "x=0 is the team's own alliance wall; WPILib y; *_norm = 0..1 of full field"},
 "sources": [{"match_key": "2026cur_qm8", "replay_id": "...", "replay_created_at": "..."}],
 "teams": {"3128": {"team_number": 3128, ...metric payload...}}}
```

- Teams are ints, as in RoboRecon's `teamNumber Int`. Matches use full TBA keys
  plus `comp_level`, `set_number` and `match_number`, because RoboRecon drops
  comp_level. Station is `alliance_position` 1–3 in TBA order, which is what
  RoboRecon's `ScoutingReport.alliancePosition` uses.
- Paths also carry `points_norm` (0..1 full-field, no timestamps), the shape of
  RoboRecon's `PathValue`.
- `RoboReconSink.push(payload)` is a stub. The future work is a RoboRecon API
  route authenticated like its cron routes (`Authorization: Bearer <secret>`),
  plus a sink here that POSTs the envelope.

## API and UI

- `GET /api/scouting/{event_key}/metrics` lists the applicable metrics and
  the matches used.
- `GET /api/scouting/{event_key}/metric/{key}?team=` returns the envelope
  (all teams, or one team).
- `GET /api/scouting/{event_key}/export` returns all applicable metrics as
  a list of envelopes.
- `GET /api/scouting/field-image/{year}` serves the AdvantageScope field2d jpg
  plus its corner pixels.
- `/scouting` page, `web/static/scouting.html` (standalone like
  `calibrate.html`): event and team pickers. An auto-paths panel draws the
  clusters in colour on the field, with a list of matches per cluster and
  quality badges. A heatmap panel has bucket and station toggles and shows the
  coverage seconds.

## Error handling and quality

- Matches without `summary.periods` (no clock) are skipped with a reason in
  `sources[].skipped`.
- A metric never invents data. Low coverage and identity conflicts are
  surfaced as fields, not filtered silently.
- Replay choice, newest first within each rank: (1) a full-match replay
  with `window_labels`, then (2) a full-match replay with legacy labels, then
  (3) an auto-only replay with window labels, then (4) any other auto-only
  replay. Auto-only means `config.auto_period_only` is set or there is no
  teleop period. Ranks (2)–(4) are flagged `legacy_labels` when the replay has
  no window labels. This keeps auto-only A/B and ground-truth runs from
  replacing a match's teleop data.

## Testing

Unit tests run on synthetic `MatchData` covering: the clock mapping, 2026 shift
classification (including the tie case), the rotation transform, path
distance with time shift and truncation, clustering of repeated synthetic
routes, heatmap accumulation by station, and the export envelope shape. One
smoke test loads real 2026cur replays when present (skipped otherwise).

## Open question for Nick

- Physical location of driver station 1/2/3 along the alliance wall (which end
  is near the scoring table). Data is split by TBA station number regardless.
  Only the station markers drawn on the heatmap depend on this. It is a
  per-season constant: `STATION_Y_ORDER` in `games/rebuilt2026.py`.

## Findings during the build (2026-11-12)

- **Clock lead.** The clock scan's `start_frame` is 0.5–1.0 s *after* the real
  period start. OCR is sampled every 0.5 s, `start_frame` is the last sample
  still showing 0:20, and the broadcast holds 0:20 for the first second. On 36
  matches, robots that start visibly at rest begin moving a median 0.27 s
  before `start_frame`. 0–5 robots per match are already moving at the first
  processed frame, which is 0.5 s before (`MATCH_CLOCK_LEAD_SECONDS`).
  Scouting applies `CLOCK_LEAD_S = 1.0` (`web/scouting/clock.py`; a 25-match
  follow-up measured a median lag of 1.04 s). The loader
  also prepends each slot's lead-window frames to its auto path. Median auto
  start moved from 5.3 m to 4.5 m from the own wall. Fully capturing auto
  starts needs the pipeline to process more lead (pipeline follow-up).
- **Mirror-aware auto grouping.** Teams run one routine from either trench
  side. For example, 2992 qm23 vs qm36 is 3.0 m apart as-is and 0.32 m apart
  reflected. Clustering compares both orientations (`mirror_aware`, default
  on) and tags members `mirrored`. On 2026cur, 6 teams show a repeated routine.
- **Shift heatmap sanity check.** Pooled over 2026cur, robots are in their own
  half 76% of hub-active time vs 55% of hub-inactive time. 29 of 31 teams
  with ≥30 s in both shift types show the drop.
- **Station geometry.** Auto start y trends high→low for stations 1→3, for
  both alliances, in alliance perspective. That is weak evidence for
  `STATION_Y_ORDER`. It is still marked unverified.
