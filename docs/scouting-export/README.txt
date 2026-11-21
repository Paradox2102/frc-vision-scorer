# Scouting export: import guide

This guide is for developers of other scouting systems, such as RoboRecon, who
want to import our data without reading our code. It covers everything in the
file.

| File | What it is |
|---|---|
| `README.md` | This guide. |
| `scouting-export.schema.json` | JSON Schema (draft 2020-12) for the whole file. Every property has a `description`. |
| `example-match.json` | A small, schema-valid sample built from real data: match `2026cur_qm8` plus team 3128 in each metric envelope. Lists are cut down, so counts such as `n_matches` do not match the list lengths. |

## What the file is

frc-vision-scorer watches FRC match videos (the TBA / YouTube broadcast and
any extra cameras). It tracks each robot on the field, works out which team
each robot is, and turns those tracks into scouting metrics. The **whole-event
export** packs one event's results into a single JSON document:

* `GET /api/scouting/{event_key}/export` on a running server, or
* `scripts/export_scouting.py <event_key>` →
  `data/exports/<event_key>_scouting_export.json`.

Add `?paths=false` (script: `--no-paths`) to leave out every robot's
whole-match path, which is more than half the file.

For **one match**, `GET /api/scouting/{event_key}/match/{match_key}` returns
that match's `matches[]` entry (lineup, video timeline, robot paths) wrapped
in a small header; add `?team=N` for a single robot. See
[Robot paths](#robot-paths). `scripts/export_scouting.py <event_key>
--per-match` writes every match's file at once, to
`data/exports/<event_key>_matches/<match_key>.json`.

The reference file is `data/exports/2026cur_scouting_export.json`: 2026 Curie
division, 136 analysed matches, 74 teams, 17.5 MB of compact JSON. The
biggest parts are the robot paths (`matches[].robots`, about 10 MB),
`auto_paths` (about 3.9 MB) and `shift_heatmap` (about 3.2 MB). Without robot
paths the file is 7.4 MB.

All the numbers come from computer vision. Expect gaps where the cameras lost a
robot, and occasional robots credited to the wrong team. Each metric has
coverage and data-quality fields. Show them to users, or use them to filter.

## Top-level layout

```jsonc
{
  "schema": "frc-vision-scorer.scouting-export",   // file-type tag
  "schema_version": 2,                              // whole-file layout version
  "event_key": "2026cur",                           // TBA event key
  "year": 2026,
  "generated_at": "2026-11-20T23:04:11+00:00",      // UTC
  "path_frame": { /* coordinate frame of matches[].robots[].path (absent with ?paths=false) */ },
  "matches":   [ /* one per analysed match: lineup + video timeline + robot paths */ ],
  "skipped":   [ /* matches whose replay could not be used */ ],
  "envelopes": [ /* one per metric: shift_heatmap, auto_paths, movement, cycles, defense, bump_stuck */ ]
}
```

### `matches[]`

Each entry is one analysed match, in play order: qm, then ef/qf/sf/f, sorted by
`set_number` and then `match_number`.

| Field | Type | Meaning |
|---|---|---|
| `match_key` | string | TBA match key, e.g. `2026cur_qm8`, `2026cur_sf4m1`, `2026cur_f1m2`. |
| `comp_level` | `"qm"\|"ef"\|"qf"\|"sf"\|"f"` | TBA comp level. |
| `set_number` | int | TBA set number (always 1 for qualifications). |
| `match_number` | int | TBA match number (the qual number, or the game within a playoff set). |
| `replay_id` | string (UUID) | Our analysis run for this match. It changes when the match is re-analysed. |
| `replay_created_at` | string \| null | When that run was made: ISO-8601 local server time with **no timezone**. Informational only. |
| `legacy_labels` | bool | `true` means an older run whose team identification is less reliable. |
| `alliances` | `{red: [int\|null ×3], blue: [...]}` | TBA lineup in station order 1, 2, 3. |
| `timeline` | object \| null | Where each phase is in the match video. See [The video timeline](#the-video-timeline). |
| `robots_version` | int | `1`. Version of the `robots` block. Absent with `?paths=false`. |
| `auto_s` | number | Match seconds of auto (2026: 20). Path times `t < auto_s` are auto, the rest teleop. Absent with `?paths=false`. |
| `robots` | array | Every robot's position over the whole match. See [Robot paths](#robot-paths). Absent with `?paths=false`. |

### `skipped[]`

`{match_key, replay_id?, reason}` lists a match that has a replay we could not
use. No data from a skipped match appears anywhere else in the file. In 2026cur
two matches are skipped, `qm32` and `qm72`, both with
`"no match clock (summary.periods)"`. A match with no replay at all is simply
absent from the file.

### `envelopes[]`

Each envelope holds one metric for the whole event. Identify it by its
`metric` key, not by its position in the array. Every envelope has these common
keys:

| Field | Meaning |
|---|---|
| `schema` / `schema_version` | `"frc-vision-scorer.scouting"` / `1`: the envelope's own layout version. |
| `metric` / `metric_version` / `metric_title` | Metric key, payload version and display name. |
| `event_key`, `year`, `generated_at` | Same meaning as the top-level fields. |
| `frame` | The coordinate frame: `{units:"m", field_m:[L,W], perspective:"alliance", symmetry, note}`. |
| `params` | The tuning parameters this metric ran with, kept so the run can be reproduced. You can usually ignore them. |
| `sources` | `[{match_key, comp_level, set_number, match_number, replay_id, replay_created_at, legacy_labels}]`. The same set of matches as `matches[]`, but **sorted by match_key string** (`f1m1, f1m2, qm1, qm10, qm100, …`), not in play order. |
| `skipped` | Same as the top-level `skipped`. |
| `teams` | `{"<team number>": {team_number, ...}}`: one entry for every team in any analysed lineup, including teams the cameras never saw. |

The remaining keys are specific to each metric. See [Metrics](#metrics).

## Identifiers

* **Event and match keys** are The Blue Alliance keys, unchanged. RoboRecon
  stores only a match number, so build its match identity from
  `comp_level` + `set_number` + `match_number`. A qualification number alone
  clashes with playoff numbers.
* **Teams** are integers wherever they are values (`team_number`, the
  `alliances` arrays). JSON object keys must be strings, so `teams` is keyed by
  the decimal team number as a string (`"3128"`). `ranking` arrays hold those
  same string keys.
* **Stations**: `station` is 1, 2 or 3 in TBA order, meaning the index in TBA's
  `alliances.<colour>.team_keys` plus 1. This is RoboRecon's
  `alliancePosition`. `alliance` is `"red"` or `"blue"`. Both are nullable in the
  schema, but are always set for lineup teams in practice.
* **Upsert keys**: use `(event_key, match_key)` for matches and
  `(event_key, metric, team_number)` for envelope team entries. If a match's
  `replay_id` changes between two exports, that match was re-analysed and its
  numbers may have changed.

## The coordinate frame

Positions are in **metres**, on a field of `frame.field_m = [L, W]`
(2026: `[16.541, 8.069]`).

**Alliance perspective** is the default, and every envelope's `frame.perspective`
says so. Under it:

* `x = 0` is the team's **own** alliance wall, and `+x` points toward the
  opponent's wall.
* `y` is WPILib's y after the symmetry transform. On a rotational field
  (2025, 2026), `y = 0` is on the team's drivers' right-hand side and `+y` is to
  their left.
* Blue positions are exactly WPILib field coordinates (origin at the blue wall,
  on the blue drivers' right). Red positions are transformed according to
  `frame.symmetry`:
  * `rotational` (2025, 2026): `(x, y) → (L − x, W − y)`
  * `mirror` (2023, 2024): `(x, y) → (L − x, y)`. Here the left/right
    handedness flips for red.

Because of this transform, a team's red-alliance and blue-alliance matches
overlay directly: one heatmap or path picture per team, with its own wall
always on the left.

Two fields use **absolute WPILib field coordinates** (blue wall at x = 0)
instead:

* `auto_paths` → `matches[].points_field`, which shows where the robot really
  drove.
* `bump_stuck` → `bumps`, the four bump rectangles.

The `*_norm` values (`auto_paths.points_norm`) are alliance-perspective
positions divided by `[L, W]`, so they run from 0 to 1 across the full field.
This is the shape of RoboRecon `PathValue` stroke points.

## The video timeline

`matches[].timeline` tells you where each phase sits **in the match video**, so
you can seek a player to "teleop start" or "shift 3" without running anything
yourself.

```jsonc
"timeline": {
  "timeline_version": 1,
  "video": {"fps": 60.0, "fps_source": "video", "frame_count": 12870, "duration_s": 214.5,
            "youtube_key": "Pialfz0joK8",
            "youtube_url": "https://www.youtube.com/watch?v=Pialfz0joK8&t=6s"},
  "marks": {
    "match_start":   {"t": 6.217,  "frame": 373,   "video_time": "0:06.22", "source": "indicator"},
    "auto_end":      {"t": 26.25,  "frame": 1575,  "video_time": "0:26.25", "source": "clock_ticks(6)"},
    "teleop_start":  {"t": 30.25,  "frame": 1815,  "video_time": "0:30.25", "source": "clock_ticks(6)"},
    "endgame_start": {"t": 141.25, "frame": 8475,  "video_time": "2:21.25", "source": "clock_ticks(13)"},
    "match_end":     {"t": 171.75, "frame": 10305, "video_time": "2:51.75", "source": "clock_ticks(7)"}
  },
  "phases": [
    {"phase": "auto",       "start_s": 6.217,  "end_s": 26.25,  "duration_s": 20.033, "start_frame": 373,  "end_frame": 1575},
    {"phase": "between",    "start_s": 26.25,  "end_s": 30.25,  "duration_s": 4.0,    "start_frame": 1575, "end_frame": 1815},
    {"phase": "teleop",     "start_s": 30.25,  "end_s": 171.75, "duration_s": 141.5,  "start_frame": 1815, "end_frame": 10305},
    {"phase": "transition", "start_s": 30.25,  "end_s": 40.25,  "duration_s": 10.0,   "start_frame": 1815, "end_frame": 2415, "within": "teleop"},
    {"phase": "shift1",     "start_s": 40.25,  "end_s": 65.75,  "duration_s": 25.5,   "start_frame": 2415, "end_frame": 3945, "within": "teleop"},
    {"phase": "shift2",     "start_s": 65.75,  "end_s": 90.75,  "duration_s": 25.0,   "start_frame": 3945, "end_frame": 5445, "within": "teleop"},
    {"phase": "shift3",     "start_s": 90.75,  "end_s": 116.25, "duration_s": 25.5,   "start_frame": 5445, "end_frame": 6975, "within": "teleop"},
    {"phase": "shift4",     "start_s": 116.25, "end_s": 141.25, "duration_s": 25.0,   "start_frame": 6975, "end_frame": 8475, "within": "teleop"},
    {"phase": "endgame",    "start_s": 141.25, "end_s": 171.75, "duration_s": 30.5,   "start_frame": 8475, "end_frame": 10305, "within": "teleop"}
  ]
}
```

### What the times mean

* **Video seconds.** Every time value `t` is `frame / video.fps`, measured from
  the first frame of the source video file (the TBA match video). The YouTube
  video named by `youtube_key` is the same footage, so seeking a YouTube player
  to `t` lands on the same moment. `youtube_url` already carries
  `&t=<floor(match_start.t)>s`, which opens the video at the start of the
  match.
* **`frame` = round(t × fps)**, a 0-based frame index into that file. Always
  use `video.fps`. It is often not exactly 60: 2026cur has 59.9401, 59.716,
  59.498 and others. Seconds are usually the more useful unit; frames are there
  for tools that step through the file frame by frame.
* **`video_time`** is `t` formatted as `M:SS.ss`, for display only.

### How the boundaries are measured (`source`)

| `source` | Meaning | Accuracy |
|---|---|---|
| `indicator` | `match_start` only: the frame where the broadcast scoreboard's start indicator lights. | frame-accurate |
| `clock_ticks(N)` | Fitted from the N on-screen clock ticks nearest this boundary, minus a 0.5 s display lag (the overlay clock trails the field). | about ±0.25 s |
| `clock_offset`, `true_start_frame` | `match_start` taken from the pipeline's own start estimate (clock-scan start minus a fixed lead). | roughly ±0.5 s |
| `clock_lead` | Fallback from the clock scan's first-sample frame. | roughly ±0.5 s |
| `<source>+<N>s` | No local clock fit was possible, so this is another mark plus N seconds of rule time, e.g. `indicator+20s`. | inherits the base's accuracy, plus clock drift |

2026cur uses `indicator` for 135 of the 136 match starts. The other one, qm76,
uses `clock_ticks(6)`. Every other mark in the file is `clock_ticks(N)`.

### Durations are video seconds, not rule lengths

Each boundary is fitted **locally** from the clock ticks around it. The
broadcast clock runs about 1% slow against the video, so phase durations come
out a little longer than the rules say. Across 2026cur (median, then range):

| phase | rule | video seconds |
|---|---|---|
| auto | 20 | 20.08 (19.47–20.90) |
| between | – | 4.03 (4.00–4.60) |
| teleop | 140 | **141.5** (141.0–144.5) |
| transition | 10 | 10.01 |
| shift1–4 | 25 each | 25.25 (25.0–26.0) |
| endgame | 30 | 30.5 (30.0–31.0) |

Do **not** compute `teleop_start + 140` to find the end of the match. Use the
marks.

### Phases

* The top-level phases are `auto`, `between` (the pause between auto and
  teleop) and `teleop`.
* Sub-phases have `"within": "teleop"`. In 2026 these are the REBUILT hub
  shifts, `transition`, `shift1`, `shift2`, `shift3`, `shift4` and `endgame`
  (teleop countdown 140→130, 130→105, 105→80, 80→55, 55→30, 30→0). Other
  seasons have only `endgame` (from 30 s remaining in 2022–2023 and 2026, or
  20 s in 2024–2025).
* Phases are sorted by `start_s`. A parent comes before the sub-phase that
  starts at the same moment.
* The `endgame` phase and the `endgame_start` mark are the same instant.

### When parts are missing

* `timeline` can be `null` when the replay has no usable match clock or frame
  rate. This does not happen in 2026cur, because those matches are in
  `skipped[]` instead.
* **Auto-only replays** have only the `match_start` and `auto_end` marks, and
  only the `auto` phase. In 2026cur this is qm76, where only auto was analysed.
  The schema enforces the grouping: `match_start` comes with `auto_end`, and
  `teleop_start`, `endgame_start` and `match_end` come together.
* `fps_source: "clock_estimate"` means the video file was not available. The
  frame rate was estimated from the clock scan (within about 2%), so times are
  rougher, and `frame_count` and `duration_s` are `null`. Every 2026cur match
  has `fps_source: "video"`.
* `youtube_key` and `youtube_url` are `null` when TBA lists no video.

### Worked example: 2026cur qm8

* The match starts at video second **6.217**, which is frame 373 at 60 fps
  (6.217 × 60 = 373.02). The source is `indicator`, so this is
  frame-accurate. `youtube_url` opens at `&t=6s`.
* Auto ends at 26.25. Teleop runs from **30.25** to **171.75**, which is
  141.5 s of video for a 140 s period. Endgame starts at **141.25**
  (`2:21.25`).
* To jump to shift 3, seek to `phases[shift3].start_s` = 90.75 (or frame 5445).
* **Converting a game-clock time from a metric into video time.** Some metric
  fields are game-clock times since auto start, with the pause removed: auto
  runs 0–20 and teleop 20–160 in 2026. Examples are
  `cycles.matches[].cycle_return_times`, `bump_stuck.episode_list[].t_start`
  and `auto_paths.points[][0]`. Interpolate inside the matching phase:

  ```
  auto   (t <= 20): video = match_start.t  + t        × auto.duration_s   / 20
  teleop (t >  20): video = teleop_start.t + (t − 20) × teleop.duration_s / 140
  ```

  Team 3128's qm8 cycle that returned at game time 142.7 is at
  30.25 + 122.7 × 141.5 / 140 ≈ **154.3 s** of video. This is good to about
  1 s, which is enough to seek a video player.

## Robot paths

Each `matches[]` entry carries `robots`: one entry per robot in the match,
with where that robot was for the **whole match** (auto and teleop). Use it
to replay a match, draw a robot's path in another app, or compute your own
metrics. The reference file has 816 robot entries; 811 have points.

```jsonc
"robots_version": 1,
"auto_s": 20.0,
"robots": [
  {"team_number": 3128, "alliance": "blue", "alliance_position": 1,
   "path": {
     "n_points": 1044,
     "seen_s": {"auto": 17.5, "teleop": 85.9},   // seconds the robot was located
     "first_t": 0.08, "last_t": 157.13,
     "auto_source": "autonomous_paths",          // or "frames", or null
     "auto_label_share": 1.0,                    // null when auto_source != "autonomous_paths"
     "confirmed_share": 0.68,
     "segments": [[0, 38], [39, 188], ...],      // index ranges, inclusive
     "n_jumps": 20,
     "t":     [0.08, 0.18, 0.28, ...],           // match seconds
     "x":     [3.36, 3.36, 3.35, ...],           // metres, absolute field frame
     "y":     [2.0, 2.0, 2.0, ...],
     "frame": [378, 384, 390, ...]               // source-video frame
   }},
  ...
]
```

**Order.** Lineup order: red stations 1, 2, 3, then blue 1, 2, 3. If TBA does
not list a match's lineup, any team we identified follows, sorted by number.
A lineup robot we never identified still has an entry, with `n_points: 0` and
empty arrays.

**Columns.** `t`, `x`, `y` and `frame` are parallel arrays of length
`n_points`, sorted by time, one point per analysed video frame (in 2026cur
about 10 per second in auto and 5 per second in teleop).

* `t` is **match time** in seconds: auto runs `0..auto_s`, and teleop runs
  `auto_s..auto_s + 140` (2026). The auto→teleop pause is removed, so `t` is
  continuous. It is the same clock as `t_start` in the metrics.
* `x`, `y` are in the file's top-level **`path_frame`**: absolute WPILib field
  coordinates in metres, for red and blue robots alike (origin at the blue
  wall's right corner as the blue drivers see it, `+x` toward the red wall,
  `+y` to the blue drivers' left). This is **not** the metrics' alliance
  perspective. Divide by `path_frame.field_m` for 0..1 values.
* `frame` is the source-video frame. Video seconds = `frame / timeline.video.fps`
  (`frame / fps` is how the timeline's own `frame` values are defined too).

**Gaps and segments.** A point exists only where the robot was located *and*
identified as this team. Do not draw a line across a gap. `segments` lists
`[first, last]` index ranges (inclusive) that are safe to draw as one
polyline. A new segment starts when either:

* consecutive points are more than **1.0 s** apart (the robot was lost), or
* the step between them is faster than **7 m/s**. No FRC robot drives that
  fast, so the label has jumped to another robot or the position is bad.
  `n_jumps` counts these breaks.

In 2026cur a robot has a median of 14 segments, the median segment covers
4.9 s, and 85% of located time is in segments of 5 s or longer. If you need
a cleaner path, drop very short segments, or segments that sit far from both
neighbours.

**Where the points come from.**

* **Auto**: when `auto_source` is `"autonomous_paths"`, auto points come from
  the pipeline's per-slot auto track. It is followed continuously from the
  start position and split wherever the slot changed robots, so it is the
  most reliable part of the path. `auto_label_share` is the share of that
  track's team readings that named this team. With `"frames"`, auto points
  are the frames labelled with this team, the same source as teleop.
* **Teleop**: frames whose tracked robot was labelled with this team.
  `confirmed_share` is the share of those labelled frames where the team
  number was read directly, rather than inferred from the lineup or filled in
  between reads.
* `seen_s` is the located time per period.

These are the same positions the metrics use. The `auto_paths` metric applies
its field rules (hub/trench clamping, start-line check) on top of them, so
its `points_field` can differ slightly from the auto part here.

### Python: write one robot's path as CSV

```python
import csv, json, sys

d = json.load(open("2026cur_scouting_export.json"))
match_key, team = "2026cur_qm8", 3128
m = next(m for m in d["matches"] if m["match_key"] == match_key)
r = next(r for r in m["robots"] if r["team_number"] == team)
p, fps = r["path"], (m["timeline"] or {}).get("video", {}).get("fps")

w = csv.writer(sys.stdout)
w.writerow(["segment", "t", "period", "x_m", "y_m", "video_s"])
for seg, (a, b) in enumerate(p["segments"]):
    for i in range(a, b + 1):
        w.writerow([seg, p["t"][i], "auto" if p["t"][i] < m["auto_s"] else "teleop",
                    p["x"][i], p["y"][i], round(p["frame"][i] / fps, 3) if fps else ""])
```

### The single-match file

`GET /api/scouting/{event_key}/match/{match_key}[?team=N]` returns:

```jsonc
{
  "schema": "frc-vision-scorer.scouting-match",
  "schema_version": 2,          // same layout version as the event file
  "event_key": "2026cur", "year": 2026,
  "generated_at": "2026-11-21T17:30:00+00:00",
  "path_frame": { ... },
  "match": { /* exactly one matches[] entry, robots included */ }
}
```

`?team=N` keeps only that robot in `match.robots`. The endpoint answers 404
for a match with no usable replay (the `detail` says why), and for a team that
is not in the match.

## Metrics

Each subsection below lists the per-team fields and the metric-level extras,
with a cut-down example from team 3128 in 2026cur. `example-match.json` has
each of them in full shape.

### Shared conventions

* **`teams[*].matches[]`** has one row per match the team was in the lineup
  for, in play order. Each row starts with the same header:
  `match_key, comp_level, set_number, match_number, alliance, station`.
* **Coverage** is tracked seconds divided by the seconds that could have been
  tracked, from 0 to 1. Cameras lose robots, so low coverage means "not seen",
  not "did nothing". Filter on it or show it.
* **The ranking metrics** (`movement`, `cycles`, `defense`, `bump_stuck`) share
  extra rules:
  * A match row has `counted: true` only when it has at least
    `params.min_tracked_s` (30 s) of valid tracking. Only counted matches feed
    the team's averages.
  * Rates are pooled over the counted matches (sum of numerators divided by sum
    of denominators).
  * Each team entry has `n_matches`, `n_counted`, `low_data`
    (`n_counted < params.min_matches`), `coverage` and `rank` (1 is best;
    `null` when the team cannot be ranked).
  * The envelope adds `ranking` (team keys, best first, unrankable teams last),
    `score_key` (the field the ranking sorts on) and `legend` (a plain-English
    explanation with the parameter values filled in).
* **Nulls**: averages, fractions and speeds are `null` when there is nothing to
  average: no counted matches, no tracked time, or no path.
* **Rounding**: seconds are usually rounded to 0.1 s, fractions to 3 decimals
  and speeds to 0.01 m/s. Integer-valued numbers can appear as `4.0`. Treat
  every non-count field as a float.
* All positions and zone bounds are in alliance perspective unless the field's
  description says otherwise.

### `shift_heatmap` (2026, metric_version 2)

This metric shows where each robot spends time during each 2026 hub **shift
bucket**:

* `auto`
* `transition`
* `active`: the team's own hub is active (shifts 1–4).
* `inactive`: the team's own hub is inactive (shifts 1–4).
* `endgame`
* `unknown`: shifts 1–4 when the auto winner is not known.

The alliance that scored more auto FUEL (from TBA) has its hub inactive in
shifts 1 and 3. A tie or missing TBA data makes those shifts `unknown`.

Per team:

| Field | Meaning |
|---|---|
| `n_matches` | Matches the team was in the lineup for. |
| `matches[]` | Header + `auto_winner` (`red`/`blue`/`null`) + `seconds` {bucket → labelled seconds in that match}. |
| `matches_by_station` | `{"1","2","3"}` → number of matches played from that station. |
| `by_station` | `"all"`, plus `"1"`..`"3"` for each station the team played. Each value maps every bucket to `{seconds, cells, summary}`. |
| `…cells` | A sparse grid: `[ix, iy, seconds]` with `ix = floor(ax / grid.cell_m)` and `iy = floor(ay / grid.cell_m)`. The cell centre is `((ix+0.5)·cell_m, (iy+0.5)·cell_m)`. |
| `…summary` | `own_zone_frac`, `neutral_zone_frac`, `opp_zone_frac` (share of the time in each of the three zones, split at `zones`), `mean_ax`, `mean_ay`, `n_samples`. These are `null` with no samples. |
| `coverage` | Bucket → labelled seconds divided by the seconds that could have been seen. `null` when nothing could have been seen, e.g. `unknown` when every match had a known winner. |

Metric-level extras:

* `grid`: `{cell_m: 0.5, nx: 34, ny: 17, field_m}`.
* `zones`: `{own_max_x: 4.029, opp_min_x: 12.512}`.
* `buckets` and `default_buckets`.
* `auto_winner_by_match`: match_key → `red` / `blue` / `null`.
* `station_layout`: `{"1":"high","2":"mid","3":"low"}`. This is a drawing hint
  only and has not been checked against the game manual.

```jsonc
"3128": {"team_number": 3128, "n_matches": 13, "matches_by_station": {"1": 3, "2": 7, "3": 3},
  "matches": [{"match_key": "2026cur_qm8", ..., "alliance": "blue", "station": 1, "auto_winner": "blue",
               "seconds": {"auto": 20.16, "transition": 6.95, "active": 19.26, "inactive": 41.9, "endgame": 21.45, "unknown": 0.0}}],
  "by_station": {"all": {"active": {"seconds": 390.87, "cells": [[0, 5, 0.6], [0, 6, 0.4], ...],
      "summary": {"own_zone_frac": 0.511, "neutral_zone_frac": 0.489, "opp_zone_frac": 0.0,
                  "mean_ax": 4.591, "mean_ay": 4.076, "n_samples": 2064}}, ...}, "1": {...}, ...},
  "coverage": {"auto": 0.761, "transition": 0.723, "active": 0.601, "inactive": 0.822, "endgame": 0.779, "unknown": null}}
```

### `auto_paths` (all seasons, metric_version 2)

This metric gives each team's autonomous path in every match, and groups
repeated paths into **routines** (clusters). Paths are compared in a
mirror-aware way: the same routine run from the left or the right trench
counts as one routine.

Per team: `n_matches`, `n_with_path`, `matches[]` and `clusters[]`.

Each row of `matches[]` has the header plus these fields:

| Field | Meaning |
|---|---|
| `points` | `[[t, ax, ay], …]`: `t` is game-clock seconds since auto start (0–20); positions are alliance-perspective metres. |
| `points_norm` | `[[ax/L, ay/W], …]`, the same points without time (RoboRecon `PathValue`). |
| `points_field` | `[[t, x, y], …]` in **absolute** field metres. Use these to draw red autos on the red side. |
| `point_kinds` | One letter per point: `s` seen, `c` clamped out of an obstacle, `d` detour point inserted to route around an obstacle, `i` inferred start inserted on the starting line. |
| `n_samples` | Number of **observed** points. `points` can be longer because of inserted `d`/`i` points (true for 411 of 816 rows), so iterate over `points.length`. |
| `coverage` | Share of the 20 s auto actually seen. |
| `source` | `autonomous_paths` (the pipeline's smoothed auto track), `frames` (rebuilt from per-frame labels) or `null` (no path). |
| `label_share`, `runner_up_share`, `identity_conflict` | How confidently the path is attributed to this team. `identity_conflict` means `runner_up_share ≥ 0.25`. |
| `start_check`, `start_offset_m` | 2026 starting-line check: `ok`, `inferred` (a start point was added on the line), `unseen` (first seen after 1.5 s), `violation` (almost certainly another robot; never clustered) or `null`. |
| `n_clamped`, `n_detour` | Points moved out of an obstacle, and detour points inserted around one. |
| `cluster`, `mirrored` | The cluster `id` this path belongs to (or `null`), and whether the path is a left/right reflection of the cluster's representative. |
| `legacy_labels` | Copied from the match. |

A team has one row per lineup match even when there is no path. Those rows
have empty arrays, `n_samples: 0`, and `source`, `start_check` and `cluster`
all `null`.

Each entry in `clusters[]` has these fields:

* `id` (0 is the most frequent routine) and `count`.
* `members` (match keys) and `representative` (the medoid match: draw this one
  as "the routine").
* `mean_spread_m` and `n_mirrored`.
* `start` and `end`: median `[ax, ay]`.
* `duration_s`.

The only metric-level extra is `legend`.

```jsonc
{"match_key": "2026cur_qm8", ..., "alliance": "blue", "station": 1, "source": "autonomous_paths",
 "coverage": 0.975, "label_share": 1.0, "runner_up_share": 0.0, "identity_conflict": false,
 "n_samples": 188, "start_check": "ok", "start_offset_m": 0.0, "n_clamped": 1, "n_detour": 2,
 "cluster": 1, "mirrored": true,
 "points": [[0.08, 3.36, 2.0], [0.18, 3.36, 2.0], ... /* 190 points */ , [19.98, 2.94, 2.17]],
 "points_norm": [[0.2032, 0.2484], ...], "points_field": [[0.08, 3.36, 2.0], ...], "point_kinds": ["s", "s", ...]}
// clusters[1]:
{"id": 1, "count": 4, "members": ["2026cur_qm8", "2026cur_qm50", "2026cur_sf4m1", "2026cur_sf10m1"],
 "representative": "2026cur_sf4m1", "mean_spread_m": 0.67, "n_mirrored": 2,
 "start": [3.7, 6.74], "end": [2.91, 5.95], "duration_s": 19.62}
```

### `movement` (2026, metric_version 1): "Time moving"

This metric measures how much of the match each robot spends driving, and how
fast. A robot is moving when its smoothed speed is at least
`params.move_speed_mps` (0.5 m/s).

* **Rank score:** `avg_speed_mps`, the average speed over all time where speed
  is known, counting stationary time as 0.
* **Per team, whole match:** `moving_s_per_match`, `stationary_s_per_match`,
  `moving_frac`, `mean_moving_mps`, `median_moving_mps`, `p90_moving_mps`,
  `avg_speed_mps`, `speed_known_s`, `tracked_s_per_match`,
  `max_sustained_mps` (median over matches of the top speed held for 1.5 s) and
  `max_sustained_best_mps`.
* **Per team, teleop only:** the same summary inside `teleop`.
* **Rows:** `tracked_s`, `coverage` (of 160 s), `held_s` (seconds the tracking
  stalled) and `counted`. Only **counted rows** also have `moving_s`,
  `stationary_s`, `moving_frac`, `avg_speed_mps`, `max_sustained_mps` and
  `teleop {moving_s, stationary_s, moving_frac, avg_speed_mps, tracked_s}`.
* **Extras:** `event_stats` {`labelled_s`, `held_s`, `held_frac`,
  `held_kept_s`, `implausible_speed_samples`} gives data quality for the whole
  event.

```jsonc
"3128": {"n_matches": 13, "n_counted": 13, "low_data": false, "rank": 1, "avg_speed_mps": 1.281,
  "moving_frac": 0.741, "mean_moving_mps": 1.73, "max_sustained_mps": 2.5, "coverage": 0.712, ...,
  "matches": [{"match_key": "2026cur_qm8", ..., "tracked_s": 108.2, "coverage": 0.676, "held_s": 1.5,
     "counted": true, "moving_s": 69.1, "stationary_s": 24.8, "moving_frac": 0.736, "avg_speed_mps": 1.378,
     "teleop": {"moving_s": 57.2, "stationary_s": 17.1, "moving_frac": 0.77, "avg_speed_mps": 1.436, "tracked_s": 88.1},
     "max_sustained_mps": 2.72}]}
```

### `cycles` (2026, metric_version 1): neutral ↔ alliance-zone trips

This metric counts collect-and-return trips. A cycle is a neutral-zone visit
followed by a visit to the team's own alliance zone. Each zone needs 1 s of
dwell to count, which stops jitter at the boundary from creating cycles. A
tracking gap longer than 3 s resets the counter (a "reset"), so
low-coverage matches **undercount**.

* **Rank score:** `cycles_per_min`, teleop cycles per tracked teleop minute.
* **Per team:** `cycles_teleop_per_match`, `cycles_auto_per_match`,
  `cycles_teleop_total`, `cycles_auto_total`, `teleop_tracked_s_per_match`,
  `mean_cycle_s`, `median_cycle_s` (neutral entry to the next neutral entry),
  `mean_neutral_visit_s`, `mean_own_visit_s` and `resets`.
* **Rows:** `teleop_tracked_s`, `coverage` (of 140 s), `counted`,
  `cycles_teleop`, `cycles_auto`, `resets`, `cycles_per_min` and
  `cycle_return_times`. The return times are game-clock seconds when each
  return to the own zone began; to turn one into video time, see the
  [worked example](#worked-example-2026cur-qm8).
* **Extras:** `zones` {`own_max_x`: 3.729, `neutral_min_x`: 5.522,
  `neutral_max_x`: 11.019}.

```jsonc
{"match_key": "2026cur_qm8", ..., "teleop_tracked_s": 88.1, "coverage": 0.629, "counted": true,
 "cycles_teleop": 2, "cycles_auto": 2, "resets": 5, "cycles_per_min": 1.36,
 "cycle_return_times": [5.9, 15.3, 48.3, 142.7]}
```

### `defense` (2026, metric_version 1): time on the opponent's side

This metric is a proxy for defense: the robot's teleop time with
`ax >= zones.opp_min_x` (11.512). That is inside the opponent's alliance zone
or within 1 m of it. Being there does not prove the robot was defending; it
could also be feeding or have been pushed there.

* **Rank score:** `opp_side_frac`, the share of tracked teleop time.
* **Per team:** `opp_side_s_per_match` (raw seconds), `opp_side_s_norm`
  (`opp_side_frac` × 140 s), `matches_long` (matches with 15 s or more there),
  `teleop_tracked_s_per_match` and `by_station`. `by_station` is keyed
  `"red1"`…`"blue3"`, with `{n_counted, opp_side_s_per_match, opp_side_frac}`
  for each key.
* **Rows:** `teleop_tracked_s`, `coverage`, `counted`, `opp_side_s` and
  `opp_side_frac`.

```jsonc
"3128": {"opp_side_frac": 0.015, "opp_side_s_per_match": 1.4, "opp_side_s_norm": 2.0, "matches_long": 0, "rank": 71,
  "by_station": {"blue1": {"n_counted": 2, "opp_side_s_per_match": 1.1, "opp_side_frac": 0.013}, ...}, ...}
```

### `bump_stuck` (2026, metric_version 2): "Stuck on a bump"

This metric measures how long a robot is stopped on a bump (beached): seconds
with the robot's smoothed centre on any of the four bumps and a smoothed speed
below 0.5 m/s, added up over the match. Driving over a bump at speed does not
count. A robot can also park on a bump on purpose, so check the video.

* **Rank score:** `stuck_s_per_match`. A larger value means the robot is
  beached more often.
* **Per team:** `stuck_s_total`, `episodes`, `matches_with_episode`,
  `longest_s`, `matches_stuck` (matches with 1 s or more), `tracked_s_per_match`
  and `episode_list[]`.
* **Each `episode_list[]` entry** is a stretch of at least 2 s stopped on one
  bump, for finding it in the video. It has the header plus these fields:
  * `period` (`auto`/`teleop`) and `t_start` (game-clock seconds).
  * `remaining_at_start`: what the on-screen clock showed.
  * `duration_s` and `stopped_s`.
  * `bump`: `own`/`opponent` + `left`/`right`, as seen from the team's own
    driver station.
  * `bump_field_side`: `red`/`blue`, the absolute half of the field.
  * `counted`.
* **Rows:** `tracked_s`, `coverage`, `counted`, `n_episodes` and `stuck_s`.
* **Extras:** `bumps`, four `[x0, y0, x1, y1]` rectangles in **absolute**
  field metres.

```jsonc
{"match_key": "2026cur_qm39", ..., "alliance": "blue", "station": 3, "period": "auto", "t_start": 7.2,
 "remaining_at_start": 12.8, "duration_s": 14.9, "stopped_s": 13.8, "bump": "own right",
 "bump_field_side": "blue", "counted": true}
```

## Versioning

There are four version numbers. Each one covers a different part of the file
and changes only when that part makes a **breaking** change: a key is removed
or renamed, a type changes, or a meaning or unit changes.

| Version | Where | Current | Covers |
|---|---|---|---|
| `schema_version` | top level | 2 | The whole-file layout: header, `matches[]`, `skipped[]`, `envelopes[]`. Version 2 added the header and `matches[]`. |
| `schema_version` | each envelope | 1 | The envelope's common keys (`frame`, `sources`, `teams` keyed by team, …). |
| `metric_version` | each envelope | shift_heatmap 2, auto_paths 2, movement 1, cycles 1, defense 1, bump_stuck 2 | That metric's own payload (team entries and extras). |
| `timeline_version` | each timeline | 1 | The `timeline` block. |
| `robots_version` | each match | 1 | The `robots` block and its `path` (and the top-level `path_frame`). |

These changes are **not breaking** and do not bump any version:

* New keys on any object.
* New metrics, i.e. a new `metric` value.
* New phase names, e.g. a future season's sub-phases.
* New `source` methods.
* Different parameter values.

Different parameter values change the numbers, so compare `params` if you need
like-for-like results.

How an importer should handle versions:

1. Check `schema == "frc-vision-scorer.scouting-export"` and
   `schema_version == 2`. If either differs, stop and report it rather than
   guess.
2. **Ignore unknown keys** everywhere. The schema allows them
   (`additionalProperties: true`).
3. Skip an envelope whose `metric` you do not know. Also skip one whose
   `metric_version` (or envelope `schema_version`) is not the version you
   coded for. Log the skip and keep importing everything else.
4. If `timeline_version` is not 1, ignore that match's timeline but import the
   rest of the match. Do the same for `robots` when `robots_version` is not 1.
5. Treat unknown string values in open lists, such as phase names and mark
   `source` methods, as data, not as errors.

The JSON Schema pins the current versions with `const`. A newer file therefore
fails validation in exactly the place that changed. An envelope with an
unknown `metric` is checked only against the common envelope keys.

## Importing

### TypeScript type sketch (top level + timeline)

```ts
type CompLevel = "qm" | "ef" | "qf" | "sf" | "f";
type Alliance = "red" | "blue";

interface ScoutingExport {
  schema: "frc-vision-scorer.scouting-export";
  schema_version: 2;
  event_key: string;            // TBA event key, e.g. "2026cur"
  year: number;
  generated_at: string;         // ISO-8601 UTC
  path_frame?: PathFrame;       // absent with ?paths=false
  matches: ExportMatch[];
  skipped: { match_key: string; replay_id?: string; reason: string }[];
  envelopes: Envelope[];
}

interface SourceRef {
  match_key: string;            // TBA match key
  comp_level: CompLevel;
  set_number: number;
  match_number: number;
  replay_id: string;
  replay_created_at: string | null;   // no timezone
  legacy_labels: boolean;
}

interface ExportMatch extends SourceRef {
  alliances: Record<Alliance, [number | null, number | null, number | null]>; // stations 1..3
  timeline: Timeline | null;
  robots_version?: 1;           // these three are absent with ?paths=false
  auto_s?: number;              // t < auto_s is auto
  robots?: Robot[];
}

interface PathFrame {
  units: "m";
  field_m: [number, number];
  perspective: "absolute";      // WPILib field frame, NOT the metrics' alliance perspective
  note: string;
}

interface Robot {
  team_number: number;
  alliance: Alliance;
  alliance_position: 1 | 2 | 3 | null;
  path: {
    n_points: number;
    seen_s: { auto: number; teleop: number };
    first_t: number | null;
    last_t: number | null;
    auto_source: "autonomous_paths" | "frames" | null;
    auto_label_share: number | null;
    confirmed_share: number | null;
    segments: [number, number][];   // inclusive index ranges, safe to draw as polylines
    n_jumps: number;
    t: number[];                    // match seconds
    x: number[];                    // metres, path_frame
    y: number[];
    frame: number[];                // source-video frame
  };
}

type MarkName = "match_start" | "auto_end" | "teleop_start" | "endgame_start" | "match_end";

interface Timeline {
  timeline_version: 1;
  video: {
    fps: number;                              // use this, never assume 60
    fps_source: "video" | "clock_estimate";
    frame_count: number | null;
    duration_s: number | null;
    youtube_key: string | null;
    youtube_url: string | null;               // opens at floor(match_start.t)
  };
  marks: Partial<Record<MarkName, Mark>>;     // auto pair and teleop trio each all-or-nothing
  phases: Phase[];
}

interface Mark {
  t: number;            // video seconds from the start of the file
  frame: number;        // round(t * fps)
  video_time: string;   // "M:SS.ss"
  source: string;       // "indicator" | "clock_ticks(N)" | "<method>+<N>s" | ...
}

interface Phase {
  phase: string;        // auto | between | teleop | transition | shift1..4 | endgame | (future names)
  start_s: number;
  end_s: number;
  duration_s: number;   // video seconds (teleop ~141.5, not 140)
  start_frame: number;
  end_frame: number;
  within?: "teleop";    // present on sub-phases only
}

interface Envelope {
  schema: "frc-vision-scorer.scouting";
  schema_version: 1;
  metric: string;       // discriminator
  metric_version: number;
  metric_title: string;
  event_key: string;
  year: number;
  generated_at: string;
  frame: { units: "m"; field_m: [number, number]; perspective: "alliance";
           symmetry: "rotational" | "mirror"; note: string };
  params: Record<string, number | boolean | string>;
  sources: SourceRef[];
  skipped: ScoutingExport["skipped"];
  teams: Record<string, { team_number: number; [k: string]: unknown }>;
  [extra: string]: unknown;
}
```

For the metric payloads, generate types from `scouting-export.schema.json`
(e.g. with `json-schema-to-typescript`), or hand-write the few fields you need
from the tables above.

### Python: load the file and print each match's phase starts

```python
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "2026cur_scouting_export.json"))
assert d["schema"] == "frc-vision-scorer.scouting-export" and d["schema_version"] == 2

for m in d["matches"]:
    red = "-".join(str(t) for t in m["alliances"]["red"])
    blue = "-".join(str(t) for t in m["alliances"]["blue"])
    tl = m["timeline"]
    if tl is None or tl.get("timeline_version") != 1:
        print(f"{m['match_key']:<16} no timeline")
        continue
    starts = {p["phase"]: p["start_s"] for p in tl["phases"]}
    cols = "  ".join(f"{name}={starts[name]:7.2f}" for name in
                     ("auto", "teleop", "shift1", "shift2", "shift3", "shift4", "endgame")
                     if name in starts)
    print(f"{m['match_key']:<16} R {red:<16} B {blue:<16} fps={tl['video']['fps']:<8} {cols}")

for e in d["envelopes"]:
    print(e["metric"], "v%d" % e["metric_version"], len(e["teams"]), "teams")
```

Output for the reference file (abridged):

```
2026cur_qm1      R 589-254-10254    B 5193-3792-7567   fps=60.0     auto=   5.58  teleop=  29.75  shift1=  39.75  shift2=  65.25  shift3=  90.25  shift4= 115.75  endgame= 140.75
2026cur_qm2      R 7421-8575-5166   B 1684-6615-2053   fps=59.872   auto=   6.38  teleop=  30.82  shift1=  41.34  shift2=  66.39  shift3=  91.95  shift4= 117.00  endgame= 142.56
...
2026cur_qm8      R 6497-2724-2637   B 3128-6773-9066   fps=60.0     auto=   6.22  teleop=  30.25  shift1=  40.25  shift2=  65.75  shift3=  90.75  shift4= 116.25  endgame= 141.25
...
2026cur_qm76     R 571-5166-3181    B 9068-3932-6773   fps=59.498   auto=   6.34
...
2026cur_f1m2     R 1714-6800-254    B 6369-8595-5907   fps=60.0     auto=   6.18  teleop=  30.75  shift1=  41.00  shift2=  66.50  shift3=  91.75  shift4= 117.00  endgame= 142.25
shift_heatmap v2 74 teams
auto_paths v2 74 teams
movement v1 74 teams
cycles v1 74 teams
defense v1 74 teams
bump_stuck v2 74 teams
```

### Validating

```sh
python3 -m venv .jsv && .jsv/bin/pip install jsonschema
.jsv/bin/python - <<'EOF'
import json
from jsonschema import Draft202012Validator, FormatChecker
schema = json.load(open("scouting-export.schema.json"))
v = Draft202012Validator(schema, format_checker=FormatChecker())
errors = list(v.iter_errors(json.load(open("2026cur_scouting_export.json"))))
print("valid" if not errors else errors[:5])
EOF
```

In TypeScript/Node, use Ajv 8's draft 2020-12 build (`import Ajv2020 from
"ajv/dist/2020"`). Add `ajv-formats` so the `date-time` and `uri` formats are
known.

## Gotchas checklist

* `envelopes[*].sources` is sorted by **string** (`qm1, qm10, qm100, …`). Use
  the top-level `matches[]` for play order.
* `teams` includes lineup teams the cameras never saw. Check `coverage`,
  `n_counted` or `n_with_path` before trusting a number.
* In `auto_paths`, `n_samples` counts observed points only. The arrays can be
  longer.
* Two fields use absolute field coordinates: `auto_paths` `points_field` and
  `bump_stuck` `bumps`. Everything else is in alliance perspective.
* Timeline durations are in video seconds, so teleop is about 141.5 s, not 140.
  `between` (the auto→teleop pause) is about 4.0 s of video.
* An auto-only replay (2026cur qm76) still appears in the teleop metrics. It
  appears with `counted: false` and lowers those teams' teleop `coverage`.
* `replay_created_at` has no timezone. `generated_at` is UTC.
