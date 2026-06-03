# Match Replay — Design

**Date:** 2026-06-03
**Status:** Approved (design); spec review pending
**Component:** new `web/replay_recorder.py`; new `replays` SQLite table in `web/database.py`; recording hook + REST endpoints in `web/app.py`; new library + player views in `web/static/app.js` (+ `index.html`, `style.css`)

## Problem

After a match video is analyzed, the rich per-frame result — the composited HUD
(main view with robot mask highlights, field map, corner cameras), running scores,
robot identities, and scoring events — is **streamed once over WebSocket and then
discarded**. Only final aggregates persist (final scores, team assignments, and
`scoring_events`) to SQLite. There is no way to watch the analysis again.

Users need to **replay a completed analysis** in the web interface — to verify
scoring attribution, present/scout, and debug the pipeline — with smooth playback,
scrubbing, variable speed, frame-stepping, and jump-to-event.

## Goals

- Every completed analysis is automatically replayable (always-on recording).
- Replay reproduces the composited HUD exactly as rendered live, with native
  play / pause / scrub / seek / variable-speed / frame-step controls.
- An interactive layer synced to playback: live-updating robot tables, a clickable
  scoring-event timeline (click → seek), and a client-side field map driven by
  per-frame field positions.
- A **library** of all past matches (revisit any prior analysis) **and** an
  immediate **Replay** affordance for the match that just finished.
- A **delete** control in the library to remove a match's replay artifacts.

## Non-Goals (YAGNI)

- Toggling overlays at replay time — overlay composition is fixed at analysis time
  via the `HUD_*` env toggles; replay shows what was rendered.
- Re-scoring / editing during replay — the existing results editor stays separate.
- Re-rendering overlays from persisted geometry (the rejected "Approach B").
- Retention caps / auto-pruning — recording is always-on; users delete manually.

## Approach

**Recorded HUD video + metadata sidecar (hybrid).** During analysis, after
`_compose_hud_frame` builds each composited frame, write it to an H.264 `.mp4` via
`cv2.VideoWriter`, and accumulate a compact per-frame metadata record (reusing the
data already built for the live WebSocket message, plus field-space positions).
Replay plays the `.mp4` in a native HTML5 `<video>` (free scrub/seek/speed) and
drives the interactive layer from the metadata synced to `video.currentTime`.

Rejected alternative — recording the raw WebSocket JSONL stream (base64 JPEGs):
several× larger storage (~tens of GB/event) and no native scrubbing. Rejected
alternative — persisting per-frame geometry and re-rendering on replay: far more
machinery, must keep replay drawing in sync with live drawing, and bulky masks.

## Storage Layout

```
web/replays/{replay_id}/
  hud.mp4        # composited HUD video (H.264; what was shown live)
  meta.json      # summary + per-frame metadata + events
  thumb.jpg      # one mid-match frame for the library card
```

`replay_id` = the existing analysis job UUID. The source video remains in
`web/uploads/` and is not duplicated or moved.

## Data Model

### `replays` SQLite table (standalone)

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | analysis job UUID |
| `created_at` | TEXT | ISO timestamp |
| `label` | TEXT | display label, e.g. "Qualification 10" or source filename |
| `event_key` | TEXT NULL | TBA event key if known |
| `match_key` | TEXT NULL | TBA match key if known |
| `comp_level` | TEXT NULL | e.g. "qm" |
| `blue_score` | INTEGER | final blue score |
| `red_score` | INTEGER | final red score |
| `blue_teams` | TEXT | JSON array of team numbers (slot order) |
| `red_teams` | TEXT | JSON array of team numbers (slot order) |
| `video_path` | TEXT | path to `hud.mp4` |
| `meta_path` | TEXT | path to `meta.json` |
| `source_video_path` | TEXT | path to original upload |
| `thumb_path` | TEXT | path to `thumb.jpg` |
| `fps` | REAL | playback fps (effective tracking fps) |
| `frame_count` | INTEGER | recorded frames |
| `duration_sec` | REAL | `frame_count / fps` |
| `width` | INTEGER | composited frame width |
| `height` | INTEGER | composited frame height |
| `codec` | TEXT | fourcc actually used (`avc1` or `mp4v`) |
| `status` | TEXT | `complete` \| `partial` |

TBA keys are nullable so ad-hoc uploads (no TBA match) still produce a replay row.
The table is independent of the existing `matches` table; no FK is required.

### `meta.json`

```jsonc
{
  "summary": {
    "fps": 8.0, "frame_count": 1583, "duration_sec": 197.9,
    "blue_score": 70, "red_score": 67,
    "robots": [{"slot": "B1", "alliance": "blue", "team_number": 581}],
    "match_label": "Qualification 10 of 124"
  },
  "frames": [
    { "i": 0, "t": 0.0, "scores": {"blue": 0, "red": 0},
      "robots": [
        { "slot": "B1", "alliance": "blue", "team_number": 581,
          "confirmed": true, "field_pos": [3.21, 4.05], "scores": 0 }
      ] }
  ],
  "events": [
    { "t": 45.2, "frame": 362, "alliance": "blue", "slot": "B2",
      "team_number": 9785, "robot_label": "B2 9785", "confidence": "high" }
  ]
}
```

- `frames[i].t` = `i / fps` (seconds) — the sync key against `video.currentTime`.
  `i` is the **recorded-frame index** (0-based count of frames written to `hud.mp4`),
  not the source `frame_num`.
- `field_pos` is the field-space track position (the same `(pos, alliance)` data the
  radar renders from), enabling the client-side field map.
- `events[].t` must be derived by mapping the event's source `frame_num` to the
  recorded-frame index `i` it was written as, then `t = i / fps`. **Do not reuse the
  in-memory `ScoringEvent.timestamp`** — that is video-time (`frame_num / video_fps`),
  whereas recorded time is `i / (video_fps / frame_skip)`; using it would make
  click-to-seek land off by the `frame_skip` factor. The recorder already knows the
  `frame_num → i` mapping (it assigns `i` as it writes frames), so it stamps each
  event's `t` at `finalize()` from that mapping.

## Components

### `web/replay_recorder.py` — `ReplayRecorder`

A focused, independently testable unit. Responsibilities:

- `open_lazy(first_composited_frame, fps, out_dir)` — opens the `cv2.VideoWriter`
  using the first composited frame's shape (HUD size is stable within a job),
  trying `avc1` then falling back to `mp4v`; records the chosen fourcc and locks
  `(width, height)`.
- `add_frame(composited_frame, record)` — writes the frame (resizing to the locked
  size if a stray frame differs) and appends the per-frame metadata `record`.
- `add_event(event)` — appends a scoring-event record.
- `finalize(status, summary)` — releases the writer, writes `meta.json` and
  `thumb.jpg` (a mid-match frame), and returns the data needed for the DB row.

Interface contract: knows nothing about FastAPI, jobs, or the DB; it consumes
frames + plain dict records and produces files. The caller (the analysis loop)
owns DB insertion.

### Recording hook (`web/app.py`)

In the analysis loop, immediately after `annotated_frame = _compose_hud_frame(...)`
and after the per-frame `robots_data` / `scores` are built for the WebSocket
message, call `recorder.add_frame(annotated_frame, record)` where `record` reuses
that same data plus each visible track's field position. Scoring events are passed
to `recorder.add_event(...)` as they are appended to `self.scoring_events`.
`fps = video_fps / frame_skip` so playback runs at true match speed. On job
completion — or on stop/error (so partial runs are still replayable) — the loop
calls `recorder.finalize(...)` and upserts the `replays` row. Recording cost is
negligible relative to SAM2/YOLO.

### REST endpoints (`web/app.py`)

- `GET /api/replays` — list rows (id, label, teams, scores, created_at,
  duration_sec, thumb URL), newest first.
- `GET /api/replays/{id}` — returns the `meta.json` contents.
- `GET /api/replays/{id}/video` — serves `hud.mp4` **with HTTP Range support**
  (returns `206 Partial Content` for ranged requests) so seeking/scrubbing works
  across browsers.
- `GET /api/replays/{id}/thumb` — serves `thumb.jpg`.
- `DELETE /api/replays/{id}` — deletes the `web/replays/{id}/` directory and the DB
  row; leaves the source upload untouched.

### Frontend (`web/static/app.js`, `index.html`, `style.css`)

- **Library view** ("Replays" tab/section): a grid of cards from `GET /api/replays`,
  each showing thumbnail, match label, alliance teams, final score, and date, with
  **Replay** and **Delete** buttons. Delete confirms, then calls `DELETE` and
  refreshes the list.
- **Player view**: reuses the existing analysis layout, replacing the live
  WebSocket-fed `<canvas>` with a native `<video controls>` whose source is
  `GET /api/replays/{id}/video`. It loads `meta.json` and, on the video's
  `timeupdate` event, maps `currentTime → frame index` (`round(currentTime * fps)`)
  to: update the robot tables, highlight the current position in the scoring-event
  timeline, and render a small client-side field map onto a `<canvas>` from that
  frame's `field_pos` values. The event timeline is clickable — clicking an event
  sets `video.currentTime` to its `t` (seek). `updateRobotTables(robots)` and
  `updateTimeline(events)` already take data as arguments, but are coupled to the
  live WebSocket payload: `updateTimeline` renders only the server's last-10 events
  and is not clickable. The refactor decouples them from the live payload shape and
  makes the timeline render the full event list with click-to-seek, so the live and
  replay paths share the same rendering.
- **Just-finished**: a **Replay** button on the results screen deep-links into the
  player for the just-completed job id (which is now in the library).

## Data Flow

1. Analysis runs; per processed frame the loop composes the HUD and builds
   `robots_data` + `scores` for the live WebSocket message (unchanged).
2. The same frame + a metadata record are handed to `ReplayRecorder`; the frame is
   written to `hud.mp4` and the record buffered.
3. On completion/stop/error, `finalize()` writes `meta.json` + `thumb.jpg`; the loop
   upserts the `replays` row.
4. Library view lists rows; selecting one opens the player.
5. Player streams `hud.mp4` (ranged) into `<video>` and fetches `meta.json`; the
   `timeupdate` handler keeps tables, timeline, and field map in sync with playback.

## Error Handling & Edge Cases

- **Codec availability:** try `avc1`, fall back to `mp4v`; persist the fourcc used.
- **Frame-size constancy:** lock `(width, height)` from the first composited frame;
  defensively resize any later frame that differs (should not happen within a job).
- **Interrupted analysis:** `finalize(status="partial")` still produces a playable
  `hud.mp4` + `meta.json` for the frames processed so far.
- **Range requests:** `/video` must honor `Range`, returning `206` with the correct
  byte slice and `Content-Range`/`Accept-Ranges` headers; a normal `GET` returns the
  whole file (`200`).
- **Delete:** removes only the `web/replays/{id}/` directory and the DB row; the
  source upload is a separate concern and is not deleted.
- **Disk:** always-on recording; users prune via the library Delete control.

## Testing

- **`ReplayRecorder` unit tests:** produces a playable `.mp4` whose frame count
  matches the number of `add_frame` calls (verified by reopening with
  `cv2.VideoCapture`); writes a `meta.json` with the expected per-frame records and
  events; the `finalize(status="partial")` path produces valid artifacts from a
  partial run; codec fallback path works when `avc1` is unavailable.
- **`replays` table CRUD tests:** insert/list/get/delete via `web/database.py`.
- **API tests:** `GET /api/replays` lists; `GET /api/replays/{id}` returns metadata;
  `DELETE` removes artifacts + row; a `Range` request to `/video` returns `206` with
  the correct bytes and a full `GET` returns `200`.
- **Frontend:** verified manually (library renders, player syncs tables/timeline/
  field map to playback, click-to-seek works, delete works).

## Open Questions

None blocking. Match-label derivation uses the TBA match label when available,
otherwise the source filename.
