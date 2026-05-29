# Diagnostic Mode Design

## Problem

The identity / re-ID / OCR pipeline has many decision points — Hungarian assignment, alliance masking, vote accumulation, audit overturn, elimination, embedding re-ID matches and drops, legibility-floored VLM crop selection — and almost none of them are visible from the UI. When the system labels a robot wrong or swaps two robots' identities, the user sees only the final output (bounding boxes + team labels) with no way to ask "why did it decide that?" The existing scoring-focused Scorer view is also too busy now that scoring is not the immediate concern.

This project adds a Diagnostic view that exposes the identity pipeline's internal state, captures a per-match decision log for after-the-fact replay, and lets the user label individual VLM reads as correct or misidentified to build a ground-truth corpus. It also strips the everyday "Scorer" view down to a "Tracker" view focused on identity rather than scoring.

## Scope

**In scope** (this project):
- A `DiagnosticRecorder` module that captures structured events and snapshots during inference.
- Per-match `diagnostic.jsonl` written to disk; live WebSocket streaming of the same records.
- A new top-level **Diagnostic** page with two tabs:
  - **Pipeline** — live decision log + state panels (robot identity cards, OCR strip, re-ID continuity).
  - **Reads** — VLM crops grouped by the team they matched, with per-crop correction UI.
- A `corrections.jsonl` sibling file capturing user-applied labels for ground truth.
- Replay endpoint: `GET /api/diagnostic/<match_id>` streams the JSONL.
- Rename and strip the Scorer view → **Tracker** view: drop scoring chrome (score chips, goals column, scoring-events timeline, post-match results section). Keep event setup, OCR setup, match selection, video upload, video canvas, inference controls, and a new compact roster panel showing identification status per team.

**Out of scope** (deferred):
- Scoring-related diagnostic data (trajectory, attribution, hub OCR). Scoring code keeps running in the backend but is not exposed in either view for now.
- Feedback loops from corrections into runtime inference (vote correction, fine-tuning input).
- Multi-user auth on corrections (single-user local tool).
- Re-running inference for replay — the video file + the diagnostic JSONL are the replay artifacts, keyed by frame.

## Compatibility with in-flight identity/alliance/OCR rework

The rework (`docs/superpowers/plans/2026-05-27-identity-alliance-ocr-rework.md`) restructures `assign_teams_from_votes`, adds `audit_overturn_confirmations`, adds `web/legibility.py`, and introduces three pure functions in `web/reid.py`. The diagnostic feature is built **against the post-rework state**:

- The rework's existing `print()` lines for `Elimination:` and `Overturn:` are replaced by `record_elimination(...)` and `record_overturn(...)` calls.
- The legibility-floor `continue` branch in `vision_identify_robots` gains a `record_vlm_filtered(...)` call immediately before it.
- The Hungarian assignment loop's per-pair branch gains `record_assignment(...)`.
- `record_reid_match` / `record_reid_drop` wrap the existing `reid.choose_canonical_match` resolution at its call site in `app.py`.

The diagnostic project ships **after** the rework. It depends on the rework's structured return values; the rework does not depend on diagnostic in any way.

## Architecture

### File boundaries

```
web/diagnostic.py                    NEW. Recorder + dataclasses + replay iter.
web/app.py                           MODIFIED. Recorder instantiation + ~7 record_* call sites.
web/static/index.html                MODIFIED. Nav update; Tracker stripped of scoring chrome.
web/static/diagnostic.html           NEW. Diagnostic page shell.
web/static/diagnostic.js             NEW. Page logic (Pipeline tab + Reads tab).
web/static/style.css                 EXTENDED. Diagnostic styles.
tests/test_diagnostic.py             NEW. Recorder unit tests + replay roundtrip.
tests/test_diagnostic_corrections.py NEW. Corrections endpoint tests.
```

### Key rules

1. **The recorder is the only place that knows the diagnostic schema.** `app.py` calls typed methods (`record_vlm_read(...)`) rather than constructing dicts. This keeps schema drift out of the 138 KB `app.py`.
2. **JSONL is appended live during inference.** One file open in `'a'` mode; periodic fsync (not per-line). A crash mid-match leaves a usable file.
3. **Scoring code keeps running.** No scoring fields appear in diagnostic events. When scoring is the focus again, add scoring event types alongside without disturbing the existing schema.
4. **The recorder never blocks inference.** Any I/O or WS error is caught; on persistent failure the recorder degrades to a `NullRecorder` for the rest of the job.

## Event & snapshot schema

Each line in `diagnostic.jsonl` is one JSON object. The on-disk file and the live WS stream share the same shapes. Records are interleaved chronologically by `frame`.

Common fields on every record: `frame` (processed frame number), `t_ms` (ms since job start), `type` (discriminator).

### Events

```jsonc
// One per crop the VLM actually read.
{"frame":142,"t_ms":4821.3,"type":"vlm_read",
 "robot_id":3,"crop_b64":"...","legibility":0.31,
 "raw_read":"1538","matched_team":"1538","alliance_read":"red",
 "confidence":0.9,"vote_weight":1.0}

// One per crop the legibility floor rejected (currently invisible — silent `continue`).
{"frame":142,"t_ms":4821.3,"type":"vlm_filtered",
 "robot_id":1,"crop_b64":"...","legibility":0.002,
 "reason":"below_legibility_floor"}

// One per (robot, team) the Hungarian assigned this tick.
{"frame":150,"t_ms":5012.0,"type":"assignment",
 "robot_id":3,"team":"1538","alliance":"red","confirmed":true,
 "votes_for_assigned":4.0,"votes_total":4.5,"top_team_share":0.89}

// One per process-of-elimination assignment.
{"frame":150,"t_ms":5012.1,"type":"elimination",
 "robot_id":5,"team":"4322","alliance":"red"}

// One per audit-overturn (unlock).
{"frame":180,"t_ms":6022.4,"type":"overturn",
 "robot_id":3,"old_team":"1538",
 "challenger_team":"294","current_votes":3.0,"challenger_votes":7.5}

// One per ByteTrack track resolved to a canonical robot.
{"frame":120,"t_ms":4001.0,"type":"reid_match",
 "track_id":7,"canonical_id":3,
 "similarity":0.82,"second_best":0.71,"gap":0.11}

// One per ByteTrack track that was dropped (not matched).
{"frame":120,"t_ms":4001.1,"type":"reid_drop",
 "track_id":7,"reason":"claimed_by_lower_index",
 "would_have_matched":3,"similarity":0.79}
```

### Snapshot

Emitted once per `IDENTITY_INTERVAL` tick. Cheap (~500 bytes); a mid-match viewer can render current state without replaying earlier snapshots.

```jsonc
{"frame":150,"t_ms":5012.5,"type":"snapshot",
 "robots":[
   {"robot_id":3,"alliance":"red","team":"1538","confirmed":true,
    "votes":{"1538":4.0,"294":0.5},
    "red_votes":12,"blue_votes":0,
    "last_seen_frame":149,
    "embedding":{"age_frames":12,"last_match_similarity":0.82,"last_match_gap":0.11}}
 ],
 "roster":{"red":["1538","294","4322"],"blue":["2102","9452","2485"]},
 "identified":{"red":["1538"],"blue":[]}}
```

### Schema design notes

- **Crops live inline in `crop_b64`.** Keeps the JSONL self-contained; per-match size stays ~5–30 MB. If files balloon later, we can add a `crop_ref` field pointing to a sidecar dir without breaking the schema.
- **`vlm_filtered` events are first-class.** They are the only way to see "robot 1 went unread because its bumper was facing away." Currently invisible.
- **`reid_drop` carries `would_have_matched` + `similarity`** so near-miss swaps surface immediately. Central to debugging collision/swap bugs.
- **Snapshot is a full picture, not a delta.** A viewer dropping into mid-match always sees current state from the latest snapshot; events between snapshots are the source of truth for "what changed."
- **No scoring fields.** Per scope cut.

## Backend recorder + call sites

### `web/diagnostic.py`

```python
class DiagnosticRecorder:
    def __init__(self, out_path: Path, ws_send: Callable | None):
        # Appends to out_path in JSONL mode. ws_send is best-effort; failure logged.
        # Holds an in-memory ring of last N=2000 records for late-joining WS clients.

    def record_vlm_read(self, frame, robot_id, crop_b64, legibility,
                       raw_read, matched_team, alliance_read, confidence, vote_weight): ...
    def record_vlm_filtered(self, frame, robot_id, crop_b64, legibility, reason): ...
    def record_assignment(self, frame, robot_id, team, alliance, confirmed,
                         votes_for_assigned, votes_total, top_team_share): ...
    def record_elimination(self, frame, robot_id, team, alliance): ...
    def record_overturn(self, frame, robot_id, old_team, challenger_team,
                       current_votes, challenger_votes): ...
    def record_reid_match(self, frame, track_id, canonical_id, similarity, second_best): ...
    def record_reid_drop(self, frame, track_id, reason, would_have_matched, similarity): ...
    def record_snapshot(self, frame, robots_state, roster, identified): ...
    # robots_state is a list of dicts pre-shaped per the snapshot schema
    # (including the `embedding` sub-dict). The caller in app.py builds it
    # from self.robot_tracks + the embedder's last-match scores. The recorder
    # validates field presence but does not introspect RobotTrack or the
    # embedder — it only knows the schema.

    def flush(self): ...   # Called on inference end + on stop.

def replay_iter(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file. Skip malformed lines (count them on close)."""

class NullRecorder:
    """Drop-in replacement that swallows all calls. Used when I/O setup fails."""
```

Each `record_*` method:
1. Builds the dict per schema.
2. Appends one line to the JSONL.
3. Pushes to the in-memory ring.
4. If `ws_send` is set, awaits the push (best-effort; failure logged, not raised).

### Call sites in `web/app.py` (post-rework line numbers approximate)

| Method | Call site | Emit |
|---|---|---|
| `vision_identify_robots` | inside the `for chosen` loop, after each Qwen result | `record_vlm_read(...)` |
| `vision_identify_robots` | inside the candidates loop, on `score < LEGIBILITY_FLOOR` continue | `record_vlm_filtered(...)` |
| `assign_teams_from_votes` | inside the `for rid, team in assignment.items()` loop (skip None) | `record_assignment(...)` |
| `assign_teams_from_votes` | inside the `for rid, team in eliminated.items()` loop | `record_elimination(...)` — replaces existing `print(...)` |
| `audit_overturn_confirmations` | when `new_team is not None` | `record_overturn(...)` — replaces existing `print(...)` |
| `get_canonical_robot_id` | wrap the `reid.choose_canonical_match` resolution | `record_reid_match(...)` / `record_reid_drop(...)` |
| Identity-interval block | right after `assign_teams_from_votes` + `audit_overturn_confirmations` | `record_snapshot(...)` once per tick |

### WebSocket integration

The existing `self.websocket.send_json({...})` path adds two more `type` values: `"diagnostic_event"` and `"diagnostic_snapshot"`. Tracker page ignores them; Diagnostic page handles them. One connection, multiplexed by `type`.

### Persistence path

`web/uploads/diagnostic_<job_id>.jsonl`. This matches the existing flat-directory convention in `UPLOAD_DIR` where the video lives as `<job_id>.mp4` and the sample frame as `frame_<job_id>.jpg`. No new directory layout is introduced.

When a match completes and is saved to the database, the diagnostic JSONL is moved to a stable name keyed by the saved `match_id`: `web/uploads/diagnostic_match_<match_id>.jsonl`. This gives Match History entries a permanent diagnostic artifact while keeping live in-progress jobs cleanly keyed by `job_id`.

### Replay endpoint

`GET /api/diagnostic/<key>` — streams the JSONL file straight back (no server-side buffering). `<key>` resolves to `diagnostic_<key>.jsonl` first (live/in-progress jobs) and falls back to `diagnostic_match_<key>.jsonl` (saved matches). Frontend reads as a stream and renders progressively, so opening a large match does not block.

## Frontend

### Tracker view (renamed from Scorer)

The existing Scorer view's scoring chrome is removed:

| Today | After |
|---|---|
| Nav: `[Scorer]` `[Match History]` | Nav: `[Tracker]` `[Diagnostic]` `[Match History]` |
| HUD: Frame · FPS · score chips | HUD: Frame · FPS only |
| Right column: Blue/Red alliance tables (Robot · Team · **Goals**), Scoring Events timeline | Right column: single "Roster" panel — each alliance shows its team chips with state: ● confirmed (team color) / ◐ tentative / ○ unread |
| Bumper crops panel under the video | **Removed.** Lives only on the Diagnostic page now. |
| Post-inference Results section (TBA ground truth + editable score tables) | **Removed** from the Tracker view. Match History keeps the saved-match list; the score-edit modal stays accessible from there. |

The video canvas stays full-size with its existing bounding-box overlay. Robot labels on the canvas read `R3 · 1538` when confirmed, `R3 · ?` when tentative/unread.

Event setup, OCR setup, match selection, video upload, and inference controls are untouched — that's setup chrome, not scoring.

### Diagnostic page

A new top-level page. Tab/segmented control at the top splits the page into **Pipeline** and **Reads** tabs over the same underlying data (live WS or replay JSONL).

```
┌─ Diagnostic ─────────────────────────────────────────────────┐
│  [● Pipeline]  [○ Reads]    Match: qual37 ▾   Filters: ...   │
└──────────────────────────────────────────────────────────────┘
```

#### Pipeline tab

```
┌──────────────────────────────────────────┬──────────────────────────────────┐
│                                          │  DECISION LOG (newest first)     │
│   ┌──────────────────────────────────┐   │  ┌────────────────────────────┐  │
│   │       VIDEO (~55% width)         │   │  │ f4231 overturn  R3 1538→? │  │
│   │       bounding boxes overlaid    │   │  │ f4229 vlm_read  R3 ✓1538  │  │
│   └──────────────────────────────────┘   │  │ f4221 reid_match T7→C3 .82│  │
│                                          │  │ f4220 vlm_filter R1  .002 │  │
│   ROBOT IDENTITY CARDS                   │  │ f4210 elim      R5→4322   │  │
│   ┌───────────┐┌───────────┐┌──────────┐ │  │ f4210 assign    R3 1538 ✓ │  │
│   │R3 RED 1538││R1 BLUE  ? ││R5 RED  ? │ │  │                           │  │
│   │ votes:    ││ votes:    ││ embed:   │ │  │                           │  │
│   │ 1538 ████ ││ 2102 ██   ││ age=42   │ │  │                           │  │
│   └───────────┘└───────────┘└──────────┘ │  │                           │  │
│                                          │  │                           │  │
│   OCR STRIP (recent reads + filtered)    │  │                           │  │
│   [crop][crop][crop][crop][crop]         │  │                           │  │
│                                          │  │                           │  │
│   RE-ID CONTINUITY                       │  │                           │  │
│   T7→C3 .82 (gap .11)  T9 dropped .79    │  │                           │  │
└──────────────────────────────────────────┴──────────────────────────────────┘
```

- **Mode bar:** Live shows active job + current frame. Replay shows match dropdown (populated from `/api/matches`) and a frame scrubber under the video. Filter chips toggle which event types appear in the log.
- **Video panel:** smaller version of the Tracker video, same overlay code. Replay mode pauses on whatever frame the scrubber is at.
- **Robot identity cards:** one per canonical robot in the latest snapshot. Each shows alliance pill, team or `?`, confirmation badge, horizontal bar chart of top-3 `team_ocr_votes`, bumper-color vote counts, and embedding age. Cards with `age_frames > 60` get a warning border so swap-risk robots surface immediately.
- **OCR strip:** recent `vlm_read` *and* `vlm_filtered` events together. Each crop shows thumbnail, robot ID, legibility, raw_read, matched team (or "filtered: <reason>"). Click → highlight matching event in log.
- **Re-ID continuity panel:** compact list of recent `reid_match` / `reid_drop`. Drops show `would_have_matched` so near-miss swaps stand out.
- **Decision log:** virtualized newest-first list, color-coded by event type. Click → in replay mode, jumps the scrubber to that frame.

#### Reads tab

```
┌─ Reads — qual37 ──────────────────────────────────────────────────────────┐
│  RED Alliance                                                              │
│  ┌────── Team 1538 (12 reads, 11 ✓ · 1 ✗) ──────────────────────────────┐  │
│  │  [crop][crop][crop][crop✗][crop]...                                    │  │
│  │   ✓     ✓     ✓     →4322   ✓                                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────── Team 294  (4 reads, 4 ✓) ──────────────────────────────────────┐  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────── Team 4322 (0 reads — unread)  ─────────────────────────────────┐  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  BLUE Alliance                                                               │
│  ... (same structure for 2102 / 9452 / 2485) ...                            │
│                                                                              │
│  ┌────── Unmatched / Filtered (8 crops) ───────────────────────────────┐  │
│  │  vlm_filtered + vlm_read where matched_team is None.                  │  │
│  │  [crop][crop][crop][crop]...                                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

One row per roster team plus a final "Unmatched / Filtered" bucket. Each row shows count + correctness summary.

#### Click → correct flow

Click any crop opens an inline editor (no modal — keeps the user in the grid):

```
┌─ crop @ frame 4229, robot R3 ──────────────────────────────┐
│  [larger crop image]                                       │
│  Read: "1538"   Matched: Team 1538 (red)   Legibility: 0.31│
│                                                            │
│  Actual team:  ( ) Correct — read matches the bumper       │
│                ( ) Misidentified, actually:  [4322 ▾]      │
│                ( ) Unreadable / not a robot                │
│                ( ) Off-roster — number visible: [____]     │
│  Note (optional): [_________________________________]      │
│                                              [Save] [Cancel]│
└────────────────────────────────────────────────────────────┘
```

Save → POST a correction. Crop gets a badge: `✓` (correct), `→4322` (misidentified), `filt` (unreadable), `off` (off-roster). Re-clicking reopens the editor with the existing label pre-filled.

### Frontend state model

A small in-memory model:
- `latestSnapshot` — replaced on each snapshot message.
- `eventLog` — append-only ring of last ~5000 events.
- `corrections` — map keyed by `(frame, robot_id)` → correction object.

All panels render purely from these. No framework added — vanilla JS to match the existing codebase style.

### Replay flow

1. User picks a match → frontend `fetch('/api/diagnostic/<match_id>')` streams the JSONL.
2. As lines arrive, they're parsed and pushed into `eventLog` / `latestSnapshot`.
3. Scrubber sets a "current frame" cursor; panels render snapshot ≤ cursor and events ≤ cursor.

## Corrections persistence

Per-job `web/uploads/corrections_<job_id>.jsonl`, separate from `diagnostic_<job_id>.jsonl` (which is the inference's own log; mixing user labels would muddy provenance). On match save, renamed to `corrections_match_<match_id>.jsonl` alongside its diagnostic file. One line per correction:

```jsonc
{"t":"2026-05-27T18:42:11Z","author":"local",
 "vlm_read":{"frame":4229,"robot_id":3},
 "verdict":"misidentified",
 "actual_team":"4322",
 "note":"crop shows 4322 not 1538"}
```

Verdicts: `correct` | `misidentified` | `unreadable` | `off_roster`.

Endpoints (same `<key>` resolution as the diagnostic endpoint: `<job_id>` for live, `<match_id>` for saved):
- `GET  /api/diagnostic/<key>/corrections` → list of corrections
- `POST /api/diagnostic/<key>/corrections` → append (idempotent on `(frame, robot_id)`: re-saving overwrites the latest via last-write-wins on load)

`actual_team` is null for `unreadable`; a roster team for `misidentified`; a free-text 1–4 digit number for `off_roster`.

A `vlm_read` is uniquely keyed by `(frame, robot_id)` — only one read per robot per frame — so corrections pin to a specific crop with no UUID plumbing.

### Useful side effects

- **Bulk export:** an "Export labels" button dumps `corrections.jsonl` joined with the matching crops as a tarball (`crops/<verdict>/<actual_team>/<frame>_<robot>.jpg`). Drop-in dataset for evaluation or fine-tuning.
- **Cross-match summary** (easy follow-up): a single list of "all misidentifications across all saved matches" — useful for spotting model-systematic confusions.

### Scope guardrails

Two deliberate exclusions for v1:
1. **No multi-user / auth.** `author` is hard-coded `"local"`.
2. **No "apply correction to votes retroactively" button.** Conflates labeling with re-running inference. Separate project.

## Error handling

| Failure | Behavior |
|---|---|
| Recorder fails to open/write JSONL | Log to stderr, set `self.diagnostic = NullRecorder()` for rest of job. Inference never blocks on diagnostic I/O. |
| WS send fails | Caught at existing `try/except` in the frame-send loop. JSONL keeps appending — the on-disk file is the canonical record. |
| Recorder called with invalid args | Each `record_*` validates required fields; logs and drops the record. Never raises into the loop. |
| Diagnostic page disconnects mid-match | Recorder is unaware; JSONL keeps appending. Page reconnects → fetches replay endpoint up to current frame, then resumes live. |
| Replay JSONL has a malformed line | `replay_iter` skips malformed lines, counts them, surfaces "N lines skipped" in the UI. Does not abort the load. |
| Correction POST references unknown `(frame, robot_id)` | 400 with clear message. Should be unreachable via the UI (clicks come from rendered crops) but the endpoint validates. |
| Concurrent correction writes (two browser tabs) | Append-only JSONL; both writes succeed; loader uses last-write-wins per key. |
| Replay opened on a match whose video file is gone | State/log/crops render fine (everything's in the JSONL); video panel shows placeholder. |
| Match with zero VLM reads (e.g., stopped at frame 5) | Reads tab shows all rosters as "0 reads — unread"; Unmatched bucket empty. No crash. |
| Missing `crop_b64` (recorder dropped due to I/O at the time) | Crop renders as placeholder; click-to-correct still works using `(frame, robot_id)`. |
| Overturn flips the same robot multiple times | Each overturn is a separate event in the log; Reads tab groups crops by each `vlm_read`'s own `matched_team` (not by the robot's current label). |
| More than 6 robots (phantom canonical) | Reads tab groups by roster team; extras show up in Pipeline tab identity cards. No special-casing. |

## Testing

### `tests/test_diagnostic.py` — recorder unit tests (no model load)

- `test_recorder_writes_valid_jsonl` — call each `record_*` once; parse the resulting file; verify shape per the schema.
- `test_recorder_survives_io_error` — patch `open` to raise; verify recorder enters null mode and subsequent calls do not raise.
- `test_recorder_drops_malformed_input` — call `record_vlm_read` with `crop_b64=None`; verify logged and dropped, not written.
- `test_replay_iter_skips_malformed_lines` — write a JSONL with valid + truncated + valid; verify replay yields 2 records.
- `test_snapshot_has_embedding_age` — call `record_snapshot` with a `robots_state` containing `embedding.age_frames=N`; parse the JSONL line back and verify the field round-trips.

### `tests/test_diagnostic_corrections.py` — corrections endpoint

- `test_post_correction_appends` — POST, then GET; verify roundtrip.
- `test_post_correction_overrides_prior` — POST verdict=A, POST verdict=B for same key; GET returns B (last-write-wins).
- `test_post_correction_rejects_unknown_read` — POST with `(frame, robot_id)` absent from diagnostic.jsonl → 400.
- `test_get_corrections_for_unknown_match` — GET on a match with no `corrections.jsonl` → 200 with empty list (not 404).

### Integration smoke (model-marked, optional in default run)

- `test_inference_emits_diagnostic_jsonl` — run a 50-frame stub inference; assert JSONL exists with ≥1 of each expected event type.

### Existing tests must still pass

The rework's `tests/test_alliance_consistency.py`, `tests/test_reid_assignment.py`, `tests/test_legibility.py`, and the existing acceptance harness on `qual37.mp4`. The recorder's call sites add lines but do not change return values, so this should hold trivially.

## Open follow-ups (not in v1)

- Cross-match misidentification summary view.
- "Re-run inference on this match with corrections applied as forced labels" — a quick way to see whether the rest of the pipeline behaves correctly given perfect OCR.
- Webhook / file-watcher to fold corrections into a continuously-updated training dataset.
