# UI Standard/Debug Modes + Reliable STOP

Branch: `ui-standard-debug-modes` (worktree `/home/nick/AI/frc-vision-scorer-ui-modes`, off `main`).

## Context

The web UI (`web/static/index.html` + `web/static/app.js`, vanilla JS class `VisionScorer`,
`web/static/style.css`) serves two audiences at once: scouts running match analysis and the
developer dialing in the pipeline. We are splitting it into two modes with a header toggle,
fixing a few defects found in review (`.ui-design/reviews/ui_ia_review_20260707.md` on the
`tracking-geometry-metrics` branch), and making the STOP button kill a job in every phase.

## Global Constraints

- Vanilla JS/HTML/CSS only — no frameworks, no build step. All frontend logic lives in the
  `VisionScorer` class in `web/static/app.js`; follow its existing patterns (fixed element
  ids, innerHTML rendering, `escapeHtml` helper).
- Mode contract (all tasks): `document.body.dataset.mode` is `'standard'` or `'debug'`,
  persisted under localStorage key `uiMode`, default `'standard'`. CSS gating:
  `body[data-mode="standard"] .debug-only { display: none !important; }`. Data keeps
  flowing to hidden nodes — toggling must never require re-fetch or reload, and must work
  mid-run.
- Backend is FastAPI in `web/app.py` (~10.5k lines); knob registry in `web/job_config.py`.
- Python tests: run with `/home/nick/AI/frc-vision-scorer/.venv/bin/python -m pytest`
  from the worktree root (the worktree has no venv of its own; bare `python` is miniconda
  and missing deps). Backend changes require pytest coverage following existing patterns
  in `tests/`. Frontend (JS) has no test harness — do not invent one; verify JS by
  careful reading and, where cheap, a syntax check (`node --check web/static/app.js`).
- Do not touch `web/static/calibrate.html` behavior.
- `job._stopped` semantics (Task 1): after `POST /api/job/{id}/stop`, the job must reach a
  terminal state within ~5 seconds no matter which phase it is in.

## Task 1: STOP kills the job in every phase (backend)

**Problem.** `POST /api/job/{job_id}/stop` sets `job._stopped = True`
(`web/app.py`, `stop_job` endpoint, ~line 9990 on main). The flag is only polled in two
places: the queue-wait (~line 820/836) and the analysis loop inside
`InferenceJob.process_video` (~line 7070-7100 on main; search `if self._stopped:`).
The long phases never check it, so STOP appears dead during: match-clock scan,
calibration verification, model load, frame export, per-view tracker propagation
(the longest phase, ~77% of wall-clock), and vLLM reader startup (~90s wait).
All heavy phases run via `await asyncio.to_thread(...)` — threads cannot be killed,
so the code inside must poll.

**Requirements.**
1. Introduce a module-level `class JobStopped(Exception)` and a small helper on
   `InferenceJob`, e.g. `def _raise_if_stopped(self): if self._stopped: raise JobStopped()`.
2. Poll points (each must make stop take effect within ~5s):
   - Between every phase in `process_video` (after clock scan, after calibration check,
     after model load, after frame export, after each view's propagation, before and
     after vLLM startup).
   - Inside the frame-export loop (per frame or per small batch).
   - Inside tracker propagation: `web/field/sam2_tracker.py` `track(...)` and the
     SAM3 equivalent in `web/field/sam3_tracker.py` gain an optional
     `should_stop: Callable[[], bool] | None` parameter, checked once per propagated
     frame (SAM2) / once per frame or chunk boundary (SAM3, whichever the loop structure
     makes natural). When it returns True, raise a dedicated exception or return early
     with whatever partial state is cheapest — `process_video` treats it as stop, not
     as tracker failure. `app.py` passes `should_stop=lambda: self._stopped` at both
     call sites (search `asyncio.to_thread(\n                            sam2_tracker.track`
     and the sam3 counterpart).
   - Inside the vLLM startup wait loop (search for the "Starting vLLM reader" status
     message ~line 6100 on main and follow to the wait/poll loop; also cover the
     queue-side wait in `vllm_lifecycle` only if the wait loop lives there — do not
     redesign the lifecycle module).
3. Semantics on stop:
   - Stopped before the analysis loop begins: terminal status `error` with
     `job.error = "Stopped by user"` and the standard error message sent to the
     websocket client (same path as other job errors). No partial results expected.
   - Stopped during the analysis loop: keep the current behavior exactly (finalize
     with partial results / `complete` message).
   - In ALL cases the existing `finally` cleanup must still run (timings written,
     vLLM `ensure_stopped()` when phase-exclusive, temp dirs cleaned). `JobStopped`
     must not escape as an unhandled 500 or leave `job.status` non-terminal.
4. The stop endpoint response and pause/resume interplay stay as they are
   (`_paused` is already cleared on stop; the pause wait-loop already checks `_stopped`).

**Tests** (pytest, follow existing patterns in `tests/` — there are existing tests that
exercise job/process_video seams; find and mirror them):
- Unit: `_raise_if_stopped` raises when flagged; tracker `track(..., should_stop=...)`
  aborts early (can be tested with a stub/fake propagation loop if the real tracker
  needs a GPU — inspect how existing sam2_tracker tests fake heavy deps; if the tracker
  has no existing test seam, cover the app.py-level phase-boundary behavior instead and
  say so in the report).
- Phase-boundary: a job flagged `_stopped` before/during a faked export phase reaches
  terminal `error` state "Stopped by user" and runs cleanup.
- Full suite green.

## Task 2: Mode toggle framework, element gating, reroutes, and small fixes (frontend)

**Toggle.** Add a two-state toggle (`Standard` | `Debug`) at the right end of the header
nav in `web/static/index.html` (`<nav id="mainNav">`). Wire in `app.js` `init()`:
read localStorage `uiMode` (default `standard`), set `document.body.dataset.mode`,
update on click, persist. Style it in `style.css` to match `.nav-btn` (a segmented
pill is fine; keep it visually distinct from view tabs).

Add the CSS gate from Global Constraints, plus (only if needed) a
`body[data-mode="debug"] .standard-only { display:none !important; }` counterpart.

If the user switches to Standard while a debug-only view (Gallery/Settings) is active,
`switchView('scorer')` automatically. The Calibration link is an `<a>`, not a view —
just gate its visibility.

**Gate as `debug-only`** (add the class; do not remove or restructure the elements):
- Nav: Gallery, Settings, Calibration buttons.
- Upload form (`index.html` ~lines 103-160): the four input-groups for
  `trackingBackend`, `sam3ChunkGroup`, `sam2ModelGroup`, `sam2VramGroup`, and the
  `fieldWireframeOverlay` group. `autoPeriodOnly` stays visible in both modes.
- Scorer: `#statusLog`, `#trackingPanel` (per-camera propagation cards),
  the BUMPER READS card (`.vlm-card`), the processing-time card
  (`#processingTimeCard`).
- HUD (`.hud-left`): the `Tracker`, `Frame`, `FPS`, `Started` items and the raw
  `Reader` item are debug-only. `Elapsed` and the score chips stay in both modes.
- Timeline events: the `Frame N | ` prefix inside each event's time line becomes a
  debug-only span; the `m:ss` timestamp shows in both modes (applies to the shared
  `updateTimeline` — live and replay).
- Replay player: the job-id span (`#replayPlayerId`), the Data (JSON) download link,
  and the replay timing card (`#replayTimingCard`).

**Standard-mode reader summary.** Add one standard-only HUD item, e.g.
`<span class="hud-item standard-only">Teams <span id="hudTeams">0 / 6</span></span>`.
Update it in `updateFrame` from `data.robots`: count robots with a confirmed
team identity (`r.confirmed`) over the roster size (6 when both alliances known, else
count of slots present). Keep the existing raw reader string updating as-is (debug).

**Stall watchdog message** (`resetRunIndicators`, ~app.js:1360): the user-facing
`setStatus` text becomes mode-agnostic ("No updates for Ns — job may be stalled;
results so far are preserved."). Additionally, when the watchdog fires, append a
status-log line (log is debug-only) containing a clickable link to `/api/debug/tasks`
(open in new tab) — build it with DOM APIs like `appendCalibrationLink` does.

**Retry-leak fix.** `resetRunIndicators()` (~app.js:1326) must also clear:
`latestVlmCrops` map + `#vlmCrops` innerHTML (restore empty-state), the timeline
container (restore empty-state), and both robot tables (restore "Waiting for
detection..." rows). `resetToMatchSelect()` (~app.js:2025) already does this —
extract the shared clearing into one helper called from both, rather than duplicating.

**Escaping fix.** Route TBA-sourced strings through `this.escapeHtml(...)` where they
are interpolated into innerHTML: `renderHistory()` (event_name, team lists, match
label parts) and `updateRobotTable()` (team_number / team_candidate). Do not
change rendering structure otherwise.

**Verification:** `node --check web/static/app.js`; careful re-read of every gated id
against index.html (typo in a class = silently ungated element).

## Task 3: Settings sections + filter

**Backend.** `web/job_config.py`: add `section: str = "misc"` to the `Knob` dataclass and
set it for every knob, formalizing the existing `# --- ... ---` comment groups (use those
comments as the section names, tidied: e.g. "Tracking backend", "Seeding & re-seeding",
"Track hygiene", "Identity & votes", "VLM reader", "Fusion & field position",
"Streaming & output"). Include `"section"` in `describe_knobs()` output. Extend the
existing job_config tests to assert every knob has a non-empty section and that
describe_knobs carries it.

**Frontend.** `loadSettings()` / settings view in `app.js`:
- Render knobs grouped under section headers (render order = first appearance order in
  the registry). Each section header shows the section name and knob count.
- Add a filter `<input>` at the top of the settings view (in `index.html`): case-insensitive
  substring match against knob name AND env var; hides non-matching rows and any section
  header whose section has no visible rows. Filtering is client-side, instant (input event).
- Keep `settingRow` / `saveSetting` / reset behavior unchanged.

## Task 4: Diagnostic viewer (debug) + header job pill (both modes)

**Job pill.** A small element in the header (both modes) that shows when a job is live and
the user is not looking at it. Poll `GET /api/job/active` every 5s (it returns
`{job_id, status, started_at, match_key}` or `{job_id: null}`). Show the pill when
`job_id` is non-null AND (the active view is not `scorer` OR `this.jobId !== job_id`);
label like `⏳ {match_key or 'job'} — {status}`; clicking it switches to the scorer view
(and, when `this.jobId` is null, triggers the existing `rejoinActiveJob()` path so a
reloaded page can re-attach). Hide when no active job. Do not poll faster than 5s and
pause polling while the tab is hidden (`document.visibilitychange`).

**Diagnostic viewer (debug-only).** A collapsible panel in the scorer view (below the
status log) plus a button in the replay player bar, both `debug-only`:
- Scorer: button "Diagnostics" appears once a job id exists; fetches
  `GET /api/diagnostic/{jobId}` (NDJSON stream; falls back 404 when absent — show
  "no diagnostic data").
- Replay player: same panel fed by the replay/job id (`openReplay` knows it).
- Rendering: parse JSONL lines; show the most recent N=2000 lines max in a scrollable
  `<pre>`-style list, one line per record (compact single-line JSON). A substring filter
  input filters lines client-side. A "Reload" button re-fetches (the file grows while a
  job runs). No auto-tail/websocket — manual reload is enough.
- Big files: read via `resp.body` reader OR `resp.text()` then split — either is fine,
  but keep only the last 2000 lines in memory/DOM and say so in a header line
  ("showing last 2000 of ~N").

**Verification:** `node --check`; confirm the pill logic against `/api/job/active`'s
actual response shape (`web/app.py` ~line 7330 on main).
