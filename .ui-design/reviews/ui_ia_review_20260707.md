# Design Review: FRC Vision Scorer — Information Architecture & Standard/Debug Split

**Reviewed:** 2026-07-07
**Target:** web/static/index.html, web/static/app.js (+ /api surface in web/app.py)
**Focus:** What is shown and where; Standard vs Debug mode proposal

## Summary

The UI is a well-built single-flow operator console, but it serves two audiences at once:
a scout who wants match results, and a pipeline developer dialing in tracking/identity.
Today every screen leans developer: raw reader telemetry in the HUD, VLM crop cards,
per-camera propagation panels, a 69-knob flat Settings list, and expert help text inline
in the upload form. A `data-mode` toggle in the header with `debug-only` / CSS-gated
elements cleanly splits the two without restructuring the app.

## Current inventory (what is shown and where)

**Nav:** Scorer | Replays | Match History | Gallery | Settings | Calibration (separate page, new tab)

**Scorer flow (index.html:24-343):**
1. Load Event (TBA URL) → 2. Scoreboard OCR region setup → 3. Match select + upload form
   (includes tracker backend, SAM3 chunk window, SAM2 model size, VRAM budget, auto-period,
   wireframe overlay — all with long expert help text) → status bar + degradation banner +
   rolling 50-line status log → start/pause/resume/stop → live view → results editor.
2. Live view: HUD (Tracker, Frame, FPS, raw Reader string, Started, Elapsed; scores),
   per-camera tracking cards, video canvas, alliance tables (slot/team ✓|?/goals),
   BUMPER READS crop card, SCORING EVENTS timeline (frame # + time + confidence badge).
3. Results editor: processing-time breakdown, TBA ground truth, editable predicted/corrected
   per robot, notes, Save / Watch Replay / New Match.

**Replays:** card grid (match title, score, teams, date, proc time, PARTIAL badge) → player
(HUD, video, timing card, alliance tables, clickable timeline, identity-correction grid,
video/JSON downloads, job id).

**History:** per-event tables — match, teams, TBA score vs predicted, status badge, detail modal.

**Gallery:** appearance exemplars per team (verify/save/delete). **Settings:** flat list of
~69 knobs (job_config.py registry) with env names + source badges.

**Backend surfaces with no UI:** `/api/diagnostic/{key}` (JSONL stream), `/api/debug/tasks`
(referenced by the stall watchdog message but only reachable via curl), server logs.

## Findings

### Major

1. **Upload form mixes scouting workflow with pipeline knobs** (index.html:103-160).
   Tracker choice, chunk window, model size, VRAM budget all say "leave empty / default is
   fine" — i.e., they self-identify as debug controls. Standard users get five decisions
   they were told not to make.

2. **HUD "Reader" string is developer shorthand** (app.js:1433):
   `vllm c12 3/4 reads 5 crops u2 0.85s`. Meaningless without reading the code. Debug: keep.
   Standard: replace with e.g. "Teams identified 4/6".

3. **Settings is a flat, unsectioned list of ~69 knobs** with env-var names as the only
   grouping cue. Even for Debug mode it needs the registry's implicit sections
   (tracking / seeding / identity / VLM / streaming) as headers, plus a filter box.

4. **No per-team aggregate view.** Scouting means comparing teams across matches; the app
   ends at per-match results. History shows totals only. This is the biggest Standard-mode
   content gap (data already exists in frc_scorer.db).

5. **No global "job running" indicator.** Navigating to Replays/History mid-run hides all
   status; the WS keeps streaming into hidden nodes. A header pill (spinner + % + click to
   return) serves both modes.

### Minor

6. **Retry path leaks prior-run artifacts**: `resetRunIndicators()` (app.js:1326) clears
   warnings/log/tracking cards but not `latestVlmCrops`, the timeline, or robot tables —
   only `resetToMatchSelect()` (app.js:2025) does. After a calibration retry the old crops
   and events linger until overwritten.

7. **Inconsistent HTML escaping**: `renderHistory()` (app.js:2129-2183) and
   `updateRobotTable()` interpolate TBA-sourced strings (event_name, team numbers) into
   innerHTML without `escapeHtml`, which is used elsewhere. Low risk, easy consistency fix.

8. **Timeline shows frame numbers first** (app.js:1596). Scouts think in match time;
   frames are debug coordinates.

9. **Timeline/crops re-render full innerHTML every frame** (~10 Hz). Fine at current sizes;
   worth keying if event counts grow.

10. **Stall watchdog tells users to "check /api/debug/tasks"** (app.js:1366) — a curl-only
    endpoint. In Debug mode, link it; in Standard, drop the reference.

## Standard / Debug assignment

Mechanism: header toggle → `document.body.dataset.mode` (persisted in localStorage) →
CSS `body[data-mode="standard"] .debug-only { display: none }`. Data keeps flowing to
hidden nodes, so flipping the toggle mid-run instantly reveals populated panels.

| Element | Standard | Debug |
|---|---|---|
| Scorer / Replays / History nav | ✓ | ✓ |
| Gallery, Settings, Calibration nav | — | ✓ |
| Event load, OCR setup, match select, upload | ✓ | ✓ |
| Tracker backend / chunk / model size / VRAM inputs | — | ✓ |
| Auto-period-only checkbox | ✓ | ✓ |
| Field wireframe overlay checkbox | — | ✓ |
| Status bar + degradation banner | ✓ | ✓ |
| Rolling status log | — | ✓ |
| Per-camera tracking cards | — (simple phase text) | ✓ |
| HUD: scores, Elapsed | ✓ | ✓ |
| HUD: Tracker, Frame, FPS, Started, raw Reader | — (friendly reader summary) | ✓ |
| Alliance tables, timeline | ✓ (time only) | ✓ (+frame #) |
| BUMPER READS crops | — | ✓ |
| Results editor + TBA ground truth | ✓ | ✓ |
| Processing-time breakdown | — | ✓ |
| Replay player + identity correction | ✓ | ✓ |
| Replay job id, JSON download, timing card | — | ✓ |
| History | ✓ | ✓ |

**Debug-mode additions (new):** diagnostic JSONL viewer (`/api/diagnostic/{key}` exists),
`/api/debug/tasks` link from the stall message, job id visible in scorer view, sectioned +
searchable Settings.

**Standard-mode additions (later):** per-team aggregate view; CSV export of scouted results.

## Positive observations

- Confirmed/tentative/candidate identity rendering (✓ / dimmed / ?) is exactly the right
  trust signal for scouts — keep in both modes.
- Degradation banner + authoritative warning list on completion is a strong "can I trust
  this" pattern.
- WS reconnect + job re-attach, stall watchdog, and the calibration-failure →
  recalibrate → continue flow are all operator-grade resilience rarely seen in tool UIs.
- Empty states exist nearly everywhere.

## Next steps (suggested order)

1. Add the mode toggle + `debug-only` class annotations (pure HTML/CSS + ~20 lines JS).
2. Reroute upload-form knobs and HUD reader string per the table.
3. Clear crops/timeline/tables in `resetRunIndicators()`.
4. Section + filter the Settings list.
5. Diagnostic viewer panel in Debug; header job pill in both.
6. (Bigger) per-team aggregate view for Standard.
