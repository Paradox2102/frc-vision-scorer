# Web Application

FastAPI app that serves the UI, manages match-processing jobs, and runs the
vision pipeline. See the top-level [README](../README.md) for setup, usage,
and architecture.

## Layout

- `app.py` — API routes, WebSocket streaming, job runner, pipeline
  orchestration
- `field/` — the vision pipeline: calibration, SAM2/SAM3 tracking,
  field-coordinate fusion, identity, appearance gallery, radar rendering
- `static/` — frontend (vanilla JS)
- `database.py` — SQLite persistence (jobs, tracks, identity, gallery)
- `job_config.py` — every pipeline knob: env-var defaults + per-job
  `config_json` overrides
- `vllm_lifecycle.py` — job-managed start/stop of the vLLM team reader

## Scripts

- `start_all.sh` / `stop_all.sh` — start/stop the app (vLLM comes up on
  demand per job); app listens on http://localhost:8004
- `start_vllm.sh` / `stop_vllm.sh` — manage the vLLM reader manually

Runtime artifacts (`uploads/`, `logs/`, `replays/`) are gitignored.

## Ground-truth annotator

Hand-annotated robot tracks for scoring the pipeline against absolute truth
(not against its own detections). One pass through the match video per robot.

    .venv/bin/python scripts/gt_annotator_server.py     # http://localhost:8766
    #   --port N   --gt-dir DIR (default data/eval/ground_truth)   --db PATH

Pick a match (only matches with a replay whose source video still exists are
listed), the video (byte-identical uploads are merged by fingerprint), then a
robot. Selecting a robot reloads its saved track and seeks to its last
keyframe. Set **auto start** (pre-filled from the latest replay's auto period)
and pick a range: Auto (t0→+20 s), Full match (t0→+163 s) or Whole video.

**Cursor convention:** the cursor is a perspective-correct robot-sized box
(0.95 × 0.95 m footprint, 0.6 m tall, projected with the event calibration for
whichever camera panel it is over). **Keep the box on the robot.** The small
crosshair near the bottom of the box is the centre of the robot's footprint on
the carpet — that ground point is what gets recorded (`uv`) and projected to
the floor plane. `-` / `=` shrink / grow the box (0.8–1.4×, remembered in the
browser only) if this season's robots are a different size.

**Follow mode (default):** press and hold with the box on the robot — the video plays at the
chosen rate (default 0.75×) and the cursor is recorded as a `visible` keyframe
every 2nd source frame while the button is held. Release to pause. For a camera
handoff (main ↔ corner panel), release, move to where the robot appears in the
other panel and press again. Hold **Shift** while following to record
`estimated` (occluded, best-guess) samples. A held stretch overwrites existing
keyframes over the frames it covers. Samples that are off-field, unprojectable,
or jump faster than 6 m/s (+0.5 m) from the previous sample are dropped with a
warning (10 consecutive self-consistent "jumps" re-anchor the track).

**Step mode (`M`, for patching):** click = `visible` keyframe + advance Δt;
Shift+click or `E` then click = `estimated`. Keys in both modes: `U` unknown +
advance, `Space` play/pause, `←/→` Δt (Shift: 1 frame), `[`/`]` prev/next
keyframe (Shift: prev/next gap), `Del`/`⌫` delete keyframe here or the
timeline selection (shift-drag on the timeline), `Ctrl+Z`/`Ctrl+Y` undo/redo,
`-`/`=` robot box size, `L` magnifier (off by default; a 2.5× view with the box,
shown in the corner opposite the cursor), `Ctrl+S` save.

The overlay shows the selected robot's last 2 s, the other robots' saved
tracks at the current frame (so later passes don't confuse robots), and the
field map can overlay a pipeline replay's slot tracks. Autosave runs ~600 ms
after an edit (every 5 s while recording, and on release); a localStorage
backup is offered on reload if the server copy is older.

Files: `data/eval/ground_truth/<match_key>/<team>.json` (format in
`web/field/ground_truth.py`; the previous version is kept as `.json.bak`).
`field_xy` is recomputed server-side on every save with the calibration epoch
the pipeline uses for that match, recorded under `calibration`. Each keyframe
carries `src` (`follow`/`click`) and follow samples carry the playback `rate`
(cursor lag vs. the robot is not corrected).
