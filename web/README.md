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
