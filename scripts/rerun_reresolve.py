"""Drive a full InferenceJob over an existing upload to exercise slice-3 re-resolution.

Usage:
  .venv/bin/python scripts/rerun_reresolve.py --check     # fast: just load calibration
  .venv/bin/python scripts/rerun_reresolve.py             # full run (~29 min on GPU)

Writes a fresh replay under web/replays/<job_id> (does NOT overwrite the original).
Uses a mock DB so it never touches the replays database.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.app import InferenceJob

EVENT = "2026hop"
SRC = ROOT / "web/uploads/43d5f2e2-4ff6-4c22-bcf6-74c0fa40c705.mp4"
JOB_ID = "rerun-43d5f2e2-slice3"


def make_job():
    job = InferenceJob(JOB_ID, SRC)
    job.event_key = EVENT
    job._mock_db = True
    return job


def main():
    if not SRC.exists():
        print(f"source video missing: {SRC}"); return 2
    job = make_job()
    if "--check" in sys.argv:
        ok = job._init_field_tracking()
        print(f"_init_field_tracking -> {ok}")
        print(f"  event_key={job.event_key} calib={'set' if job.calib is not None else None} "
              f"camera_center={getattr(job.calib, 'camera_center', None) is not None} "
              f"views={[v.name for v in getattr(job, 'views', [])]}")
        return 0 if ok else 1
    asyncio.run(job.process_video())
    print(f"status={job.status} error={getattr(job, 'error', None)}")
    rec = getattr(job, "_replay", None)
    if rec is not None:
        print(f"frames={len(rec.frames)} identity_log={len(getattr(job, '_identity_log', []))}")
    import json as _json
    log = getattr(job, "_identity_log", [])
    outp = ROOT / "web/replays" / JOB_ID / "identity_log.json"
    if outp.parent.exists():
        outp.write_text(_json.dumps(log))
        print(f"identity_log dumped: {len(log)} entries -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
