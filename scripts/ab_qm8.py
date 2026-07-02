"""Automated A/B harness for the red-robot identity-collapse fix on 2026cur qm8.

For each config: POST the qm8 video + config_json to /api/infer, drive the job
to completion over its websocket, then score the resulting replay against Manta
(aggregate mean-NN + per-red-robot coverage). Configs toggle the spatial-gate
knobs. Runs sequentially (GPU is serial). Results stream to a JSON file.

    .venv/bin/python scripts/ab_qm8.py
"""
from __future__ import annotations
import asyncio, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import websockets

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.compare_manta import (  # noqa: E402
    load_our_auto_paths, load_manta_auto_paths, assign_alliance, summarize_our_path,
)

HOST = "localhost:8004"
# Defaults target qm8; override via CLI for other matches.
MATCH_KEY = "2026cur_qm8"
VIDEO = REPO / "web/uploads/ff4be1d9-b8e5-49ff-ac78-4a6a0e77a697.mp4"  # cached qm8
MATCH = 8
RESULTS = Path("/tmp/frc-ab-qm8.json")

TRACKER_SIDE = {"merge_main_guard": True, "sticky_max_dist_m": 2.5}
PROFILES = {
    "full": [
        ("baseline", {}),
        ("relink-gate", {"relink_max_dist_m": 3.0}),
        ("tracker-side", TRACKER_SIDE),
        ("both", {"relink_max_dist_m": 3.0, **TRACKER_SIDE}),
    ],
    "validate": [("baseline", {}), ("tracker-side", TRACKER_SIDE)],
    # qm10 blue-side failure: SAM3 masks hijacked by static field equipment
    # (B1 green machine at ~(1.4,2.7), B3 tan machine at ~(7.9,2.6)). Tests the
    # opt-in hijack cull (default-off since the qm24 gate) and mask re-anchoring.
    "static-cull": [
        ("baseline", {}),
        ("cull-hijack", {"cull_hijack_enabled": True}),
        ("reanchor", {"reanchor_enabled": True}),
        ("cull+reanchor", {"cull_hijack_enabled": True, "reanchor_enabled": True}),
    ],
    # Occlusion re-acquisition guard (slot-level; complements the mask-level
    # cull above): suspect-zone detections need motion proof to claim a slot
    # that went into coast while hub-occluded.
    "occl-guard": [
        ("baseline", {}),
        ("occl-guard", {"occlusion_rebind_guard": True}),
        ("guard+cull", {"occlusion_rebind_guard": True, "cull_hijack_enabled": True}),
    ],
    # Corner close-range fix: scope the too-large image-fraction backstop to
    # uncalibrated views only, so corner cameras can detect/seed the near
    # robots that fill most of their short panels (qm10: 4027 dead zone).
    # qm11 red identity collapse: one fused tid bound to R1/R2/R3 in rapid
    # alternation. The relink distance gate was A/B'd for qm8 but left
    # default-off; this isolates it against the (already default-on) tracker
    # side.
    # One plain-defaults job (for validating default-on fixes on any match).
    "single": [("baseline", {})],
    "relink": [
        ("baseline", {}),
        ("relink-gate", {"relink_max_dist_m": 3.0}),
    ],
    # Single-config probe for seed-decision forensics (sam3.* diagnostics).
    "corner-probe": [
        ("frac-fix", {"det_frac_backstop": "uncalibrated"}),
    ],
    "corner-close": [
        ("baseline", {}),
        ("frac-fix", {"det_frac_backstop": "uncalibrated"}),
        ("frac-fix+guard", {"det_frac_backstop": "uncalibrated",
                            "occlusion_rebind_guard": True}),
    ],
    # qm11 fast-robot blackout: pixel-motion evidence unblocks the seed
    # rejection gates (already_tracked vs a stationary claimant; roster_cap
    # eviction tier 2.5). motion+cull adds the static-equipment cull on top.
    "motion": [
        ("baseline", {}),
        ("motion", {"motion_override": True}),
        ("motion+cull", {"motion_override": True, "cull_hijack_enabled": True}),
    ],
    # Resume variant: baseline already scored before a server restart.
    "motion-only": [
        ("motion", {"motion_override": True}),
        ("motion+cull", {"motion_override": True, "cull_hijack_enabled": True}),
    ],
    # Motion-gated low-conf admission on top of the override (qm11 6800: dark
    # blurred bumper reads 0.26-0.34, below the 0.4 seed conf; the low band is
    # admitted only where pixels moved).
    "motion-lowconf": [
        ("baseline", {}),
        ("override", {"motion_override": True}),
        ("override+lowconf", {"motion_override": True, "motion_gated_conf": True}),
        ("all", {"motion_override": True, "motion_gated_conf": True,
                 "cull_hijack_enabled": True}),
    ],
    # New-configs-only variant (baseline and override already scored).
    "lowconf-only": [
        ("override+lowconf", {"motion_override": True, "motion_gated_conf": True}),
        ("all", {"motion_override": True, "motion_gated_conf": True,
                 "cull_hijack_enabled": True}),
    ],
    # qm11 sprint fix, full stack: seed at auto start (offset 0 — the 2s
    # boundary reprompts are the "second seed", deduped against propagated
    # masks), motion-gated low-conf admission, motion override, and
    # detector-independent diff-blob candidates.
    "sprint-fix": [
        ("dual-seed", {"sam2_prompt_offset_seconds": 0.0,
                       "motion_override": True, "motion_gated_conf": True}),
        ("dual-seed+blobs", {"sam2_prompt_offset_seconds": 0.0,
                             "motion_override": True, "motion_gated_conf": True,
                             "motion_seed_blobs": True}),
    ],
    # Blob taming: blobs scoped to the first motion_blob_window_s (6s) of the
    # active period — keeps the sprint rescue, drops the match-long churn.
    "blob-window": [
        ("dual-seed+blobs6s", {"sam2_prompt_offset_seconds": 0.0,
                               "motion_override": True, "motion_gated_conf": True,
                               "motion_seed_blobs": True}),
    ],
    # Seed-prompt quality: dedupe cross-class bumper double-reads and adopt
    # whole-robot union extents (a 24px sliver prompt on a blurred robot
    # dies in propagation; the union box is the prompt SAM can hold onto).
    "seed-quality": [
        ("union", {"sam2_prompt_offset_seconds": 0.0,
                   "motion_override": True, "motion_gated_conf": True,
                   "seed_union_extent": True}),
        ("union+blobs6s", {"sam2_prompt_offset_seconds": 0.0,
                           "motion_override": True, "motion_gated_conf": True,
                           "motion_seed_blobs": True,
                           "seed_union_extent": True}),
    ],
}
CONFIGS = PROFILES["full"]


def load_event() -> None:
    """Load 2026cur so the server has a calibrated event (jobs read event_key
    from the global current_event); without it inference aborts immediately."""
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
         "-d", '{"event_key":"2026cur"}', f"http://{HOST}/api/event/load"],
        capture_output=True, text=True, timeout=180).stdout
    if '"event_key"' not in out:
        raise RuntimeError(f"event load failed: {out[:200]}")


def post_job(cfg: dict) -> str:
    """Create a job via /api/infer (curl multipart); return job_id."""
    config_json = json.dumps({"auto_period_only": True, **cfg})
    out = subprocess.run(
        ["curl", "-s", "-F", f"video_file=@{VIDEO}",
         "-F", f"match_key={MATCH_KEY}", "-F", f"config_json={config_json}",
         f"http://{HOST}/api/infer"],
        capture_output=True, text=True, timeout=300).stdout
    return json.loads(out)["job_id"]


async def drive(job_id: str) -> str:
    """Connect the job websocket and drain until it completes; return status.

    The server signals success with a ``type:"complete"`` message and then
    closes the socket, so a close AFTER seeing 'complete' is a success, not a
    failure. Errors arrive as a status/error message before the close."""
    uri = f"ws://{HOST}/ws/infer/{job_id}"
    saw_complete = False
    async with websockets.connect(uri, max_size=None, ping_interval=20,
                                  ping_timeout=1200) as ws:
        while True:
            try:
                msg = json.loads(await ws.recv())
            except websockets.ConnectionClosed:
                return "completed" if saw_complete else "closed"
            except (TypeError, ValueError):
                continue
            t = msg.get("type")
            if t == "complete":
                saw_complete = True                      # drain until the server closes
            if t == "status" and msg.get("status") in ("completed", "error"):
                return msg.get("status")
            if t == "error":
                return "error"


def score() -> dict:
    """Score the newest 2026cur_qm8 replay vs Manta: aggregate mean-NN and
    per-alliance red coverage/tracked-count."""
    ours = load_our_auto_paths("2026cur")
    tbf = {False: load_manta_auto_paths("2026cur", False),
           True: load_manta_auto_paths("2026cur", True)}
    pairs = []
    for alliance in ("blue", "red"):
        pairs += [dict(p, alliance=alliance)
                  for p in assign_alliance(ours, tbf, MATCH, alliance, "off")]
    if not pairs:
        return {"error": "no pairs"}
    red = [p for p in pairs if p["alliance"] == "red"]
    blue = [p for p in pairs if p["alliance"] == "blue"]
    return {
        "agg_meanNN": round(float(np.mean([p["mean_nn_m"] for p in pairs])), 3),
        "blue_meanNN": round(float(np.mean([p["mean_nn_m"] for p in blue])), 3) if blue else None,
        "red_meanNN": round(float(np.mean([p["mean_nn_m"] for p in red])), 3) if red else None,
        "n_blue": len(blue), "n_red": len(red),
        "red_cov": {str(p["manta_team"]): round(100 * p["coverage_frac"]) for p in red},
        "blue_cov": {str(p["manta_team"]): round(100 * p["coverage_frac"]) for p in blue},
        # Seconds each slot's path starts after the match window opens — a
        # late start is an untracked opening (qm11 6800's first sweep), which
        # span-relative coverage alone cannot see.
        "red_lag": {str(p["manta_team"]): round(p["start_lag_s"], 1) for p in red
                    if p.get("start_lag_s") is not None},
        "blue_lag": {str(p["manta_team"]): round(p["start_lag_s"], 1) for p in blue
                     if p.get("start_lag_s") is not None},
    }


def main():
    load_event()
    print("event 2026cur loaded", flush=True)
    results = []
    for name, cfg in CONFIGS:
        t0 = time.time()
        print(f"\n=== [{name}] config={cfg} ===", flush=True)
        try:
            jid = post_job(cfg)
            print(f"  job_id={jid}; driving...", flush=True)
            status = asyncio.run(drive(jid))
            print(f"  status={status} in {time.time()-t0:.0f}s", flush=True)
            time.sleep(3)  # let the replay finish writing
            metrics = score() if status == "completed" else {"error": status}
        except Exception as e:
            metrics = {"error": repr(e)}
        row = {"config": name, "cfg": cfg, "metrics": metrics}
        results.append(row)
        RESULTS.write_text(json.dumps(results, indent=2))  # stream partial
        print(f"  metrics={metrics}", flush=True)
    # summary table
    print("\n\n===== A/B SUMMARY (2026cur qm8) =====", flush=True)
    print(f"{'config':>13} {'aggMeanNN':>10} {'redMeanNN':>10} {'nRed':>5} {'redCov%':>28}", flush=True)
    for r in results:
        m = r["metrics"]
        if "error" in m:
            print(f"{r['config']:>13}  ERROR: {m['error']}", flush=True); continue
        print(f"{r['config']:>13} {m['agg_meanNN']:>10} {str(m['red_meanNN']):>10} "
              f"{m['n_red']:>5} {str(m['red_cov']):>28}", flush=True)
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-key", default=MATCH_KEY)
    ap.add_argument("--match", type=int, default=MATCH)
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--profile", choices=list(PROFILES), default="full")
    a = ap.parse_args()
    MATCH_KEY, MATCH, VIDEO, RESULTS = a.match_key, a.match, Path(a.video), Path(a.results)
    CONFIGS = PROFILES[a.profile]
    main()
