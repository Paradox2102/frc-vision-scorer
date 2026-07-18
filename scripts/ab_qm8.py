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
# Validated qm11 sprint recipe (meanNN 1.105->0.738): seed at the auto-period
# open, motion override + motion-gated low-conf admission, whole-robot union
# seed extents. Shared by the endgame profiles and the backfill A/B below.
SPRINT = {"sam2_prompt_offset_seconds": 0.0, "motion_override": True,
          "motion_gated_conf": True, "seed_union_extent": True}
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
    # Hijack pair fix (2026-07-06 deep dive): person-overlap veto (positive
    # hijack signal — qm11 R3 spent all of auto on a ref) and the absence
    # cull scoped to the main view (corner views were qm24's false-cull
    # victims; mid-field equipment is main-visible so absence is strong).
    "hijack-fix": [
        ("person-veto", {"person_veto_enabled": True}),
        ("cull-main", {"cull_hijack_enabled": True}),
        ("veto+cull", {"person_veto_enabled": True,
                       "cull_hijack_enabled": True}),
    ],
    # Post-detector-retrain validation (hub_zone_yolo26s_2026cur_v2 wired in
    # via MODEL_PATH): baseline on the new model, then the full hijack stack.
    "newmodel": [
        ("baseline", {}),
        ("veto+cull", {"person_veto_enabled": True,
                       "cull_hijack_enabled": True}),
    ],
    # OVERFLOW-bin endgame: tight zones_2026cur.json (bin + rig pole) are now
    # event config (both configs get the seed gate); the contrast is the
    # corroboration blinding that lets the scoped cull finally fire on the
    # bumper-FP-corroborated bin mask.
    "binfix": [
        ("zones-only", {}),
        ("full-stack", {"zone_blind_corroboration": True,
                        "cull_hijack_enabled": True,
                        "person_veto_enabled": True}),
    ],
    # Endgame: the validated sprint recipe (qm11 memory: meanNN 1.105->0.738;
    # it was NOT enabled in any of today's runs) as the true baseline, vs
    # sprint + the full hijack stack (3 tight zones incl. the pole tote,
    # zone-blind corroboration, main-scoped cull, person veto). Opus root
    # cause: 3181's 26 already_tracked seed rejections are exactly what
    # motion_override promotes; the stack frees the seats clutter holds.
    # Single sprint+stack run (vote-fix validation etc.).
    "endgame-stack": [
        ("sprint+stack", {**SPRINT,
                          "zone_blind_corroboration": True,
                          "cull_hijack_enabled": True,
                          "person_veto_enabled": True}),
    ],
    "endgame": [
        ("sprint", SPRINT),
        ("sprint+stack", {**SPRINT,
                          "zone_blind_corroboration": True,
                          "cull_hijack_enabled": True,
                          "person_veto_enabled": True}),
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
    # qm12 (86d6e154) root cause: a static vertical sliver on the blue hub
    # face (box aspect h/w=4.47) seeded a SAM object and EVICTED a real robot
    # from the capped roster. seed_box_max_aspect rejects implausibly tall
    # seed boxes before they can seed/re-anchor/evict; evict_requires_motion
    # keeps a non-moving candidate from evicting when motion evidence exists.
    "seed-gate": [
        ("baseline", {}),
        ("seed-gate", {"seed_box_max_aspect": 3.0,
                       "evict_requires_motion": True}),
    ],
    # qm12 lever 2 (opening-window dense reprompt) stacked on lever 1. B-only:
    # the baseline A was scored in the seed-gate run just prior. Seeds at period
    # start (offset 0) and re-prompts every 0.5s for the first 6s so robots
    # streaming into center during 0:20-0:16 seed promptly instead of waiting a
    # 2s boundary; lever-1 gates keep static junk out of the freed roster slots.
    "seed-open": [
        ("lever1+2", {"seed_box_max_aspect": 3.0, "evict_requires_motion": True,
                      "sam2_prompt_offset_seconds": 0.0,
                      "opening_reprompt_seconds": 0.5, "opening_window_seconds": 6.0}),
    ],
    # Full A/B pair (baseline + lever1+2) for cross-match regression checking
    # before landing the seed-gate + opening-dense-reprompt config as a default.
    # qm12 was the target match; run this on matches WITHOUT the hub-face sliver
    # failure (qm8/qm10/qm11) to confirm the gates don't regress them.
    "seed-open-ab": [
        ("baseline", {}),
        ("lever1+2", {"seed_box_max_aspect": 3.0, "evict_requires_motion": True,
                      "sam2_prompt_offset_seconds": 0.0,
                      "opening_reprompt_seconds": 0.5, "opening_window_seconds": 6.0}),
    ],
    # qm11/qm10/qm12 gated leak cleanup: mask leaked onto a bump; the gated
    # cleaner drops a component only if BOTH small AND detached, tightening the
    # box (and field pos) without dropping a legit arm / occlusion-split half.
    # Baseline here = current shipped default (lever1+2).
    "leak-clean": [
        ("baseline", {}),
        ("leak-clean", {"mask_leak_cleanup": True}),
    ],
    # Phase-1 3D-geometry hardening (branch geometry-tracking-hardening): the
    # four behavior-changing tracking knobs enabled together vs current shipped
    # defaults. vote_ray_gate drops misattributed bumper votes (cross-ray +
    # stale-pos), size_prior_cull evicts masks geometrically too wide/narrow for
    # a robot at their field position (catches detector-corroborated equipment
    # hijacks), start_zone_seed admits low-conf bumpers in the alliance-wall
    # bands during the opening window (fast-robot seeding), kin_gate rejects
    # impossible-speed associations. Baseline = defaults (already lever1+2).
    "geo-hardening": [
        ("baseline", {}),
        ("geo4", {"vote_ray_gate": True, "size_prior_cull": True,
                  "start_zone_seed": True, "kin_gate": True}),
    ],
    # Per-knob isolation of the geo4 stack: the stacked run recovered
    # missed/late robots (start_zone_seed) but regressed existing tracks on
    # qm11 (3181 94->44%, 6800, 1477) — the signature of a cull/gate evicting
    # real tracks. Each knob alone vs baseline isolates the regressor.
    "geo-sweep": [
        ("baseline", {}),
        ("start_zone_seed", {"start_zone_seed": True}),
        ("size_prior_cull", {"size_prior_cull": True}),
        ("kin_gate", {"kin_gate": True}),
        ("vote_ray_gate", {"vote_ray_gate": True}),
    ],
    # sam3_compile speed A/B: torch.compile on the propagation stage (now the
    # dominant job cost). The first compile run pays inductor warm-up, so the
    # arm runs twice in the same server process; compare compile-warm against
    # baseline, and confirm Manta scores are unchanged (compile must be
    # numerically neutral or it doesn't ship).
    "sam3-compile": [
        ("baseline", {}),
        ("compile-cold", {"sam3_compile": True}),
        ("compile-warm", {"sam3_compile": True}),
    ],
    # VLM equipment adjudicator (branch vlm-quality-checks), observe-only:
    # long-static tracks are classified robot/equipment/person and written as
    # adjudication diagnostic records without touching tracking. Run on a FULL
    # match (NOT auto_period_only) so parked/hijacked tracks accumulate the
    # static window — auto-only sprints never sit still long enough to fire.
    # Manta score should match baseline since nothing acts on the verdicts.
    # (The full-field census was prototyped and dropped — see git history.)
    "vlm-qc": [
        ("adjudicate", {"equipment_adjudicate": True}),
    ],
    # Mask-footprint field position: derive each robot's ground contact from
    # the SAM mask's carpet-contact band instead of the tracked box bottom-
    # center (an extended intake/arm stretches the box off the true contact).
    # Payload-neutral to identity; the expected win is tighter field positions,
    # so watch the Manta mean-NN / weighted-NN drop vs baseline.
    "mask-footprint": [
        ("baseline", {}),
        ("mask_footprint", {"mask_footprint_pos": True}),
    ],
    # yolo-sam3-fusion branch: per-feature isolation of the auto-measurable
    # fusion knobs vs current shipped defaults (baseline). Each alone — not
    # stacked — so a regressor is attributable (the geo4 stack masked which
    # knob hurt qm11). corrob_conf_floor admits a sub-seed-conf corroborate-
    # only band (weak/blurry fast robots stop accruing absences); neg_point_
    # prompts conditions SAM against bleeding onto neighbors at seed/re-anchor;
    # misalign_reanchor re-prompts a drifted healthy mask (between the dark and
    # sliver cases); evict_prob_immune spares high-tracker_prob objects from
    # weak-tier eviction. vote_prob_weight + det_gated_vlm_reads are identity
    # features (weak signal on ~15s auto) — evaluate those on full matches.
    "fusion-sweep": [
        ("baseline", {}),
        ("corrob-floor", {"corrob_conf_floor": 0.25}),
        ("neg-points", {"neg_point_prompts": True}),
        ("misalign", {"misalign_reanchor": True}),
        ("evict-prob", {"evict_prob_immune": 0.8}),
    ],
    # Full-match A/B for the "other" fusion features the auto sweep left
    # inconclusive: corrob_conf_floor + evict_prob_immune were inert/mixed on
    # ~15s auto, and vote_prob_weight + det_gated_vlm_reads are IDENTITY
    # features that need teleop's VLM reads to exercise at all. auto_period_
    # only:False processes the whole match. Split into single-config profiles
    # so the driver can restart the server between each — full-match jobs grow
    # SAM3 session VRAM and 20+ back-to-back jobs OOM-hung the auto sweep at
    # qm11. (Note: score() still measures AUTO-period Manta position; identity
    # gains show up as team-coverage/pairing shifts, weaker than a pure
    # label-accuracy metric — read those configs' per-team coverage.)
    "full-base":    [("baseline",     {"auto_period_only": False})],
    "full-corrob":  [("corrob-floor", {"auto_period_only": False, "corrob_conf_floor": 0.25})],
    "full-evict":   [("evict-prob",   {"auto_period_only": False, "evict_prob_immune": 0.8})],
    "full-vote":    [("vote-prob",    {"auto_period_only": False, "vote_prob_weight": True})],
    "full-detgate": [("det-gated",    {"auto_period_only": False, "det_gated_vlm_reads": True})],
    # late_seed_backfill A/B (qm11): recovers the pre-seed history of a robot
    # first detected at a late reprompt boundary by bounded reverse propagation
    # at end of track. Runs on the sprint recipe — with plain defaults the
    # sprinting robot never seeds late, leaving backfill inert. Expect per-team
    # auto coverage gains for late-seeded teams; meanNN should not regress.
    # Two repeats per arm to gauge the known run-to-run nondeterminism.
    "backfill": [
        ("sprint", SPRINT),
        ("sprint+backfill", {**SPRINT, "late_seed_backfill": True}),
        ("sprint-b", SPRINT),
        ("sprint+backfill-b", {**SPRINT, "late_seed_backfill": True}),
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


def _replay_saved(job_id: str, tries: int = 10, delay: float = 0.5) -> bool:
    """True if a replay row with id == job_id exists (success writes one).
    Polls briefly: the row can lag the websocket close by a moment."""
    import sqlite3
    db = REPO / "frc_scorer.db"
    for _ in range(tries):
        try:
            con = sqlite3.connect(db)
            row = con.execute("SELECT 1 FROM replays WHERE id = ?", (job_id,)).fetchone()
            con.close()
            if row:
                return True
        except sqlite3.Error:
            pass
        time.sleep(delay)
    return False


async def drive(job_id: str) -> str:
    """Connect the job websocket and drain until it completes; return status.

    The server signals success with a ``type:"complete"`` message and then
    closes the socket, so a close AFTER seeing 'complete' is a success, not a
    failure. Errors arrive as a status/error message before the close.

    The server can occasionally close the socket before the client reads the
    terminal message (observed on longer jobs). On an unexpected close we fall
    back to the durable record: a successful job saves a replay whose id equals
    the job_id, so a matching replay row means the job completed."""
    uri = f"ws://{HOST}/ws/infer/{job_id}"
    saw_complete = False
    async with websockets.connect(uri, max_size=None, ping_interval=20,
                                  ping_timeout=1200) as ws:
        while True:
            try:
                msg = json.loads(await ws.recv())
            except websockets.ConnectionClosed:
                if saw_complete or _replay_saved(job_id):
                    return "completed"
                return "closed"
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
