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
AUTO_ONLY = True  # module global; --full-match sets False (per-arm cfg still wins, see post_job)
# Defaults target qm8; override via CLI for other matches.
MATCH_KEY = "2026cur_qm8"
EVENT_KEY = "2026cur"
VIDEO = REPO / "web/uploads/ff4be1d9-b8e5-49ff-ac78-4a6a0e77a697.mp4"  # cached qm8
MATCH = 8
RESULTS = Path("/tmp/frc-ab-qm8.json")

TRACKER_SIDE = {"merge_main_guard": True, "sticky_max_dist_m": 2.5}
# Validated qm11 sprint recipe (meanNN 1.105->0.738): seed at the auto-period
# open, motion override + motion-gated low-conf admission, whole-robot union
# seed extents. Shared by the endgame profiles and the backfill A/B below.
SPRINT = {"sam3_prompt_offset_seconds": 0.0, "motion_override": True,
          "motion_gated_conf": True, "seed_union_extent": True}
# Reader base for the qwen35-escape profile: local vLLM reader with the Oct 31
# Clef combo's census + keep-evicted read settings (see that profile).
_QWEN35_R = {"team_reader": "vllm", "read_census": True, "read_keep_evicted": True}
# Reader base for the "speed" profile: the "jev" profile's reader config.
_SPEED_R = {"team_reader": "jev", "read_census": True, "read_keep_evicted": True}
_SPEED_B2 = {**_SPEED_R, "sam3_track_fps": 5, "corner_track_stride": 3,
             "selective_mask_prop": True, "selective_app_weight": 0.3}
_S4_AB = {"team_reader": "jev", "window_labels": True, "vllm_phase_exclusive": False}
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
    # Bumper-reader model A/B/C/D (vlm_model knob, web/vlm_models.py): the
    # incumbent Qwen3-VL-4B vs the three 2026-07-22 candidates, geometry knobs
    # all default so only the reader differs. Read quality shows up in the
    # identity columns (mislabel / slot_map / coverage); read speed comes from
    # the replay's timings JSON (vlm.pass_async) keyed by each arm's job_id.
    # Huihui Mistral 24B was dropped: no vision-intact non-GGUF quant exists.
    "vlm-models": [
        ("qwen3-vl-4b", {}),
        ("glm-4.6v-flash", {"vlm_model": "glm-4.6v-flash"}),
        ("qwen3-vl-8b", {"vlm_model": "qwen3-vl-8b"}),
        ("lfm2.5-vl", {"vlm_model": "lfm2.5-vl"}),
    ],
    # Contenders only (LFM eliminated by the crop bench: 2 confident misreads).
    # Use with --full-match for read-rich comparisons — auto-only windows on
    # qm8/qm12 yielded 0-2 bindings for every model (too few legible moments).
    "vlm-abc": [
        ("qwen3-vl-4b", {}),
        ("glm-4.6v-flash", {"vlm_model": "glm-4.6v-flash"}),
        ("qwen3-vl-8b", {"vlm_model": "qwen3-vl-8b"}),
    ],
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
        ("dual-seed", {"sam3_prompt_offset_seconds": 0.0,
                       "motion_override": True, "motion_gated_conf": True}),
        ("dual-seed+blobs", {"sam3_prompt_offset_seconds": 0.0,
                             "motion_override": True, "motion_gated_conf": True,
                             "motion_seed_blobs": True}),
    ],
    # Blob taming: blobs scoped to the first motion_blob_window_s (6s) of the
    # active period — keeps the sprint rescue, drops the match-long churn.
    "blob-window": [
        ("dual-seed+blobs6s", {"sam3_prompt_offset_seconds": 0.0,
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
        ("union", {"sam3_prompt_offset_seconds": 0.0,
                   "motion_override": True, "motion_gated_conf": True,
                   "seed_union_extent": True}),
        ("union+blobs6s", {"sam3_prompt_offset_seconds": 0.0,
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
                      "sam3_prompt_offset_seconds": 0.0,
                      "opening_reprompt_seconds": 0.5, "opening_window_seconds": 6.0}),
    ],
    # Full A/B pair (baseline + lever1+2) for cross-match regression checking
    # before landing the seed-gate + opening-dense-reprompt config as a default.
    # qm12 was the target match; run this on matches WITHOUT the hub-face sliver
    # failure (qm8/qm10/qm11) to confirm the gates don't regress them.
    "seed-open-ab": [
        ("baseline", {}),
        ("lever1+2", {"seed_box_max_aspect": 3.0, "evict_requires_motion": True,
                      "sam3_prompt_offset_seconds": 0.0,
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
    # Ground-centre offset: push every projected robot ground point 0.475 m
    # (half a 0.95 m robot) away from its view's camera centre — the box/mask
    # bottom is the NEAR bumper edge, not the footprint centre. qm8 hand GT
    # post-hoc: coverage 58->74%, median err 0.66->0.39 m.
    "ground-offset": [
        ("baseline", {}),
        ("ground_offset_0.475", {"ground_center_offset_m": 0.475}),
    ],
    # Repeat of the offset arm alone: identity outcomes on qm8 hinge on a few
    # binding decisions, so one arm can't separate a regression from a swap.
    "ground-offset-repeat": [("ground_offset_0.475_r2", {"ground_center_offset_m": 0.475})],
    # Two more interleaved pairs on the Nov 7 lens-refit calibration: auto
    # labels swing 2-66% at identical settings, so the label question needs n>=3/arm.
    "ground-offset-x2": [("baseline_r3", {}), ("ground_offset_0.475_r3", {"ground_center_offset_m": 0.475}),
                         ("baseline_r4", {}), ("ground_offset_0.475_r4", {"ground_center_offset_m": 0.475})],
    # alliance_by_territory=False: unknown-colour detections no longer take
    # their alliance from the field half (qm8 GT: opponent-half robots ride a
    # wrong-alliance slot 63% of the time). Two interleaved pairs.
    "territory-free": [("baseline_t1", {}), ("territory_free_t1", {"alliance_by_territory": False}),
                       ("baseline_t2", {}), ("territory_free_t2", {"alliance_by_territory": False})],
    # bumper_color_read: per-frame main-view bumper colour read on every
    # main-backed tracker input (qm8 GT offline: own half 74% correct / 0.1%
    # wrong, opponent half 32% / 2.3%). override = decided reads replace the
    # pipeline colour; strict = colour := read (drops stale SAM colours),
    # paired with territory-free so undecided inputs don't fall back to the
    # field half.
    # Identity combo from the Nov 7 GT work: strict per-frame colour + no
    # territory alliance + duplicate-slot guards (offline replay vs GT: slot
    # purity 50 -> 61%, switches 178 -> 88). Two runs (labels are noisy).
    "identity-combo": [("combo_i1", {"bumper_color_read": "strict", "alliance_by_territory": False,
                                     "dup_seed_guard_m": 1.0, "dup_reap_m": 0.7, "dup_reap_updates": 8}),
                       ("combo_i2", {"bumper_color_read": "strict", "alliance_by_territory": False,
                                     "dup_seed_guard_m": 1.0, "dup_reap_m": 0.7, "dup_reap_updates": 8})],
    # Same with OVERRIDE colour (keeps the seed colour when the read is
    # undecided): offline vs GT keeps auto purity at baseline (91%) while
    # full-match purity 50 -> 66%; strict lost auto purity (91 -> 80%).
    "identity-combo-override": [
        ("combo_ovr_i1", {"bumper_color_read": "override", "alliance_by_territory": False,
                          "dup_seed_guard_m": 1.0, "dup_reap_m": 0.7, "dup_reap_updates": 8}),
        ("combo_ovr_i2", {"bumper_color_read": "override", "alliance_by_territory": False,
                          "dup_seed_guard_m": 1.0, "dup_reap_m": 0.7, "dup_reap_updates": 8})],
    # Label stage: hand-GT read analysis (Nov 7) shows team-naming reads are
    # 94-100% right but the whole-match per-slot resolve wastes them; nearest
    # read within the same slot+identity binding (else slot majority) took
    # label correctness 35-46% -> 58-71% offline. identity_segment_resolve is
    # the existing segment-level labeller (default OFF, never run live).
    "segment-labels": [
        ("segments", {"identity_segment_resolve": True}),
        ("segments_strict_combo", {"identity_segment_resolve": True, "bumper_color_read": "strict",
                                   "alliance_by_territory": False, "dup_seed_guard_m": 1.0,
                                   "dup_reap_m": 0.7, "dup_reap_updates": 8})],
    # Window labels (docs/superpowers/specs/2026-11-09-jev-window-labels-design.md):
    # finalize-stage relabel from Jev reads on short track windows. Score each
    # arm with scripts/score_ground_truth.py --replay <job_id> --score-shift
    # 0.475 (this harness scores Manta auto only). Run with --full-match.
    # Deinterlace bumper crops before the reader upscale (bumper_register.deinterlace_crop).
    "deinterlace": [("baseline", {}), ("deinterlace", {"bumper_deinterlace": True})],
    # Treatment arm alone (pair with existing baseline runs of the same code).
    "deinterlace-on": [("deinterlace", {"bumper_deinterlace": True})],
    # Single window-labels arm (event-lineup baseline collection, Nov 10).
    "wl-on": [("wl_08", {"window_labels": True})],
    # Same, plus the post-finalize HUD render (first knob-on end-to-end runs, Nov 10).
    "wl-hud": [("wl_08_hud", {"window_labels": True, "hud_post_render": True})],
    # Window-label tiles deinterlaced before the Jev digit reads
    # (window_labels_deinterlace) vs plain window labels.
    "wl-deinterlace": [("wl_08", {"window_labels": True}),
                       ("wl_08_di", {"window_labels": True, "window_labels_deinterlace": True})],
    # Window-label digit reader: Jev (default) vs the local S1 student first
    # (Jev only for windows S1 leaves undecided) vs S1 alone in the digit pass
    # (window_labels_reader; teacher-distillation spec section 14).
    "wl-student": [("wl_08", {"window_labels": True, "window_labels_reader": "jev"}),
                   ("wl_08_s1", {"window_labels": True, "window_labels_reader": "student"}),
                   ("wl_08_s1only", {"window_labels": True,
                                     "window_labels_reader": "student_only"})],
    # Window-label run fill (window_labels_fill, default ON): unread windows
    # of a track chain take the chain's named team vs stay unlabelled.
    "wl-fill": [("wl_nofill", {"window_labels": True, "window_labels_fill": False}),
                ("wl_fill", {"window_labels": True, "window_labels_fill": True})],
    # Window-label chain roster elimination (window_labels_eliminate,
    # experimental, default OFF): colour-gated third-team labels after the fill.
    "wl-elim": [("wl_noelim", {"window_labels": True, "window_labels_eliminate": False}),
                ("wl_elim", {"window_labels": True, "window_labels_eliminate": True})],
    "window-labels": [
        ("off", {}),
        ("wl_08", {"window_labels": True}),
        ("wl_065", {"window_labels": True, "window_labels_lineup_min_p": 0.65})],
    "bumper-colour": [("baseline_c1", {}),
                      ("colour_override_c1", {"bumper_color_read": "override"}),
                      ("colour_strict_tfree_c1", {"bumper_color_read": "strict",
                                                  "alliance_by_territory": False})],
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
    # Selective mask propagation (arXiv 2606.13033 re-implementation): cheap
    # per-frame TBD everywhere, SAM3 mask propagation only inside ambiguous
    # windows. Attacks the ~77%-of-wall-clock propagation cost; expect Manta
    # scores to hold vs baseline (TBD-only frames still emit boxes, just no
    # mask) while sam_frames/frames in the [selective] log line drops hard.
    "selective": [
        ("baseline", {}),
        ("selective", {"selective_mask_prop": True}),
    ],
    # Margin-entry sweep: selective_margin_entry is in 1 - expanded-IoU cost
    # units (not the paper's Eq. 1 units), so the paper's default is only a
    # start-point. Sweep around it to find this codebase's cost scale.
    "selective-sweep": [
        ("m02", {"selective_mask_prop": True, "selective_margin_entry": 0.02}),
        ("m05", {"selective_mask_prop": True, "selective_margin_entry": 0.05}),
        ("m10", {"selective_mask_prop": True, "selective_margin_entry": 0.10}),
        ("budget08", {"selective_mask_prop": True, "selective_sam_budget_frac": 0.8}),
    ],
    # Same-alliance mask-convergence gap (docs/selective-mask-prop-blue-gap.md):
    # selective mode's qm11 `killed` outcomes fire on partially-overlapping
    # same-alliance robots, not just genuine mask collapse, and the TBD
    # fallback has no appearance term to recover blue coverage. Two candidate
    # fixes: raise the kill threshold (selective_mask_overlap_exit, Task 1) so
    # only near-total collapse kills, and/or coast the last-confident SAM box
    # instead of reverting to raw TBD on a kill (selective_kill_coast, Task 2).
    # Single-arm probe for the kill-coast diagnostics (coast= in the
    # [selective] log line): did _apply_kill_coast write entries, and why did
    # each kill's coast end (no_seed / range / reclaim / budget)?
    "coast-probe": [
        ("sel-coast", {"selective_mask_prop": True, "selective_kill_coast": True}),
    ],
    # Appearance-term sweep (blue-gap fix #3, selective_app_weight): recover
    # blue_meanNN toward the 0.594 baseline without losing the red win
    # (~0.73) or the cost win (main sam_frames <= 100% of uniform). Weight is
    # in combined-cost units (1-cos in [0,2] vs 1-eIoU in [0,1]) -- sweep it.
    # Full-match validation of the winning appearance weight (0.3 — the sweep
    # saturates there; 0.6/1.0 byte-identical on qm11 auto). Run with
    # --full-match --restart-between: Manta position metrics still score the
    # auto window; read teleop value from mislabel/slot_map, per-team
    # coverage, killed counts, wall clock, and VRAM stability across arms.
    "app-full": [
        ("baseline", {}),
        ("selective", {"selective_mask_prop": True}),
        ("sel-app03", {"selective_mask_prop": True, "selective_app_weight": 0.3}),
    ],
    "app-sweep": [
        ("baseline", {}),
        ("selective", {"selective_mask_prop": True}),
        ("sel-app03", {"selective_mask_prop": True, "selective_app_weight": 0.3}),
        ("sel-app06", {"selective_mask_prop": True, "selective_app_weight": 0.6}),
        ("sel-app10", {"selective_mask_prop": True, "selective_app_weight": 1.0}),
    ],
    "blue-gap": [
        ("baseline", {}),
        ("selective", {"selective_mask_prop": True}),
        ("sel-ov95", {"selective_mask_prop": True, "selective_mask_overlap_exit": 0.95}),
        ("sel-ov97", {"selective_mask_prop": True, "selective_mask_overlap_exit": 0.97}),
        ("sel-coast", {"selective_mask_prop": True, "selective_kill_coast": True}),
        ("sel-coast-ov95", {"selective_mask_prop": True, "selective_kill_coast": True,
                            "selective_mask_overlap_exit": 0.95}),
    ],
    # Starvation-triggered windows (blue-gap v2 fix #1, spec docs/superpowers/
    # specs/2026-07-21-selective-starvation-windows-design.md): recover the
    # blue coverage cliffs (3181 parks behind the hub at t=37s -> 52%
    # coverage; 2992's 2s occlusion respawn -> 73%) without losing the
    # appearance-term red win (0.639) or the cost bar (main sam_frames <=
    # 100% of uniform). Run with --full-match --restart-between; watch trig=
    # and outcomes in the [selective] log line, per-team blue coverage, and
    # whether 2992's post-occlusion stream inherits its identity via swap.
    "starve": [
        ("baseline", {}),
        ("sel-app03", {"selective_mask_prop": True,
                       "selective_app_weight": 0.3}),
        ("sel-app03-starve", {"selective_mask_prop": True,
                              "selective_app_weight": 0.3,
                              "selective_starve_windows": True}),
    ],
    # Jul 21 starve A/B verdict: 3181 FIXED (cov 52->99, meanNN 0.455 beats
    # baseline 0.614) but UNCAPPED starve windows blew the corner budgets to
    # 100% of uniform (26%/42% in control), main dropped 34 windows on
    # budget, and 4786 regressed 2.31->3.11 with a 9.7s start lag (budget
    # cannibalization and/or junk-track starve swaps; renames 7->17). Probe:
    # cap starve windows at 240 frames (~28s at full-match fps) -- long
    # enough to carry 3181 through the scored window, short enough to stop
    # corner junk tracks from eating the whole budget.
    "starve-cap": [
        ("sel-app03-starve240", {"selective_mask_prop": True,
                                 "selective_app_weight": 0.3,
                                 "selective_starve_windows": True,
                                 "selective_starve_max_frames": 240}),
    ],
    # Junk-gate probe (starve v1.1): min-obs gate should keep 3181's recovery
    # (cov 52->99, meanNN 0.455) while restoring 4786 (~2.31 @ 26, lag 0) and
    # the corner budgets (26%/42% of uniform in control) by refusing starve
    # windows to transient junk tracks. Single arm; compare against the
    # existing baseline / sel-app03 / sel-app03-starve rows from
    # /tmp/frc-ab-starve-qm11.json.
    "starve-gate": [
        ("sel-app03-starve-gate", {"selective_mask_prop": True,
                                   "selective_app_weight": 0.3,
                                   "selective_starve_windows": True}),
    ],
    # Default-ON candidacy (2026-07-20): the three merged-but-OFF tech-upgrade
    # features, each isolated vs baseline (the geo4 lesson: stacked-only runs
    # can't attribute a regressor), plus the stacked arm twice — the stack IS
    # the proposed new default, and the repeat gauges rerun stability
    # (global_identity_resolve's determinism claim; qm11 baseline reproduced
    # 0.816 three times, so any drift is the arm's).
    # rts_smooth: offline path rewrite + gap fill -> expect meanNN/coverage
    # gains at ~flat job time. global_identity_resolve: meanNN-blind identity
    # feature -> read mislabel/slot_map. sam3_multiplex: NOT a pure perf
    # toggle (different builder heuristics) -> quality parity + wall-clock.
    "default-on": [
        ("baseline", {}),
        ("rts", {"rts_smooth": True}),
        ("gir", {"global_identity_resolve": True}),
        ("mux", {"sam3_multiplex": True}),
        ("all3", {"rts_smooth": True, "global_identity_resolve": True,
                  "sam3_multiplex": True}),
        ("all3-b", {"rts_smooth": True, "global_identity_resolve": True,
                    "sam3_multiplex": True}),
    ],
    # Cross-match confirmation (qm11 verdicts: rts regresses the headline
    # metric via gap-fill insertions, mux trades blue collapse for red gains,
    # gir needed the collision-sweep fix): per-feature isolation again, no
    # stack arm — nothing earned a stacked default on qm11.
    "default-on-cross": [
        ("baseline", {}),
        ("rts", {"rts_smooth": True}),
        ("gir", {"global_identity_resolve": True}),
        ("mux", {"sam3_multiplex": True}),
    ],
    # Single gir arm: live re-validation of the merge collision-sweep fix on
    # the match that produced the duplicate label (expect R2 assigned AND the
    # colliding stale label stripped -> 2 changes, no dup teams in replay).
    "gir-single": [
        ("gir-fixed", {"global_identity_resolve": True}),
    ],
    # Single mux arm: re-validation after the init_state offload-kwarg shim
    # (before it, EVERY view raised TypeError at session start and the whole
    # job silently ran the no-SAM3 fallback — the 2026-07-20 "mux" numbers
    # measured that, not multiplex). Grep the job log for "Error during
    # propagation" = 0 before reading the metrics.
    "mux-single": [
        ("mux-fixed", {"sam3_multiplex": True}),
    ],
    # Qwen3.5-4B + escape read prompt (vlm_models "qwen3.5-4b", read_prompt=
    # "escape"; 2026-11-05 200-crop bench: 70/90, 0 wrong, 2% FP at 0.9 s/crop)
    # vs the incumbent Qwen3-VL-4B on the same vLLM read path. Run with
    # --full-match --restart-between. R matches the Oct 31 Clef combo baselines
    # (3e95c939 qm8 / 73d2a195 qm11: team_reader=clef + read_census +
    # read_keep_evicted), so against those only the reader differs; the two
    # arms here differ ONLY in vlm_model. Audit escape reads via the `subject`
    # field on the vlm_read diagnostic rows.
    "qwen35-escape": [
        ("q35-4b-escape", {**_QWEN35_R, "vlm_model": "qwen3.5-4b"}),
        ("qwen3-vl-4b", {**_QWEN35_R}),
    ],
    # TypeSafe Jev 1.14 cloud reader (web/jev_reader.py; 200-crop bench
    # 2026-11-05: argmax 59/90 legible, 0 wrong, 2% FP; + tentative reads at
    # p >= 0.17 on an "unreadable" top, voting p x camera weight and never
    # locking). Same census + keep-evicted read settings as the Oct 31 Clef
    # combo baselines (3e95c939 qm8 / 73d2a195 qm11: team_reader=clef +
    # read_census + read_keep_evicted) and the qwen35-escape profile, so only
    # the reader differs. Run with --full-match. Audit tentative reads via
    # the `tentative` / `runner_up` fields on the vlm_read diagnostic rows;
    # cost lands in replay timings (jev.requests / jev.cost_cents).
    "jev": [
        ("jev", {"team_reader": "jev", "read_census": True, "read_keep_evicted": True}),
    ],
    # Live-event speed A/B (2026-11-05): full-match wall NEED < 12 min, WANT
    # < 8 min (from ~24 min) without a meaningful accuracy drop. Arm A is the
    # "jev" profile run above (same code, all defaults: sam3_track_fps 8,
    # corner_track_stride 2) -- not re-run here. Every arm shares the jev
    # reader config, so only the speed levers differ:
    #   B1  track at 5 fps, corners at stride 3 (dense head unchanged);
    #   B2  + selective mask propagation (SAM only in ambiguous windows) with
    #       the 0.3 appearance term. CAUTION: selective's frame-denominated
    #       knobs (gap_trigger 7, min_streak 5, window_max 120, starve/coast
    #       frames) are NOT rescaled for fps/stride the way the uniform
    #       tracker's are -- at 5 fps / stride 3 a corner "frame" is 0.6 s
    #       (vs 0.125 s at the 8 fps / stride 1 it was tuned at);
    #   B3  + stream_analysis (loop's OCR / main YOLO / person det prefetched
    #       during SAM propagation; web/analysis_prefetch.py).
    # Run with --full-match. Wall time per stage (timings.total,
    # timings.sam3_tracking, timings.analysis_loop, stream.*) lands in each
    # result row's "timings", read from the replays table by job_id.
    "speed": [
        ("B1-fps5-cs3", {**_SPEED_R, "sam3_track_fps": 5, "corner_track_stride": 3}),
        ("B2-+selective", _SPEED_B2),
        ("B3-+stream", {**_SPEED_B2, "stream_analysis": True}),
    ],
    # Nov 6: selective cut auto coverage at fps5 (qm8 2724 100->32%), so
    # trade it for a lower track rate instead: fps4 + corner stride 3 + stream.
    "speed-b4": [
        ("B4-fps4-cs3-stream", {**_SPEED_R, "sam3_track_fps": 4,
                                "corner_track_stride": 3, "stream_analysis": True}),
    ],
    # Nov 6: B4 + SAM 3.1 Object Multiplex (joint multi-object propagation).
    # SAM costs ~500 ms/frame/view regardless of fps because the base
    # predictor propagates objects one at a time; multiplex is the remaining
    # lever toward < 8 min. Compare against speed-b4 (qm8 754 s / 0.795,
    # qm11 704 s / 0.751). Run with --full-match --restart-between.
    "speed-mux": [
        ("B4-mux", {**_SPEED_R, "sam3_track_fps": 4, "corner_track_stride": 3,
                    "stream_analysis": True, "sam3_multiplex": True}),
    ],
    # Full-match multiplex OOMs at the default 200-frame chunk (10-12
    # singleton object states per session); 100-frame chunks halve the
    # per-session accumulation.
    "speed-mux-c100": [
        ("B4-mux-c100", {**_SPEED_R, "sam3_track_fps": 4, "corner_track_stride": 3,
                         "stream_analysis": True, "sam3_multiplex": True,
                         "sam3_chunk_frames": 100}),
    ],
    # Nov 11: S4 same-robot embedder (teacher distillation §16) vs the
    # default DINOv3 embedder, everything else at the live defaults
    # (window_labels ON with the local S1 reader). window_labels is set
    # explicitly so the harness waits for the finalize stage;
    # vllm_phase_exclusive off so a second-server run never stops :8006
    # (the jev reader needs no vLLM). Run with --full-match; score vs hand GT
    # with scripts/relabel_replay.score_numbers.
    "s4-embedder": [
        ("emb-dinov3", {**_S4_AB, "robot_embedder": "dinov3"}),
        ("emb-s4", {**_S4_AB, "robot_embedder": "s4"}),
    ],
    "s4-embedder-dinov3": [("emb-dinov3", {**_S4_AB, "robot_embedder": "dinov3"})],
    "s4-embedder-s4": [("emb-s4", {**_S4_AB, "robot_embedder": "s4"})],
}
CONFIGS = PROFILES["full"]


def load_event() -> None:
    """Load EVENT_KEY (default 2026cur) so the server has a calibrated event (jobs read event_key
    from the global current_event); without it inference aborts immediately."""
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
         "-d", json.dumps({"event_key": EVENT_KEY}), f"http://{HOST}/api/event/load"],
        capture_output=True, text=True, timeout=180).stdout
    if '"event_key"' not in out:
        raise RuntimeError(f"event load failed: {out[:200]}")


def restart_server() -> None:
    """Stop/start the app between arms (full-match VRAM hygiene) and reload
    the event. Mirrors web/stop_all.sh + web/start_all.sh + load_event()."""
    subprocess.run([str(REPO / "web/stop_all.sh")], check=False, timeout=120)
    subprocess.run([str(REPO / "web/start_all.sh")], check=True, timeout=120)
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 f"http://{HOST}/"], capture_output=True, text=True,
                timeout=10).stdout
            if out in ("200", "404"):
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("server did not come back after restart")
    load_event()


def post_job(cfg: dict) -> str:
    """Create a job via /api/infer (curl multipart); return job_id."""
    config_json = json.dumps({"auto_period_only": AUTO_ONLY, **cfg})
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


def _wait_window_labels(job_id: str, timeout_s: float = 600.0, delay: float = 2.0) -> bool:
    """With window_labels on, the job reports "completed" before the finalize
    stage runs, so the replay row and meta.json appear minutes later. Wait for
    the replay and for its summary to carry the stage's stats. 600 s = the
    stage's 360 s deadline plus the re-resolve, reader shutdown and the
    replay's video encode. Warns (but still returns True) when the stage
    aborted."""
    meta = REPO / "web" / "replays" / job_id / "meta.json"
    end = time.time() + timeout_s
    while time.time() < end:
        if _replay_saved(job_id, tries=1, delay=0.0) and meta.exists():
            try:
                summary = json.loads(meta.read_text()).get("summary") or {}
                if "window_labels" in summary:
                    reason = (summary["window_labels"] or {}).get("aborted")
                    if reason:
                        print(f"  WARNING: window_labels aborted for {job_id}: {reason}",
                              flush=True)
                    return True
            except (OSError, ValueError):
                pass
        time.sleep(delay)
    print(f"  WARNING: replay {job_id} has no window_labels summary after {timeout_s:.0f}s",
          flush=True)
    return False


# Per-stage wall-time keys copied from the replay's timings JSON into each
# result row (prefix match for the stream_analysis "stream.*" stats).
# "sam2_tracking" is the pre-rename name of "sam3_tracking" (older replays).
TIMING_KEYS = ("total", "sam3_tracking", "sam2_tracking", "analysis_loop",
               "frame_export", "model_load")


def replay_timings(job_id: str | None) -> dict:
    """Stage wall times for ``job_id`` from the replays table (read-only):
    TIMING_KEYS plus any ``stream.*`` keys. {} if missing/unreadable."""
    import sqlite3
    if not job_id:
        return {}
    db = REPO / "frc_scorer.db"
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT timings FROM replays WHERE id = ?",
                              (job_id,)).fetchone()
        finally:
            con.close()
        t = json.loads(row[0]) if row and row[0] else {}
    except (sqlite3.Error, ValueError, TypeError):
        return {}
    return {k: v for k, v in t.items()
            if k in TIMING_KEYS or k.startswith("stream.")}


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


def score(expect_replay: str | None = None) -> dict:
    """Score the newest 2026cur_qm8 replay vs Manta: aggregate mean-NN and
    per-alliance red coverage/tracked-count.

    ``expect_replay``: the arm's job_id. load_our_auto_paths keeps the newest
    replay PER KEY THAT HAS PATHS, so a job that exports zero autonomous_paths
    silently inherits the previous replay's score (2026-07-20: the post-shim
    mux arm scored the prior gir arm's replay bit-for-bit). Refuse that."""
    # Score ONLY this job's replay: load_our_auto_paths is newest-per-SLOT,
    # so a replay missing a slot silently inherits it from an older run.
    ours = load_our_auto_paths("2026cur", only_replay=expect_replay)
    if expect_replay is not None:
        rids = {rec["replay_id"] for key, rec in ours.items() if key[0] == MATCH}
        if expect_replay not in rids:
            return {"error": f"stale replay scored (job {expect_replay[:8]} "
                             f"exported no paths; newest with paths: "
                             f"{sorted(r[:8] for r in rids)})"}
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
        # Identity metrics: scoring pairs geometrically, so a label-only
        # feature (global_identity_resolve) is invisible to meanNN. Count
        # slot-label vs Manta-DS-position mismatches and keep the mapping
        # for cross-run stability diffs.
        "mislabel": sum(1 for p in pairs if p["our_slot"][1:] != str(p["manta_pos"])),
        "slot_map": {p["our_slot"]: f"{p['manta_pos']}:{p['manta_team']}"
                     for p in pairs},
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


def main(args) -> None:
    global AUTO_ONLY
    AUTO_ONLY = not args.full_match
    load_event()
    print("event 2026cur loaded", flush=True)
    results = []
    for idx, (name, cfg) in enumerate(CONFIGS):
        if args.restart_between and idx > 0:
            print("  restarting server between arms...", flush=True)
            restart_server()
        t0 = time.time()
        print(f"\n=== [{name}] config={cfg} ===", flush=True)
        jid, job_s = None, None
        try:
            jid = post_job(cfg)
            print(f"  job_id={jid}; driving...", flush=True)
            status = asyncio.run(drive(jid))
            job_s = round(time.time() - t0)
            print(f"  status={status} in {job_s}s", flush=True)
            if status == "completed" and cfg.get("window_labels"):
                _wait_window_labels(jid)
            time.sleep(3)  # let the replay finish writing
            metrics = score(expect_replay=jid) if status == "completed" else {"error": status}
        except Exception as e:
            metrics = {"error": repr(e)}
        row = {"config": name, "cfg": cfg, "job_id": jid, "job_s": job_s,
               "timings": replay_timings(jid), "metrics": metrics}
        results.append(row)
        RESULTS.write_text(json.dumps(results, indent=2))  # stream partial
        print(f"  metrics={metrics}", flush=True)
        if row["timings"]:
            print(f"  timings={row['timings']}", flush=True)
    # summary table
    print(f"\n\n===== A/B SUMMARY ({MATCH_KEY}) =====", flush=True)
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
    ap.add_argument("--event", default=EVENT_KEY, help="event loaded before the jobs (default 2026cur)")
    ap.add_argument("--host", default=HOST, help="app server host:port (default localhost:8004)")
    ap.add_argument("--profile", choices=list(PROFILES), default="full")
    ap.add_argument("--full-match", action="store_true",
                    help="process whole matches (auto_period_only=False) unless an arm overrides it")
    ap.add_argument("--restart-between", action="store_true",
                    help="restart the app server between arms (VRAM hygiene for full-match sweeps)")
    a = ap.parse_args()
    MATCH_KEY, MATCH, VIDEO, RESULTS = a.match_key, a.match, Path(a.video), Path(a.results)
    HOST = a.host
    EVENT_KEY = a.event
    if a.restart_between and HOST != "localhost:8004":
        sys.exit("--restart-between restarts the :8004 server via web/*_all.sh; not allowed with --host")
    CONFIGS = PROFILES[a.profile]
    main(a)
