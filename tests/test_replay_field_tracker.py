"""scripts/replay_field_tracker.py: a TRACKER_TRACE dump replays exactly, and the
A/B metrics count what they claim."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import replay_field_tracker as R  # noqa: E402
from web.field.tracker import FieldDetection, FieldTracker  # noqa: E402

DT = 8 / 60.0


def _trace_from_run(frames, record_occ=True, **kw):
    """Run a tracker over ``frames`` [(dets, occ)] and log it like app.py."""
    trk = FieldTracker(16.541, **kw)
    trace = []
    for f, (dets, occ) in enumerate(frames):
        out = trk.update(dets, occ, DT)
        rec = {
            "frame": 100 + 8 * f, "dt": DT,
            "fused": [{"pos": [round(d.pos[0], 4), round(d.pos[1], 4)],
                       "cams": ["main"] if d.box is not None else ["corner_right"],
                       "has_main_box": d.box is not None, "color": d.color,
                       "tid": d.track_id} for d in dets],
            "tracks": [{"slot": t.slot_id, "pos": [round(t.pos[0], 4), round(t.pos[1], 4)],
                        "state": t.state, "corner_only": t.corner_only} for t in out],
            "det_to_slot": {int(k): int(v) for k, v in trk.last_det_to_slot.items()},
            "events": [dict(e) for e in trk.binding_events],
            "raw": {"main": [{"pos": [round(d.pos[0], 4), round(d.pos[1], 4)],
                              "color": d.color, "has_box": True, "tid": 50 + i}
                             for i, d in enumerate(dets) if d.box is not None]},
        }
        if record_occ:
            rec["occ"] = {int(s): bool(v.get("view_occluded")) for s, v in occ.items()}
        if f == 0:
            rec["tracker_cfg"] = trk.config_dict()
        trace.append(rec)
    return trace


def _frames():
    """Two blue robots; robot B hides for 14 frames while 'occluded' (coast
    budget 90) and comes back; a red robot drifts across."""
    frames = []
    for f in range(40):
        dets = [FieldDetection(pos=(2 + 0.05 * f, 2.0), color="blue", box=(0, 0, 1, 1)),
                FieldDetection(pos=(12 - 0.1 * f, 6.0), color="red", box=None)]
        if not 10 <= f < 24:
            dets.append(FieldDetection(pos=(5.0, 5.0 + 0.02 * f), color="blue",
                                       box=(0, 0, 1, 1)))
        occ = {1: {"view_occluded": 10 <= f < 24}}
        frames.append((dets, occ))
    return frames


def test_replay_reproduces_traced_run_exactly():
    trace = _trace_from_run(_frames(), sticky_max_dist_m=2.5)
    cfg = R.trace_tracker_cfg(trace)
    rep = R.replay(trace, FieldTracker(**cfg), R.occlusion_traced(trace))
    f = R.fidelity(trace, rep)
    assert f["det_to_slot_frames"] == 1.0
    assert f["track_state_frames"] == 1.0
    assert f["track_pos_frames"] == 1.0


def test_oracle_occlusion_reproduces_untraced_occlusion():
    # Old traces have no occlusion flags; the oracle infers them from which
    # slots the log kept alive, so the 14-frame occluded coast survives.
    trace = _trace_from_run(_frames(), record_occ=False, sticky_max_dist_m=2.5)
    rep = R.replay(trace, R.make_tracker(sticky_max_dist_m=2.5), R.occlusion_oracle(trace))
    assert R.fidelity(trace, rep)["det_to_slot_frames"] == 1.0
    assert R.fidelity(trace, rep)["track_state_frames"] == 1.0
    # Without occlusion the hidden slot expires after 8 missed frames.
    rep2 = R.replay(trace, R.make_tracker(sticky_max_dist_m=2.5), lambda k, p: {})
    assert R.fidelity(trace, rep2)["track_state_frames"] < 1.0


def test_position_occlusion_transfers_to_whoever_holds_the_spot():
    trace = _trace_from_run(_frames(), record_occ=False, sticky_max_dist_m=2.5)
    rep = R.replay(trace, R.make_tracker(sticky_max_dist_m=2.5), R.occlusion_position(trace))
    assert R.fidelity(trace, rep)["track_state_frames"] == 1.0


def test_evaluate_read_purity_and_coverage():
    trace = _trace_from_run(_frames(), sticky_max_dist_m=2.5)
    rep = R.replay(trace, FieldTracker(**R.trace_tracker_cfg(trace)),
                   R.occlusion_traced(trace))
    # Fused det 0 is blue robot A every frame; det 2 (when present) robot B.
    reads = [{"frame": trace[k]["frame"], "team": "111", "conf": 0.9, "det": 0}
             for k in (2, 12, 30)]
    reads += [{"frame": trace[k]["frame"], "team": "222", "conf": 0.9, "det": 2}
              for k in (3, 30)]
    reads.append({"frame": trace[5]["frame"], "team": "333", "conf": 0.3, "det": 0})
    m = R.evaluate(trace, rep, reads)
    assert m["conf_reads"] == 5
    assert m["pair_agree"] == (3, 3)            # A: 2 pairs, B: 1 pair
    assert m["slot_majority_share"] == 1.0
    assert m["reads_lost"] == 0
    assert m["visible_frac"]["B1"] == 1.0
    assert m["coast_frac"]["B2"] > 0           # B's occluded coast
    assert m["far_seeds"] == 0 and m["oid_steals"] == 0


def test_evaluate_counts_far_reseed_as_discontinuity():
    # Blue robot vanishes at (2, 2) for 10 frames (slot expires), then a blue
    # robot appears at (14, 7): the lowest free slot teleports 13 m.
    frames = []
    for f in range(30):
        dets = [] if 5 <= f < 15 else [FieldDetection(
            pos=(2.0, 2.0) if f < 5 else (14.0, 7.0), color="blue", box=(0, 0, 1, 1))]
        frames.append((dets, {}))
    trace = _trace_from_run(frames)
    rep = R.replay(trace, R.make_tracker(), R.occlusion_traced(trace))
    reads = [{"frame": trace[2]["frame"], "team": "111", "conf": 0.9, "det": 0},
             {"frame": trace[20]["frame"], "team": "999", "conf": 0.9, "det": 0}]
    m = R.evaluate(trace, rep, reads)
    assert m["far_seeds"] == 1
    assert m["across_agree"] == (0, 1)
    assert m["within_agree"] == (0, 0)
    assert m["segments_multi_team"] == 0
