"""Unit tests for web/field/gt_scoring.py (pure; synthetic replays + GT docs)."""
import numpy as np
import pytest

from web.field.gt_scoring import (
    AlignmentError, align_gt, gt_track, pipeline_at, replay_from_meta, score_match,
    select_samples,
)

FP = "123:abc"


def meta(frames, *, ranges=None, auto=(0, 1200), out_fps=7.5):
    """frames: {frame_num: [(slot, team, (x, y)), ...]}"""
    fr = [{"frame_num": f, "robots": [
        {"slot": s, "alliance": "blue" if s[0] == "B" else "red",
         "team_number": t, "confidence": True, "field_pos": list(p)} for s, t, p in rs]}
        for f, rs in sorted(frames.items())]
    fns = sorted(frames)
    return {"frames": fr, "summary": {
        "fps": out_fps,
        "processing_ranges": ranges or [[fns[0], fns[-1]]],
        "periods": [{"label": "auto", "start_frame": auto[0], "end_frame": auto[1]}] if auto else [],
    }}


def gt(team, kfs, *, fp=FP, fps=60, t0=None, alliance="blue", station=1):
    """kfs: [(frame, state, (x, y) | None, extra-dict?)]"""
    out = []
    for k in kfs:
        f, st, xy = k[:3]
        kf = {"frame": f, "state": st}
        if xy is not None:
            kf["uv"] = [0, 0]
            kf["field_xy"] = list(xy)
        if len(k) > 3:
            kf.update(k[3])
        out.append(kf)
    return {"match_key": "m", "team": team, "alliance": alliance, "station": station,
            "video": {"name": "v.mp4", "fingerprint": fp, "fps": fps},
            "match_t0_frame": t0, "keyframes": out}


def line(x0, vx, f):
    """Position (m) at source frame f moving vx m/s in x."""
    return (x0 + vx * f / 60.0, 2.0)


def straight_replay(slots, frames=range(0, 1201, 8), **kw):
    """slots: {slot: (team, x0, vx)} -> robots moving along y=2 (+ slot offset)."""
    fr = {}
    for f in frames:
        fr[f] = [(s, t, (x0 + vx * f / 60.0, 2.0 + 2 * i))
                 for i, (s, (t, x0, vx)) in enumerate(sorted(slots.items()))]
    return replay_from_meta(meta(fr, **kw), fingerprint=FP)


def score(docs, rt, **kw):
    kw.setdefault("eval_step_s", 0)
    return score_match(docs, rt, **kw)


def robot(res, team):
    return next(r for r in res["robots"] if r["team"] == team)


def test_src_fps_inferred_from_output_fps_and_spacing():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)})
    assert rt.spacing == 8 and rt.src_fps == 60.0


def test_perfect_track():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)})
    doc = gt("100", [(f, "visible", (1.0 + f / 60.0, 2.0)) for f in range(0, 1201, 15)])
    m = robot(score([doc], rt), "100")["visible"]
    assert m["coverage"] == 1.0 and m["err"]["mean"] == 0.0
    assert m["label_correct"] == 1.0 and m["slot_switches"] == 0
    assert m["label_err"]["mean"] == 0.0 and m["label_on_wrong_robot"] == 0.0


def test_constant_offset():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)})
    doc = gt("100", [(f, "visible", (1.0 + f / 60.0, 2.3)) for f in range(0, 1201, 15)])
    m = robot(score([doc], rt), "100")["visible"]
    assert m["coverage"] == 1.0
    assert abs(m["err"]["mean"] - 0.3) < 1e-6 and abs(m["err"]["p90"] - 0.3) < 1e-6
    # outside the radius: unmatched, but the label is still found 1.3 m away
    doc2 = gt("100", [(f, "visible", (1.0 + f / 60.0, 3.3)) for f in range(0, 1201, 15)])
    m2 = robot(score([doc2], rt), "100")["visible"]
    assert m2["coverage"] == 0.0 and m2["label_present"] == 1.0
    assert m2["label_on_wrong_robot"] == 1.0 and abs(m2["label_err"]["mean"] - 1.3) < 1e-6


def test_interpolation_between_replay_frames():
    # replay every 8 frames, GT on odd frames: linear motion -> zero error
    rt = straight_replay({"B1": ("100", 1.0, 3.0)})
    doc = gt("100", [(f, "visible", (1.0 + 3.0 * f / 60.0, 2.0)) for f in range(3, 1190, 7)])
    m = robot(score([doc], rt), "100")["visible"]
    assert m["coverage"] == 1.0 and m["err"]["p90"] < 1e-6


def test_pipeline_gap_rules():
    fr = {f: [("B1", "1", (f, 0))] for f in (0, 8, 100, 108, 116, 124)}
    rt = replay_from_meta(meta(fr, ranges=[[0, 124]]))
    assert rt.spacing == 8
    has, xy, _ = pipeline_at(rt, [4, 6, 8, 11, 30, 97])
    assert list(has) == [True, True, True, True, False, True]
    assert xy[0, 0, 0] == 4 and xy[1, 0, 0] == 6        # interpolated
    assert xy[3, 0, 0] == 8 and xy[5, 0, 0] == 100      # nearest within half spacing
    assert np.isnan(xy[4, 0, 0])


def test_label_swap_mid_match():
    # Two robots; at frame 600 the pipeline swaps which slot carries which label
    # *and* the B1 slot jumps onto robot 200's path.
    fr = {}
    for f in range(0, 1201, 8):
        a, b = line(1.0, 1.0, f), (line(1.0, 1.0, f)[0], 5.0)
        if f < 600:
            fr[f] = [("B1", "100", a), ("B2", "200", b)]
        else:
            fr[f] = [("B1", "200", a), ("B2", "100", b)]
    rt = replay_from_meta(meta(fr), fingerprint=FP)
    d100 = gt("100", [(f, "visible", line(1.0, 1.0, f)) for f in range(0, 1201, 15)])
    d200 = gt("200", [(f, "visible", (line(1.0, 1.0, f)[0], 5.0)) for f in range(0, 1201, 15)],
              station=2)
    res = score([d100, d200], rt)
    m = robot(res, "100")["visible"]
    assert m["coverage"] == 1.0 and m["err"]["mean"] == 0.0
    assert 0.45 < m["label_correct"] < 0.55 and 0.45 < m["label_wrong"] < 0.55
    assert m["slot_switches"] == 0          # geometry stayed on slot B1
    # labelled track: before the swap on the right robot, after it 3 m away
    assert 0.45 < m["label_on_wrong_robot"] < 0.55

    # now the slot itself hops: GT 100 is matched B1 then B2
    fr2 = {}
    for f in range(0, 1201, 8):
        a, b = line(1.0, 1.0, f), (line(1.0, 1.0, f)[0], 5.0)
        fr2[f] = ([("B1", "100", a), ("B2", "200", b)] if f < 600
                  else [("B1", "100", b), ("B2", "200", a)])
    res2 = score([d100, d200], replay_from_meta(meta(fr2), fingerprint=FP))
    m2 = robot(res2, "100")["visible"]
    assert m2["slot_switches"] == 1 and 0.45 < m2["label_wrong"] < 0.55
    assert res2["headline"]["slot_switches"] == 2


def test_unprocessed_excluded():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)}, frames=range(0, 601, 8),
                         ranges=[[0, 600]])
    doc = gt("100", [(f, "visible", line(1.0, 1.0, f)) for f in range(0, 1201, 15)])
    m = robot(score([doc], rt), "100")["visible"]
    assert m["n_unprocessed"] == len([f for f in range(0, 1201, 15) if f > 600])
    assert m["coverage"] == 1.0


def test_fingerprint_mismatch_aligns_by_match_t0():
    # replay auto starts at source frame 420; GT video is a 30 fps download whose
    # auto starts at frame 100 -> rf = 420 + (g - 100) * 2
    rt = straight_replay({"B1": ("100", 1.0, 1.0)}, frames=range(400, 1601, 8),
                         auto=(420, 1650))
    kfs = [(g, "visible", line(1.0, 1.0, 420 + (g - 100) * 2)) for g in range(100, 690, 8)]
    doc = gt("100", kfs, fp="other", fps=30, t0=100)
    res = score([doc], rt)
    r = robot(res, "100")
    assert r["alignment"] == "match_t0"
    assert r["visible"]["coverage"] == 1.0 and r["visible"]["err"]["p90"] < 1e-6
    with pytest.raises(AlignmentError):
        score([gt("100", kfs, fp="other", fps=30, t0=None)], rt)
    forced = score([gt("100", kfs, fp="other", fps=30)], rt, force=True)
    assert forced["warnings"] and robot(forced, "100")["alignment"] == "forced"


def test_hungarian_prevents_double_claim():
    # one pipeline robot between two GT robots, nearer to A only by a bit; B
    # must not also match it.
    fr = {f: [("B1", "100", (5.0, 2.0))] for f in range(0, 401, 8)}
    rt = replay_from_meta(meta(fr), fingerprint=FP)
    a = gt("100", [(f, "visible", (5.0, 2.4)) for f in range(0, 401, 15)])
    b = gt("200", [(f, "visible", (5.0, 1.5)) for f in range(0, 401, 15)], station=2)
    res = score([a, b], rt)
    ma, mb = robot(res, "100")["visible"], robot(res, "200")["visible"]
    # both are within the 1 m gate; the cheaper pairing (A, 0.4 m) wins, B is unmatched
    assert ma["coverage"] == 1.0 and abs(ma["err"]["mean"] - 0.4) < 1e-6
    assert mb["coverage"] == 0.0
    # B's keyframes are on a different phase: A is interpolated at B's times
    b2 = gt("200", [(f, "visible", (5.0, 1.5)) for f in range(7, 401, 15)], station=2)
    res2 = score([a, b2], rt)
    assert robot(res2, "100")["visible"]["coverage"] + \
        robot(res2, "200")["visible"]["coverage"] == pytest.approx(1.0, abs=0.05)


def test_estimated_vs_visible_separated():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)})
    kfs = [(f, "visible" if f < 600 else "estimated",
            line(1.0, 1.0, f) if f < 600 else (line(1.0, 1.0, f)[0], 2.5))
           for f in range(0, 1201, 15)]
    kfs.append((1203, "unknown", None))
    r = robot(score([gt("100", kfs)], rt), "100")
    assert r["visible"]["err"]["mean"] == 0.0
    assert abs(r["estimated"]["err"]["mean"] - 0.5) < 1e-6
    assert r["all"]["n"] == r["visible"]["n"] + r["estimated"]["n"]
    assert r["n_unknown"] == 1


def test_eval_step_makes_density_irrelevant():
    rt = straight_replay({"B1": ("100", 1.0, 1.0)})
    dense = gt("100", [(f, "visible", line(1.0, 1.0, f)) for f in range(0, 1201, 2)])
    sparse = gt("100", [(f, "visible", line(1.0, 1.0, f)) for f in range(0, 1201, 15)])
    md = robot(score([dense], rt, eval_step_s=0.25), "100")["visible"]
    ms = robot(score([sparse], rt, eval_step_s=0.25), "100")["visible"]
    assert md["n"] == ms["n"] == 81
    tr = gt_track(dense, rt)
    assert len(select_samples(tr, 0.25, 0.0, 60.0)) == 81
    assert len(select_samples(tr, 0, 0.0, 60.0)) == 601


def test_lag_shift_follow_keyframes():
    # A follow pass that trails the robot by 200 ms of wall time at 0.5x
    # playback = 100 ms of video = 6 frames.
    rt = straight_replay({"B1": ("100", 1.0, 3.0)})
    kfs = [(f + 6, "visible", line(1.0, 3.0, f), {"src": "follow", "rate": 0.5})
           for f in range(0, 1190, 2)]
    doc = gt("100", kfs)
    lagged = robot(score([doc], rt), "100")["visible"]
    fixed = robot(score([doc], rt, lag_ms=200), "100")["visible"]
    assert abs(lagged["err"]["median"] - 0.3) < 1e-6      # 3 m/s * 0.1 s
    assert fixed["err"]["p90"] < 1e-6


def test_auto_period_window():
    rt = straight_replay({"B1": ("100", 1.0, 0.5)}, frames=range(0, 2401, 8),
                         auto=(420, 1650))
    doc = gt("100", [(f, "visible", line(1.0, 0.5, f)) for f in range(0, 2401, 15)])
    full = robot(score([doc], rt), "100")["visible"]
    auto = robot(score([doc], rt, period="auto"), "100")["visible"]
    assert auto["n"] == len([f for f in range(0, 2401, 15) if 420 <= f <= 1650])
    assert full["n"] == len(range(0, 2401, 15))


def test_align_gt_identity():
    al = align_gt(gt("1", []), replay_fingerprint=FP, replay_auto_start=None,
                  replay_src_fps=60)
    assert al["mode"] == "fingerprint" and al["scale"] == 1 and al["offset"] == 0
