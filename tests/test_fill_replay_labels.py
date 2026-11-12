"""scripts/fill_replay_labels.py: label / point reconstruction from a
finalised replay, the refusals, the atomic write with a one-time backup,
--force from the backup, segments and --render. Everything under tmp_path;
the DB and the renderer are stubbed."""
import copy
import json

import pytest

import scripts.fill_replay_labels as fr
from web.field import window_labels as wl
from tests.field.test_window_labels import FPS, _frames, _still

ROSTER = ["2637", "2724", "6497", "3128", "6773", "9066"]


def _live_meta(point_calls=None, sources="s1"):
    """A replay as a fill-less live stage leaves it: B1 tracked on frames
    0-59 (three 20-frame windows), only window 2 named 3128 (digits); B2 on
    frames 0-19 far away, named 6773 by the lineup, with frame 5 cleared."""
    frames = _frames({"B1": _still(range(60)), "B2": _still(range(20), (12.0, 6.0))}, 60)
    ws = wl.build_windows(frames, FPS)
    b1 = [w for w in ws if w.slot == "B1"]
    b2 = next(w for w in ws if w.slot == "B2")
    lab, _ = wl.assemble_labels(ws, {b1[2].wid: ("3128", wl.RANK_DIGITS, 0.9),
                                     b2.wid: ("6773", wl.RANK_LINEUP, 0.85)})
    lab[(5, "B2")] = None
    wl.apply_labels(frames, lab)
    calls = point_calls or {}
    raw = {"calls": {str(w): c for w, c in calls.items()}, "fingerprints": {},
           "roster": ROSTER, "model_key": "jev-1.14.0", "reader": "student_only",
           "sources": {str(w): sources for w in calls}}
    meta = {"summary": {"window_labels": {"aborted": None, "windows": len(ws)},
                        "window_labels_digit_calls": raw},
            "frames": frames,
            "events": [{"frame_num": 10, "slot": "B1", "team_number": None, "robot_label": "B1"}]}
    return meta, ws


def test_reconstructed_labels_match_the_replay():
    meta, ws = _live_meta()
    lab = fr.reconstruct_lab(meta["frames"], ws)
    assert lab[(45, "B1")] == ("3128", wl.RANK_DIGITS, 1.0, 20)
    assert lab[(3, "B2")] == ("6773", wl.RANK_LINEUP, 1.0, 20)
    assert lab[(5, "B2")] is None                              # cleared inside a named window
    assert (10, "B1") not in lab                               # unread window
    frames = copy.deepcopy(meta["frames"])
    wl.apply_labels(frames, lab)
    assert frames == meta["frames"]                            # round trip


def test_points_s1_exact_approximate_and_jev_from_fingerprints():
    meta, ws = _live_meta()
    b1 = [w for w in ws if w.slot == "B1"]
    w0, w1 = b1[0], b1[1]
    raw = {"calls": {str(w0.wid): [{"3128": 0.9}] * 4, str(w1.wid): [{"6773": 0.8}] * 2},
           "sources": {str(w0.wid): "s1", str(w1.wid): "s1"}}
    point, info = fr.reconstruct_point(meta["frames"], ws, raw)
    assert point[w0.wid] == {k: {"3128": 0.9} for k in (1, 2, 3, 4)}
    assert point[w1.wid] == {1: {"6773": 0.8}, 2: {"6773": 0.8}}
    assert info == {"points_exact": 1, "points_approx": 1, "points_mismatch": 0}
    reqs = [[kind, k] for k in (1, 2) for kind in ("wide", "zoom")] + [["grid", 0]]
    fp = {"slot": "B1", "first_frame": meta["frames"][w0.idx[0]]["frame_num"], "requests": reqs}
    raw = {"calls": {str(w0.wid): [{"3128": 0.7}, {"x": 1}, None, {"x": 1}, {"x": 1}]},
           "fingerprints": {str(w0.wid): fp}}
    point, info = fr.reconstruct_point(meta["frames"], ws, raw)
    assert point == {w0.wid: {1: {"3128": 0.7}}} and info["points_exact"] == 1
    raw["fingerprints"][str(w0.wid)] = {**fp, "first_frame": -1}       # a different window
    point, info = fr.reconstruct_point(meta["frames"], ws, raw)
    assert point == {} and info["points_mismatch"] == 1


def test_fill_meta_fills_backward_and_restamps_events():
    meta, _ws = _live_meta()
    stats, _lab = fr.fill_meta(meta, FPS)
    rows = {i: {r["slot"]: r for r in f["robots"]} for i, f in enumerate(meta["frames"])}
    assert rows[0]["B1"]["team_number"] == "3128" and rows[0]["B1"]["confirmed"] is False
    assert rows[45]["B1"]["confirmed"] is True
    assert rows[5]["B2"]["team_number"] is None                # cleared stays cleared
    assert stats["fill_frames"] == 40 and stats["offline"] is True
    assert meta["summary"]["window_labels"]["fill"] == stats
    assert meta["events"][0]["team_number"] == "3128"


def test_fill_meta_honours_a_conflicting_s1_sample():
    meta, ws = _live_meta()
    w1 = [w for w in ws if w.slot == "B1"][1]
    meta["summary"]["window_labels_digit_calls"]["calls"][str(w1.wid)] = [
        {"6773": 0.1}, {"6773": 0.1}, {"6773": 0.1}, {"2637": 0.9}]      # sample 4 = frame 39
    meta["summary"]["window_labels_digit_calls"]["sources"][str(w1.wid)] = "s1"
    fr.fill_meta(meta, FPS)
    b1 = [next(r for r in f["robots"] if r["slot"] == "B1") for f in meta["frames"]]
    # the 2637 sample at frame 39 is now the chain's first waypoint: the 3128
    # read no longer reaches back past it, and (the measured rule) the edge
    # before it takes the sample's team
    assert all(b1[i]["team_number"] == "2637" and not b1[i]["confirmed"] for i in range(40))
    assert all(b1[i]["team_number"] == "3128" for i in range(40, 60))


@pytest.fixture()
def replay(tmp_path, monkeypatch):
    meta, _ws = _live_meta()
    d = tmp_path / "replays" / "rid-1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(fr, "REPLAYS_DIR", tmp_path / "replays")
    monkeypatch.setattr(fr, "replay_row", lambda rid: {"id": "rid-1", "match_key": "2026cur_qm8",
                                                       "meta_path": None,
                                                       "source_video_path": "v.mp4"})
    monkeypatch.setattr(fr, "_resolve", lambda p: tmp_path / p if p else None)
    monkeypatch.setattr(fr, "_probe_fps", lambda v: FPS)
    seen = {"segments": [], "render": []}
    monkeypatch.setattr(fr, "store_segments", lambda rid, rows: seen["segments"].append((rid, rows)))
    monkeypatch.setattr(fr, "render", lambda rid: seen["render"].append(rid) or (0, {"ok": 1}))
    return d, seen


def _load(d, name="meta.json"):
    return json.loads((d / name).read_text())


def test_dry_run_writes_nothing(replay):
    d, seen = replay
    before = (d / "meta.json").read_text()
    assert fr.main(["--replay", "rid", "--dry-run"]) == 0
    assert (d / "meta.json").read_text() == before
    assert not (d / fr.BACKUP_NAME).exists() and seen["segments"] == []


def test_write_backs_up_once_stores_segments_renders_and_then_refuses(replay):
    d, seen = replay
    original = (d / "meta.json").read_text()
    assert fr.main(["--replay", "rid", "--render"]) == 0
    assert (d / fr.BACKUP_NAME).read_text() == original
    assert not list(d.glob("*.tmp"))
    filled = _load(d)
    assert filled["summary"]["window_labels"]["fill"]["fill_frames"] == 40
    rid, rows = seen["segments"][0]
    assert rid == "rid-1" and {r["source"] for r in rows if r["slot"] == 0} == {
        "window_fill", "window_digits"}
    assert all(r["slot_label"] == {0: "B1", 1: "B2"}[r["slot"]] for r in rows)
    assert seen["render"] == ["rid-1"]
    assert fr.main(["--replay", "rid"]) == 2                     # idempotent: refused
    assert fr.main(["--replay", "rid", "--force"]) == 0          # redone from the backup
    assert (d / fr.BACKUP_NAME).read_text() == original
    assert _load(d)["summary"]["window_labels"]["fill"] == filled["summary"]["window_labels"]["fill"]
    assert _load(d)["frames"] == filled["frames"]


@pytest.mark.parametrize("wls", [None, {"aborted": "no windows"}, {"aborted": None,
                                                                   "fill_frames": 3}])
def test_refusals(replay, wls):
    d, seen = replay
    meta = _load(d)
    if wls is None:
        del meta["summary"]["window_labels"]
    else:
        meta["summary"]["window_labels"] = wls
    (d / "meta.json").write_text(json.dumps(meta))
    assert fr.main(["--replay", "rid"]) == 2
    assert fr.main(["--replay", "rid", "--force"]) == 2           # a live fill has no backup
    assert _load(d) == meta and seen["segments"] == []


def test_no_source_fps_is_refused(replay, monkeypatch):
    monkeypatch.setattr(fr, "_probe_fps", lambda v: None)
    assert fr.main(["--replay", "rid"]) == 2
