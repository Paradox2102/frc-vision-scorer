"""Pure parts of scripts/backfill_det_conf.py: box mapping, IoU matching,
camera combination, draw-log parsing, meta patching, skip/force logic."""
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backfill_det_conf as bf  # noqa: E402


def test_export_to_full_inverts_export_transform():
    # export = (full - y_off) * scale  ->  full = export / scale (+ y_off on y)
    assert bf.export_to_full((100, 50, 200, 150), 0.5, 30) == (200, 130, 400, 330)
    assert bf.export_to_full((10, 10, 20, 20), 1.0, 0) == (10, 10, 20, 20)


def test_box_iou():
    assert bf.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bf.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert bf.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_match_box_best_iou_threshold_and_none_conf():
    dets = [((0, 0, 10, 10), 0.9), ((1, 0, 11, 10), 0.4), ((50, 50, 60, 60), None)]
    conf, iou, ok = bf.match_box((1, 0, 11, 10), dets, 0.5)
    assert ok and conf == 0.4 and iou == pytest.approx(1.0)
    conf, iou, ok = bf.match_box((3, 0, 13, 10), dets, 0.5)     # 0.67 det 2 > 0.54 det 1
    assert ok and conf == 0.4 and iou == pytest.approx(80 / 120)
    conf, _iou, ok = bf.match_box((8, 0, 18, 10), dets, 0.5)    # best ~0.25 < 0.5
    assert not ok and conf is None
    conf, _iou, ok = bf.match_box((50, 50, 60, 60), dets, 0.5)  # matched, no conf
    assert ok and conf is None
    assert bf.match_box((0, 0, 1, 1), [], 0.5) == (None, 0.0, False)


def test_combine_cams_weight_and_max():
    cands = [("main", 0.9, 1.0), ("corner_left", 0.5, 2.5), ("corner_right", None, 2.5)]
    # fusion rule: max-weight member WITH a conf -> corner_left's 0.5
    assert bf.combine_cams(cands, "weight") == (0.5, "corner_left")
    assert bf.combine_cams(cands, "max") == (0.9, "main")
    # attenuated corner below main weight -> main wins
    assert bf.combine_cams([("main", 0.7, 1.0), ("corner_left", 0.8, 0.3)],
                           "weight") == (0.7, "main")
    # weight 0 members are dropped (fusion skips corner_weight <= 0)
    assert bf.combine_cams([("corner_left", 0.8, 0.0)], "weight") == (None, None)
    assert bf.combine_cams([("main", None, 1.0)], "weight") == (None, None)
    # ties on weight go to the larger conf, like max((w, conf))
    assert bf.combine_cams([("a", 0.3, 1.0), ("b", 0.6, 1.0)], "weight")[0] == 0.6


def _line(fn, main=(), corners=None, radar=None):
    d = {"frame_num": fn, "main_thumb": "xxxx", "main": list(main),
         "corners": corners or {}}
    if radar is not None:
        d["radar"] = radar
    return json.dumps(d)


def test_draw_targets_filters_ghosts_and_unslotted():
    lines = [
        _line(10, main=[{"slot": "R1", "box": [0, 0, 10, 10], "ghost": False},
                        {"slot": "B1", "box": [5, 5, 9, 9], "ghost": True}],
              corners={"corner_left": [{"slot": None, "box": [1, 1, 2, 2]},
                                       {"slot": "R2", "box": [3, 3, 4, 4]}]},
              radar=[{"slot": "R1", "state": "visible"}, {"slot": "B1", "state": "coasting"}]),
        _line(16, main=[{"slot": "B1", "box": [5, 5, 9, 9], "ghost": True}]),  # all ghosts
        "not json",
    ]
    t = bf.draw_targets(lines)
    assert set(t) == {10}
    assert t[10]["boxes"]["main"] == [("R1", (0.0, 0.0, 10.0, 10.0))]
    assert t[10]["boxes"]["corner_left"] == [("R2", (3.0, 3.0, 4.0, 4.0))]
    assert t[10]["state"] == {"R1": "visible", "B1": "coasting"}


def test_read_draw_targets_gz(tmp_path):
    p = tmp_path / "draw.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(_line(7, main=[{"slot": "R3", "box": [1, 2, 3, 4], "ghost": False}]) + "\n")
    assert bf.read_draw_targets(p)[7]["state"] is None


def test_slot_values_visible_gate():
    targets = {10: {"boxes": {}, "state": {"R1": "visible", "R2": "coasting"}},
               20: {"boxes": {}, "state": None}}
    per_cam = {(10, "R1"): [("main", 0.8, 1.0)], (10, "R2"): [("corner_left", 0.6, 2.5)],
               (20, "R1"): [("main", 0.7, 1.0)]}
    v = bf.slot_values(per_cam, targets, "weight")
    assert v[(10, "R1")] == (0.8, "main")
    assert v[(10, "R2")] == (None, None)        # coasting -> None like live
    assert v[(20, "R1")] == (0.7, "main")       # no radar -> no gate


def _meta():
    return {"summary": {"autonomous_paths": {
                "R1": [{"frame_num": 10, "field_pos": [1, 1]},
                       {"frame_num": 13, "field_pos": [1, 1], "interpolated": True},
                       {"frame_num": 16, "field_pos": [1, 1]}],
                "B1": [{"frame_num": 10, "field_pos": [2, 2]}]}},
            "frames": [
                {"frame_num": 10, "robots": [{"slot": "R1"}, {"slot": "B1"}]},
                {"frame_num": 16, "robots": [{"slot": "R1"}]}]}


def test_patch_meta_robots_and_autonomous_paths():
    meta = _meta()
    vals = {(10, "R1"): 0.812345, (16, "R1"): None, (13, "R1"): 0.5}
    assert bf.patch_meta(meta, vals) == (3, 1, 4, 1)
    r = meta["frames"][0]["robots"]
    assert r[0]["det_conf"] == 0.8123 and r[1]["det_conf"] is None
    assert meta["frames"][1]["robots"][0]["det_conf"] is None
    ap = meta["summary"]["autonomous_paths"]
    assert ap["R1"][0]["det_conf"] == 0.8123
    assert ap["R1"][1]["det_conf"] is None       # interpolated stays None
    assert ap["R1"][2]["det_conf"] is None
    assert ap["B1"][0]["det_conf"] is None
    # tracker_prob / state are not added
    assert "tracker_prob" not in r[0] and "state" not in r[0]


def test_skip_and_force_logic():
    meta = _meta()
    assert bf.skip_reason(meta, force=False) is None
    assert bf.skip_reason({"frames": []}, force=True) == "no frames"
    meta["frames"][0]["robots"][0]["det_conf"] = 0.5
    assert "live det_conf" in bf.skip_reason(meta, force=False)
    meta["summary"]["det_conf_backfill"] = {"script_version": 1}
    assert "already backfilled" in bf.skip_reason(meta, force=False)
    assert bf.skip_reason(meta, force=True) is None


def test_write_meta_atomic_keeps_first_backup(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"v": 0}))
    bf.write_meta_atomic(p, {"v": 1})
    bf.write_meta_atomic(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}
    assert json.loads((tmp_path / "meta.json.pre-detconf").read_text()) == {"v": 0}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["meta.json", "meta.json.pre-detconf"]


def test_view_selection():
    summary = {"calibration_version": "M1",
               "corner_calibration": {"corner_left": {"version_id": None, "status": "dropped"},
                                      "corner_right": {"version_id": "R9", "status": "kept"}}}
    sel, skip = bf.view_selection(summary, ["main", "corner_left", "corner_right", "x"],
                                  lambda n: "E-" + n)
    assert sel == {"main": "M1", "corner_right": "R9", "x": "E-x"}
    assert skip == ["corner_left"]
    sel, skip = bf.view_selection({}, ["main"], lambda n: "E-" + n)
    assert sel == {"main": "E-main"} and skip == []


def test_compare_stats():
    meta = {"frames": [{"frame_num": 10, "robots": [
        {"slot": "R1", "det_conf": 0.80, "state": "visible"},
        {"slot": "R2", "det_conf": 0.50, "state": "visible"},
        {"slot": "R3", "det_conf": None, "state": "visible"},
        {"slot": "B1", "det_conf": None, "state": "coasting"}]}]}
    values = {(10, "R1"): (0.805, "main"), (10, "R2"): (None, None),
              (10, "R3"): (0.4, "corner_left")}
    cams_of = {(10, "R1"): {"main"}, (10, "R3"): {"corner_left", "main"}}
    c = bf.compare_stats(meta, values, cams_of)
    t = c["total"]
    assert (t["n"], t["live"], t["bf_on_live"], t["bf_where_live_null"]) == (4, 2, 1, 1)
    assert t["fill_on_live"] == 0.5
    assert t["diff_max"] == pytest.approx(0.005)
    assert t["within_0.01"] == 1.0
    assert c["live_visible_but_null"] == 1
    assert c["by_cam_set"]["corner_left+main"]["bf_where_live_null"] == 1
    assert c["by_picked_cam"]["main"]["bf_on_live"] == 1


def test_known_overrides_drops_branch_knobs():
    ov, unknown = bf.known_overrides({"window_labels": True, "corner_robot_model": "/x.pt"})
    assert ov == {"window_labels": True} and unknown == ["corner_robot_model"]


def test_compare_stats_auto_paths_and_frame_confs():
    meta = {"frames": [],
            "summary": {"autonomous_paths": {"R1": [
                {"frame_num": 10, "det_conf": 0.8},
                {"frame_num": 16, "det_conf": 0.7},
                {"frame_num": 22, "det_conf": None, "interpolated": True}]}}}
    values = {(10, "R1"): (0.8, "main"), (16, "R1"): (0.9, "main")}
    c = bf.compare_stats(meta, values, {}, "auto_paths",
                         frame_confs={10: {0.8}, 16: {0.9, 0.7}})
    t = c["total"]
    assert (t["n"], t["live"], t["bf_on_live"], t["live_in_frame_dets"]) == (2, 2, 2, 2)
    assert t["within_0.01"] == 0.5
    assert bf.has_det_conf(meta)
    assert not bf.has_det_conf({"frames": [{"robots": [{"slot": "R1"}]}]})
