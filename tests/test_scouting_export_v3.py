"""Export v3: unidentified tracks, per-point cameras, identity provenance and
camera calibration details, all on synthetic data."""
import gzip
import json
import math

import numpy as np
import pytest

from web.field import store
from web.field.calibration import CameraCalibration
from web.scouting import export, loader, provenance
from web.scouting.model import MatchData

FPS = 60.0
PERIODS = [{"label": "auto", "start_frame": 100, "end_frame": 1300, "start_seconds": 20,
            "end_seconds": 0, "true_start_frame": 100}]


def _frame(k):
    return 100 + 6 * k                     # 0.1 s apart, t = k / 10


def _meta():
    """Red slot R1: team 1 (k 0-9), unlabelled (k 10-19), unlabelled after a
    6 m jump (k 20-29), team 2 (k 30-39). R2 carries team 2 for k 0-24. Blue
    B1 is never labelled; a slot-less unlabelled row is ignored."""
    frames = []
    for k in range(40):
        robots = []
        x = 2 + 0.01 * k if k < 20 else 8 + 0.01 * k
        team = 1 if k < 10 else (2 if k >= 30 else None)
        robots.append({"slot": "R1", "alliance": "red", "team_number": team and str(team),
                       "confirmed": False, "field_pos": [x, 3.0]})
        if k < 25:
            robots.append({"slot": "R2", "alliance": "red", "team_number": "2",
                           "confirmed": True, "field_pos": [5.0, 5.0]})
        robots.append({"slot": "B1", "alliance": "blue", "team_number": None,
                       "confirmed": False, "field_pos": [12.0, 2.0 + 0.01 * k]})
        robots.append({"slot": None, "alliance": "blue", "team_number": None,
                       "field_pos": [1.0, 1.0]})
        frames.append({"frame_num": _frame(k), "robots": robots})
    return {"summary": {"periods": PERIODS, "calibration_version": "v-main",
                        "corner_calibration": {"corner_left": {"status": "kept",
                                                               "version_id": "v-cl",
                                                               "reason": "holds"}}},
            "frames": frames}


def _md():
    lineup = {"alliances": {"red": [1, 2, 3], "blue": [None, None, None]}}
    md, why = loader.build_match("2026tst", "2026tst_qm1", {"id": "rep"}, _meta(), False,
                                 lineup, 2026)
    assert why is None
    return md


def _write_draw_log(path, lines):
    with gzip.open(path, "wt") as f:
        for d in lines:
            f.write(json.dumps({**d, "main_thumb": "QUJD"}) + "\n")


# ----------------------------------------------------------------- unidentified
def test_loader_keeps_unlabelled_rows_per_slot():
    md = _md()
    assert sorted(md.unidentified) == ["B1", "R1"]
    assert len(md.unidentified["R1"]) == 20 and len(md.unidentified["B1"]) == 40
    assert md.unidentified["B1"][0].alliance == "blue"
    assert md.unidentified_s["auto"] == pytest.approx(6.0)    # 60 rows x 0.1 s
    assert md.frame_labels[_frame(0)] == (("R1", 1), ("R2", 2))
    assert md.cameras["main"]["version_id"] == "v-main"
    assert md.cameras["corner_left"] == {"version_id": "v-cl", "status": "kept",
                                         "reason": "holds"}


def test_unidentified_pieces_hints_and_candidates():
    md = _md()
    out = {e["track_id"]: e for e in export.match_unidentified(md)}
    assert sorted(out) == ["B1#1", "R1#1", "R1#2"]            # the 6 m jump splits R1
    a, b = out["R1#1"], out["R1#2"]
    assert (a["first_t"], a["last_t"], a["n_points"]) == (1.0, 1.9, 10)
    assert a["duration_s"] == pytest.approx(1.0)
    assert (a["team_before"], a["gap_before_s"]) == (1, 0.1)
    assert (a["team_after"], a["gap_after_s"]) == (2, 1.1)
    assert a["candidates"] == [1, 3]           # 2 is on R2 in every frame of the piece
    assert (b["team_before"], b["gap_before_s"]) == (1, 1.1)
    assert (b["team_after"], b["gap_after_s"]) == (2, 0.1)
    assert b["candidates"] == [1, 3]           # 2 elsewhere in 5 of 10 frames: >= 50%
    assert b["x"][0] == 8.2 and b["frame"][0] == _frame(20)
    assert a["cams"] is None and a["camera_s"] is None    # no draw log
    blue = out["B1#1"]
    assert blue["alliance"] == "blue" and blue["candidates"] == []   # lineup unknown
    assert blue["team_before"] is None and blue["gap_before_s"] is None


def test_split_path_is_shared_with_robot_paths():
    pts = [(0.0, 1.0, 1.0), (0.1, 1.1, 1.0), (0.2, 9.0, 1.0), (2.0, 9.0, 1.0)]
    assert export._split_path(pts) == ([[0, 1], [2, 2], [3, 3]], 1)


# ----------------------------------------------------------------- cameras
def test_slot_cameras_bitmask_from_draw_log(tmp_path):
    _write_draw_log(tmp_path / "draw.jsonl.gz", [
        {"frame_num": 20, "main": [{"slot": "B1", "ghost": True}],
         "corners": {"corner_left": [], "corner_right": [], "corner_far": [{"slot": "B1"}]}},
        {"frame_num": 10,
         "main": [{"slot": "B1", "ghost": False}, {"slot": "R1", "ghost": True}],
         "corners": {"corner_left": [{"slot": "B1"}, {"slot": None}],
                     "corner_right": [{"slot": "R1"}]}},
    ])
    sc = provenance.slot_cameras(tmp_path)
    assert sc.legend == ("main", "corner_left", "corner_right", "corner_far")
    assert sc.mask(10, "B1") == 0b011          # main + corner_left (null-slot entry ignored)
    assert sc.mask(10, "R1") == 0b100          # main was coasting (ghost): corner_right only
    assert sc.mask(20, "B1") == 0b1000         # ghost on main, seen by the extra camera
    assert sc.mask(20, "R1") == 0              # nobody saw it
    assert sc.mask(30, "B1") is None           # frame not in the log
    assert provenance.slot_cameras(tmp_path / "nope") is None

    legend = provenance.event_legend([sc.legend, ("main", "corner_aaa")])
    assert legend == ["main", "corner_left", "corner_right", "corner_aaa", "corner_far"]
    assert provenance.remap(0b1001, sc.legend, legend) == 0b10001
    assert provenance.remap(0b011, sc.legend, legend) == 0b011


def test_match_row_cams_column_and_camera_seconds(tmp_path):
    md = _md()
    lines = []
    for k in range(40):
        main = [{"slot": "R1", "ghost": k >= 35}, {"slot": "R2", "ghost": False}]
        corners = {"corner_left": [{"slot": "R1"}] if k >= 30 else [], "corner_right": []}
        lines.append({"frame_num": _frame(k), "main": main, "corners": corners})
    _write_draw_log(tmp_path / "draw.jsonl.gz", lines)
    md.replay_dir = str(tmp_path)
    row = export.match_row(md, None, paths=True)
    r2 = next(r for r in row["robots"] if r["team_number"] == 2)["path"]
    # team 2: R2 for k 0-24 (main), then R1 for k 30-39 (main+left, then left only)
    assert r2["cams"][:25] == [1] * 25 and r2["cams"][25:30] == [3] * 5
    assert r2["cams"][30:] == [2] * 5
    assert r2["camera_s"] == {"main": 3.0, "corner_left": 1.0, "corner_right": 0.0,
                              "none": 0.0}
    blue = next(e for e in row["unidentified"] if e["slot"] == "B1")
    assert set(blue["cams"]) == {0} and blue["camera_s"]["none"] == pytest.approx(4.0)
    assert row["unidentified_s"] == {"auto": 6.0, "teleop": 0.0}


# ----------------------------------------------------------------- identity
def test_identity_rows_collapse_and_seconds():
    md = MatchData(event_key="2026tst", match_key="2026tst_qm1", comp_level="qm",
                   match_number=1, set_number=1, year=2026, replay_id="r",
                   replay_created_at=None, legacy_labels=False,
                   alliances={"red": [1, 2, 3], "blue": [4, 5, 6]}, hub_auto={},
                   frame_period_s=0.1, periods=PERIODS)
    seg = lambda f0, f1, team, src, score=0.9, conf=False, slot="R1": {  # noqa: E731
        "slot": slot, "start_frame": f0, "end_frame": f1, "team": team,
        "confirmed": conf, "score": score, "source": src}
    md.identity_segments = [
        seg(40, 100, 1, "window_lineup"),                      # starts before the clock
        seg(106, 154, 1, "window_digits", 1.0, True),
        seg(160, 214, 1, "window_digits", 0.8, True),          # merges with the row above
        seg(220, 274, None, "window_none", None),
        seg(280, 334, 1, "window_fill"),
        seg(340, 394, 2, "window_digits", slot="R2"),
        seg(400, 454, 1, "stitched", slot="R2"),
    ]
    rows, secs = export.robot_identity(md, 1)
    assert [(r["slot"], r["source"], r["start_frame"], r["end_frame"]) for r in rows] == [
        ("R1", "window_lineup", 40, 100), ("R1", "window_digits", 106, 214),
        ("R1", "window_fill", 280, 334), ("R2", "stitched", 400, 454)]
    assert rows[0]["start_t"] is None and rows[0]["end_t"] == 0.0
    assert rows[1]["start_t"] == 0.1 and rows[1]["end_t"] == 1.9
    assert rows[1]["score"] == pytest.approx((49 * 1.0 + 55 * 0.8) / 104, abs=1e-4) and rows[1]["confirmed"] is True
    assert [r["read_camera"] for r in rows] == ["main", "main", None, None]
    # in-clock seconds: frames [f0, f1 + 6) at 60 fps, clipped at the auto start
    assert secs == {"window_lineup": 0.1, "window_digits": 1.9, "window_fill": 1.0,
                    "stitched": 1.0}
    robots = export.match_robots(md)
    assert robots[0]["identity_s"] == secs and robots[3]["identity"] == []


def test_identity_segments_from_db(tmp_path):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "x.db")
    assert provenance.identity_segments(conn, "job") == []       # no table
    conn.execute("CREATE TABLE match_robot_identity_segment (id INTEGER PRIMARY KEY, "
                 "job_id TEXT, slot INTEGER, slot_label TEXT, start_frame INTEGER, "
                 "end_frame INTEGER, team_number TEXT, confirmed INTEGER, score REAL, "
                 "source TEXT)")
    conn.executemany("INSERT INTO match_robot_identity_segment (job_id, slot, slot_label, "
                     "start_frame, end_frame, team_number, confirmed, score, source) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", [
                         ("job", 3, "R1", 50, 60, "254", 1, 0.99, "window_digits"),
                         ("job", 0, "B1", 10, 20, None, 0, None, "window_none"),
                         ("other", 0, "B1", 10, 20, "1", 0, None, "read")])
    segs = provenance.identity_segments(conn, "job")
    assert [(s["slot"], s["team"], s["source"]) for s in segs] == [
        ("B1", None, "window_none"), ("R1", 254, "window_digits")]


# ----------------------------------------------------------------- calibration
def test_pose_round_trip_and_conventions():
    pos, yaw, pitch, roll = (8.2, -5.1, 3.4), 95.0, -20.0, 3.0
    rvec, tvec = provenance.pose_from_angles(pos, yaw, pitch, roll)
    p = provenance.camera_pose(rvec, tvec)
    assert p["position_m"] == pytest.approx(list(pos), abs=1e-4)
    assert (p["yaw_deg"], p["pitch_deg"], p["roll_deg"]) == pytest.approx((yaw, pitch, roll),
                                                                          abs=1e-3)

    # Level camera at the origin looking down +x: camera x = -y world, y = -z, z = +x.
    rvec, tvec = provenance.pose_from_angles((0, 0, 1), 0, 0, 0)
    R = np.array(provenance.camera_pose(rvec, tvec)["rotation_matrix"])
    assert R == pytest.approx(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), abs=1e-9)

    # Independent checks through a real projection (OpenCV convention).
    import cv2
    K = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])

    def project(pt, rv, tv):
        img, _ = cv2.projectPoints(np.array([[pt]], float), rv, tv, K, None)
        return img.reshape(2)

    rvec, tvec = provenance.pose_from_angles((0, 0, 2), 90, -30, 0)  # look +y, down 30
    ahead = (0, 5 * math.cos(math.radians(30)), 2 - 5 * math.sin(math.radians(30)))
    assert project(ahead, rvec, tvec) == pytest.approx([640, 360], abs=1e-6)
    rvec, tvec = provenance.pose_from_angles((0, 0, 2), 0, 0, 10)    # rolled clockwise
    u, v = project((5, 0, 3), rvec, tvec)          # a point straight above the axis
    assert v < 360 and u < 640                     # ...appears up and to the LEFT


def test_distortion_shift_fisheye_matches_equidistant_model():
    w, h, f = 640, 480, 300.0
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    cal = CameraCalibration(K=K, D=np.zeros(4), rvec=np.zeros((3, 1)),
                            tvec=np.zeros((3, 1)), image_size=(w, h), mode="camera")
    sh = provenance.distortion_shift(cal)
    # equidistant: distorted radius = f*theta, ideal pinhole radius = f*tan(theta)
    th = ((w - 1) / 2) / f
    assert sh["edge_shift_px"] == pytest.approx(f * (math.tan(th) - th), abs=0.05)
    th = math.hypot((w - 1) / 2, (h - 1) / 2) / f
    assert sh["corner_shift_px"] == pytest.approx(f * (math.tan(th) - th), abs=0.05)
    # A wider lens: the corners lie past 90 degrees off-axis -> null, never raises.
    cal.K = np.array([[230.0, 0, (w - 1) / 2], [0, 230, (h - 1) / 2], [0, 0, 1]])
    sh = provenance.distortion_shift(cal)
    assert sh["edge_shift_px"] is not None and sh["corner_shift_px"] is None


def test_distortion_shift_pinhole():
    K = np.array([[1000.0, 0, 500], [0, 1000, 300], [0, 0, 1]])
    cal = CameraCalibration(K=K, D=np.array([-0.2, 0, 0, 0, 0]), image_size=(1001, 601),
                            rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), mode="pinhole")
    sh = provenance.distortion_shift(cal)
    rd = 0.5                                   # mid-left edge: 500 px / f
    ru = rd
    for _ in range(100):                       # solve ru (1 - 0.2 ru^2) = rd
        ru = rd / (1 - 0.2 * ru * ru)
    assert sh["edge_shift_px"] == pytest.approx(1000 * (ru - rd), abs=0.05)
    assert sh["corner_shift_px"] > sh["edge_shift_px"]


def test_camera_detail_reads_versions_and_reports_bad_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "UPLOAD_DIR", tmp_path)
    rvec, tvec = provenance.pose_from_angles((8.0, -5.0, 3.5), 92, -18, 1)
    cal = CameraCalibration(K=np.array([[1300.0, 0, 960], [0, 1300, 360], [0, 0, 1]]),
                            D=np.array([-0.3, 0.1, 0, 0, 0]), rvec=rvec, tvec=tvec,
                            image_size=(1920, 723), crop_region=(0, 0, 1920, 723),
                            field_season="2026", rms_error=1.234, mode="pinhole")
    vid = store.save_version("2026tst", "main", cal, start_match="2026tst_qm1", label="lbl")
    d = provenance.camera_detail("2026tst", "main", vid)
    assert d["version_id"] == vid and d["label"] == "lbl" and d["rejected"] is False
    assert d["start_match"] == "2026tst_qm1" and d["rms_error_px"] == 1.234
    assert d["intrinsics"]["fx"] == 1300.0 and d["distortion"]["model"] == "opencv_pinhole"
    assert d["distortion"]["coefficients"] == [-0.3, 0.1, 0.0, 0.0, 0.0]
    assert d["pose"]["position_m"] == pytest.approx([8.0, -5.0, 3.5], abs=1e-3)
    assert d["pose"]["yaw_deg"] == pytest.approx(92, abs=1e-3) and d["homography"] is None
    bad = provenance.camera_detail("2026tst", "main", "missing")
    assert bad["version_id"] == "missing" and "error" in bad and len(bad) == 2

    hcal = CameraCalibration(image_size=(960, 377), mode="homography", H=np.eye(3),
                             field_season="2026")
    hv = store.save_version("2026tst", "corner_left", hcal)
    hd = provenance.camera_detail("2026tst", "corner_left", hv)
    assert hd["pose"] is None and hd["homography"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert hd["distortion"]["model"] is None and hd["intrinsics"] is None


# ----------------------------------------------------------------- file shape
def test_paths_false_keeps_cameras_and_drops_tracks(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "UPLOAD_DIR", tmp_path)
    md = _md()
    md.replay_dir = str(tmp_path / "no-such-replay")
    row = export.match_row(md, None, paths=False)
    for k in ("robots", "unidentified", "robots_version", "auto_s"):
        assert k not in row
    assert row["cameras"]["corner_left"]["status"] == "kept"
    assert row["unidentified_s"] == {"auto": 6.0, "teleop": 0.0}

    out = export.event_export("2026tst", 2026, [md], [], [], {}, paths=False)
    assert out["schema_version"] == 3
    assert "camera_legend" not in out and "path_frame" not in out
    assert list(out["cameras"]) == ["main", "corner_left"]      # versions the match used
    assert out["cameras"]["main"]["versions"]["v-main"]["error"]  # no such version file

    full = export.event_export("2026tst", 2026, [md], [], [], {}, paths=True)
    assert full["camera_legend"] == ["main", "corner_left", "corner_right"]
    m = full["matches"][0]
    assert m["robots_version"] == 2 and len(m["unidentified"]) == 3
    assert all(r["path"]["cams"] is None for r in m["robots"])   # no draw log
    assert {"identity", "identity_s"} <= set(m["robots"][0])


def test_auto_path_claimed_frames_are_not_unidentified():
    """An unlabelled auto frame inside a slot span that _attach_auto_paths gave
    to a team is in that team's auto path, so it must not be unidentified too."""
    from web.scouting.loader import build_match
    periods = [{"label": "auto", "start_frame": 0, "end_frame": 600, "start_seconds": 20,
                "end_seconds": 0, "true_start_frame": 0},
               {"label": "teleop", "start_frame": 720, "end_frame": 4920, "start_seconds": 140,
                "end_seconds": 0}]
    frames, path = [], []
    for k in range(10):
        fn = k * 6
        team = None if k in (4, 5) else "254"
        frames.append({"frame_num": fn, "robots": [
            {"slot": "R1", "alliance": "red", "team_number": team, "field_pos": [10.0 + k * 0.1, 4.0]}]})
        path.append({"frame_num": fn, "field_pos": [10.0 + k * 0.1, 4.0]})
    meta = {"summary": {"periods": periods, "autonomous_paths": {"R1": path}}, "frames": frames}
    lineup = {"alliances": {"red": [254, None, None], "blue": [None, None, None]}}
    md, why = build_match("2026tst", "2026tst_qm1", {"id": "r1"}, meta, False, lineup, 2026)
    assert md is not None, why
    assert md.teams[254].auto_source == "autonomous_paths"
    assert {s.frame_num for s in md.teams[254].auto_path} >= {24, 30}
    assert not md.unidentified
    assert abs(md.unidentified_s["auto"]) < 1e-9
