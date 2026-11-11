"""Unit tests for the teacher crop harvester (synthetic inputs; no GPU, no network)."""
import pytest

from scripts.teacher import events
from scripts.teacher.harvest import (
    MANIFEST_KEYS, Det, interlaced_from_idet, parse_idet, Tracklet, build_tracklets, bumper_in_robot, link_bumpers_to_robots,
    manifest_row, pad_box, parse_matches, parse_roster, robot_links, select_crops, spread_pick,
)


def det(sample, box, cls="red_bumper", conf=0.9, fps=2.0):
    return Det(sample=sample, frame=sample * 15, t=sample / fps, box=tuple(box), conf=conf, cls=cls)


# ---- match-range parsing ---------------------------------------------------

def test_parse_matches_range_and_list():
    assert parse_matches("qm1-qm3") == ["qm1", "qm2", "qm3"]
    assert parse_matches("qm1-3") == ["qm1", "qm2", "qm3"]
    assert parse_matches("qm3,qm7") == ["qm3", "qm7"]
    assert parse_matches("qm1-qm2, qm9,qm2") == ["qm1", "qm2", "qm9"]
    assert len(parse_matches("qm1-qm15")) == 15


@pytest.mark.parametrize("bad", ["qm5-qm1", "qm1-sf2", "foo", "qm"])
def test_parse_matches_rejects(bad):
    with pytest.raises(ValueError):
        parse_matches(bad)


# ---- tracklets ---------------------------------------------------------------

def test_tracklets_follow_moving_box_and_split_on_jump():
    dets = [det(s, (100 + 5 * s, 100, 160 + 5 * s, 130)) for s in range(4)]
    dets.append(det(4, (900, 500, 960, 530)))  # far away -> new tracklet
    trs = build_tracklets(dets, "bumper", "b")
    assert [len(t.dets) for t in trs] == [4, 1]
    assert [t.tid for t in trs] == ["b0", "b1"]


def test_tracklets_bridge_one_missing_sample_not_two():
    box = (100, 100, 160, 130)
    one_gap = build_tracklets([det(0, box), det(2, box)], "bumper", "b")
    two_gap = build_tracklets([det(0, box), det(3, box)], "bumper", "b")
    assert len(one_gap) == 1 and len(two_gap) == 2


def test_tracklets_respect_class_and_max_length():
    box = (100, 100, 160, 130)
    mixed = build_tracklets([det(0, box, "red_bumper"), det(1, box, "blue_bumper")], "bumper", "b")
    assert len(mixed) == 2
    # 2 fps, 3 s max -> samples 0..6 fit in one tracklet, sample 7 starts another
    long = build_tracklets([det(s, box) for s in range(10)], "bumper", "b")
    assert [len(t.dets) for t in long] == [7, 3]


def test_tracklets_greedy_highest_iou_first():
    a = det(0, (0, 0, 100, 100))
    b = det(0, (200, 0, 300, 100))
    # next frame: one box overlaps a strongly, one overlaps b strongly
    a2 = det(1, (5, 0, 105, 100))
    b2 = det(1, (195, 0, 295, 100))
    trs = build_tracklets([a, b, b2, a2], "bumper", "b")
    assert len(trs) == 2
    assert trs[0].dets[1] is a2 and trs[1].dets[1] is b2


# ---- crop spreading ------------------------------------------------------------

def test_spread_pick():
    assert spread_pick(3, 6) == [0, 1, 2]
    assert spread_pick(11, 6) == [0, 2, 4, 6, 8, 10]
    assert spread_pick(10, 1) == [5]
    assert spread_pick(0, 6) == []
    assert len(spread_pick(7, 6)) == 6


def test_select_crops_spreads_and_skips_small():
    dets = [det(s, (0, 0, 60, 30)) for s in range(12)]
    dets[0] = det(0, (0, 0, 60, 15))       # short side 15 px -> skipped
    tr = Tracklet("b0", "bumper", dets)
    picks = select_crops(tr, k=6, min_side=20)
    assert len(picks) == 6
    assert picks[0].sample == 1 and picks[-1].sample == 11
    assert all(min(p.box[2] - p.box[0], p.box[3] - p.box[1]) >= 20 for p in picks)
    assert [p.t for p in picks] == sorted(p.t for p in picks)


# ---- linking ------------------------------------------------------------------

def test_bumper_in_robot_lower_60pct():
    robot = (100, 100, 300, 300)               # lower 60% = y in [180, 300]
    assert bumper_in_robot((150, 250, 250, 290), robot)
    assert not bumper_in_robot((150, 110, 250, 150), robot)   # centre y=130: top 40%
    assert not bumper_in_robot((350, 250, 400, 290), robot)   # outside in x


def test_link_bumpers_to_robots_most_shared_frames():
    r_a = Tracklet("r0", "robot", [det(s, (100, 100, 300, 300), "robot") for s in range(4)])
    r_b = Tracklet("r1", "robot", [det(s, (120, 100, 320, 300), "robot") for s in range(2, 4)])
    b = Tracklet("b0", "bumper", [det(s, (150, 250, 250, 290)) for s in range(4)])
    lone = Tracklet("b1", "bumper", [det(s, (800, 800, 860, 830), "blue_bumper") for s in range(3)])
    links = link_bumpers_to_robots([b, lone], [r_a, r_b])
    assert links == {"b0": "r0", "b1": None}
    rl = robot_links([b, lone], links)
    assert rl == {"r0": ("b0", "red", ["b0"])}


def test_robot_links_colour_majority():
    b_red = Tracklet("b0", "bumper", [det(s, (0, 0, 1, 1), "red_bumper") for s in range(3)])
    b_blue = Tracklet("b1", "bumper", [det(s, (0, 0, 1, 1), "blue_bumper") for s in range(1)])
    rl = robot_links([b_red, b_blue], {"b0": "r0", "b1": "r0"})
    assert rl["r0"] == ("b0", "red", ["b0", "b1"])


# ---- manifest -----------------------------------------------------------------

def test_pad_box_clips_to_frame():
    assert pad_box((10, 10, 110, 50), 0.25, 1920, 1080) == (0, 0, 135, 60)
    assert pad_box((1900, 1070, 1920, 1080), 0.25, 1920, 1080)[2:] == (1920, 1080)


def test_parse_roster_strips_frc():
    m = {"alliances": {"red": {"team_keys": ["frc254", "frc1678", "frc118"]},
                       "blue": {"team_keys": ["frc971", "frc2056", "frc4414"]}}}
    assert parse_roster(m) == {"red": ["254", "1678", "118"], "blue": ["971", "2056", "4414"]}
    assert parse_roster(None) is None


def test_manifest_row_schema():
    d = det(3, (10.04, 20, 70, 50), conf=0.87654321)
    row = manifest_row(event="2025cur", match_key="2025cur_qm11", det=d, video_fps=29.97,
                       detector="bumper", tracklet_id="b4", linked_tracklet_id="r2",
                       alliance_colour="red", comb_ratio_before=0.5, deinterlaced=False,
                       path="2025cur/2025cur_qm11/crops/b4_f45.jpg",
                       roster={"red": ["1"], "blue": ["2"]})
    assert tuple(k for k in row if k in MANIFEST_KEYS) == MANIFEST_KEYS
    assert row["year"] == 2025 and row["frame"] == 45 and row["t_sec"] == 1.5
    assert row["px_w"] == 60 and row["px_h"] == 30 and row["box"][0] == 10.0
    assert row["conf"] == 0.8765
    import json
    assert json.loads(json.dumps(row)) == row


# ---- events / splits ------------------------------------------------------------

def test_event_splits():
    assert events.split_of("2026cur_qm8") == "excluded"
    assert events.split_of("2026cur_qm11") == "excluded"
    assert all(k not in events.GT_EXCLUDE for k in events.all_match_keys())
    assert events.split_of("2026alhu_qm1") == "test"
    assert events.split_of("2025casd_qm6") == "test"
    assert {e[:4] for e in events.TEST_EVENTS} == {"2026", "2025"}
    for ev in events.EVENTS:
        assert len(events.EVENTS[ev]) == len(set(events.EVENTS[ev]))
        val = events.val_matches(ev)
        if ev in events.TEST_EVENTS:
            assert val == []
        else:
            assert 1 <= len(val) <= 2 and all(events.split_of(k) == "val" for k in val)
    assert events.split_of("2025cur_qm1") in ("train", "val")


# ---- video-level interlace gate ---------------------------------------------------

IDET_TFF = """[Parsed_idet_0 @ 0x1] Single frame detection: TFF: 0 BFF: 0 Progressive: 0 Undetermined: 0
[Parsed_idet_0 @ 0x1] Multi frame detection: TFF:     0 BFF:     0 Progressive:     0 Undetermined:     0
[Parsed_idet_0 @ 0x2] Multi frame detection: TFF:   313 BFF:     0 Progressive:     0 Undetermined:     1
"""
IDET_UND = "[Parsed_idet_0 @ 0x2] Multi frame detection: TFF:     0 BFF:     0 Progressive:     2 Undetermined:   314\n"


def test_idet_parsing_and_decision():
    c = parse_idet(IDET_TFF)
    assert c == {"tff": 313, "bff": 0, "progressive": 0, "undetermined": 1}
    assert interlaced_from_idet(c)
    assert not interlaced_from_idet(parse_idet(IDET_UND))
    assert not interlaced_from_idet(parse_idet("no idet output"))
