"""Unit tests for per-crop robot team labels (pure logic; no GPU, no network)."""
import json

from scripts.teacher.robot_labels import (
    Bumper, Params, box_at, bumper_inside, crop_label, label_match_rows, run_match,
)

P = Params()
ROBOT = (100.0, 100.0, 200.0, 200.0)          # lower 60% starts at y = 140
IN = (120.0, 170.0, 180.0, 195.0)              # a bumper along the robot's bottom edge
OUT = (300.0, 170.0, 360.0, 195.0)             # a bumper beside the robot


def bumper(tid, start, end, crops, decision="unsure", team=None, calls=None, off=False, robot=None):
    return Bumper(tid=tid, start=start, end=end, crops=sorted(crops), decision=decision, team=team,
                  off_alliance=off, robot=robot, call_teams=dict(calls or {}))


# ---- geometry ------------------------------------------------------------------------

def test_box_at_interpolates_and_respects_tolerance():
    crops = [(10.0, (0, 0, 10, 10)), (11.0, (10, 0, 20, 10))]
    assert box_at(crops, 10.5, 1.0) == (5, 0, 15, 10)
    assert box_at(crops, 10.0, 0.1) == (0, 0, 10, 10)
    assert box_at(crops, 11.8, 1.0) == (10, 0, 20, 10)          # nearest within tol
    assert box_at(crops, 13.0, 1.0) is None                     # nothing within tol
    assert box_at([], 1.0, 1.0) is None


def test_bumper_inside_lower_part_only():
    assert bumper_inside(IN, ROBOT, P)
    assert not bumper_inside(OUT, ROBOT, P)
    assert not bumper_inside((120, 100, 180, 120), ROBOT, P)    # top of the box (centre y 110 < 140)
    # centre inside but most of the bumper outside -> not this robot's bumper
    assert not bumper_inside((190, 170, 290, 195), ROBOT, P)


# ---- crop label ------------------------------------------------------------------------

def test_decided_witness_in_box_at_crop_time_names_the_crop():
    b = bumper("b1", 9.0, 11.0, [(9.5, IN), (10.5, IN)], "team", "254")
    d = crop_label(10.0, ROBOT, "r1", [b], P)
    assert d["team"] == "254" and d["support"] == {"254": ["b1"]}


def test_decided_bumper_from_another_time_does_not_name_the_crop():
    # the robot tracklet switched robots: the decided bumper ended 3 s before this crop
    b = bumper("b1", 5.0, 7.0, [(6.5, IN)], "team", "254", robot="r1")
    d = crop_label(10.0, ROBOT, "r1", [b], P)
    assert d["team"] is None and d["reason"] == "no_witness"


def test_decided_bumper_outside_the_crop_box_does_not_count():
    b = bumper("b1", 9.0, 11.0, [(10.0, OUT)], "team", "254", robot="r1")
    assert crop_label(10.0, ROBOT, "r1", [b], P)["team"] is None


def test_two_decided_teams_in_box_abstain():
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "254")
    b = bumper("b2", 9.0, 11.0, [(10.0, IN)], "team", "1678")
    assert crop_label(10.0, ROBOT, "r1", [a, b], P)["reason"] == "witnesses_disagree"


def test_other_team_just_outside_support_window_still_conflicts():
    # neighbour's decided bumper left the box 0.5 s ago: too old to support, recent enough to veto
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "254")
    b = bumper("b2", 8.0, 9.5, [(9.5, IN)], "team", "1678")
    d = crop_label(10.0, ROBOT, "r1", [a, b], P)
    assert d["team"] is None and d["reason"] == "conflict" and "b2:team:1678" in d["conflict"]


def test_unsure_witness_naming_another_team_vetoes():
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "254")
    b = bumper("b2", 9.0, 11.0, [(10.0, IN)], "unsure", calls={"1678": 0.6, "254": 0.55})
    d = crop_label(10.0, ROBOT, "r1", [a, b], P)
    assert d["team"] is None and d["conflict"] == ["b2:call:1678"]


def test_off_alliance_decision_never_supports():
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "971", off=True)
    d = crop_label(10.0, ROBOT, "r1", [a], P)
    assert d["team"] is None and d["reason"] == "no_decided_witness"


def test_weak_witness_supports_only_with_one_named_team_at_threshold():
    one = bumper("b1", 9.0, 11.0, [(10.0, IN)], calls={"254": 0.9})
    assert crop_label(10.0, ROBOT, "r1", [one], P)["support"] == {"254": ["b1:weak"]}
    low = bumper("b1", 9.0, 11.0, [(10.0, IN)], calls={"254": 0.7})
    assert crop_label(10.0, ROBOT, "r1", [low], P)["team"] is None
    two = bumper("b1", 9.0, 11.0, [(10.0, IN)], calls={"254": 0.9, "1678": 0.6})
    assert crop_label(10.0, ROBOT, "r1", [two], P)["team"] is None
    off = Params(weak_support_p=None)
    assert crop_label(10.0, ROBOT, "r1", [one], off)["team"] is None


def test_unpositioned_linked_bumper_only_when_enabled():
    b = bumper("b1", 5.0, 15.0, [(5.0, IN), (15.0, IN)], "team", "254", robot="r1")
    assert crop_label(10.0, ROBOT, "r1", [b], P)["team"] is None          # no crop within 1 s
    assert crop_label(10.0, ROBOT, "r1", [b], Params(use_unpositioned=True))["team"] == "254"
    assert crop_label(10.0, ROBOT, "r2", [b], Params(use_unpositioned=True))["team"] is None


# ---- match rows / files ----------------------------------------------------------------------

ROSTER = {"red": ["254", "1678", "118"], "blue": ["971", "2056", "4414"]}


def mrow(det, tid, t, box, **kw):
    r = {"event": "2026x", "match_key": "2026x_qm1", "detector": det, "tracklet_id": tid,
         "t_sec": t, "frame": int(t * 60), "box": list(box), "px_w": int(box[2] - box[0]),
         "path": f"2026x/2026x_qm1/crops/{tid}_f{int(t * 60)}.jpg", "roster": ROSTER,
         "tracklet_start_t": kw.pop("start", t), "tracklet_end_t": kw.pop("end", t)}
    r.update(kw)
    return r


def test_label_match_rows_per_crop_not_per_tracklet(tmp_path):
    rows = [
        mrow("bumper", "b1", 1.0, IN, start=0.5, end=2.0, linked_tracklet_id="r1", alliance_colour="red"),
        mrow("bumper", "b2", 6.0, IN, start=5.5, end=7.0, linked_tracklet_id="r1", alliance_colour="red"),
        # one robot tracklet that switched robots between t=2 and t=5
        mrow("robot", "r1", 1.0, ROBOT, start=0.5, end=7.0, linked_bumper_tracklets=["b1", "b2"],
             alliance_colour="red"),
        mrow("robot", "r1", 6.0, ROBOT, start=0.5, end=7.0, linked_bumper_tracklets=["b1", "b2"],
             alliance_colour="red"),
        mrow("robot", "r1", 3.5, ROBOT, start=0.5, end=7.0, linked_bumper_tracklets=["b1", "b2"],
             alliance_colour="red"),
    ]
    jev = [
        {"detector": "bumper", "tracklet_id": "b1", "decision": "team", "team": "254", "calls": []},
        {"detector": "bumper", "tracklet_id": "b2", "decision": "unsure", "team": None,
         "calls": [{"top": "unreadable", "top_p": 0.9}]},
        {"detector": "robot", "tracklet_id": "r1", "decision": "team", "team": "254"},
    ]
    out = label_match_rows(rows, jev)
    by_t = {r["t_sec"]: r for r in out}
    assert by_t[1.0]["team"] == "254"
    assert by_t[6.0]["team"] is None and by_t[6.0]["reason"] == "no_decided_witness"
    assert by_t[3.5]["team"] is None and by_t[3.5]["reason"] == "no_witness"
    assert all(r["tracklet_team_old"] == "254" for r in out)     # the old rule labelled all three

    mdir = tmp_path / "2026x" / "2026x_qm1"
    mdir.mkdir(parents=True)
    (mdir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (mdir / "labels_jev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jev))
    _, summ = run_match(mdir, Params())
    written = [json.loads(l) for l in (mdir / "labels_robot.jsonl").open()]
    assert len(written) == 3 and summ["team"] == 1 and summ["old_team"] == 3
    assert {"path", "team", "reason", "support", "conflict", "witnesses"} <= set(written[0])


# ---- vetoes added after spot checks ---------------------------------------------------------

RA = {"254": "red", "1678": "red", "118": "red", "971": "blue", "2056": "blue", "4414": "blue"}


def test_other_alliance_bumper_in_box_vetoes():
    # a blue neighbour's unreadable bumper next to a red decided one -> two robots in the crop
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "254")
    a.colour = "red"
    b = bumper("b2", 9.0, 11.0, [(10.0, (130.0, 160.0, 190.0, 190.0))])
    b.colour = "blue"
    d = crop_label(10.0, ROBOT, "r1", [a, b], P, RA)
    assert d["team"] is None and d["conflict"] == ["b2:colour:blue"]
    assert crop_label(10.0, ROBOT, "r1", [a, b], Params(colour_conflict=False), RA)["team"] == "254"


def test_linked_bumper_of_other_team_vetoes_even_without_position():
    a = bumper("b1", 9.0, 11.0, [(10.0, IN)], "team", "254")
    b = bumper("b2", 5.0, 15.0, [(5.0, IN)], "team", "1678", robot="r1")   # no crop near t
    d = crop_label(10.0, ROBOT, "r1", [a, b], P)
    assert d["team"] is None and d["conflict"] == ["b2:team:1678"]
    assert crop_label(10.0, ROBOT, "r9", [a, b], P)["team"] == "254"       # linked elsewhere


def test_narrow_bumper_does_not_support():
    narrow = (150.0, 170.0, 170.0, 195.0)                                   # 20% of the robot width
    b = bumper("b1", 9.0, 11.0, [(10.0, narrow)], "team", "254")
    assert crop_label(10.0, ROBOT, "r1", [b], P)["team"] is None
    assert crop_label(10.0, ROBOT, "r1", [b], Params(min_w_frac=0.0))["team"] == "254"
