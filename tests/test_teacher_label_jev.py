"""Unit tests for the Jev tracklet labeller (pure logic; no network, no GPU)."""
import json

from scripts.teacher.label_jev import (
    call_summary, choose_crops, decide, finish_match, robot_decision, roster_alliance_of,
)

RA = {"254": "red", "1678": "red", "118": "red", "971": "blue", "2056": "blue", "4414": "blue"}


def row(t, w, sharp, path=None):
    return {"t_sec": t, "px_w": w, "px_h": w // 3, "sharpness": sharp,
            "path": path or f"c/{t}_{w}.jpg"}


def call(top, p, error=None):
    return {"top": top, "top_p": p, "error": error}


# ---- crop choice ----------------------------------------------------------------

def test_choose_crops_largest_then_sharpest_spread_in_time():
    rows = [row(0.0, 100, 50), row(0.5, 98, 300), row(1.0, 97, 80),
            row(4.0, 95, 200), row(6.0, 60, 999)]
    picks = choose_crops(rows)
    # large set = px_w >= 80: sharpest is t=0.5 (300); next best >= 1 s away: t=4.0 (200)
    assert [p["t_sec"] for p in picks] == [0.5, 4.0]


def test_choose_crops_near_only_and_small_inputs():
    # nothing >= 1 s away -> the farthest in time
    near = [row(0.0, 100, 10), row(0.1, 100, 5), row(0.5, 100, 1)]
    assert [p["t_sec"] for p in choose_crops(near)] == [0.0, 0.5]
    assert len(choose_crops([row(0.0, 100, 1)])) == 1
    assert choose_crops([]) == []


def test_choose_crops_second_is_by_rank_among_far():
    rows = [row(0.0, 100, 10), row(0.2, 50, 500), row(3.0, 40, 900)]
    picks = choose_crops(rows)
    # (0.2, 50 px) is not >= 1 s from the first pick; (3.0, 40 px) is
    assert [p["t_sec"] for p in picks] == [0.0, 3.0]


# ---- per-call summary --------------------------------------------------------------

def test_call_summary():
    s = call_summary({"254": 0.7, "971": 0.1, "unreadable": 0.2}, RA)
    assert s == {"top": "254", "top_p": 0.7, "best_team": "254", "best_team_p": 0.7}
    s = call_summary({"254": 0.2, "unreadable": 0.8}, RA)
    assert s["top"] == "unreadable" and s["best_team"] == "254"
    assert call_summary({}, RA)["top"] is None


# ---- tracklet decision ---------------------------------------------------------------

def test_decide_two_calls_agree():
    d = decide([call("254", 0.6), call("254", 0.5)], RA, "red")
    assert d["decision"] == "team" and d["team"] == "254" and not d["off_alliance"]


def test_decide_two_calls_one_weak_or_disagree():
    assert decide([call("254", 0.9), call("254", 0.45)], RA, "red")["decision"] == "unsure"
    d = decide([call("254", 0.9), call("1678", 0.9)], RA, "red")
    assert d["decision"] == "unsure" and "teams_disagree" in d["reasons"]
    assert decide([call("254", 0.9), call("unreadable", 0.9)], RA, "red")["decision"] == "unsure"


def test_decide_one_call_threshold():
    assert decide([call("254", 0.86)], RA, "red")["decision"] == "team"
    assert decide([call("254", 0.84)], RA, "red")["decision"] == "unsure"


def test_decide_not_bumper():
    assert decide([call("not_robot", 0.4), call("printed", 0.6)], RA, "red")["decision"] == "not_bumper"
    assert decide([call("not_robot", 0.4), call("unreadable", 0.6)], RA, "red")["decision"] == "unsure"
    assert decide([call("not_robot", 0.9)], RA, "blue")["decision"] == "not_bumper"
    assert decide([call("not_robot", 0.5)], RA, "blue")["decision"] == "unsure"


def test_decide_off_alliance_flag_and_roster_guard():
    d = decide([call("971", 0.9), call("971", 0.9)], RA, "red")
    assert d["decision"] == "team" and d["off_alliance"] and "off_alliance" in d["reasons"]
    assert not decide([call("971", 0.9)], RA, None)["off_alliance"]
    # a non-roster "team" top never comes out as a team
    assert decide([call("9999", 0.99), call("9999", 0.99)], RA, "red")["team"] is None
    assert decide([call("other", 0.99)], RA, "red")["decision"] == "unsure"


def test_decide_errors_and_missing_roster():
    d = decide([call("254", 0.9), call(None, 0.0, error="HTTP 500")], RA, "red")
    assert d["decision"] == "unsure" and "call_error" in d["reasons"]
    assert decide([call("254", 0.9)], {}, "red")["reasons"] == ["no_roster"]
    assert decide([], RA, "red")["reasons"] == ["no_calls"]


# ---- robot inheritance --------------------------------------------------------------------

def test_robot_decision():
    labels = {"b0": {"decision": "team", "team": "254"}, "b1": {"decision": "team", "team": "254"},
              "b2": {"decision": "unsure", "team": None}, "b3": {"decision": "team", "team": "118"}}
    d = robot_decision(["b0", "b1", "b2"], labels)
    assert d["decision"] == "team" and d["team"] == "254" and d["votes"] == {"254": 2}
    c = robot_decision(["b0", "b1", "b3"], labels)
    assert c["decision"] == "conflict" and c["team"] is None and c["majority_team"] == "254"
    assert robot_decision(["b2", "bX"], labels)["decision"] == "none"
    assert robot_decision([], labels)["decision"] == "none"


def test_roster_alliance_of():
    assert roster_alliance_of({"red": ["1", "2"], "blue": ["3"]}) == {"1": "red", "2": "red", "3": "blue"}
    assert roster_alliance_of(None) == {}


# ---- end-to-end write from cached calls (no network) -----------------------------------------

def test_finish_match_writes_labels_and_summary(tmp_path):
    roster = {"red": ["254", "1678", "118"], "blue": ["971", "2056", "4414"]}
    info = {"match_key": "2025cur_qm1", "event": "2025cur", "roster": roster}
    b0 = [dict(row(0.0, 100, 10, "a.jpg"), alliance_colour="red", tracklet_len=20),
          dict(row(2.0, 100, 10, "b.jpg"), alliance_colour="red", tracklet_len=20),
          dict(row(3.0, 90, 10, "b2.jpg"), alliance_colour="red", tracklet_len=20)]
    b1 = [dict(row(5.0, 80, 10, "c.jpg"), alliance_colour="blue", tracklet_len=3)]
    r0 = [{"t_sec": 0.0, "px_w": 200, "alliance_colour": "red", "tracklet_len": 30,
           "linked_bumper_tracklets": ["b0"], "path": "r.jpg"}]
    chosen = {"b0": b0[:2], "b1": b1}
    mk = lambda top, p: {"top": top, "top_p": p, "error": None, "input_tokens": 1000,
                         "latency_s": 0.2, "probs": {top: p}}
    calls = {"a.jpg": mk("254", 0.8), "b.jpg": mk("254", 0.7), "c.jpg": mk("unreadable", 0.9)}
    s = finish_match(tmp_path, info, {"b0": b0, "b1": b1}, {"r0": r0}, chosen, calls)
    rows = [json.loads(l) for l in (tmp_path / "labels_jev.jsonl").read_text().splitlines()]
    by = {(r["detector"], r["tracklet_id"]): r for r in rows}
    assert by[("bumper", "b0")]["decision"] == "team" and by[("bumper", "b0")]["team"] == "254"
    assert len(by[("bumper", "b0")]["calls"]) == 2
    assert by[("bumper", "b1")]["decision"] == "unsure"
    assert by[("robot", "r0")]["team"] == "254" and by[("robot", "r0")]["off_alliance"] is False
    assert s["bumper_decisions"] == {"team": 1, "unsure": 1}
    assert s["bumper_crops"] == 4 and s["bumper_crops_team"] == 3
    assert s["bumper_crop_team_coverage"] == 0.75
    assert s["calls"] == 3 and s["input_tokens"] == 3000
    assert json.loads((tmp_path / "labels_jev_summary.json").read_text())["match_key"] == "2025cur_qm1"
