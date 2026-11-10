"""Pure logic of scripts/audit_replay.py: sampling, scoring math, GT matching."""
import random
from collections import Counter

import pytest

from scripts import audit_replay as ar

PERIODS = [
    {"label": "auto", "processing_start_frame": 0, "processing_end_frame": 99},
    {"label": "teleop", "processing_start_frame": 120, "processing_end_frame": 999},
]


def _frames(n=80, step=12, slots=("B1", "B2", "R1"), unlabelled=("B2",)):
    out = []
    for i in range(n):
        fn = i * step
        robots = []
        for k, s in enumerate(slots):
            team = None if (s in unlabelled and i % 2) else f"{100 + k}"
            robots.append({"slot": s, "alliance": "blue" if s[0] == "B" else "red",
                           "team_number": team, "field_pos": [float(k), float(i) * 0.1]})
        out.append({"frame_num": fn, "robots": robots})
    return out


# ---------------------------------------------------------------- sampling --

def test_period_of():
    assert ar.period_of(50, PERIODS) == "auto"
    assert ar.period_of(110, PERIODS) is None
    assert ar.period_of(500, PERIODS) == "teleop"


def test_slot_units_skips_out_of_period_and_dupes_get_suffix():
    frames = [{"frame_num": 110, "robots": [{"slot": "B1", "field_pos": [1, 1]}]},
              {"frame_num": 120, "robots": [{"slot": "B1", "field_pos": [1, 1], "team_number": 5},
                                            {"slot": "B1", "field_pos": [2, 2]},
                                            {"slot": "R1", "field_pos": [None, 1]}]}]
    u = ar.slot_units(frames, PERIODS)
    assert [x["slot_key"] for x in u] == ["B1", "B1#2"]
    assert u[0]["team"] == "5" and u[1]["team"] is None and u[1]["dup"] == 2


def test_allocate_proportional_with_floor_and_cap():
    a = ar.allocate({"a": 1000, "b": 10, "c": 1}, 20)
    assert sum(a.values()) == 20 and a["b"] >= 1 and a["c"] == 1
    assert a["a"] > a["b"]
    assert ar.allocate({"a": 3, "b": 2}, 50) == {"a": 3, "b": 2}       # capped at sizes
    assert sum(ar.allocate({"a": 5, "b": 5, "c": 5}, 2).values()) == 2  # n below the floor


def test_systematic_spreads_over_time():
    seq = list(range(100))
    picks = ar.systematic(seq, 10, random.Random(0))
    assert len(picks) == 10 and len(set(picks)) == 10
    gaps = [b - a for a, b in zip(picks, picks[1:])]
    assert all(g == 10 for g in gaps)


def test_plan_sample_strata_and_auto_share():
    units = ar.slot_units(_frames(), PERIODS)
    samples, strata = ar.plan_sample(units, 40, seed=3, unlabelled_frac=0.1, auto_min_share=0.25)
    assert len(samples) == 40
    assert len({s["sample_id"] for s in samples}) == 40
    by = Counter(s["stratum"] for s in samples)
    assert sum(v for k, v in by.items() if k.startswith("unlabelled|")) == 4
    lab = [s for s in samples if s["team"] is not None]
    assert sum(s["period"] == "auto" for s in lab) >= round(0.25 * 36)
    for k, v in strata.items():                     # bookkeeping is consistent
        assert v["n"] == by.get(k, 0) and v["n"] <= v["N"]
    assert all(not k.startswith("unlabelled|") or s["team"] is None
               for s in samples for k in [s["stratum"]])
    # every labelled (period, slot) stratum is represented
    assert {k for k in strata if not k.startswith("unlabelled|")} <= set(by)
    # deterministic
    again, _ = ar.plan_sample(units, 40, seed=3, unlabelled_frac=0.1, auto_min_share=0.25)
    assert [(s["sample_id"], s["frame"], s["slot_key"]) for s in again] == \
        [(s["sample_id"], s["frame"], s["slot_key"]) for s in samples]


def test_plan_sample_with_no_unlabelled_units():
    units = ar.slot_units(_frames(unlabelled=()), PERIODS)
    samples, _ = ar.plan_sample(units, 30, seed=0)
    assert len(samples) == 30 and all(s["team"] is not None for s in samples)


# ----------------------------------------------------------------- answers --

def test_parse_answer_and_conf():
    roster = {"254", "1678"}
    assert ar.parse_answer({"team": "254", "confidence": "high"}, roster) == \
        {"kind": "team", "team": "254", "conf": 0.9}
    assert ar.parse_answer({"team": "frc1678", "confidence": 80}, roster)["conf"] == 0.8
    assert ar.parse_answer({"team": "9999", "confidence": 1}, roster)["kind"] == "none"
    assert ar.parse_answer({"team": "no_robot", "confidence": 0.9}, roster)["kind"] == "no_robot"
    assert ar.parse_answer({"team": "unsure"}, roster) == {"kind": "unsure", "team": None,
                                                           "conf": 0.0}
    assert ar.parse_answer({"team": 254, "confidence": "0.75"}, roster)["conf"] == 0.75


# ------------------------------------------------------------- scoring math --

def _rec(sid, stratum, w, pipe, team, conf=0.9, kind="team"):
    return {"sample_id": sid, "stratum": stratum, "w": w, "pipe": pipe, "slot": "B1",
            "ans": {"kind": kind, "team": team, "conf": conf}}


def test_sample_weights_renormalise_missing_items():
    strata = {"a": {"N": 100, "n": 4}, "b": {"N": 10, "n": 2}}
    samples = [{"sample_id": "x", "stratum": "a"}, {"sample_id": "y", "stratum": "a"},
               {"sample_id": "z", "stratum": "b"}]
    w = ar.sample_weights(samples, strata, 0.2)
    assert w == {"x": pytest.approx(10.0), "y": pytest.approx(10.0), "z": pytest.approx(2.0)}


def test_misid_estimate_weighted():
    recs = [_rec("1", "a", 9.0, "254", "254"),           # agree, heavy
            _rec("2", "b", 1.0, "254", "1678"),          # disagree, light
            _rec("3", "b", 1.0, "254", "1678", conf=0.3),  # not confident
            _rec("4", "b", 1.0, "254", None, kind="unsure"),
            _rec("5", "b", 1.0, "254", None, kind="no_robot")]
    e = ar.misid_estimate(recs, 0.7)
    assert e["n_confident"] == 2 and e["n_disagree"] == 1
    assert e["misid"] == pytest.approx(1.0 / 10.0)
    assert e["misid_unw"] == pytest.approx(0.5)
    assert e["coverage"] == pytest.approx(10.0 / 13.0)
    assert e["no_robot_rate"] == pytest.approx(1.0 / 13.0)
    assert e["misid_s"] == pytest.approx(0.1 * 13.0)


def test_misid_estimate_no_confident():
    e = ar.misid_estimate([_rec("1", "a", 1.0, "254", None, kind="unsure")], 0.7)
    assert e["misid"] is None and e["misid_s"] is None


def test_bootstrap_ci_brackets_estimate():
    recs = [_rec(str(i), "a" if i % 2 else "b", 1.0, "254", "1678" if i % 5 == 0 else "254")
            for i in range(60)]
    m = ar.misid_estimate(recs, 0.7)["misid"]
    lo, hi = ar.bootstrap_ci(recs, 0.7, reps=300, seed=1)
    assert lo <= m <= hi and hi - lo < 0.4


def test_breakdown_by_slot():
    recs = [_rec("1", "a", 1.0, "254", "254"), _rec("2", "a", 1.0, "254", "1678")]
    recs[1]["slot"] = "R1"
    b = ar.breakdown(recs, "slot", 0.7)
    assert b == {"B1": (1, 0, 0.0), "R1": (1, 1, 1.0)}


# -------------------------------------------------------------- GT matching --

def test_gt_match_nearest_within_gate():
    pipe = {"B1": (0.0, 0.0), "B2": (5.0, 5.0)}
    gt = {"254": (0.3, 0.0), "1678": (5.2, 5.0)}
    assert ar.gt_match("B1", pipe, gt)[0] == "254"
    assert ar.gt_match("B2", pipe, gt)[0] == "1678"


def test_gt_match_gate_and_hungarian():
    # nothing within 1 m
    t, d = ar.gt_match("B1", {"B1": (0.0, 0.0)}, {"254": (2.0, 0.0)})
    assert t is None and d == pytest.approx(2.0)
    # two slots near one robot: the nearer slot gets it, the other none
    pipe = {"B1": (0.0, 0.0), "B2": (0.5, 0.0)}
    gt = {"254": (0.45, 0.0)}
    assert ar.gt_match("B2", pipe, gt)[0] == "254"
    assert ar.gt_match("B1", pipe, gt)[0] is None
    # Hungarian: B1 is nearer 254 but 254 is the only robot B2 can take
    pipe = {"B1": (0.0, 0.0), "B2": (1.6, 0.0)}
    gt = {"254": (0.8, 0.0), "1678": (-0.5, 0.0)}
    assert ar.gt_match("B1", pipe, gt)[0] == "1678"
    assert ar.gt_match("B2", pipe, gt)[0] == "254"
    assert ar.gt_match("R1", pipe, gt) == (None, None)


def test_calibration_table_bins():
    recs = [dict(_rec("1", "a", 1, "254", "254", conf=0.95), gt="254"),
            dict(_rec("2", "a", 1, "254", "1678", conf=0.95), gt="254"),
            dict(_rec("3", "a", 1, "254", "254", conf=0.6), gt="254"),
            dict(_rec("4", "a", 1, "254", "254", conf=0.6), gt=None)]
    t = ar.calibration_table(recs)
    assert (0.9, 1.01, 2, 1) in t and (0.5, 0.7, 1, 1) in t


# -------------------------------------------------------- corner snap + jev --

def test_snap_to_detection_unique_only():
    t = {"u": 100.0, "v": 100.0, "bumper_px": 50.0}
    assert ar.snap_to_detection(t, [(130.0, 100.0), (400.0, 100.0)])[:2] == (130.0, 100.0)
    assert ar.snap_to_detection(t, [(200.0, 100.0)]) is None             # 2 bw away
    assert ar.snap_to_detection(t, [(120.0, 100.0), (80.0, 100.0)]) is None  # ambiguous
    assert ar.snap_to_detection(t, []) is None


def test_dets_near():
    vd = {100: [(1, 1)], 124: [(2, 2)]}
    assert ar.dets_near(vd, 100, 12) == [(1, 1)]
    assert ar.dets_near(vd, 112, 12) == [(1, 1)]
    assert ar.dets_near(vd, 150, 12) is None


def test_jev_to_answer_and_fuse():
    roster = ["254", "1678"]
    assert ar.jev_to_answer({"254": 0.8, "unreadable": 0.2}, roster) == \
        {"team": "254", "confidence": 0.8}
    assert ar.jev_to_answer({"254": 0.1, "not_robot": 0.9}, roster)["team"] == "no_robot"
    assert ar.jev_to_answer({"other": 0.6, "254": 0.4}, roster)["team"] == "none"
    assert ar.jev_to_answer(None, roster)["team"] == "unsure"
    f = ar.fuse_views({"main": {"254": 0.6, "unreadable": 0.4},
                       "corner_left": {"unreadable": 0.9, "254": 0.1}}, roster)
    assert f == {"team": "254", "confidence": 0.6}
    f = ar.fuse_views({"main": {"254": 0.6}, "corner_left": {"1678": 0.7}}, roster)
    assert f["team"] == "unsure" and f["conflict"] == ["1678", "254"]
