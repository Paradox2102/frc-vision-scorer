"""Pure-function tests for scripts/build_bumper_crop_bench.py (synthetic data only)."""
import json
import random
from collections import Counter

from scripts.build_bumper_crop_bench import (
    allocate_quota,
    classify_row,
    dedupe_candidates,
    make_crop_id,
    scan_diagnostic,
    select_bench,
    spread_pick,
    stratum_targets,
    _strip_crop,
)


def _cand(match, frame, rid, stratum="matched", job="j1", team="254", offset=0):
    return {"job_id": job, "match_key": match, "frame": frame, "robot_id": rid,
            "stratum": stratum, "matched_team": team if stratum == "matched" else None,
            "offset": offset, "t_ms": 0.0}


def test_classify_row():
    assert classify_row({"type": "vlm_read", "matched_team": "254"}) == "matched"
    assert classify_row({"type": "vlm_read", "matched_team": None}) == "null"
    assert classify_row({"type": "vlm_filtered"}) == "filtered"
    assert classify_row({"type": "frame"}) is None


def test_strip_crop_removes_payload_only():
    line = b'{"frame":1,"type":"vlm_read","robot_id":2,"crop_b64":"QUJD+/==","raw_read":"9"}\n'
    slim, n = _strip_crop(line)
    assert n == len("QUJD+/==")
    d = json.loads(slim)
    assert d["crop_b64"] == "" and d["raw_read"] == "9" and d["robot_id"] == 2


def test_scan_diagnostic_streams_and_keeps_offsets(tmp_path):
    rows = [
        {"frame": 1, "type": "frame_stats", "x": 1},
        {"frame": 5, "t_ms": 1.0, "type": "vlm_read", "robot_id": 3, "crop_b64": "QUJD",
         "raw_read": "254", "matched_team": "254", "alliance_read": "red", "confidence": 0.95},
        {"frame": 6, "t_ms": 2.0, "type": "vlm_filtered", "robot_id": 4, "crop_b64": "",
         "legibility": 0.0, "reason": "below_legibility_floor"},  # empty crop -> skipped
        {"frame": 7, "t_ms": 3.0, "type": "vlm_filtered", "robot_id": 4, "crop_b64": "REVG",
         "legibility": 0.0, "reason": "below_legibility_floor"},
    ]
    p = tmp_path / "diagnostic_x.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = list(scan_diagnostic(p, "x", "2026cur_qm1"))
    assert [(c["stratum"], c["frame"]) for c in out] == [("matched", 5), ("filtered", 7)]
    with open(p, "rb") as fh:  # offsets point at the right lines
        fh.seek(out[1]["offset"])
        assert json.loads(fh.readline())["crop_b64"] == "REVG"


def test_dedupe_is_order_independent():
    cands = [_cand("m1", 10, 1, job="a"), _cand("m1", 10, 1, job="b"),
             _cand("m1", 11, 1, job="a"), _cand("m2", 10, 1, job="a")]
    d1 = dedupe_candidates(cands, seed=0)
    d2 = dedupe_candidates(list(reversed(cands)), seed=0)
    assert len(d1) == 3
    assert [c["job_id"] for c in d1] == [c["job_id"] for c in d2]


def test_stratum_targets_sum_and_shares():
    t = stratum_targets(200)
    assert t == {"matched": 100, "null": 70, "filtered": 30}
    assert sum(stratum_targets(7).values()) == 7


def test_allocate_quota_water_fills():
    alloc = allocate_quota({"a": 1, "b": 10, "c": 10}, 9, random.Random(0))
    assert sum(alloc.values()) == 9 and alloc["a"] == 1
    assert abs(alloc["b"] - alloc["c"]) <= 1
    # Total capped by capacity
    assert sum(allocate_quota({"a": 2}, 9, random.Random(0)).values()) == 2


def test_spread_pick_covers_time_and_diversifies_teams():
    cands = [_cand("m", f, f % 7, team=("A" if f < 90 else "B")) for f in range(100)]
    picks = spread_pick(cands, 4, random.Random(1), diversity_key="matched_team")
    frames = sorted(p["frame"] for p in picks)
    assert len(picks) == 4
    assert frames[0] < 25 and frames[-1] >= 75  # one per quantile bin
    assert {p["matched_team"] for p in picks} == {"A", "B"}


def test_select_bench_stratified_deterministic():
    cands = []
    for m in ("m1", "m2", "m3"):
        for f in range(40):
            cands.append(_cand(m, f, 0, "matched", team=str(f % 3)))
            cands.append(_cand(m, f, 1, "null"))
            if f % 4 == 0:
                cands.append(_cand(m, f, 2, "filtered"))
        cands.append(_cand(m, 0, 0, "matched", job="dup"))  # re-run duplicate
    a = select_bench(cands, 20, seed=3)
    b = select_bench(list(reversed(cands)), 20, seed=3)
    assert [make_crop_id(c["match_key"], c["frame"], c["robot_id"]) for c in a] == \
           [make_crop_id(c["match_key"], c["frame"], c["robot_id"]) for c in b]
    assert Counter(c["stratum"] for c in a) == {"matched": 10, "null": 7, "filtered": 3}
    assert len({c["match_key"] for c in a}) == 3
    ids = [make_crop_id(c["match_key"], c["frame"], c["robot_id"]) for c in a]
    assert len(ids) == len(set(ids))


def test_select_bench_hands_shortfall_to_other_strata():
    cands = [_cand("m1", f, 0, "matched") for f in range(30)]  # no null/filtered at all
    assert len(select_bench(cands, 10, seed=0)) == 10


def test_make_crop_id():
    assert make_crop_id("2026cur_qm11", 733, 5) == "2026cur_qm11_f00733_r005"
