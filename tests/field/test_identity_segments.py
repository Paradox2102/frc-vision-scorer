"""identity_segment_resolve core (web/field/identity_segments.py): cut slots at
discontinuities, label segments by their own reads, stitch unread segments
along feasible links with team exclusivity; plus the Clef evidence floor."""
from web.field.identity_resolve import apply_evidence_floor, resolve_slot_labels
from web.field.identity_segments import (
    SegmentParams,
    boundary_cuts,
    boundary_rows_from_rejects,
    build_segments,
    exclusivity_violations,
    format_segment_summary,
    is_cut,
    labels_at,
    presence_cuts,
    resolve_segment_labels,
    slot_presence,
)

BLUE = ["1111", "2222", "3333"]
RED = ["4444", "5555", "6666"]
FPS = 60.0
P = SegmentParams(fps=FPS)


def _rows(slot, frames, pos, iid=None):
    """identity_log rows for one slot; ``pos`` is (x, y) or f -> (x, y)."""
    return [{"frame_num": f, "identity_id": 100 + slot if iid is None else iid,
             "slot_id": slot, "alliance": "blue" if slot < 3 else "red",
             "pos": list(pos(f) if callable(pos) else pos)} for f in frames]


def _read(frame, iid, team, w=1.0, conf=0.95, probs=None):
    r = {"type": "vlm_read", "frame": frame, "robot_id": iid, "matched_team": team,
         "vote_weight": w, "confidence": conf}
    if probs is not None:
        r["clef_probs"] = probs
    return r


def _steady_match():
    """Six slots, one robot each all match, far apart; 2 reads per slot."""
    frames = list(range(0, 800, 8))
    log, reads = [], []
    for s in range(6):
        log += _rows(s, frames, (2.0 + 2.5 * s, 1.0 + 1.0 * s))
        team = (BLUE + RED)[s]
        reads += [_read(104, 100 + s, team), _read(600, 100 + s, team)]
    return log, reads


def _segs(res, slot):
    return [(g["start_frame"], g["end_frame"], g["team_number"], g["source"])
            for g in res["segments"][slot]]


# --------------------------------------------------------------------------- #
# Required behaviours
# --------------------------------------------------------------------------- #
def test_no_discontinuity_slot_is_one_segment_equal_to_gir():
    log, reads = _steady_match()
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    gir = resolve_slot_labels(reads, log, {}, {}, lambda s: "blue" if s < 3 else "red",
                              BLUE, RED, {})["slot_meta"]
    for s in range(6):
        assert len(res["segments"][s]) == 1
        seg = res["segments"][s][0]
        assert seg["team_number"] == gir[s]["team_number"] == (BLUE + RED)[s]
        assert seg["source"] == "read" and seg["start_frame"] == 0 and seg["end_frame"] == 792
    assert res["stats"]["coverage"] == 1.0
    assert exclusivity_violations(res["segments"], log)["violations"] == 0


def test_discontinuity_with_conflicting_reads_gives_two_labels():
    """R1 teleports 6 m at f400 (a reseed onto another robot): before it reads
    4444, after it 6666 -> two segments, two labels. R3 carried 6666 before
    and 4444 after (the two robots traded slots), far from R1."""
    frames = list(range(0, 800, 8))
    log = _rows(3, frames, lambda f: (3.0, 2.0) if f < 400 else (9.0, 2.0))
    log += _rows(4, frames, (12.0, 8.0))
    log += _rows(5, frames, lambda f: (9.0, 6.0) if f < 400 else (3.0, 6.0))
    reads = [_read(104, 103, "4444"), _read(200, 103, "4444"),
             _read(600, 103, "6666"), _read(704, 103, "6666"),
             _read(104, 104, "5555"),
             _read(104, 105, "6666"), _read(704, 105, "4444")]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    assert _segs(res, 3) == [(0, 392, "4444", "read"), (400, 792, "6666", "read")]
    assert _segs(res, 5) == [(0, 392, "6666", "read"), (400, 792, "4444", "read")]
    assert labels_at(res["segments"], 3, 300) == "4444"
    assert labels_at(res["segments"], 3, 500) == "6666"
    assert exclusivity_violations(res["segments"], log)["violations"] == 0


def _handoff(seed_pos):
    """R1 carries 4444 (read) until f400 and expires; R2 is seeded at f448 at
    ``seed_pos`` and never read. R3 elsewhere reads 6666."""
    log = _rows(3, range(0, 401, 8), (5.0, 4.0))
    log += _rows(4, range(448, 800, 8), seed_pos)
    log += _rows(5, range(0, 800, 8), (14.0, 1.0))
    reads = [_read(104, 103, "4444"), _read(296, 103, "4444"), _read(504, 105, "6666")]
    return log, reads


def test_unread_segment_is_stitched_along_a_feasible_link():
    log, reads = _handoff((5.8, 4.3))           # 0.85 m away, 0.8 s later
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    seg = res["segments"][4][0]
    assert (seg["team_number"], seg["source"], seg["hops"]) == ("4444", "stitched", 1)
    assert seg["support"] == 2.0 and seg["confirmed"] is False
    assert (3, 4) in {(a_slot, b_slot) for a_slot, b_slot in
                      [(_slot_of(res, a), _slot_of(res, b)) for a, b, _c in res["links"]]}
    assert res["stats"]["stitched"] == 1


def _slot_of(res, idx):
    k = 0
    for slot in sorted(res["segments"]):
        for _g in res["segments"][slot]:
            if k == idx:
                return slot
            k += 1
    return None


def test_infeasible_link_is_refused():
    """Same handoff but the new slot appears 12 m away 0.8 s later (> 5 m/s
    x 0.8 s + 1.5 m): no link, the unread segment stays unlabelled."""
    log, reads = _handoff((5.0 + 12.0, 4.0))
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    seg = res["segments"][4][0]
    assert (seg["team_number"], seg["source"]) == (None, "none")
    assert res["stats"]["links"] == 0
    # ...and a link beyond link_max_gap_s is refused even when close.
    log2 = _rows(3, range(0, 401, 8), (5.0, 4.0)) + _rows(4, range(600, 800, 8), (5.2, 4.0))
    res2 = resolve_segment_labels(reads[:2], log2, BLUE, RED,
                                  params=SegmentParams(fps=FPS, link_max_gap_s=2.0))
    assert res2["segments"][4][0]["team_number"] is None


def test_exclusivity_refuses_a_stitch_that_puts_one_team_in_two_places():
    """R2's only feasible predecessor is R1 (4444), but R3 -- 8 m away over
    the same frames -- READS 4444 there: R2 may not become 4444 too."""
    log = _rows(3, range(0, 401, 8), (5.0, 4.0))
    log += _rows(4, range(448, 800, 8), (5.5, 4.0))
    log += _rows(5, range(448, 800, 8), (13.0, 4.0))
    reads = [_read(104, 103, "4444"), _read(600, 105, "4444"), _read(704, 105, "4444")]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    assert res["segments"][5][0]["team_number"] == "4444"
    assert res["segments"][4][0]["team_number"] is None
    assert res["stats"]["refused_exclusive"] == 1
    assert exclusivity_violations(res["segments"], log)["violations"] == 0


def test_exclusive_anchors_weaker_falls_back_to_its_next_read_team():
    """Two slots both read 4444 at the same time far apart: the stronger keeps
    it, the weaker takes its next-best read team (gir would do the same)."""
    frames = list(range(0, 400, 8))
    log = _rows(3, frames, (3.0, 2.0)) + _rows(4, frames, (10.0, 6.0))
    reads = [_read(104, 103, "4444", 2.5),
             _read(104, 104, "4444", 1.0), _read(200, 104, "5555", 0.5)]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    assert res["segments"][3][0]["team_number"] == "4444"
    seg = res["segments"][4][0]
    assert (seg["team_number"], seg["demoted"]) == ("5555", True)
    # No fallback read -> unlabelled.
    res = resolve_segment_labels(reads[:2], log, BLUE, RED, params=P)
    assert res["segments"][4][0]["team_number"] is None
    assert res["stats"]["demoted"] == 1


def test_duplicate_pair_shares_a_label_and_is_flagged():
    """Two slots on one robot (co-located within 1 m) both read 4444: allowed,
    flagged as duplicates, not an exclusivity violation."""
    frames = list(range(0, 400, 8))
    log = _rows(3, frames, (5.0, 4.0)) + _rows(4, frames, (5.4, 4.3))
    reads = [_read(104, 103, "4444"), _read(200, 104, "4444")]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    a, b = res["segments"][3][0], res["segments"][4][0]
    assert a["team_number"] == b["team_number"] == "4444"
    assert a["dup_with"] == [4] and b["dup_with"] == [3]
    ex = exclusivity_violations(res["segments"], log)
    assert ex["violations"] == 0 and ex["duplicates"] == len(frames)
    assert res["stats"]["dup_segments"] == 2


def test_floor_drops_low_conf_clef_reads():
    probs_lo = {"4444": 0.58, "6666": 0.10, "none": 0.30, "other": 0.02}
    probs_hi = {"6666": 0.90, "none": 0.05, "4444": 0.03, "other": 0.02}
    rows = [_read(104, 103, "4444", 0.5, 0.58, probs_lo),     # blurry misread
            _read(200, 103, "6666", 1.0, 0.90, probs_hi),
            _read(296, 103, None, 0.0, 0.0, {"none": 0.9}),    # no team: kept
            _read(400, 103, "4444", 0.5, 0.60)]                # non-Clef: kept
    kept, dropped = apply_evidence_floor(rows, min_conf=0.65)
    assert dropped == 1 and [r["frame"] for r in kept] == [200, 296, 400]
    assert apply_evidence_floor(rows, 0.0) == (rows, 0)
    assert apply_evidence_floor(None, 0.65) == (None, 0)
    kept, dropped = apply_evidence_floor(rows, min_margin=0.4)    # 0.58 - 0.30
    assert dropped == 1
    # The floor flips a segment the misread was deciding.
    frames = list(range(0, 400, 8))
    log = _rows(3, frames, (5.0, 4.0))
    two = [_read(104, 103, "4444", 1.0, 0.58, probs_lo),
           _read(200, 103, "6666", 0.5, 0.80, {**probs_hi, "6666": 0.80})]
    assert resolve_segment_labels(two, log, BLUE, RED, params=P)[
        "segments"][3][0]["team_number"] == "4444"
    floored = apply_evidence_floor(two, min_conf=0.65)[0]
    assert resolve_segment_labels(floored, log, BLUE, RED, params=P)[
        "segments"][3][0]["team_number"] == "6666"


# --------------------------------------------------------------------------- #
# Details
# --------------------------------------------------------------------------- #
def test_tied_reads_leave_a_segment_contested():
    log = _rows(3, range(0, 400, 8), (5.0, 4.0))
    reads = [_read(104, 103, "4444"), _read(200, 103, "6666")]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    seg = res["segments"][3][0]
    assert seg["team_number"] is None and seg["contested"] is True


def test_cross_alliance_and_off_roster_reads_are_ignored():
    log = _rows(3, range(0, 400, 8), (5.0, 4.0))
    reads = [_read(104, 103, "1111", 2.5), _read(152, 103, "9999", 2.5),
             _read(200, 103, "5555", 0.5)]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    assert res["segments"][3][0]["team_number"] == "5555"
    assert res["segments"][3][0]["reads"] == {"5555": 0.5}


def test_propagation_stops_where_a_chain_meets_a_different_anchor():
    """A(4444 read) -> U1 -> U2 -> B(6666 read) linked on one slot with
    gaps; the costliest link (U1->U2, 1.4 m) splits the chain: U1 takes A's
    label, U2 takes B's."""
    log = _rows(3, range(0, 201, 8), (5.0, 4.0))
    log += _rows(3, range(240, 401, 8), (5.3, 4.0))
    log += _rows(3, range(440, 601, 8), (6.7, 4.0))
    log += _rows(3, range(640, 801, 8), (7.0, 4.0))
    reads = [_read(104, 103, "4444"), _read(704, 103, "6666")]
    res = resolve_segment_labels(reads, log, BLUE, RED, params=P,
                                 grid=list(range(0, 801, 8)))
    assert [g[2:] for g in _segs(res, 3)] == [
        ("4444", "read"), ("4444", "stitched"), ("6666", "stitched"), ("6666", "read")]


def test_presence_gap_and_step_cut_but_noise_does_not():
    log = _rows(3, range(0, 201, 8), lambda f: (5.0 + 0.004 * f, 4.0))   # smooth
    log += _rows(3, range(208, 401, 8), (9.0, 4.0))                      # 3.2 m step
    log += _rows(3, range(424, 601, 8), (9.2, 4.0))                      # gap of 2 frames
    pres = slot_presence(log)
    grid = sorted({r["frame_num"] for r in log} | {408, 416})
    cuts = presence_cuts(pres, grid, P)
    assert cuts == {3: {208, 424}}
    segs = build_segments(pres, cuts)
    assert [(s.start, s.end) for s in segs] == [(0, 200), (208, 400), (424, 600)]


def test_exact_boundary_rows_cut_rules():
    base = {"frame": 80, "slot_id": 3}
    assert is_cut({**base, "kinds": ["seed"], "seed_dist_m": 0.3}, P)
    assert is_cut({**base, "kinds": ["reacquire"], "pred_jump_m": 3.1, "jump_m": 1.0}, P)
    assert not is_cut({**base, "kinds": ["reacquire"], "pred_jump_m": 0.4, "jump_m": 3.5}, P)
    assert is_cut({**base, "kinds": ["birth"], "birth_disrupted": True,
                   "birth_distance_m": 0.2}, P)
    assert is_cut({**base, "kinds": ["birth"], "birth_disrupted": False,
                   "birth_distance_m": 2.4}, P)
    assert not is_cut({**base, "kinds": ["birth"], "birth_disrupted": False,
                       "birth_distance_m": 0.6}, P)
    assert not is_cut({**base, "kinds": ["feed_change", "corner_only_on"]}, P)
    assert is_cut({**base, "kinds": ["feed_change"]}, SegmentParams(cut_feed_change=True))
    assert boundary_cuts([{**base, "kinds": ["seed"]}, {**base, "kinds": ["expire"]}],
                         P) == {3: {80}}
    rows = boundary_rows_from_rejects([
        {"type": "identity_reject", "frame": 96, "slot_id": 4, "slot_disrupted": True,
         "distance_m": 0.5}, {"type": "identity_reject", "frame": 104, "slot_id": None}])
    assert rows == [{"frame": 96, "slot_id": 4, "kinds": ["birth"],
                     "birth_disrupted": True, "birth_distance_m": 0.5}]


def test_exact_birth_boundary_splits_a_continuous_slot():
    """No gap, no step: only the logged boundary row separates the robots."""
    log = _rows(3, range(0, 800, 8), (5.0, 4.0))
    reads = [_read(104, 103, "4444"), _read(704, 103, "6666")]
    no_rows = resolve_segment_labels(reads, log, BLUE, RED, params=P)
    assert _segs(no_rows, 3) == [(0, 792, None, "none")]               # tie
    rows = [{"frame": 400, "slot_id": 3, "kinds": ["birth"], "birth_disrupted": True}]
    res = resolve_segment_labels(reads, log, BLUE, RED, boundary_rows=rows, params=P)
    assert _segs(res, 3) == [(0, 392, "4444", "read"), (400, 792, "6666", "read")]


def test_no_reads_leaves_everything_unlabelled():
    log, _ = _steady_match()
    res = resolve_segment_labels(None, log, BLUE, RED, params=P)
    assert all(g["team_number"] is None for v in res["segments"].values() for g in v)
    assert res["stats"]["coverage"] == 0.0
    line = format_segment_summary(res)
    assert line.startswith("[segment-resolve] segments=6 read=0 stitched=0 none=6")
    assert "R1=None[0-792]" in line
