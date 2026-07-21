"""Tests for web.field.identity_resolve: exact global slot->team assignment.

See web/field/identity_resolve.py module docstring for the design rationale
(exhaustive enumeration over ~34 candidates/alliance is exact and cheap here).
"""
from __future__ import annotations

import pytest

from web.field.identity_resolve import (
    SlotEvidence,
    assign_teams_global,
    build_slot_evidence,
    combine_scores,
    merge_global_assignment,
)


BLUE_ROSTER = ["1111", "2222", "3333"]
RED_ROSTER = ["4444", "5555", "6666"]


def _ev(slot_id, alliance, read_scores=None, gallery_scores=None, frames_seen=0):
    return SlotEvidence(
        slot_id=slot_id,
        alliance=alliance,
        read_scores=read_scores or {},
        gallery_scores=gallery_scores or {},
        frames_seen=frames_seen,
    )


# ---------------------------------------------------------------------------
# combine_scores
# ---------------------------------------------------------------------------


def test_combine_scores_linear_blend():
    ev = _ev(0, "blue", read_scores={"1111": 2.0}, gallery_scores={"1111": 1.0, "2222": 0.5})
    out = combine_scores(ev, w_reads=1.0, w_gallery=0.4)
    assert out["1111"] == pytest.approx(2.0 * 1.0 + 1.0 * 0.4)
    assert out["2222"] == pytest.approx(0.0 * 1.0 + 0.5 * 0.4)


# ---------------------------------------------------------------------------
# assign_teams_global
# ---------------------------------------------------------------------------


def test_collision_impossible():
    # Two blue slots both prefer "1234"; only one can get it.
    evidence = {
        0: _ev(0, "blue", read_scores={"1234": 5.0, "9999": 1.0}),
        1: _ev(1, "blue", read_scores={"1234": 4.5}),
    }
    roster = ["1234", "9999"]
    result = assign_teams_global(evidence, blue_roster=roster, red_roster=[])
    assigned_teams = [result[s] for s in (0, 1, 2) if result[s] is not None]
    # "1234" appears in at most one slot.
    assert assigned_teams.count("1234") == 1
    # slot 2 has no evidence -> None.
    assert result[2] is None
    # Every returned key covers the fixed 6-slot universe.
    assert set(result.keys()) == {0, 1, 2, 3, 4, 5}


def test_global_beats_greedy_lock():
    # Classic misattribution case: greedy-by-max assigns slot A -> 1111 (5.0),
    # starving slot B (which only has evidence for 1111, at 4.8) down to None.
    # The globally optimal total is A->2222 (4.9) + B->1111 (4.8) = 9.7,
    # beating greedy's A->1111 (5.0) + B->None = 5.0.
    evidence = {
        0: _ev(0, "blue", read_scores={"1111": 5.0, "2222": 4.9}),
        1: _ev(1, "blue", read_scores={"1111": 4.8}),
    }
    roster = ["1111", "2222"]
    result = assign_teams_global(evidence, blue_roster=roster, red_roster=[])
    assert result[0] == "2222"
    assert result[1] == "1111"
    assert result[2] is None


def test_deterministic_under_dict_order():
    evidence_a = {
        0: _ev(0, "blue", read_scores={"1111": 5.0, "2222": 4.9}),
        1: _ev(1, "blue", read_scores={"1111": 4.8}),
        2: _ev(2, "blue", read_scores={"3333": 3.0}),
    }
    # Same content, reversed insertion order.
    evidence_b = {
        2: _ev(2, "blue", read_scores={"3333": 3.0}),
        1: _ev(1, "blue", read_scores={"1111": 4.8}),
        0: _ev(0, "blue", read_scores={"1111": 5.0, "2222": 4.9}),
    }
    roster = ["1111", "2222", "3333"]
    result_a = assign_teams_global(evidence_a, blue_roster=roster, red_roster=[])
    result_b = assign_teams_global(evidence_b, blue_roster=roster, red_roster=[])
    assert result_a == result_b


def test_min_score_floor():
    evidence = {
        0: _ev(0, "blue", read_scores={"1111": 0.3}),
    }
    result = assign_teams_global(
        evidence, blue_roster=["1111"], red_roster=[], min_score=0.5
    )
    assert result[0] is None


def test_alliance_separation():
    # Blue slot's read_scores contains a red team; it must never win a blue slot.
    evidence = {
        0: _ev(0, "blue", read_scores={"4444": 9.0, "1111": 1.0}),
    }
    result = assign_teams_global(
        evidence, blue_roster=BLUE_ROSTER, red_roster=RED_ROSTER
    )
    assert result[0] != "4444"
    for slot in (0, 1, 2):
        assert result[slot] not in RED_ROSTER
    for slot in (3, 4, 5):
        assert result[slot] not in BLUE_ROSTER


def test_tie_break_prefers_more_assignments_then_lex():
    # --- Clause 1: prefer MORE assigned slots on a genuine total-score tie. ---
    #
    # Slot scores (reads only, w_gallery contributes nothing):
    #   slot0: 1111->1.0, 2222->0.5   slot1: 1111->0.5   slot2: (none)
    # Valid (>= min_score=0.5) pairs: (0,1111)=1.0, (0,2222)=0.5, (1,1111)=0.5.
    #
    # The two maximal candidates tie at total 1.0 but differ in assigned count:
    #   A = {0: 1111}            -> 1.0, 1 slot
    #   B = {0: 2222, 1: 1111}   -> 0.5 + 0.5 = 1.0, 2 slots
    # No candidate exceeds 1.0: 1111 (the only >0.5 slot1 team) cannot occupy
    # both slot0 and slot1, so the best 2-slot total is exactly 2222+1111=1.0.
    # The assigned-count clause MUST pick B. A count-blind solver would pick A
    # (its lex key (0,"1111") sorts before B's, so lex alone favors A) and
    # leave slot 1 None -- which is exactly the bug this test guards against.
    evidence = {
        0: _ev(0, "blue", read_scores={"1111": 1.0, "2222": 0.5}),
        1: _ev(1, "blue", read_scores={"1111": 0.5}),
    }
    roster = ["1111", "2222", "3333"]
    result = assign_teams_global(evidence, blue_roster=roster, red_roster=[], min_score=0.5)
    assert result == {0: "2222", 1: "1111", 2: None, 3: None, 4: None, 5: None}

    # --- Clause 2: on an EQUAL-count tie, lexicographically smallest wins. ---
    # Only slot 0 has evidence; both teams tie at 5.0. Both candidates are
    # 1-slot, so the count clause can't break it -> lex: (0,"1111") < (0,"2222").
    evidence2 = {
        0: _ev(0, "blue", read_scores={"1111": 5.0, "2222": 5.0}),
    }
    result2 = assign_teams_global(
        evidence2, blue_roster=["1111", "2222"], red_roster=[], min_score=0.5
    )
    assert result2[0] == "1111"


def test_weights_thread_through_solver():
    # Reads favor "1111", gallery favors "2222". At the default w_gallery=0.4
    # the read weight dominates and "1111" wins; crank w_gallery up and the
    # blend flips the winner to "2222" -- proving both weights actually thread
    # through assign_teams_global (not just combine_scores in isolation).
    ev = _ev(
        0, "blue",
        read_scores={"1111": 2.0},
        gallery_scores={"2222": 3.0},
    )
    # Default weights: 1111 -> 2.0, 2222 -> 0.4 * 3.0 = 1.2 -> 1111 wins.
    default = assign_teams_global(
        {0: ev}, blue_roster=["1111", "2222"], red_roster=[], min_score=0.5
    )
    assert default[0] == "1111"
    # High gallery weight: 1111 -> 2.0, 2222 -> 1.0 * 3.0 = 3.0 -> 2222 wins.
    weighted = assign_teams_global(
        {0: ev}, blue_roster=["1111", "2222"], red_roster=[], w_gallery=1.0, min_score=0.5
    )
    assert weighted[0] == "2222"


# ---------------------------------------------------------------------------
# merge_global_assignment
# ---------------------------------------------------------------------------


def test_merge_rules():
    slot_meta = {
        0: {"team_number": "1111", "confirmed": True, "extra": "keep-me"},
        1: {"team_number": "9999", "confirmed": False},
        2: {"team_number": "8888", "confirmed": True},
        3: {"team_number": None, "confirmed": False},
    }
    assignment = {
        0: "1111",  # same as existing -> untouched
        1: "2222",  # differs, margin >= 2.0 -> confirmed True
        2: "7777",  # differs, margin < 2.0 -> confirmed False
        3: None,    # unassigned -> leave untouched
    }
    scores = {
        0: {"1111": 5.0, "2222": 1.0},
        1: {"2222": 5.0, "9999": 1.0},          # margin = 5.0 - 1.0 = 4.0 >= 2.0
        2: {"7777": 3.0, "8888": 2.0},          # margin = 3.0 - 2.0 = 1.0 < 2.0
        3: {},
    }
    new_meta, changes = merge_global_assignment(slot_meta, assignment, scores, confirm_margin=2.0)

    # Original untouched.
    assert slot_meta[0]["team_number"] == "1111"
    assert slot_meta[1]["team_number"] == "9999"
    assert slot_meta[2]["team_number"] == "8888"
    assert slot_meta[3]["team_number"] is None

    # Slot 0: same team -> untouched, including confirmed + extra fields.
    assert new_meta[0] == {"team_number": "1111", "confirmed": True, "extra": "keep-me"}

    # Slot 1: overridden, margin 4.0 >= 2.0 -> confirmed True.
    assert new_meta[1]["team_number"] == "2222"
    assert new_meta[1]["confirmed"] is True

    # Slot 2: overridden, margin 1.0 < 2.0 -> confirmed False.
    assert new_meta[2]["team_number"] == "7777"
    assert new_meta[2]["confirmed"] is False

    # Slot 3: assignment None -> untouched entirely.
    assert new_meta[3] == {"team_number": None, "confirmed": False}

    # Change list: exactly the two overrides, exact contents.
    assert len(changes) == 2
    by_slot = {c["slot_id"]: c for c in changes}
    assert by_slot[1] == {
        "slot_id": 1,
        "old_team": "9999",
        "new_team": "2222",
        "score": 5.0,
        "margin": 4.0,
    }
    assert by_slot[2] == {
        "slot_id": 2,
        "old_team": "8888",
        "new_team": "7777",
        "score": 3.0,
        "margin": 1.0,
    }


def test_merge_does_not_mutate_input():
    slot_meta = {0: {"team_number": "1111", "confirmed": True}}
    assignment = {0: "2222"}
    scores = {0: {"2222": 5.0, "1111": 1.0}}
    original_copy = {0: dict(slot_meta[0])}
    merge_global_assignment(slot_meta, assignment, scores, confirm_margin=2.0)
    assert slot_meta == original_copy


def test_merge_strips_kept_label_colliding_with_assignment():
    # qm11 debut collision (2026-07-20 A/B): the exact assignment moved 1477
    # onto slot 2, slot 3's assignment came back None (1477 already taken),
    # and the v1 keep-on-None rule preserved slot 3's stale greedy "1477" —
    # two 1477s on the overlay. A team assigned to one slot must not survive
    # as a kept label on any other slot.
    slot_meta = {
        2: {"team_number": None, "confirmed": False},
        3: {"team_number": "1477", "confirmed": False},
    }
    assignment = {2: "1477", 3: None}
    scores = {2: {"1477": 1.1}, 3: {}}
    new_meta, changes = merge_global_assignment(slot_meta, assignment, scores)

    assert new_meta[2]["team_number"] == "1477"
    assert new_meta[3]["team_number"] is None       # stripped, not duplicated
    assert new_meta[3]["confirmed"] is False
    teams = [m["team_number"] for m in new_meta.values() if m["team_number"]]
    assert len(teams) == len(set(teams))
    strip = {c["slot_id"]: c for c in changes}[3]
    assert strip["old_team"] == "1477" and strip["new_team"] is None
    # Original still unmutated.
    assert slot_meta[3]["team_number"] == "1477"


def test_merge_strips_collision_when_slot_absent_from_assignment():
    # Same hazard when the stale slot never entered the solve at all (no
    # evidence rows -> absent from ``assignment``, not merely None).
    slot_meta = {
        2: {"team_number": None, "confirmed": False},
        3: {"team_number": "1477", "confirmed": True},
    }
    new_meta, _ = merge_global_assignment(
        slot_meta, {2: "1477"}, {2: {"1477": 3.0}})
    assert new_meta[2]["team_number"] == "1477"
    assert new_meta[3]["team_number"] is None
    assert new_meta[3]["confirmed"] is False


# ---------------------------------------------------------------------------
# Extra edge cases
# ---------------------------------------------------------------------------


def test_empty_roster_yields_all_none():
    evidence = {0: _ev(0, "blue", read_scores={"1111": 9.0})}
    result = assign_teams_global(evidence, blue_roster=[], red_roster=[])
    assert result == {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}


def test_evidence_for_unknown_slot_id_raises():
    evidence = {9: _ev(9, "blue", read_scores={"1111": 9.0})}
    with pytest.raises(ValueError):
        assign_teams_global(evidence, blue_roster=BLUE_ROSTER, red_roster=RED_ROSTER)


def test_alliance_mismatch_raises():
    # slot_id=0 is blue by convention, but evidence claims "red" -> caller bug.
    evidence = {0: _ev(0, "red", read_scores={"4444": 9.0})}
    with pytest.raises(ValueError):
        assign_teams_global(evidence, blue_roster=BLUE_ROSTER, red_roster=RED_ROSTER)


# ---------------------------------------------------------------------------
# build_slot_evidence
# ---------------------------------------------------------------------------


def _slot_alliance(slot_id):
    return "blue" if slot_id in (0, 1, 2) else "red"


def _id_row(frame_num, identity_id, slot_id, alliance):
    return {
        "frame_num": frame_num,
        "identity_id": identity_id,
        "slot_id": slot_id,
        "alliance": alliance,
    }


def _vlm_row(frame, robot_id, matched_team, vote_weight=1.0, **extra):
    row = {
        "type": "vlm_read",
        "frame": frame,
        "robot_id": robot_id,
        "matched_team": matched_team,
        "confidence": 0.9,
        "vote_weight": vote_weight,
        "alliance_read": "blue",
        "surface": "front",
        "legibility": 0.8,
    }
    row.update(extra)
    return row


def test_build_slot_evidence_sums_reads_by_slot_and_team():
    identity_log = [
        _id_row(10, 1, 0, "blue"),
        _id_row(20, 1, 0, "blue"),
        _id_row(10, 2, 3, "red"),
        _id_row(20, 2, 3, "red"),
    ]
    vlm_read_rows = [
        _vlm_row(10, 1, "1111", vote_weight=2.0),
        _vlm_row(20, 1, "1111", vote_weight=1.5),
        _vlm_row(10, 2, "4444", vote_weight=3.0),
        _vlm_row(20, 2, "5555", vote_weight=1.0),
    ]
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot={}, gallery_scores_by_slot={},
        slot_alliance=_slot_alliance,
    )
    assert evidence[0].read_scores == {"1111": 3.5}
    assert evidence[3].read_scores == {"4444": 3.0, "5555": 1.0}
    assert evidence[0].frames_seen == 2
    assert evidence[3].frames_seen == 2
    assert evidence[0].alliance == "blue"
    assert evidence[3].alliance == "red"
    assert stats == {
        "reads_used": 4,
        "reads_dropped_misattributed": 0,
        "fallback_used": False,
    }


def test_build_slot_evidence_drops_misattributed_read():
    # identity 1 is only bound to slot 0 at frame 10; a read at frame 99 for
    # the same identity has no binding and must be dropped, not attributed.
    identity_log = [_id_row(10, 1, 0, "blue")]
    vlm_read_rows = [
        _vlm_row(10, 1, "1111", vote_weight=2.0),  # bound -> counted
        _vlm_row(99, 1, "2222", vote_weight=5.0),  # unbound at frame 99 -> dropped
    ]
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot={}, gallery_scores_by_slot={},
        slot_alliance=_slot_alliance,
    )
    assert evidence[0].read_scores == {"1111": 2.0}
    assert "2222" not in evidence[0].read_scores
    assert stats["reads_used"] == 1
    assert stats["reads_dropped_misattributed"] == 1
    assert stats["fallback_used"] is False


def test_build_slot_evidence_fallback_when_rows_none():
    team_ocr_votes_by_slot = {0: {"1111": 4.0}, 3: {"4444": 2.0}}
    evidence, stats = build_slot_evidence(
        None, identity_log=[], team_ocr_votes_by_slot=team_ocr_votes_by_slot,
        gallery_scores_by_slot={}, slot_alliance=_slot_alliance,
    )
    assert evidence[0].read_scores == {"1111": 4.0}
    assert evidence[3].read_scores == {"4444": 2.0}
    assert stats == {
        "reads_used": 0,
        "reads_dropped_misattributed": 0,
        "fallback_used": True,
    }


def test_build_slot_evidence_fallback_when_rows_empty_list():
    team_ocr_votes_by_slot = {0: {"1111": 4.0}}
    evidence, stats = build_slot_evidence(
        [], identity_log=[], team_ocr_votes_by_slot=team_ocr_votes_by_slot,
        gallery_scores_by_slot={}, slot_alliance=_slot_alliance,
    )
    assert evidence[0].read_scores == {"1111": 4.0}
    assert stats["fallback_used"] is True
    assert stats["reads_dropped_misattributed"] == 0


def test_build_slot_evidence_falls_back_when_all_reads_misattributed():
    # Every row either has no matched_team or has no identity_log binding at
    # its frame -> zero usable reads -> must fall back to OCR votes, but the
    # misattribution drop still gets counted before the fallback kicks in.
    identity_log = [_id_row(10, 1, 0, "blue")]
    vlm_read_rows = [
        _vlm_row(99, 1, "1111", vote_weight=5.0),  # unbound -> dropped
        _vlm_row(10, 1, None, vote_weight=0.0),  # no team vote -> skipped, not a drop
    ]
    team_ocr_votes_by_slot = {0: {"9999": 1.0}}
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot=team_ocr_votes_by_slot,
        gallery_scores_by_slot={}, slot_alliance=_slot_alliance,
    )
    assert evidence[0].read_scores == {"9999": 1.0}
    assert stats["fallback_used"] is True
    assert stats["reads_used"] == 0
    assert stats["reads_dropped_misattributed"] == 1


def test_build_slot_evidence_attaches_gallery_to_every_slot_incl_gallery_only():
    identity_log = [_id_row(10, 1, 0, "blue")]
    vlm_read_rows = [_vlm_row(10, 1, "1111", vote_weight=2.0)]
    gallery_scores_by_slot = {0: {"1111": 0.5}, 1: {"2222": 0.9}}
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot={},
        gallery_scores_by_slot=gallery_scores_by_slot, slot_alliance=_slot_alliance,
    )
    # slot 1 has no reads and no identity_log rows, but gallery evidence alone
    # is enough to make it a candidate slot.
    assert set(evidence.keys()) == {0, 1}
    assert evidence[0].gallery_scores == {"1111": 0.5}
    assert evidence[1].gallery_scores == {"2222": 0.9}
    assert evidence[1].read_scores == {}
    assert evidence[1].frames_seen == 0
    assert evidence[1].alliance == "blue"
    assert stats["fallback_used"] is False


def test_build_slot_evidence_fallback_slot_universe_excludes_ocr_only_when_reads_used():
    # Read path is active (>=1 usable read); an OCR-vote slot with no read
    # binding and no gallery evidence must NOT appear in the evidence dict --
    # team_ocr_votes_by_slot is only consulted on fallback.
    identity_log = [_id_row(10, 1, 0, "blue")]
    vlm_read_rows = [_vlm_row(10, 1, "1111", vote_weight=2.0)]
    team_ocr_votes_by_slot = {5: {"6666": 9.0}}
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot=team_ocr_votes_by_slot,
        gallery_scores_by_slot={}, slot_alliance=_slot_alliance,
    )
    assert set(evidence.keys()) == {0}
    assert stats["fallback_used"] is False


def test_build_slot_evidence_does_not_mutate_inputs():
    identity_log = [_id_row(10, 1, 0, "blue"), _id_row(99, 2, 3, "red")]
    vlm_read_rows = [
        _vlm_row(10, 1, "1111", vote_weight=2.0),
        _vlm_row(50, 2, "2222", vote_weight=1.0),  # dropped (unbound)
    ]
    team_ocr_votes_by_slot = {0: {"1111": 4.0}}
    gallery_scores_by_slot = {0: {"1111": 0.5}, 3: {"4444": 0.2}}

    identity_log_snapshot = [dict(r) for r in identity_log]
    vlm_rows_snapshot = [dict(r) for r in vlm_read_rows]
    ocr_votes_snapshot = {k: dict(v) for k, v in team_ocr_votes_by_slot.items()}
    gallery_snapshot = {k: dict(v) for k, v in gallery_scores_by_slot.items()}

    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, team_ocr_votes_by_slot=team_ocr_votes_by_slot,
        gallery_scores_by_slot=gallery_scores_by_slot, slot_alliance=_slot_alliance,
    )

    assert identity_log == identity_log_snapshot
    assert vlm_read_rows == vlm_rows_snapshot
    assert team_ocr_votes_by_slot == ocr_votes_snapshot
    assert gallery_scores_by_slot == gallery_snapshot

    # Mutating the returned evidence's dicts must not reach back into the
    # caller's gallery/OCR-vote dicts (no aliasing).
    evidence[0].gallery_scores["9999"] = 123.0
    evidence[0].read_scores["8888"] = 456.0
    assert "9999" not in gallery_scores_by_slot[0]
    assert "8888" not in team_ocr_votes_by_slot.get(0, {})
