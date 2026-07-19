"""Tests for web.field.identity_resolve: exact global slot->team assignment.

See web/field/identity_resolve.py module docstring for the design rationale
(exhaustive enumeration over ~34 candidates/alliance is exact and cheap here).
"""
from __future__ import annotations

import pytest

from web.field.identity_resolve import (
    SlotEvidence,
    assign_teams_global,
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
    # Two candidates tie on total score: {0: "1111"} alone (score 3.0) vs
    # {0: "1111", 1: "2222"} (score 3.0 + 0.0 -> but 2222 must clear min_score).
    # Construct so that assigning slot 1 costs nothing (adds a qualifying,
    # score-equal pair) so total score ties but assigned-count differs.
    evidence = {
        0: _ev(0, "blue", read_scores={"1111": 3.0}),
        1: _ev(1, "blue", read_scores={"2222": 3.0}),
    }
    roster = ["1111", "2222"]
    result = assign_teams_global(evidence, blue_roster=roster, red_roster=[], min_score=0.5)
    # Assigning both slots (total 6.0) strictly beats assigning only one (3.0),
    # so this exercises the "more assigned slots" path via the score ordering
    # already, but also confirms the deterministic single optimum:
    assert result[0] == "1111"
    assert result[1] == "2222"

    # Now force an actual tie in total score with a genuine choice: two ways
    # to reach the same max total, prefer more assigned slots, then lex order.
    evidence2 = {
        0: _ev(0, "blue", read_scores={"1111": 5.0, "2222": 5.0}),
    }
    # Only slot 0 has evidence; both candidate teams tie at 5.0. Assigning
    # either is a 1-slot candidate with equal score -> lexicographically
    # smallest (slot_id, team) tuple wins: (0, "1111") < (0, "2222").
    result2 = assign_teams_global(
        evidence2, blue_roster=["1111", "2222"], red_roster=[], min_score=0.5
    )
    assert result2[0] == "1111"


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
