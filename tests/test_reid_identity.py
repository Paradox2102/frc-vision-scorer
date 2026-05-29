import numpy as np
import pytest

from web.reid import assign_identities

MAX_COST = 0.45  # accept a label only if (1 - cosine) <= 0.45


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_never_assigns_same_team_twice():
    # Two tracks both look most like team "A"; only one may get it.
    a = _e(1, 0, 0)
    tracks = {1: a, 2: _e(0.9, 0.1, 0)}
    refs = {"A": a, "B": _e(0, 1, 0)}
    alliance_track = {1: "blue", 2: "blue"}
    alliance_team = {"A": "blue", "B": "blue"}
    result = assign_identities(tracks, refs, alliance_track, alliance_team,
                               locked_tracks=set(), locked_teams=set(), max_cost=MAX_COST)
    assigned_teams = [t for t in result.values() if t is not None]
    assert len(assigned_teams) == len(set(assigned_teams))  # unique


def test_alliance_hard_constraint_forbids_cross():
    # Track 1 is red but visually identical to a blue reference; must NOT match it.
    v = _e(1, 0, 0)
    tracks = {1: v}
    refs = {"BLUE_TEAM": v}
    result = assign_identities(tracks, refs, {1: "red"}, {"BLUE_TEAM": "blue"},
                               set(), set(), MAX_COST)
    assert result.get(1) is None


def test_locked_tracks_and_teams_excluded():
    a, b = _e(1, 0, 0), _e(0, 1, 0)
    tracks = {1: a, 2: b}
    refs = {"A": a, "B": b}
    # Track 1 already locked to "A"; only track 2 / team "B" remain.
    result = assign_identities(tracks, refs, {1: "blue", 2: "blue"},
                               {"A": "blue", "B": "blue"},
                               locked_tracks={1}, locked_teams={"A"}, max_cost=MAX_COST)
    assert 1 not in result        # locked track not reassigned
    assert result.get(2) == "B"


def test_cost_above_threshold_unassigned():
    tracks = {1: _e(1, 0, 0)}
    refs = {"FAR": _e(0, 1, 0)}  # cosine ~0 -> cost ~1 > MAX_COST
    result = assign_identities(tracks, refs, {1: "blue"}, {"FAR": "blue"},
                               set(), set(), MAX_COST)
    assert result.get(1) is None


def test_rectangular_more_tracks_than_refs():
    a = _e(1, 0, 0)
    tracks = {1: a, 2: _e(0.8, 0.2, 0), 3: _e(0.6, 0.4, 0)}
    refs = {"A": a}
    result = assign_identities(tracks, refs,
                               {1: "blue", 2: "blue", 3: "blue"}, {"A": "blue"},
                               set(), set(), MAX_COST)
    assert sum(1 for v in result.values() if v == "A") == 1  # only one gets it
