import numpy as np
import pytest

from web.reid import (
    assign_teams_alliance_masked,
    eliminate_last_team,
    audit_overturn,
)

BLUE = ["2102", "9452", "2485"]
RED  = ["1538", "294", "4322"]


def _alliance_of(spec: dict[int, str]) -> dict[int, str]:
    return dict(spec)


# --- assign_teams_alliance_masked --------------------------------------------

def test_assignment_never_crosses_alliance():
    """A red robot cannot be assigned a blue team, even if its votes prefer one."""
    votes = {0: {"2102": 10.0}}              # robot 0 has only blue-team votes
    alliance = _alliance_of({0: "red"})       # but bumper says red
    out = assign_teams_alliance_masked(
        votes, alliance, BLUE, RED, locked_teams=set(), locked_robots=set(),
    )
    assert out[0] is None, "cross-alliance assignment must be forbidden"


def test_unknown_alliance_is_unconstrained():
    votes = {0: {"2102": 5.0}}
    alliance = _alliance_of({0: "unknown"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assert out[0] == "2102"


def test_no_team_assigned_twice():
    """Two robots both vote highest for the same team; only one gets it."""
    votes = {0: {"9452": 5.0}, 1: {"9452": 3.0, "2485": 1.0}}
    alliance = _alliance_of({0: "blue", 1: "blue"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assigned = [t for t in out.values() if t is not None]
    assert len(assigned) == len(set(assigned)), f"duplicate team in {out}"
    assert out[0] == "9452"  # strongest vote keeps the team


def test_no_votes_means_no_assignment():
    votes = {0: {}}
    alliance = _alliance_of({0: "blue"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assert out[0] is None


def test_locked_robots_and_teams_excluded():
    """Confirmed robots and their teams must not appear in the new assignment."""
    votes = {0: {"9452": 5.0}, 1: {"9452": 4.0, "2485": 1.0}}
    alliance = _alliance_of({0: "blue", 1: "blue"})
    out = assign_teams_alliance_masked(
        votes, alliance, BLUE, RED,
        locked_teams={"9452"}, locked_robots={0},
    )
    assert 0 not in out                        # locked robot excluded
    assert out.get(1) == "2485"                # 9452 locked → robot 1 takes 2485


# --- eliminate_last_team -----------------------------------------------------

def test_elimination_fires_when_one_team_and_one_robot_remain():
    """Two of three red teams already assigned; one red-bumper robot still unlabeled."""
    alliance = _alliance_of({0: "red", 1: "red", 2: "red"})
    team_of = {0: "1538", 1: "294", 2: None}
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {2: "4322"}


def test_elimination_does_not_fire_with_two_unread_robots():
    alliance = _alliance_of({0: "red", 1: "red", 2: "red"})
    team_of = {0: "1538", 1: None, 2: None}  # two unread on red
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {}


def test_elimination_does_not_cross_alliance():
    """A blue-bumper robot must never be eliminated into a red team."""
    alliance = _alliance_of({0: "red", 1: "red", 2: "blue"})
    team_of = {0: "1538", 1: "294", 2: None}  # robot 2 is blue, last red team open
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {}


# --- audit_overturn ----------------------------------------------------------

def test_audit_overturns_when_other_team_dominates():
    out = audit_overturn(current_team="9452",
                         votes={"9452": 3.0, "2485": 7.0},
                         margin=3.0)
    assert out == "2485"


def test_audit_sticky_below_margin():
    out = audit_overturn(current_team="9452",
                         votes={"9452": 3.0, "2485": 5.0},  # only +2 over current
                         margin=3.0)
    assert out is None


def test_audit_no_other_candidates():
    out = audit_overturn("9452", {"9452": 5.0}, margin=3.0)
    assert out is None
