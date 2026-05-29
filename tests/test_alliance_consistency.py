"""Regression guard for alliance consistency.

SP2 change: alliance is owned by the field-tracker SLOT (bumper color), set every
frame by the loop sync. Team assignment is masked to the slot's alliance and must
NEVER flip it — bumper color is the physical ground truth, so a contradictory team
read is the suspect, not the alliance. (This supersedes the old "alliance follows the
team roster" rule, whose flip-flop root cause — embedding clustering — is removed.)
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from web.app import InferenceJob, RobotTrack
from web.diagnostic import NullRecorder


def _make_job():
    """A lightweight stand-in exposing only the attributes
    `assign_teams_from_votes` actually reads. Lets us call the method as an
    unbound function (`InferenceJob.assign_teams_from_votes(job)`) without
    paying the cost of `InferenceJob.__init__` (which loads YOLO weights)."""
    return SimpleNamespace(
        match_blue_teams=["2102", "9452", "2485"],
        match_red_teams=["1538", "294", "4322"],
        match_teams=["2102", "9452", "2485", "1538", "294", "4322"],
        robot_tracks={},
        scoring_events=[],
        _current_frame=0,
        diagnostic=NullRecorder(),
    )


def test_team_assignment_respects_slot_alliance_and_never_flips_it():
    """A blue-bumper slot is assigned a BLUE team, and assign_teams_from_votes never
    changes the slot's alliance."""
    job = _make_job()
    job.robot_tracks = {
        0: RobotTrack(robot_id=0, alliance="blue", team_number=None,
                      last_position=(0, 0), positions_history=deque(maxlen=30)),
    }
    job.robot_tracks[0].team_ocr_votes = {"2102": 5.0}   # 2102 is BLUE

    InferenceJob.assign_teams_from_votes(job)

    r = job.robot_tracks[0]
    assert r.team_number == "2102"
    assert r.alliance == "blue"   # slot alliance must not change


def test_cross_alliance_read_is_rejected_not_flipped():
    """A confident RED-team read on a blue slot must NOT be assigned to it and must NOT
    flip the slot to red — bumper color wins, the contradictory read is the suspect."""
    job = _make_job()
    job.robot_tracks = {
        0: RobotTrack(robot_id=0, alliance="blue", team_number=None,
                      last_position=(0, 0), positions_history=deque(maxlen=30)),
    }
    job.robot_tracks[0].team_ocr_votes = {"4322": 5.0}   # 4322 is RED

    InferenceJob.assign_teams_from_votes(job)

    r = job.robot_tracks[0]
    assert r.alliance == "blue"        # unchanged
    assert r.team_number != "4322"     # red team not placed on a blue slot


def test_no_embedding_alliance_calls_in_loop():
    """Structural guard: the loop must not call _assign_alliances_by_embedding
    or assign_identities_now (Component D removal). They caused the flip-flop."""
    src = Path("web/app.py").read_text()
    # Function bodies may exist, but no `self.<func>()` calls should remain in the loop.
    assert "self._assign_alliances_by_embedding()" not in src, (
        "_assign_alliances_by_embedding() must not be called from the loop."
    )
    assert "self.assign_identities_now()" not in src, (
        "assign_identities_now() must not be called from the loop."
    )
