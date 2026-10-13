"""Live-only team assignment (always on).

qm11 full match (job 80a1cd4c): robot_tracks is never pruned (575 entries by
the end), and assign_teams_from_votes ran its 1:1 Hungarian over all of them.
All 395 assignment rows named only 8 identities, none alive at the end — e.g.
live identity 337 with {4256: 1.0} lost 4256 to dead identity 28. "Live" is
membership in the IdentityRegistry: eviction (cap enforcement) pops the
identity, its RobotTrack stays behind as history.
"""
from collections import deque
from types import SimpleNamespace

from web.app import InferenceJob, RobotTrack
from web.diagnostic import NullRecorder
from web.field.identity import IdentityRegistry, RobotIdentity, live_robot_tracks

BLUE = ["4786", "3181", "2992"]
RED = ["6800", "1477", "4256"]


def _track(rid, alliance, votes=None, team=None, confirmed=False):
    t = RobotTrack(robot_id=rid, alliance=alliance, team_number=team,
                   last_position=(0, 0), positions_history=deque(maxlen=30))
    t.team_ocr_votes = dict(votes or {})
    t.team_number_confirmed = confirmed
    return t


def _job(live_ids, tracks):
    reg = IdentityRegistry()
    for rid in live_ids:
        reg.identities[rid] = RobotIdentity(identity_id=rid, alliance=tracks[rid].alliance)
    job = SimpleNamespace(
        match_blue_teams=list(BLUE), match_red_teams=list(RED),
        match_teams=BLUE + RED, robot_tracks=tracks, scoring_events=[],
        _current_frame=0, diagnostic=NullRecorder(), identity_registry=reg,
    )
    job._clear_team_from_other_robots = (
        InferenceJob._clear_team_from_other_robots.__get__(job))
    return job


def test_live_robot_tracks_filters_by_registry_membership():
    tracks = {1: _track(1, "red"), 2: _track(2, "red")}
    reg = IdentityRegistry()
    reg.identities[2] = RobotIdentity(identity_id=2, alliance="red")
    assert set(live_robot_tracks(tracks, reg)) == {2}
    # No registry (legacy stand-ins): everything counts as live.
    assert set(live_robot_tracks(tracks, None)) == {1, 2}


def test_dead_identity_cannot_steal_team_from_live_one():
    """Regression (80a1cd4c R1): dead 28 out-votes live 337 for 4256."""
    tracks = {
        28: _track(28, "red", {"4256": 3.0}),      # evicted long ago
        337: _track(337, "red", {"4256": 1.0}),    # live, same team
    }
    job = _job([337], tracks)
    InferenceJob.assign_teams_from_votes(job)
    assert tracks[337].team_number == "4256"
    assert tracks[28].team_number is None          # dead track never assigned


def test_dead_confirmed_team_does_not_lock_out_live_identity():
    tracks = {
        28: _track(28, "red", {"4256": 3.0}, team="4256", confirmed=True),
        337: _track(337, "red", {"4256": 1.0}),
    }
    job = _job([337], tracks)
    InferenceJob.assign_teams_from_votes(job)
    assert tracks[337].team_number == "4256"


def test_elimination_counts_only_live_identities():
    """Process-of-elimination needs exactly one unidentified robot on the
    alliance; dead tracks no longer count as unidentified robots."""
    tracks = {
        0: _track(0, "blue", {"4786": 2.0}),
        1: _track(1, "blue", {"3181": 2.0}),
        2: _track(2, "blue"),                     # live, unread
        9: _track(9, "blue"),                     # dead, unread
    }
    job = _job([0, 1, 2], tracks)
    InferenceJob.assign_teams_from_votes(job)
    assert tracks[2].team_number == "2992" and tracks[2].team_number_confirmed
    assert tracks[9].team_number is None


def test_hud_claim_arbitration_ignores_dead_tracks():
    tracks = {
        28: _track(28, "red", {"4256": 3.0}),
        337: _track(337, "red", {"4256": 1.0}),
    }
    job = _job([337], tracks)
    label, confirmed = InferenceJob._team_hud_label(job, tracks[337], "R1")
    assert label.startswith("4256?") and not confirmed


def test_retire_evicted_tracks_drops_team_keeps_votes():
    """A dead track must not keep a (confirmed) team: the read roster
    (_remaining_rosters), the consecutive-lock 'already assigned' check and
    the Hungarian's locked teams all treat confirmed teams as taken, and a
    frozen tentative team would duplicate the live holder's label in the
    all-tracks lists (snapshot 'identified', replay team lists)."""
    tracks = {
        28: _track(28, "red", {"4256": 3.0}, team="4256", confirmed=True),
        30: _track(30, "red", {"1477": 1.0}, team="1477"),
        31: _track(31, "red", {"6800": 1.0}, team="6800"),   # still live
    }
    job = SimpleNamespace(robot_tracks=tracks)
    InferenceJob._retire_evicted_tracks(job, [28, 30, 999])
    assert tracks[28].team_number is None and not tracks[28].team_number_confirmed
    assert tracks[30].team_number is None
    assert tracks[28].team_ocr_votes == {"4256": 3.0}     # evidence kept
    assert tracks[31].team_number == "6800"               # untouched
