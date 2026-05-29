"""Pure re-identification decision logic (no app state, no model, no I/O).

Two functions:
- choose_canonical_match: continuity — which canonical track is a new detection?
- assign_identities: identity — one-to-one matching of tracks to references.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from web.embedder import cosine_similarity


@dataclass
class TrackView:
    """A read-only view of a canonical track for the continuity decision."""
    robot_id: int
    alliance: str
    position: tuple
    embedding: Optional[np.ndarray]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _dist(p, q) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _merge_score(det_pos, det_embed, track, pos_gate, w_pos, w_app, app_floor):
    """Combined position+appearance score for merging det into a track.

    Returns None when the track is out of the position gate or vetoed by the
    appearance floor; otherwise w_pos*pos_score + w_app*app_sim.
    """
    d = _dist(det_pos, track.position)
    if d > pos_gate:
        return None
    pos_score = max(0.0, 1.0 - d / pos_gate)
    # Neutral similarity (0.0) when a track has no embedding yet.
    app_sim = _clamp01(cosine_similarity(det_embed, track.embedding)) if track.embedding is not None else 0.0
    # Appearance veto: never absorb a visually different detection.
    if track.embedding is not None and det_embed is not None and app_sim < app_floor:
        return None
    return w_pos * pos_score + w_app * app_sim


def choose_canonical_match(
    det_pos, det_embed, det_alliance, tracks,
    app_floor: float, pos_gate: float, w_pos: float, w_app: float,
    cap_per_alliance: int, cap_total: int, claimed=None,
):
    """Decide what a new detection is. Returns one of:
        ("merge", robot_id)   merge into an existing track (within gate + floor)
        ("create", None)      genuinely new robot (caps permitting)
        ("assign", robot_id)  caps full — must be an existing robot; best embedding
        ("drop", None)        caps full AND every track already claimed this frame

    Same-alliance tracks are preferred for merge; the cap-full fallback considers
    all tracks (alliance/color can be misread). `claimed` is the set of canonical
    robot_ids already bound to another detection THIS frame — they cannot be
    reused, which keeps each physical robot on exactly one track per frame.
    """
    claimed = claimed or set()
    avail = [t for t in tracks if t.robot_id not in claimed]
    same = [t for t in avail if t.alliance == det_alliance]
    other = [t for t in avail if t.alliance != det_alliance]

    def best_merge(candidates):
        best, best_score = None, -1.0
        for t in candidates:
            s = _merge_score(det_pos, det_embed, t, pos_gate, w_pos, w_app, app_floor)
            if s is not None and s > best_score:
                best, best_score = t, s
        return best

    # Primary path: merge by combined position + appearance score (with veto).
    # This uses BOTH signals even when the alliance is full — position is a
    # valuable tiebreaker among visually similar robots.
    merged = best_merge(same) or best_merge(other)
    if merged is not None:
        return ("merge", merged.robot_id)

    # Caps count ALL tracks (claimed or not) — they reflect physical robots.
    alliance_count = sum(1 for t in tracks if t.alliance == det_alliance)
    if alliance_count < cap_per_alliance and len(tracks) < cap_total:
        return ("create", None)

    if avail:  # caps full — assign to the best UNCLAIMED match (embedding first)
        if det_embed is not None and any(t.embedding is not None for t in avail):
            best = max(avail, key=lambda t: cosine_similarity(det_embed, t.embedding))
        else:
            best = min(avail, key=lambda t: _dist(det_pos, t.position))
        return ("assign", best.robot_id)

    if tracks:  # caps full AND every track already claimed this frame
        return ("drop", None)

    return ("create", None)  # absolute first-detection fallback


def best_candidate_score(det_pos, det_embed, tracks, pos_gate, w_pos, w_app, app_floor) -> float:
    """Best merge score of this detection against any track (claimed-agnostic).

    Used to order detections best-match-first so the strongest claimant binds a
    contested canonical track before weaker ones. Returns -1.0 if no track
    qualifies (out of gate or vetoed).
    """
    best = -1.0
    for t in tracks:
        s = _merge_score(det_pos, det_embed, t, pos_gate, w_pos, w_app, app_floor)
        if s is not None and s > best:
            best = s
    return best


FORBIDDEN = 1e6  # effectively-infinite cost for alliance-forbidden pairs


def assign_identities(
    track_embeds: dict, ref_embeds: dict,
    alliance_of_track: dict, alliance_of_team: dict,
    locked_tracks: set, locked_teams: set, max_cost: float,
) -> dict:
    """One-to-one assignment of tracks to reference teams (Hungarian).

    Args:
        track_embeds: {robot_id: embedding}
        ref_embeds:   {team_number: embedding}
        alliance_of_track: {robot_id: "blue"|"red"|None}
        alliance_of_team:  {team_number: "blue"|"red"}
        locked_tracks: robot_ids already confirmed (excluded from the problem)
        locked_teams:  team_numbers already taken (excluded)
        max_cost: accept a match only if (1 - cosine) <= max_cost

    Returns {robot_id: team_number or None} for every UNLOCKED track with an
    embedding. Uniqueness is guaranteed by construction.
    """
    rows = [rid for rid, e in track_embeds.items()
            if rid not in locked_tracks and e is not None]
    cols = [team for team in ref_embeds
            if team not in locked_teams and ref_embeds[team] is not None]
    result = {rid: None for rid in rows}
    if not rows or not cols:
        return result

    cost = np.full((len(rows), len(cols)), FORBIDDEN, dtype=np.float64)
    for i, rid in enumerate(rows):
        ta = alliance_of_track.get(rid)
        for j, team in enumerate(cols):
            if ta is not None and ta != alliance_of_team.get(team):
                continue  # alliance hard constraint
            cost[i, j] = 1.0 - cosine_similarity(track_embeds[rid], ref_embeds[team])

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] <= max_cost:  # rejects FORBIDDEN and weak matches
            result[rows[i]] = cols[j]
    return result


def assign_teams_alliance_masked(
    votes: dict, alliance_of_robot: dict,
    blue_teams: list, red_teams: list,
    locked_teams: set, locked_robots: set,
) -> dict:
    """Alliance-masked one-to-one assignment of robots to teams by accumulated
    VLM read votes. A single Hungarian solve over the whole field with
    cross-alliance (robot, team) cells forbidden — unknown-alliance robots are
    open to all teams. Returns {robot_id: team_or_None} for every unlocked robot.
    Robots with no positive votes for any allowed team get None.
    """
    rows = [rid for rid in votes if rid not in locked_robots]
    cols = [t for t in (blue_teams + red_teams) if t not in locked_teams]
    result = {rid: None for rid in rows}
    if not rows or not cols:
        return result

    cost = np.full((len(rows), len(cols)), FORBIDDEN, dtype=np.float64)
    for i, rid in enumerate(rows):
        ra = alliance_of_robot.get(rid, "unknown")
        rvotes = votes.get(rid) or {}
        for j, team in enumerate(cols):
            ta = "red" if team in red_teams else "blue"
            if ra != "unknown" and ra != ta:
                continue  # cross-alliance forbidden
            cost[i, j] = -float(rvotes.get(team, 0.0))

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        # Skip cells with no positive evidence (cost == 0) and forbidden cells.
        if cost[i, j] >= 0:
            continue
        result[rows[i]] = cols[j]
    return result


def eliminate_last_team(
    alliance_of_robot: dict, team_of_robot: dict,
    blue_teams: list, red_teams: list,
) -> dict:
    """Process-of-elimination: for each alliance, if exactly one roster team is
    unassigned AND exactly one robot has that alliance's color with no team,
    assign that robot that team. Returns {robot_id: team} for newly-eliminated
    pairs (empty dict if no alliance qualifies).
    """
    result: dict = {}
    for alliance, roster in (("red", red_teams), ("blue", blue_teams)):
        assigned = {team_of_robot.get(rid) for rid in team_of_robot}
        unassigned = [t for t in roster if t not in assigned]
        unidentified = [
            rid for rid, a in alliance_of_robot.items()
            if a == alliance and team_of_robot.get(rid) is None
        ]
        if len(unassigned) == 1 and len(unidentified) == 1:
            result[unidentified[0]] = unassigned[0]
    return result


def audit_overturn(current_team: str, votes: dict, margin: float):
    """Return a different team if its accumulated votes exceed `current_team`'s
    by `margin`; else None. Sticky-by-default so a confirmed label flips only
    when a competitor genuinely dominates.
    """
    current = float((votes or {}).get(current_team, 0.0))
    best_t, best_v = None, -1.0
    for t, v in (votes or {}).items():
        if t == current_team:
            continue
        if v > best_v:
            best_t, best_v = t, float(v)
    if best_t is not None and best_v > current + margin:
        return best_t
    return None
