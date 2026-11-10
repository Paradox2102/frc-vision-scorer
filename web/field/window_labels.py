"""Window labels: relabel a finished replay from reads on short track windows.

Spec: docs/superpowers/specs/2026-11-09-jev-window-labels-design.md.

A slot's track is cut into windows of at most MAX_WINDOW_S (split at gaps and
jumps); a reader names each window from a few native-resolution crops; the
window's team goes to every frame of that slot in the window. Unread windows
stay unlabelled and a team sits on at most one slot per frame.

Everything above ``run_window_labels`` is pure (no I/O)."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from web.field.track_evidence import slot_label_to_id

MAX_WINDOW_S = 4.0
GAP_FRAME_PERIODS = 2.5
JUMP_SPEED_M_S = 7.5
SAMPLES_PER_WINDOW = 4
DIGIT_MIN_P = 0.5
DIGIT_CONFLICT_P = 0.4
DIGIT_MIN_TOP_CALLS = 2
ANCHOR_MIN_P = 0.85
ANCHORS_PER_TEAM = 6
MIN_ANCHORS = 2
MIN_LINEUP_ROWS = 2
MAX_ERROR_FRACTION = 0.10
STAGE_TIMEOUT_S = 360.0
REQUEST_BATCH = 200
RANK_DIGITS, RANK_LINEUP = 2, 1
SOURCE = {RANK_DIGITS: "window_digits", RANK_LINEUP: "window_lineup"}
SOURCE_NONE = "window_none"
LETTERS = "ABCDEF"


@dataclass
class Window:
    wid: int
    slot: str                     # replay slot label, "B1".."R3"
    run: int                      # continuous-run id within the slot
    idx: list                     # indices into the replay frames list
    samples: list = field(default_factory=list)   # <= SAMPLES_PER_WINDOW of idx


def frame_period_s(frames: list, fps: float) -> Optional[float]:
    """Seconds between processed frames: median frame_num step / source fps."""
    if not fps or fps <= 0:
        return None
    nums = [f["frame_num"] for f in frames if f.get("frame_num") is not None]
    steps = sorted(b - a for a, b in zip(nums, nums[1:]) if b > a)
    return steps[len(steps) // 2] / float(fps) if steps else None


def _even_picks(n: int, m: int) -> list:
    """m positions spread evenly over range(n), first and last included."""
    if n <= 0:
        return []
    if m <= 1 or n == 1:
        return [0]
    m = min(m, n)
    return sorted({round(j * (n - 1) / (m - 1)) for j in range(m)})


def _split(seq: list, n: int) -> list:
    k, extra = divmod(len(seq), n)
    out, start = [], 0
    for j in range(n):
        end = start + k + (1 if j < extra else 0)
        out.append(seq[start:end])
        start = end
    return out


def build_windows(frames: list, fps: float) -> list:
    """Cut every slot's track into windows. ``frames``: replay frames
    ({frame_num, robots: [{slot, field_pos, ...}]}), in order."""
    period = frame_period_s(frames, fps)
    if period is None:
        return []
    tracks: dict = {}
    for i, f in enumerate(frames):
        if f.get("frame_num") is None:
            continue
        for r in f.get("robots") or []:
            pos = r.get("field_pos")
            if (not r.get("slot") or not pos or len(pos) < 2
                    or pos[0] is None or pos[1] is None):
                continue
            tracks.setdefault(r["slot"], []).append(
                (i, f["frame_num"] / float(fps), float(pos[0]), float(pos[1])))
    out: list = []
    for slot in sorted(tracks):
        seq = tracks[slot]
        runs = [[seq[0]]]
        for a, b in zip(seq, seq[1:]):
            dt = b[1] - a[1]
            dist = math.hypot(b[2] - a[2], b[3] - a[3])
            if dt > GAP_FRAME_PERIODS * period or (dt > 0 and dist / dt > JUMP_SPEED_M_S):
                runs.append([b])
            else:
                runs[-1].append(b)
        for run_id, run in enumerate(runs):
            if len(run) < 2:
                continue                      # an isolated sample stays unlabelled
            # a run of N contiguous frames lasts N periods
            dur = run[-1][1] - run[0][1] + period
            n = max(1, math.ceil(dur / MAX_WINDOW_S - 1e-6))
            n = max(1, min(n, len(run) // 2))  # every window keeps >= 2 samples
            for part in _split(run, n):
                idx = [p[0] for p in part]
                out.append(Window(len(out), slot, run_id, idx,
                                  [idx[j] for j in _even_picks(len(idx), SAMPLES_PER_WINDOW)]))
    return out


def digit_decision(calls: list, roster: list) -> Optional[tuple]:
    """``calls``: one probability dict per request sent for a window (None or
    {} for a failed request). Returns (team, p) when the best team reaches
    DIGIT_MIN_P, no other team reaches DIGIT_CONFLICT_P and the team is the
    top option in at least DIGIT_MIN_TOP_CALLS calls; else None."""
    roster = [str(t) for t in roster]
    best = {t: 0.0 for t in roster}
    tops = {t: 0 for t in roster}
    for p in calls:
        if not p:
            continue
        for t in roster:
            best[t] = max(best[t], float(p.get(t, 0.0)))
        top = max(p, key=p.get)
        if top in tops:
            tops[top] += 1
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return None
    team, p = ranked[0]
    if p < DIGIT_MIN_P:
        return None
    if len(ranked) > 1 and ranked[1][1] >= DIGIT_CONFLICT_P:
        return None
    if tops[team] < DIGIT_MIN_TOP_CALLS:
        return None
    return team, p


def pick_anchors(labels: dict, t0: dict) -> dict:
    """``labels``: window id -> (team, p) digit labels; ``t0``: window id ->
    start seconds. Returns team -> up to ANCHORS_PER_TEAM window ids spread
    evenly in time, for teams with at least MIN_ANCHORS windows at
    ANCHOR_MIN_P or above."""
    by: dict = {}
    for wid, (team, p) in labels.items():
        if p >= ANCHOR_MIN_P:
            by.setdefault(team, []).append((t0[wid], wid))
    out = {}
    for team, cands in by.items():
        if len(cands) < MIN_ANCHORS:
            continue
        cands.sort()
        out[team] = [cands[j][1] for j in _even_picks(len(cands), ANCHORS_PER_TEAM)]
    return out


def lineup_decision(probs: Optional[dict], letters: dict, min_p: float) -> Optional[tuple]:
    """``letters``: row letter -> team. Returns (team, p) when the most likely
    letter reaches ``min_p``; else None."""
    if not probs or not letters:
        return None
    letter = max(letters, key=lambda l: float(probs.get(l, 0.0)))
    p = float(probs.get(letter, 0.0))
    return (letters[letter], p) if p >= min_p else None


def assemble_labels(windows: list, evidence: dict) -> tuple:
    """``evidence``: window id -> (team, rank, p). Returns (lab, cleared) where
    ``lab[(frame index, slot)]`` is (team, rank, p, window length) or None for
    an entry cleared by the one-team-per-frame rule. Keys that are absent are
    unlabelled. Precedence when two slots carry one team in a frame: higher
    rank, higher p, longer window, lower slot id."""
    lab: dict = {}
    for w in windows:
        ev = evidence.get(w.wid)
        if not ev:
            continue
        for i in w.idx:
            lab[(i, w.slot)] = (ev[0], ev[1], float(ev[2]), len(w.idx))
    by: dict = {}
    for (i, slot), e in lab.items():
        by.setdefault((i, e[0]), []).append(slot)
    cleared = 0
    for (i, _team), slots in by.items():
        if len(slots) < 2:
            continue

        def strength(s, i=i):
            e = lab[(i, s)]
            sid = slot_label_to_id(s)
            return (-e[1], -e[2], -e[3], 99 if sid is None else sid)

        keep = min(slots, key=strength)
        for s in slots:
            if s != keep:
                lab[(i, s)] = None
                cleared += 1
    return lab, cleared


def apply_labels(frames: list, lab: dict) -> None:
    """Write the assembled labels into replay frames, in place. Every robot
    row is rewritten: a labelled entry gets its team, everything else None."""
    for i, f in enumerate(frames):
        for r in f.get("robots") or []:
            e = lab.get((i, r.get("slot")))
            r["team_number"] = e[0] if e else None
            r["confirmed"] = bool(e and e[1] == RANK_DIGITS)


def _extend_rows(rows: list, start: int, end: int, e) -> None:
    """Append frames [start, end] to the segment rows: extend the last row
    when team and source match, else start a new one. ``e`` is a label
    entry (team, rank, p, ...) or None for unlabelled."""
    team = e[0] if e else None
    source = SOURCE[e[1]] if e else SOURCE_NONE
    score = round(e[2], 4) if e else None
    if rows and rows[-1]["team_number"] == team and rows[-1]["source"] == source:
        rows[-1]["end_frame"] = end
        if score is not None:
            rows[-1]["score"] = max(rows[-1]["score"], score)
    else:
        rows.append({"start_frame": start, "end_frame": end, "team_number": team,
                     "confirmed": bool(e and e[1] == RANK_DIGITS),
                     "score": score, "source": source})


def window_segments(frames: list, lab: dict) -> dict:
    """Slot id -> contiguous segment rows for the match_robot_identity_segment
    table. Readers of that table look a frame up by "last start at or before
    it", so every unlabelled stretch gets its own row (team None, source
    SOURCE_NONE): unnamed windows, frames cleared by the one-team-per-frame
    rule, and frames where the slot is absent between two of its runs."""
    nums = [f.get("frame_num") for f in frames]
    per: dict = {}
    for i, f in enumerate(frames):
        if nums[i] is None:
            continue
        for r in f.get("robots") or []:
            sid = slot_label_to_id(r.get("slot"))
            if sid is None:
                continue
            per.setdefault(sid, []).append((i, int(nums[i]), lab.get((i, r["slot"]))))
    out = {}
    for sid, seq in sorted(per.items()):
        rows: list = []
        prev_i = None
        for i, frame_num, e in seq:
            if prev_i is not None and i > prev_i + 1:
                absent = [nums[j] for j in range(prev_i + 1, i) if nums[j] is not None]
                if absent:
                    _extend_rows(rows, int(absent[0]), int(absent[-1]), None)
            _extend_rows(rows, frame_num, frame_num, e)
            prev_i = i
        out[sid] = rows
    return out


def restamp_events(frames: list, events: list) -> int:
    """Re-stamp each replay event's team from the frame labels: the team on
    that slot's row in ``frames[event index]``, or None when the row is
    absent or unlabelled. An event's ``frame_num`` is the processed-frame
    index. An index outside the recording is left alone. Returns the number
    of events whose team changed."""
    changed = 0
    for ev in events or []:
        idx = ev.get("frame_num")
        if not isinstance(idx, int) or not 0 <= idx < len(frames):
            continue
        row = next((r for r in frames[idx].get("robots") or []
                    if r.get("slot") == ev.get("slot")), None)
        team = row.get("team_number") if row else None
        if ev.get("team_number") != team:
            changed += 1
        ev["team_number"] = team
        ev["robot_label"] = team or ev.get("slot")
    return changed
