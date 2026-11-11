"""Window labels: relabel a finished replay from reads on short track windows.

Spec: docs/superpowers/specs/2026-11-09-jev-window-labels-design.md.

A slot's track is cut into windows of at most MAX_WINDOW_S (split at gaps and
jumps); a reader names each window from a few native-resolution crops; the
window's team goes to every frame of that slot in the window. Unread windows
stay unlabelled and a team sits on at most one slot per frame.

Everything above ``run_window_labels`` is pure (no I/O). Event-store access happens in the callers (web/app.py, scripts/relabel_replay.py)."""
from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from web.field.track_evidence import slot_label_to_id
from web.field.event_anchors import KIND_DIGITS, KIND_ELIM, Anchor

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
MIN_ERRORS_TO_ABORT_EARLY = 20
STAGE_TIMEOUT_S = 360.0
REQUEST_BATCH = 32
RANK_DIGITS, RANK_LINEUP = 2, 1
SOURCE = {RANK_DIGITS: "window_digits", RANK_LINEUP: "window_lineup"}
SOURCE_NONE = "window_none"
LETTERS = "ABCDEF"
BORROW_PER_MATCH = 2              # borrowed anchors from any one source match, per row

# Elimination anchors (spec section 4)
ELIM_OVERLAP_S = 1.0              # each partner window overlaps W3 by at least this
ELIM_MAX_TEAM_P = 0.4             # no roster team at or above this in any W3 call
ELIM_MIN_SEPARATION_M = 1.5       # W3's median distance to each partner
ELIM_MIN_WIDE_CALLS = 3           # returned wide calls W3 needs
ELIM_MAX_NOT_ROBOT_P = 0.3        # every returned wide call below this p(not_robot)
ELIM_TILES_PER_WINDOW = 3
ELIM_PER_TEAM = 3                 # per team per match


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


def store_anchors(labels: dict, t0: dict) -> dict:
    """Like pick_anchors without the MIN_ANCHORS floor: team -> up to
    ANCHORS_PER_TEAM window ids at ANCHOR_MIN_P or above, spread evenly in
    time. A team's single good read is kept: it goes to the event store and,
    with event_lineup on, counts toward its own row."""
    by: dict = {}
    for wid, (team, p) in labels.items():
        if p >= ANCHOR_MIN_P:
            by.setdefault(team, []).append((t0[wid], wid))
    out = {}
    for team, cands in by.items():
        cands.sort()
        out[team] = [cands[j][1] for j in _even_picks(len(cands), ANCHORS_PER_TEAM)]
    return out


def best_tile_k(wide_p: dict, tiles: dict, wid: int, team: str) -> int:
    """The sample whose wide call gave ``team`` the highest p (the first of
    a tie); the lowest sample number when no wide call returned."""
    per_k = wide_p.get(wid) or {}
    return max(per_k, key=lambda j: per_k[j].get(team, 0.0)) if per_k else min(tiles[wid])


def lineup_decision(probs: Optional[dict], letters: dict, min_p: float) -> Optional[tuple]:
    """``letters``: row letter -> team. Returns (team, p) when the most likely
    letter reaches ``min_p``; else None."""
    if not probs or not letters:
        return None
    letter = max(letters, key=lambda l: float(probs.get(l, 0.0)))
    p = float(probs.get(letter, 0.0))
    return (letters[letter], p) if p >= min_p else None


def borrow_key(a: Anchor, colour: Optional[str]) -> tuple:
    """Rank of a borrowed anchor for a row (spec section 3): digits before
    elimination, then this match's bumper colour, then p descending, then
    the most recent match, then the earliest frame; match key and store id last, so the
    order never depends on the input order."""
    return (a.kind != KIND_DIGITS, a.alliance_color != colour, -float(a.p),
            -int(a.match_number or 0), int(a.frame_num or 0), str(a.match_key or ""),
            int(a.id or 0))


def fill_row(n_own: int, borrowed: list, colour: Optional[str],
             use_elimination: bool = True) -> list:
    """Borrowed anchors for one team's lineup row, in card order. ``n_own``:
    the team's own digit anchors in this match. Fills toward ANCHORS_PER_TEAM
    by borrow_key, at most BORROW_PER_MATCH from any one source match (both
    kinds together). Elimination anchors are used only when own plus
    borrowed digit anchors are fewer than MIN_ANCHORS."""
    need = ANCHORS_PER_TEAM - n_own
    if need <= 0:
        return []
    ranked = sorted(borrowed, key=lambda a: borrow_key(a, colour))
    picked: list = []
    per_match: dict = {}

    def take(kind):
        for a in ranked:
            if len(picked) >= need:
                return
            if a.kind != kind or per_match.get(a.match_key, 0) >= BORROW_PER_MATCH:
                continue
            picked.append(a)
            per_match[a.match_key] = per_match.get(a.match_key, 0) + 1

    take(KIND_DIGITS)
    if use_elimination and n_own + len(picked) < MIN_ANCHORS:
        take(KIND_ELIM)
    return picked


def slot_alliance(slot) -> Optional[str]:
    """"red" for an R slot, "blue" for a B slot, else None."""
    s = str(slot or "")[:1]
    return "red" if s == "R" else "blue" if s == "B" else None


def _slot_xy(frames: list, i: int, slot: str) -> Optional[tuple]:
    for r in frames[i].get("robots") or []:
        pos = r.get("field_pos")
        if (r.get("slot") == slot and pos and len(pos) >= 2
                and pos[0] is not None and pos[1] is not None):
            return float(pos[0]), float(pos[1])
    return None


def _median_gap_m(frames: list, shared, slot_a: str, slot_b: str) -> Optional[float]:
    """Median distance between two slots' tracked positions over the
    ``shared`` frame indices (None when no frame has both)."""
    d = []
    for i in shared:
        a, b = _slot_xy(frames, i, slot_a), _slot_xy(frames, i, slot_b)
        if a and b:
            d.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    return statistics.median(d) if d else None


def elimination_anchors(windows: list, frames: list, digit: dict, calls: dict, wide_p: dict,
                        red: list, blue: list, period: float) -> dict:
    """Spec section 4. A window W3 on alliance X, with no digit label, shows
    the third X team T3 when every guard holds:
      1. windows W1, W2 on X's other two slots each overlap W3 by at least
         ELIM_OVERLAP_S and carry digit labels T1 != T2 of X at ANCHOR_MIN_P;
      2. no roster team reaches ELIM_MAX_TEAM_P in any of W3's returned
         calls, and X's teams' summed p is at least the other alliance's;
      3. no other slot carries a digit label T3 at any frame of W3;
      4. W3's median distance to W1 and to W2 over their shared frames is at
         least ELIM_MIN_SEPARATION_M;
      5. W3 has at least ELIM_MIN_WIDE_CALLS returned wide calls, each with
         p(not_robot) below ELIM_MAX_NOT_ROBOT_P.
    W3 is never labelled. Returns team -> [(wid, k, p)]: up to
    ELIM_TILES_PER_WINDOW samples per window, at most ELIM_PER_TEAM per team
    spread in time; p is the smaller of the two partner labels' p."""
    teams = {"red": [str(t) for t in red], "blue": [str(t) for t in blue]}
    roster = teams["red"] + teams["blue"]
    idx = {w.wid: set(w.idx) for w in windows}
    by_slot: dict = {}
    for w in windows:
        by_slot.setdefault(w.slot, []).append(w)
    found: dict = {}
    for w3 in windows:
        x = slot_alliance(w3.slot)
        if x is None or w3.wid in digit or len(teams[x]) != 3:
            continue
        wides = wide_p.get(w3.wid) or {}                                      # guard 5
        if (len(wides) < ELIM_MIN_WIDE_CALLS
                or any(float(p.get("not_robot", 0.0)) >= ELIM_MAX_NOT_ROBOT_P
                       for p in wides.values())):
            continue
        returned = [p for p in calls.get(w3.wid) or [] if p]                    # guard 2
        if any(float(p.get(t, 0.0)) >= ELIM_MAX_TEAM_P for p in returned for t in roster):
            continue
        other = "blue" if x == "red" else "red"
        mass_x = sum(float(p.get(t, 0.0)) for p in returned for t in teams[x])
        mass_o = sum(float(p.get(t, 0.0)) for p in returned for t in teams[other])
        if mass_x < mass_o:
            continue
        partners = []                                                           # guard 1
        for s in sorted({f"{w3.slot[0]}{n}" for n in (1, 2, 3)} - {w3.slot}):
            best = None
            for w in by_slot.get(s, ()):
                d = digit.get(w.wid)
                if not d or d[1] < ANCHOR_MIN_P or d[0] not in teams[x]:
                    continue
                shared = idx[w3.wid] & idx[w.wid]
                if len(shared) * period < ELIM_OVERLAP_S - 1e-9:
                    continue
                if best is None or (len(shared), -w.wid) > (len(best[1]), -best[0].wid):
                    best = (w, shared)
            if best is None:
                break
            partners.append(best)
        if len(partners) != 2:
            continue
        (w1, sh1), (w2, sh2) = partners
        t1, t2 = digit[w1.wid][0], digit[w2.wid][0]
        rest = [t for t in teams[x] if t not in (t1, t2)]
        if t1 == t2 or len(rest) != 1:
            continue
        t3 = rest[0]
        if any(w.slot != w3.slot and digit.get(w.wid, (None,))[0] == t3          # guard 3
               and idx[w.wid] & idx[w3.wid] for w in windows):
            continue
        gaps = (_median_gap_m(frames, sh1, w3.slot, w1.slot),                   # guard 4
                _median_gap_m(frames, sh2, w3.slot, w2.slot))
        if any(g is None or g < ELIM_MIN_SEPARATION_M for g in gaps):
            continue
        p = min(digit[w1.wid][1], digit[w2.wid][1])
        ks = sorted(wides)
        for j in _even_picks(len(ks), ELIM_TILES_PER_WINDOW):
            found.setdefault(t3, []).append((w3.samples[ks[j] - 1], w3.wid, ks[j], p))
    out = {}
    for team, cands in found.items():
        cands.sort()
        out[team] = [cands[j][1:] for j in _even_picks(len(cands), ELIM_PER_TEAM)]
    return out


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


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class Params:
    fps: float                                 # source video fps
    red: list
    blue: list
    view: tuple = (0, 0, 1920, 723)            # main camera's crop of the composite
    ground_center_offset_m: float = 0.0        # the job's knob (positions already pushed)
    lineup_min_p: float = 0.8                  # > 1 disables the lineup pass
    deadline: Optional[float] = None           # time.monotonic() value; None = no limit
    seed: int = 0                              # lineup letter order
    model_key: str = "jev-1.14.0"
    ball_note: Optional[str] = None            # None = jev_reader.BALL_NOTE_2026
    # Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md).
    # Every default keeps the stage exactly as it was.
    event_lineup: bool = False                 # borrowed rows, colour bar, anchors_out
    borrowed: dict = field(default_factory=dict)      # team -> [Anchor], fetched by the caller
    this_colors: dict = field(default_factory=dict)   # team -> "red" / "blue" in this match
    use_elimination: bool = True               # ablation: elimination anchors
    rows_alliance_only: bool = False           # ablation: one card per alliance
    digit_calls: Optional[dict] = None         # digit-call cache, reused per window fingerprint
    deinterlace: bool = False                  # deinterlace combed tiles (window_crops.wide_tile)
    # Digit reader (window_labels_reader knob; teacher-distillation spec section 14):
    # "jev" = every digit request goes to Jev (today's stage, unchanged);
    # "student" = S1 reads every sample first, Jev only for windows S1 leaves
    # undecided; "student_only" = no Jev in the digit pass. The lineup pass is
    # Jev in every mode.
    reader: str = "jev"
    cancelled: bool = False                    # a user stop: aborts at the next check, never partial


READERS = ("jev", "student", "student_only")
STUDENT_DETECT_BATCH = 16          # main-view frames per bumper-detector call


class StageAbort(Exception):
    """The stage cannot finish; the caller keeps the labels it had."""


class DeadlinePassed(StageAbort):
    """The stage's time budget ran out (a stop is a plain StageAbort)."""


def _stats(event_lineup: bool = False, deinterlace: bool = False) -> dict:
    out = {"windows": 0, "digit_named": 0, "lineup_named": 0, "cleared_by_uniqueness": 0,
           "unlabelled": 0, "lineup_rows": 0, "samples": 0, "tiles": 0, "requests": 0,
           "input_tokens": 0, "cost_cents": 0.0, "request_errors": 0, "aborted": None}
    if event_lineup:
        out.update(rows_own=0, rows_borrowed=0, rows_elim=0)
    if deinterlace:                    # knob off: the stats stay exactly as they were
        out["tiles_deinterlaced"] = 0
    return out


def _student_stats(reader: str) -> dict:
    return {"reader": reader, "fallback": None, "model": None, "samples": 0, "bumper_found": 0,
            "crops": 0, "named_samples": 0, "conflict_samples": 0, "windows_s1": 0,
            "windows_jev": 0, "jev_digit_calls": 0, "detect_s": 0.0, "read_s": 0.0}


def _open_student(student) -> tuple:
    """(reader or None, owned, why): ``student`` is a ready reader (tests,
    a caller that keeps one), a zero-argument factory, or None for
    window_student.StudentReader(). A reader this function built is owned:
    the stage closes it."""
    try:
        if student is None:
            from web.field.window_student import StudentReader
            return StudentReader(), True, None
        if callable(student) and not hasattr(student, "read"):
            return student(), True, None
        return student, False, None
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"


def run_window_labels(frames: list, video_path, cal, session, params: Params, *,
                      frame_source=None, clock=time.monotonic, student=None) -> dict:
    """Run the whole stage. Blocking; reads ``frames`` but never changes them.

    ``session``: a ``JevSession`` (anything with ``ask``, ``available``,
    ``disabled_reason`` and ``stats``). ``frame_source(video_path, frame_nums,
    view)`` yields (frame_num, main-view image); defaults to
    ``window_crops.iter_view_frames``.

    Returns {"aborted": reason or None, "labels": lab or None, "segments":
    {slot id: rows} or None, "stats": {...}, "lineup_jpeg": the card (the red
    card with rows_alliance_only) or None, "lineup_jpeg_blue": the blue card
    with rows_alliance_only else None, "anchors_out": [Anchor] for the event
    store (event_lineup only), "digit_calls": the digit-pass cache
    {"calls", "wide_p", "fingerprints"} (None on abort), "row_sources": team
    -> {"own", "borrowed", "elim", "from"} (event_lineup only)}. On abort,
    labels and segments are None and nothing should be applied or stored.

    ``params.reader`` "student" / "student_only": ``student`` is the S1
    reader (see _open_student; default: loaded here on the GPU and released
    before returning). When it cannot be loaded the stage runs as "jev" and
    stats["student"]["fallback"] says why."""
    stats = _stats(params.event_lineup, params.deinterlace)
    before = _usage(session)
    reader, stu, owned = "jev", None, False
    if params.reader != "jev":
        stats["student"] = _student_stats(params.reader)
        if params.reader not in READERS:
            why = f"unknown reader {params.reader!r}"
        else:
            stu, owned, why = _open_student(student)
        if stu is None:
            print(f"[window-labels] student reader unavailable, Jev reads every window: {why}")
            stats["student"]["fallback"] = why
        else:
            reader = params.reader
            stats["student"]["model"] = getattr(stu, "model_key", None)
    try:
        out = _run_stage(frames, video_path, cal, session, params, frame_source, clock, stats,
                         reader, stu)
    except StageAbort as e:
        stats["aborted"] = str(e)
        out = None
    except Exception as e:                       # never let the stage break a job
        stats["aborted"] = f"error: {type(e).__name__}: {e}"
        out = None
    finally:
        if owned:
            try:
                stu.close()
            except Exception:
                pass
    after = _usage(session)
    # HTTP attempts (retries included), not payloads
    stats["requests"] = after[0] - before[0]
    stats["input_tokens"] = after[1] - before[1]
    stats["cost_cents"] = round((after[2] - before[2]) * 100.0, 2)
    if out is None:
        return {"aborted": stats["aborted"], "labels": None, "segments": None,
                "stats": stats, "lineup_jpeg": None, "lineup_jpeg_blue": None,
                "anchors_out": [], "digit_calls": None, "row_sources": {}}
    return {"aborted": None, "stats": stats, **out}


def _usage(session) -> tuple:
    try:
        st = session.stats()
        return (int(st.get("requests", 0)), int(st.get("input_tokens", 0)),
                float(st.get("cost_usd", 0.0)))
    except Exception:
        return (0, 0, 0.0)


def _check(params: Params, clock, where: str) -> None:
    if params.cancelled:
        raise StageAbort(f"cancelled {where}")
    if params.deadline is not None and clock() > params.deadline:
        raise DeadlinePassed(f"deadline passed {where}")


def _ask_batched(session, specs: list, build, question: str, params: Params, clock,
                 stats: dict, where: str, partial_ok: bool = False) -> list:
    """Send ``specs`` in batches of REQUEST_BATCH, checking the deadline and
    the session before each. Returns one result per spec (probs dict or
    None). Aborts when more than MAX_ERROR_FRACTION of the pass failed (checked
    early after every batch against what was sent so far, with a floor of
    MIN_ERRORS_TO_ABORT_EARLY, and against the whole pass after the last
    batch), when the reader latched off during a batch, or when a batch's
    answer count does not match what was sent. With ``partial_ok`` a deadline
    (not a stop) ends the pass early instead of aborting: the answers so far are
    returned, so the result is shorter than ``specs``."""
    results: list = []
    errors = 0
    for start in range(0, len(specs), REQUEST_BATCH):
        try:
            _check(params, clock, where)
        except DeadlinePassed:
            if not partial_ok:
                raise
            break
        if not session.available:
            raise StageAbort(f"reader unavailable: {session.disabled_reason}")
        batch = specs[start:start + REQUEST_BATCH]
        answers = session.ask([build(s) for s in batch], question)
        if not session.available:                # latched off (401/403) mid-batch
            stats["request_errors"] += errors + sum(1 for r in answers if "probs" not in r)
            raise StageAbort(f"reader unavailable: {session.disabled_reason}")
        if len(answers) != len(batch):
            stats["request_errors"] += errors
            raise StageAbort(f"reader returned {len(answers)} answers for {len(batch)} "
                             f"requests {where}")
        for res in answers:
            if "probs" in res:
                results.append(res["probs"])
            else:
                results.append(None)
                errors += 1
        sent = start + len(batch)
        if errors > max(MIN_ERRORS_TO_ABORT_EARLY, MAX_ERROR_FRACTION * sent):
            stats["request_errors"] += errors
            raise StageAbort(f"request errors: {errors} of {sent} sent so far "
                             f"(pass of {len(specs)}) {where}")
        if sent >= len(specs) and errors > MAX_ERROR_FRACTION * len(specs):
            stats["request_errors"] += errors
            raise StageAbort(f"request errors: {errors} of {len(specs)} {where}")
    stats["request_errors"] += errors
    return results


def window_fingerprint(frames: list, w: Window, requests: list,
                       deinterlace: bool = False) -> dict:
    """What a cached digit-call list must match to be reused (spec section 5):
    the slot, the window's first source frame, its sample frames and the
    (kind, k) requests the stage would send now. Window ids are positional
    and the in-view filter depends on calibration and offset, so a mismatch
    means the cached calls belong to a different window. ``deinterlace``
    adds a key (only when on, so knob-off caches stay valid): calls made on
    combed tiles are not reused for deinterlaced ones, nor the reverse."""
    fp = {"slot": w.slot, "first_frame": int(frames[w.idx[0]]["frame_num"]),
          "samples": [int(frames[i]["frame_num"]) for i in w.samples],
          "requests": [[kind, int(k)] for kind, k in requests]}
    if deinterlace:
        fp["deinterlace"] = True
    return fp


def load_digit_calls(obj) -> dict:
    """A digit-call cache as JSON gives it back (string keys) -> int window
    ids and sample numbers. An already-converted cache passes unchanged."""
    obj = obj or {}
    return {"calls": {int(w): list(c) for w, c in (obj.get("calls") or {}).items()},
            "wide_p": {int(w): {int(k): p for k, p in (d or {}).items()}
                       for w, d in (obj.get("wide_p") or {}).items()},
            "fingerprints": {int(w): fp for w, fp in (obj.get("fingerprints") or {}).items()}}


def _requests(windows: list, tiles: dict) -> dict:
    """wid -> the digit requests for that window, in send order: wide and
    zoom per tile (by sample number), then one grid."""
    return {w.wid: [r for k in sorted(tiles[w.wid]) for r in (("wide", k), ("zoom", k))]
            + [("grid", 0)] for w in windows if tiles.get(w.wid)}


def _wide_p(requests: dict, calls: dict) -> dict:
    """wid -> {k: probs} of the wide calls that returned."""
    out: dict = {}
    for wid, reqs in requests.items():
        for (kind, k), probs in zip(reqs, calls.get(wid) or []):
            if kind == "wide" and probs:
                out.setdefault(wid, {})[k] = probs
    return out


def _digit_pass(frames, windows, tiles, params, session, digit_item, clock, stats,
                reader: str = "jev", sent: Optional[list] = None) -> dict:
    """Digit calls for every window with tiles. A window's calls come from
    ``params.digit_calls`` when its fingerprint matches and the call count
    fits (cached failures stay failures); every other window is requested.
    Returns the cache for exactly the calls used: {"calls": wid -> [probs or
    None] in request order, "wide_p": wid -> {k: probs}, "fingerprints", plus
    the "roster" and "model_key" it was made for; a cache made for another
    roster or model, or one that fails to load, is ignored whole. So is a
    cache made by another ``reader`` mode (a cache without a "reader" key is
    a "jev" cache). ``sent`` (a list) gets the number of requests sent."""
    requests = _requests(windows, tiles)
    by_wid = {w.wid: w for w in windows}
    fingerprints = {wid: window_fingerprint(frames, by_wid[wid], reqs, params.deinterlace)
                    for wid, reqs in requests.items()}
    roster = [str(t) for t in params.red] + [str(t) for t in params.blue]
    cache = None
    try:                    # a cache from another roster / model, or a broken one, is no cache
        if (params.digit_calls and params.digit_calls.get("roster") == roster
                and params.digit_calls.get("model_key") == params.model_key
                and params.digit_calls.get("reader", "jev") == reader):
            cache = load_digit_calls(params.digit_calls)
    except Exception:
        cache = None
    calls: dict = {}
    specs = []
    for wid, reqs in requests.items():
        old = cache["calls"].get(wid) if cache else None
        if (old is not None and len(old) == len(reqs)
                and cache["fingerprints"].get(wid) == fingerprints[wid]):
            calls[wid] = list(old)
        else:
            specs += [(wid, kind, k) for kind, k in reqs]
    if sent is not None:
        sent.append(len(specs))
    answers = _ask_batched(session, specs, digit_item, "team", params, clock, stats,
                           "in the digit pass")
    for (wid, _kind, _k), probs in zip(specs, answers):
        calls.setdefault(wid, []).append(probs)
    calls = {wid: calls[wid] for wid in requests}           # window order
    return {"calls": calls, "wide_p": _wide_p(requests, calls), "fingerprints": fingerprints,
            "roster": roster, "model_key": params.model_key}


def _lineup_rows(digit, start_s, tiles, wide_p, roster, colours, params, crops) -> tuple:
    """(team -> row tiles, team -> row provenance, own anchors). Knob off:
    pick_anchors rows exactly as before, no provenance, no own anchors. Knob
    on: store_anchors windows (a single good read counts) filled with
    borrowed anchors by fill_row; a row needs MIN_ANCHORS tiles in all. A
    borrowed tile that does not decode is dropped."""
    if not params.event_lineup:
        anchors = pick_anchors(digit, start_s)
        return ({t: [tiles[wid][best_tile_k(wide_p, tiles, wid, t)] for wid in wids]
                 for t, wids in anchors.items()}, {}, {})
    own = store_anchors(digit, start_s)
    rows, sources = {}, {}
    for t in roster:
        mine = [tiles[wid][best_tile_k(wide_p, tiles, wid, t)] for wid in own.get(t, [])]
        got = [(a, crops.decode_tile(a.tile))
               for a in fill_row(len(mine), params.borrowed.get(t) or [], colours[t],
                                 params.use_elimination)]
        got = [(a, img) for a, img in got if img is not None]
        if len(mine) + len(got) < MIN_ANCHORS:
            continue
        rows[t] = mine + [img for _a, img in got]
        sources[t] = {"own": len(mine),
                      "borrowed": sum(a.kind == KIND_DIGITS for a, _img in got),
                      "elim": sum(a.kind == KIND_ELIM for a, _img in got),
                      "from": sorted({a.match_key for a, _img in got if a.match_key})}
    return rows, sources, own


def _anchors_out(frames, windows, by_wid, digit, own, calls, wide_p, tiles, colours,
                 red, blue, params, crops) -> list:
    """This match's anchors for the event store: every own digit anchor
    (store_anchors) and, unless disabled, the elimination anchors. The match
    fields stay None; the store stamps them."""
    out = []

    def add(team, wid, k, kind, p):
        w = by_wid[wid]
        out.append(Anchor(team=team, kind=kind, p=round(float(p), 4),
                          frame_num=int(frames[w.samples[k - 1]]["frame_num"]),
                          alliance_color=colours.get(team), tile=crops.jpeg(tiles[wid][k])))

    for team, wids in own.items():
        for wid in wids:
            add(team, wid, best_tile_k(wide_p, tiles, wid, team), KIND_DIGITS, digit[wid][1])
    if params.use_elimination:
        period = frame_period_s(frames, params.fps) or 0.0
        for team, picks in elimination_anchors(windows, frames, digit, calls, wide_p,
                                               red, blue, period).items():
            for wid, k, p in picks:
                if k in tiles.get(wid, {}):
                    add(team, wid, k, KIND_ELIM, p)
    return out


class _StudentPass:
    """S1 inputs cut while the video pass runs: frames are buffered, the
    bumper detector runs per STUDENT_DETECT_BATCH frames (on one field when
    deinterlacing, like the harvester on woven video), and the padded crops
    of the bumpers near each sample's target are kept."""

    def __init__(self, stu, deinterlace: bool, st: dict):
        self.stu, self.deinterlace, self.st = stu, deinterlace, st
        self.pending: list = []                # (main-view image, [(wid, k, u, v, side)])
        self.crops: list = []
        self.colours: list = []
        self.by_sample: dict = {}              # (wid, k) -> crop indices, nearest first

    def add(self, img, items) -> None:
        if items:
            self.pending.append((img, list(items)))
        if len(self.pending) >= STUDENT_DETECT_BATCH:
            self.flush()

    def flush(self) -> None:
        from web.field import window_student as ws
        if not self.pending:
            return
        t0 = time.perf_counter()
        imgs = [ws.field_frame(img) if self.deinterlace else img for img, _ in self.pending]
        dets = self.stu.detect(imgs)
        if len(dets) != len(imgs):
            raise StageAbort(f"bumper detector returned {len(dets)} results for {len(imgs)} frames")
        for (img, items), d in zip(self.pending, dets):
            h, w = img.shape[:2]
            for wid, k, u, v, side in items:
                self.st["samples"] += 1
                near = ws.near_bumpers(d, u, v, side)
                if near:
                    self.st["bumper_found"] += 1
                for _dist, box, _conf, colour in near:
                    x0, y0, x1, y1 = ws.pad_box(box, ws.PAD_BUMPER, w, h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    crop = img[y0:y1, x0:x1].copy()
                    if self.deinterlace:
                        from web.bumper_register import deinterlace_crop
                        crop = deinterlace_crop(crop)
                    self.by_sample.setdefault((wid, k), []).append(len(self.crops))
                    self.crops.append(crop)
                    self.colours.append(colour)
        self.pending = []
        self.st["detect_s"] = round(self.st["detect_s"] + time.perf_counter() - t0, 3)

    def calls(self, tiles: dict, red: list, blue: list) -> dict:
        """wid -> one call per tile (sample number order): S1's read of
        the sample's bumpers (window_student.sample_call), None where no
        bumper was found."""
        from web.field import window_student as ws
        self.flush()
        t0 = time.perf_counter()
        roster = red + blue
        reads = self.stu.read(self.crops, [{"red": red, "blue": blue}] * len(self.crops),
                              self.colours) if self.crops else []
        if len(reads) != len(self.crops):
            raise StageAbort(f"S1 returned {len(reads)} reads for {len(self.crops)} crops")
        self.st["crops"] = len(self.crops)
        self.st["read_s"] = round(time.perf_counter() - t0, 3)
        out = {}
        for wid in tiles:
            row = []
            for k in sorted(tiles[wid]):
                mine = [reads[j] for j in self.by_sample.get((wid, k), ())]
                call = ws.sample_call(mine, roster)
                if call is not None and call != ws.ABSTAIN:
                    self.st["named_samples"] += 1
                elif any(r.get("team") is not None for r in mine):
                    self.st["conflict_samples"] += 1
                row.append(call)
            out[wid] = row
        return out


def _student_digit_calls(frames, windows, tiles, params, session, digit_item, clock, stats,
                         reader: str, s1: dict) -> dict:
    """The digit pass for reader "student" / "student_only": S1's calls
    name a window when digit_decision accepts them; "student" sends Jev the
    usual requests for every other window. Same cache shape as _digit_pass
    plus "reader", "s1_model", "sources" (wid -> "s1" | "jev") and
    "s1_calls" (every window's S1 calls). Fingerprints cover the Jev windows
    only, so S1 calls are never reused as Jev calls."""
    st = stats["student"]
    roster = [str(t) for t in params.red] + [str(t) for t in params.blue]
    decided = {wid for wid, c in s1.items() if digit_decision(c, roster)}
    jdc, sent = {"calls": {}, "fingerprints": {}}, []
    if reader == "student":
        rest = [w for w in windows if w.wid not in decided and tiles.get(w.wid)]
        jdc = _digit_pass(frames, rest, tiles, params, session, digit_item, clock, stats,
                          reader, sent)
    calls, sources, wide_p = {}, {}, {}
    for w in windows:
        if not tiles.get(w.wid):
            continue
        if w.wid in jdc["calls"]:
            calls[w.wid], sources[w.wid] = jdc["calls"][w.wid], "jev"
        else:
            calls[w.wid], sources[w.wid] = list(s1[w.wid]), "s1"
            if w.wid in decided:     # only named windows get "wide" calls (lineup tiles)
                per_k = {k: c for k, c in zip(sorted(tiles[w.wid]), s1[w.wid])
                         if c and any(c.get(t, 0.0) > 0.0 for t in roster)}
                if per_k:
                    wide_p[w.wid] = per_k
    wide_p.update(jdc.get("wide_p") or {})
    st["windows_s1"] = sum(1 for wid in decided if sources.get(wid) == "s1")
    st["windows_jev"] = sum(1 for s in sources.values() if s == "jev")
    st["jev_digit_calls"] = sum(sent)
    return {"calls": calls, "wide_p": wide_p, "fingerprints": jdc["fingerprints"],
            "roster": roster, "model_key": params.model_key, "reader": reader,
            "s1_model": st.get("model"), "sources": sources,
            "s1_calls": {wid: s1[wid] for wid in calls}}


def _run_stage(frames, video_path, cal, session, params, frame_source, clock, stats,
               reader: str = "jev", stu=None) -> dict:
    from web import jev_reader
    from web.field import window_crops as crops

    _check(params, clock, "before start")
    red, blue = [str(t) for t in params.red], [str(t) for t in params.blue]
    roster = red + blue
    windows = build_windows(frames, params.fps)
    stats["windows"] = len(windows)
    if not windows:
        raise StageAbort("no windows")
    if not session.available:                    # e.g. no token: fail before decoding video
        raise StageAbort(f"reader unavailable: {session.disabled_reason}")
    if not roster:                               # only escape options: nothing could be named
        raise StageAbort("no roster: the match has no red or blue teams")

    # 1. where each sample is in the main view
    view_wh = (params.view[2], params.view[3])
    want: dict = {}                                  # frame_num -> [(wid, k, u, v, side)]
    for w in windows:
        for k, i in enumerate(w.samples, 1):
            row = next(r for r in frames[i]["robots"] if r.get("slot") == w.slot)
            tg = crops.target(cal, row["field_pos"], params.ground_center_offset_m)
            if tg is None or not crops.in_view(tg[0], tg[1], view_wh):
                continue
            want.setdefault(int(frames[i]["frame_num"]), []).append((w.wid, k) + tg)
    stats["samples"] = sum(len(v) for v in want.values())
    if not want:
        raise StageAbort("no window sample is in the main view")

    # 2. one sequential video pass; keep only the tiles
    tiles: dict = {}                                 # wid -> {k: tile}
    spass = _StudentPass(stu, params.deinterlace, stats["student"]) if reader != "jev" else None
    source = (frame_source or crops.iter_view_frames)(video_path, list(want), params.view)
    try:
        for n, (frame_num, img) in enumerate(source, 1):
            for wid, k, u, v, side in want.get(frame_num, ()):
                tile, _, fired = crops.wide_tile(img, u, v, side, params.deinterlace)
                tiles.setdefault(wid, {})[k] = tile
                stats["tiles"] += 1
                if fired:
                    stats["tiles_deinterlaced"] += 1
            if spass is not None:
                spass.add(img, want.get(frame_num, ()))
            if n % 50 == 0:
                _check(params, clock, "during the video pass")
    finally:
        if hasattr(source, "close"):             # release the video capture promptly
            source.close()
    if not tiles:
        raise StageAbort("no frames decoded from the source video")
    _check(params, clock, "after the video pass")

    # 3. digit pass: wide + zoom per tile, one grid per window
    def digit_item(spec):
        wid, kind, k = spec
        t = tiles[wid]
        img = (crops.send_wide(t[k]) if kind == "wide" else crops.zoom_tile(t[k])
               if kind == "zoom" else crops.grid_image([t[j] for j in sorted(t)]))
        return jev_reader.build_jev_window_payload(params.model_key, crops.jpeg(img), kind,
                                                   red, blue)

    if reader == "jev":
        dc = _digit_pass(frames, windows, tiles, params, session, digit_item, clock, stats)
    else:
        s1 = spass.calls(tiles, red, blue)
        _check(params, clock, "after the student pass")
        dc = _student_digit_calls(frames, windows, tiles, params, session, digit_item, clock,
                                  stats, reader, s1)
    calls, wide_p = dc["calls"], dc["wide_p"]
    digit = {}
    for wid, c in calls.items():
        d = digit_decision(c, roster)
        if d:
            digit[wid] = d
    stats["digit_named"] = len(digit)

    # 4. lineup rows from the most confident digit labels (knob on: plus rows
    # borrowed from the event store); one card, or one per alliance
    by_wid = {w.wid: w for w in windows}
    start_s = {w.wid: frames[w.idx[0]]["frame_num"] / float(params.fps) for w in windows}
    colours = {t: params.this_colors.get(t) or ("red" if t in red else "blue") for t in roster}
    rows, row_sources, own = _lineup_rows(digit, start_s, tiles, wide_p, roster, colours,
                                          params, crops)
    cards = [("red", red), ("blue", blue)] if params.rows_alliance_only else [("all", roster)]
    todo_all = [w.wid for w in windows if w.wid not in digit and tiles.get(w.wid)]
    note = params.ball_note if params.ball_note is not None else jev_reader.BALL_NOTE_2026
    lineup, jpegs, lettered = {}, {}, []
    for name, teams in cards:
        order = [t for t in teams if t in rows]
        random.Random(params.seed).shuffle(order)
        letters = {LETTERS[j]: t for j, t in enumerate(order[:len(LETTERS)])}
        stats["lineup_rows"] += len(letters)
        lettered += list(letters.values())
        if len(letters) < MIN_LINEUP_ROWS or params.lineup_min_p > 1.0:
            continue
        colors = [colours[t] for t in letters.values()] if params.event_lineup else None
        card_jpeg = crops.jpeg(crops.lineup_card([(l, rows[t]) for l, t in letters.items()],
                                                 ANCHORS_PER_TEAM, colors=colors))
        jpegs[name] = card_jpeg
        card_asset = jev_reader.jev_asset(card_jpeg)   # encoded once, shared by every payload
        todo = [wid for wid in todo_all
                if name == "all" or slot_alliance(by_wid[wid].slot) == name]

        def lineup_item(wid, card_jpeg=card_jpeg, card_asset=card_asset, letters=letters):
            t = tiles[wid]
            grid = crops.jpeg(crops.grid_image([t[j] for j in sorted(t)]))
            return jev_reader.build_jev_lineup_payload(params.model_key, card_jpeg, grid,
                                                       list(letters), note,
                                                       lineup_asset=card_asset,
                                                       colour_note=params.event_lineup)

        # 5. lineup pass on the windows the digit pass did not name
        answered = _ask_batched(session, todo, lineup_item, "who", params, clock, stats,
                                "in the lineup pass", partial_ok=True)
        if len(answered) < len(todo):            # deadline mid-pass: keep what we have
            stats["deadline_partial"] = "lineup"
            stats["lineup_skipped"] = stats.get("lineup_skipped", 0) + len(todo) - len(answered)
        for wid, probs in zip(todo, answered):
            d = lineup_decision(probs, letters, params.lineup_min_p)
            if d:
                lineup[wid] = d
    stats["lineup_named"] = len(lineup)
    anchors_out: list = []
    if params.event_lineup:
        src = [row_sources[t] for t in lettered]
        stats["rows_own"] = sum(1 for s in src if not s["borrowed"] and not s["elim"])
        stats["rows_borrowed"] = sum(1 for s in src if s["borrowed"] and not s["elim"])
        stats["rows_elim"] = sum(1 for s in src if s["elim"])
        anchors_out = _anchors_out(frames, windows, by_wid, digit, own, calls, wide_p, tiles,
                                   colours, red, blue, params, crops)

    # 6. assemble
    evidence = {wid: (t, RANK_DIGITS, p) for wid, (t, p) in digit.items()}
    evidence.update({wid: (t, RANK_LINEUP, p) for wid, (t, p) in lineup.items()})
    lab, cleared = assemble_labels(windows, evidence)
    stats["cleared_by_uniqueness"] = cleared
    stats["unlabelled"] = len(windows) - len(evidence)
    return {"labels": lab, "segments": window_segments(frames, lab),
            "lineup_jpeg": jpegs.get("all", jpegs.get("red")),
            "lineup_jpeg_blue": jpegs.get("blue"),
            "anchors_out": anchors_out, "digit_calls": dc, "row_sources": row_sources}
