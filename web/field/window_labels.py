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
MIN_ERRORS_TO_ABORT_EARLY = 20
STAGE_TIMEOUT_S = 360.0
REQUEST_BATCH = 32
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


class StageAbort(Exception):
    """The stage cannot finish; the caller keeps the labels it had."""


def _stats() -> dict:
    return {"windows": 0, "digit_named": 0, "lineup_named": 0, "cleared_by_uniqueness": 0,
            "unlabelled": 0, "lineup_rows": 0, "samples": 0, "tiles": 0, "requests": 0,
            "input_tokens": 0, "cost_cents": 0.0, "request_errors": 0, "aborted": None}


def run_window_labels(frames: list, video_path, cal, session, params: Params, *,
                      frame_source=None, clock=time.monotonic) -> dict:
    """Run the whole stage. Blocking; reads ``frames`` but never changes them.

    ``session``: a ``JevSession`` (anything with ``ask``, ``available``,
    ``disabled_reason`` and ``stats``). ``frame_source(video_path, frame_nums,
    view)`` yields (frame_num, main-view image); defaults to
    ``window_crops.iter_view_frames``.

    Returns {"aborted": reason or None, "labels": lab or None, "segments":
    {slot id: rows} or None, "stats": {...}, "lineup_jpeg": bytes or None}.
    On abort, labels and segments are None and nothing should be applied."""
    stats = _stats()
    before = _usage(session)
    try:
        out = _run_stage(frames, video_path, cal, session, params, frame_source, clock, stats)
    except StageAbort as e:
        stats["aborted"] = str(e)
        out = None
    except Exception as e:                       # never let the stage break a job
        stats["aborted"] = f"error: {type(e).__name__}: {e}"
        out = None
    after = _usage(session)
    # HTTP attempts (retries included), not payloads
    stats["requests"] = after[0] - before[0]
    stats["input_tokens"] = after[1] - before[1]
    stats["cost_cents"] = round((after[2] - before[2]) * 100.0, 2)
    if out is None:
        return {"aborted": stats["aborted"], "labels": None, "segments": None,
                "stats": stats, "lineup_jpeg": None}
    return {"aborted": None, "stats": stats, **out}


def _usage(session) -> tuple:
    try:
        st = session.stats()
        return (int(st.get("requests", 0)), int(st.get("input_tokens", 0)),
                float(st.get("cost_usd", 0.0)))
    except Exception:
        return (0, 0, 0.0)


def _check(params: Params, clock, where: str) -> None:
    if params.deadline is not None and clock() > params.deadline:
        raise StageAbort(f"deadline passed {where}")


def _ask_batched(session, specs: list, build, question: str, params: Params, clock,
                 stats: dict, where: str) -> list:
    """Send ``specs`` in batches of REQUEST_BATCH, checking the deadline and
    the session before each. Returns one result per spec (probs dict or
    None). Aborts when more than MAX_ERROR_FRACTION of the pass failed (checked
    early after every batch against what was sent so far, with a floor of
    MIN_ERRORS_TO_ABORT_EARLY, and against the whole pass after the last
    batch), when the reader latched off during a batch, or when a batch's
    answer count does not match what was sent."""
    results: list = []
    errors = 0
    for start in range(0, len(specs), REQUEST_BATCH):
        _check(params, clock, where)
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


def _run_stage(frames, video_path, cal, session, params, frame_source, clock, stats) -> dict:
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
    source = (frame_source or crops.iter_view_frames)(video_path, list(want), params.view)
    try:
        for n, (frame_num, img) in enumerate(source, 1):
            for wid, k, u, v, side in want.get(frame_num, ()):
                tiles.setdefault(wid, {})[k] = crops.wide_tile(img, u, v, side)[0]
                stats["tiles"] += 1
            if n % 50 == 0:
                _check(params, clock, "during the video pass")
    finally:
        if hasattr(source, "close"):             # release the video capture promptly
            source.close()
    if not tiles:
        raise StageAbort("no frames decoded from the source video")
    _check(params, clock, "after the video pass")

    # 3. digit pass: wide + zoom per tile, one grid per window
    specs = []
    for w in windows:
        t = tiles.get(w.wid)
        if not t:
            continue
        for k in sorted(t):
            specs += [(w.wid, "wide", k), (w.wid, "zoom", k)]
        specs.append((w.wid, "grid", 0))

    def digit_item(spec):
        wid, kind, k = spec
        t = tiles[wid]
        img = (crops.send_wide(t[k]) if kind == "wide" else crops.zoom_tile(t[k])
               if kind == "zoom" else crops.grid_image([t[j] for j in sorted(t)]))
        return jev_reader.build_jev_window_payload(params.model_key, crops.jpeg(img), kind,
                                                   red, blue)

    answers = _ask_batched(session, specs, digit_item, "team", params, clock, stats,
                           "in the digit pass")
    calls: dict = {}                                 # wid -> [probs or None]
    wide_p: dict = {}                                # wid -> {k: probs}
    for (wid, kind, k), probs in zip(specs, answers):
        calls.setdefault(wid, []).append(probs)
        if kind == "wide" and probs:
            wide_p.setdefault(wid, {})[k] = probs
    digit = {}
    for wid, c in calls.items():
        d = digit_decision(c, roster)
        if d:
            digit[wid] = d
    stats["digit_named"] = len(digit)

    # 4. lineup card from the most confident digit labels
    start_s = {w.wid: frames[w.idx[0]]["frame_num"] / float(params.fps) for w in windows}
    anchors = pick_anchors(digit, start_s)
    order = [t for t in roster if t in anchors]
    random.Random(params.seed).shuffle(order)
    letters = {LETTERS[j]: t for j, t in enumerate(order[:len(LETTERS)])}
    stats["lineup_rows"] = len(letters)
    lineup = {}
    lineup_jpeg = None
    if len(letters) >= MIN_LINEUP_ROWS and params.lineup_min_p <= 1.0:
        rows = []
        for letter, team in letters.items():
            row_tiles = []
            for wid in anchors[team]:
                row_tiles.append(tiles[wid][best_tile_k(wide_p, tiles, wid, team)])
            rows.append((letter, row_tiles))
        lineup_jpeg = crops.jpeg(crops.lineup_card(rows, ANCHORS_PER_TEAM))
        card_asset = jev_reader.jev_asset(lineup_jpeg)   # encoded once, shared by every payload
        todo = [w.wid for w in windows if w.wid not in digit and tiles.get(w.wid)]
        note = params.ball_note if params.ball_note is not None else jev_reader.BALL_NOTE_2026

        def lineup_item(wid):
            t = tiles[wid]
            grid = crops.jpeg(crops.grid_image([t[j] for j in sorted(t)]))
            return jev_reader.build_jev_lineup_payload(params.model_key, lineup_jpeg, grid,
                                                       list(letters), note,
                                                       lineup_asset=card_asset)

        # 5. lineup pass on the windows the digit pass did not name
        for wid, probs in zip(todo, _ask_batched(session, todo, lineup_item, "who", params,
                                                 clock, stats, "in the lineup pass")):
            d = lineup_decision(probs, letters, params.lineup_min_p)
            if d:
                lineup[wid] = d
    stats["lineup_named"] = len(lineup)

    # 6. assemble
    evidence = {wid: (t, RANK_DIGITS, p) for wid, (t, p) in digit.items()}
    evidence.update({wid: (t, RANK_LINEUP, p) for wid, (t, p) in lineup.items()})
    lab, cleared = assemble_labels(windows, evidence)
    stats["cleared_by_uniqueness"] = cleared
    stats["unlabelled"] = len(windows) - len(evidence)
    return {"labels": lab, "segments": window_segments(frames, lab), "lineup_jpeg": lineup_jpeg}
