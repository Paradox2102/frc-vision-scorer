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
