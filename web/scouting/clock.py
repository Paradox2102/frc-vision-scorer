"""Source-video frame -> game-clock time, from a replay's ``summary.periods``.

``summary.periods`` comes from the clock scan:
  {"label": "auto"|"teleop", "start_frame", "end_frame",
   "start_seconds", "end_seconds"}   (seconds = the countdown shown on screen)

``MatchClock.at(frame_num)`` returns ``(period, t_match, remaining)`` or None
for frames outside both periods (pre-match, the auto→teleop pause, post-match).

  t_match   game seconds since auto start, pause removed:
            auto   0 .. auto_len
            teleop auto_len .. auto_len + teleop_len
  remaining the countdown within the period (true time, not the displayed digit)

CLOCK LEAD. ``start_frame`` is NOT the true period start: the clock scan
samples OCR every 0.5 s and takes the last sample still showing the start
value ("0:20") before the first tick, and the broadcast clock holds 0:20 for
the whole first second. So the real start is 0.5..1.0 s before
``start_frame``. Measured on 2026cur/2026hop (36 matches): robots that start
visibly at rest begin moving a median 0.27 s BEFORE ``start_frame`` and 0-5
robots per match are already moving at the first processed frame (0.5 s
before). A follow-up measurement over 25 matches (pipeline clock-lead work,
Nov 12) put the lag at a median 1.04 s (range 0.65-1.72), so ``CLOCK_LEAD_S``
= 1.0 s, applied to both periods (one clock, one bias).

TRUE START. Newer replays carry ``true_start_frame`` on the auto period (the
frame the 2026 scoreboard indicator lights, ``start_method == "indicator"``,
or ``start_frame`` - 1.0 s with ``"clock_offset"``). When present it replaces
the lead for that period; teleop keeps ``CLOCK_LEAD_S`` (its OCR sampling
phase is independent of auto's).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CLOCK_LEAD_S = 1.0


@dataclass(frozen=True)
class Period:
    label: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    true_start_frame: Optional[int] = None

    @property
    def length_s(self) -> float:
        return float(self.start_seconds - self.end_seconds)

    @property
    def frames_per_s(self) -> float:
        span = self.end_frame - self.start_frame
        return span / self.length_s if self.length_s > 0 and span > 0 else 0.0


class MatchClock:
    def __init__(self, periods: list, lead_s: float = CLOCK_LEAD_S):
        self.lead_s = float(lead_s)
        self.periods: dict = {}
        for p in periods or []:
            label = p.get("label")
            if label not in ("auto", "teleop"):
                continue
            try:
                tsf = p.get("true_start_frame")
                per = Period(label, int(p["start_frame"]), int(p["end_frame"]),
                             float(p["start_seconds"]), float(p["end_seconds"]),
                             int(tsf) if tsf is not None else None)
            except (KeyError, TypeError, ValueError):
                continue
            if per.frames_per_s > 0:
                self.periods[label] = per

    @property
    def ok(self) -> bool:
        return "auto" in self.periods or "teleop" in self.periods

    @property
    def auto_len(self) -> float:
        p = self.periods.get("auto")
        return p.start_seconds if p else 0.0

    def frames_per_s(self) -> float:
        for label in ("teleop", "auto"):
            if label in self.periods:
                return self.periods[label].frames_per_s
        return 0.0

    def at(self, frame_num: int) -> Optional[tuple]:
        for label in ("auto", "teleop"):
            p = self.periods.get(label)
            if p is None:
                continue
            if p.true_start_frame is not None:
                elapsed = (frame_num - p.true_start_frame) / p.frames_per_s
            else:
                elapsed = (frame_num - p.start_frame) / p.frames_per_s + self.lead_s
            if not (0.0 <= elapsed <= p.length_s):
                continue
            remaining = max(p.end_seconds, p.start_seconds - elapsed)
            if label == "auto":
                t = p.start_seconds - remaining
            else:
                t = self.auto_len + (p.start_seconds - remaining)
            return label, float(t), float(remaining)
        return None
