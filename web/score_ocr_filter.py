"""Misread guard for the scoreboard alliance-score OCR.

The analysis loop OCRs each alliance score about once a second. A single read
that jumps more than ``max_jump`` points from the last accepted value is
treated as a misread and skipped (tesseract occasionally returns garbage such
as 117 when the score is 2).

The old guard compared each read against the last ACCEPTED value only and
never moved that reference while rejecting. 2026 scores can climb by more than
``max_jump`` between two reads, so once one real jump was rejected every later
read was rejected too and the score froze for the rest of the match
(2026cur_qm5 red: frozen at 4, TBA 785).

This filter keeps the single-read guard but re-syncs: when ``confirm_reads``
consecutive rejected reads form a plausible upward run (each step
non-decreasing and at most ``run_max_step``), the newest one is accepted as
the new score. A lone garbage read never forms such a run.

Re-syncs only go UP. An alliance score never decreases within a match, so a
run of reads well below the current score is a different screen, not a
correction (2026cur_qm76: the broadcast cut to a side camera and a run of
garbage 8s pulled blue down from 435). For the same reason the score starts
at 0: a large first read must be confirmed by a run like any other jump.

``run_max_step`` is looser than ``max_jump`` because the end of a 2026 match
adds points fast (tower climbs, the last fuel), so consecutive true reads can
step by 20-35 (2026cur_qm85 red: 311, 345, 368, 371 never confirmed and the
final score stayed at 311). Garbage reads jump by hundreds, so a few dozen
points still separates them.

Because re-syncs only go up, a run of garbage that confirms upward would lock
the score high for the rest of the match (2026cur_qm85 blue: 75, then a burst
of 661/689 reads one second later). So a re-sync candidate must also be
reachable from the last accepted score in the source-video time since it was
accepted: ``max_jump`` plus a short burst at ``burst_rate`` for the first
``burst_s`` seconds, then ``sustained_rate``. Bursts are real (qm85 red +34
in one second at the end); sustained they are not (TBA's busiest 25-30 s
windows at 2026cur are ~8 points/s, e.g. sf4m1 red 207 in shift 2).
"""
from __future__ import annotations

from typing import List, Optional

SCORE_OCR_MAX_JUMP = 20
SCORE_OCR_CONFIRM_READS = 3
SCORE_OCR_RUN_MAX_STEP = 60
SCORE_OCR_BURST_RATE = 40.0        # points/s an alliance can climb briefly
SCORE_OCR_BURST_S = 3.0
SCORE_OCR_SUSTAINED_RATE = 12.0    # points/s it can keep up


class ScoreOcrFilter:
    """Per-alliance accept/reject state for scoreboard score reads."""

    def __init__(self, max_jump: int = SCORE_OCR_MAX_JUMP,
                 confirm_reads: int = SCORE_OCR_CONFIRM_READS,
                 run_max_step: int = SCORE_OCR_RUN_MAX_STEP,
                 burst_rate: float = SCORE_OCR_BURST_RATE,
                 burst_s: float = SCORE_OCR_BURST_S,
                 sustained_rate: float = SCORE_OCR_SUSTAINED_RATE):
        self.max_jump = max_jump
        self.confirm_reads = max(1, confirm_reads)
        self.run_max_step = max(max_jump, run_max_step)
        self.burst_rate = burst_rate
        self.burst_s = burst_s
        self.sustained_rate = sustained_rate
        self.last: Optional[int] = None
        self.last_t: Optional[float] = None
        self._pending: List[int] = []
        self.rejected = 0
        self.resyncs = 0

    def accept(self, value: Optional[int], t: Optional[float] = None) -> Optional[int]:
        """Feed one OCR read taken at source-video time ``t`` (seconds).
        Returns the value to adopt as the score, or None when the read is
        rejected (or missing). A None read leaves the state untouched, so
        unreadable frames neither break nor build a run. Without ``t`` the
        reachability bound is skipped."""
        if value is None:
            return None
        ref = 0 if self.last is None else self.last
        if abs(value - ref) <= self.max_jump:
            self._adopt(value, t)
            return value
        if value < ref or not self._reachable(value, ref, t):
            # Far below the current score, or higher than the alliance could
            # have scored since: never a re-sync candidate. Leave any upward
            # run that is building intact.
            self.rejected += 1
            return None
        if self._pending and not (0 <= value - self._pending[-1] <= self.run_max_step):
            self._pending.clear()
        self._pending.append(value)
        if len(self._pending) >= self.confirm_reads:
            self._adopt(value, t)
            self.resyncs += 1
            return value
        self.rejected += 1
        return None

    def _adopt(self, value: int, t: Optional[float]) -> None:
        self.last = value
        if t is not None:
            self.last_t = t
        self._pending.clear()

    def _reachable(self, value: int, ref: int, t: Optional[float]) -> bool:
        if t is None or self.last_t is None:
            return True
        dt = max(0.0, t - self.last_t)
        reach = (self.max_jump + self.burst_rate * min(dt, self.burst_s)
                 + self.sustained_rate * max(0.0, dt - self.burst_s))
        return value - ref <= reach
