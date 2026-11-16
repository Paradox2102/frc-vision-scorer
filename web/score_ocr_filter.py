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
consecutive rejected reads form a plausible run (each step non-decreasing and
at most ``max_jump``), the newest one is accepted as the new score. A lone
garbage read never forms such a run, and two identical misreads in a row are
corrected the same way once the true value is read consistently again.
"""
from __future__ import annotations

from typing import List, Optional

SCORE_OCR_MAX_JUMP = 20
SCORE_OCR_CONFIRM_READS = 3


class ScoreOcrFilter:
    """Per-alliance accept/reject state for scoreboard score reads."""

    def __init__(self, max_jump: int = SCORE_OCR_MAX_JUMP,
                 confirm_reads: int = SCORE_OCR_CONFIRM_READS):
        self.max_jump = max_jump
        self.confirm_reads = max(1, confirm_reads)
        self.last: Optional[int] = None
        self._pending: List[int] = []
        self.rejected = 0
        self.resyncs = 0

    def accept(self, value: Optional[int]) -> Optional[int]:
        """Feed one OCR read. Returns the value to adopt as the score, or None
        when the read is rejected (or missing). A None read leaves the state
        untouched, so unreadable frames neither break nor build a run."""
        if value is None:
            return None
        if self.last is None or abs(value - self.last) <= self.max_jump:
            self.last = value
            self._pending.clear()
            return value
        if self._pending and not (0 <= value - self._pending[-1] <= self.max_jump):
            self._pending.clear()
        self._pending.append(value)
        if len(self._pending) >= self.confirm_reads:
            self.last = value
            self._pending.clear()
            self.resyncs += 1
            return value
        self.rejected += 1
        return None
