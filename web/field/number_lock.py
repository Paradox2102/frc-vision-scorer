"""Consecutive-agreement number lock logic for FRC robot identity tracking.

A stable team number is locked to a slot (track) only after N consecutive
consistent reads (resets on disagreement). A direct contradiction unlock
recovers from wrong locks.
"""
from __future__ import annotations

from typing import Optional


class ConsecutiveValueTracker:
    def __init__(self, n_consecutive: int = 3) -> None:
        """Initialize tracker with a target consecutive agreement threshold.

        Args:
            n_consecutive: Number of consecutive consistent reads required to lock.
        """
        self.n_consecutive = n_consecutive
        self.current_value: Optional[str] = None
        self.consecutive_count: int = 0
        self.locked_value: Optional[str] = None
        self.confirmed: bool = False

    def update(self, value: Optional[str]) -> tuple[Optional[str], bool]:
        """Update tracker with a new value read.

        A None read is ignored (does not increment or reset the count).
        A different non-None value resets the count to 1.

        Args:
            value: The read team number value.

        Returns:
            Tuple of (locked_value, confirmed).
        """
        if value is None:
            return self.locked_value, self.confirmed

        value_str = str(value).strip()
        if not value_str:
            return self.locked_value, self.confirmed

        if self.confirmed:
            # Once confirmed, it stays confirmed unless unlocked via contradiction
            return self.locked_value, self.confirmed

        if value_str == self.current_value:
            self.consecutive_count += 1
        else:
            self.current_value = value_str
            self.consecutive_count = 1

        if self.consecutive_count >= self.n_consecutive:
            self.locked_value = value_str
            self.confirmed = True

        return self.locked_value, self.confirmed

    def unlock(self) -> None:
        """Unlock the confirmed status and reset the tracking state."""
        self.current_value = None
        self.consecutive_count = 0
        self.locked_value = None
        self.confirmed = False
