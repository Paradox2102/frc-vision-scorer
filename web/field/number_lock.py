"""Consecutive-agreement number lock logic for FRC robot identity tracking.

A team number is (re)locked to a slot (track) after N consecutive consistent
reads (resets on disagreement). The lock is never permanent: a sustained run of
a *different* value overrides a prior lock so a wrong identity self-corrects.
"""
from __future__ import annotations

from typing import Optional


class ConsecutiveValueTracker:
    def __init__(self, n_consecutive: int = 3, n_override: int = 2) -> None:
        """Initialize tracker with consecutive-agreement thresholds.

        Args:
            n_consecutive: Consecutive consistent reads required to establish (or
                reaffirm) a lock from an unconfirmed state.
            n_override: Consecutive reads of a *different* value required to
                override an already-confirmed lock. Smaller than ``n_consecutive``
                so a wrong lock corrects quickly. Clamped to ``n_consecutive``.
        """
        self.n_consecutive = n_consecutive
        self.n_override = max(1, min(n_override, n_consecutive))
        self.current_value: Optional[str] = None
        self.consecutive_count: int = 0
        self.locked_value: Optional[str] = None
        self.confirmed: bool = False

    def update(self, value: Optional[str]) -> tuple[Optional[str], bool]:
        """Update tracker with a new value read.

        A None or blank read is ignored (does not increment or reset the count).
        A different non-None value resets the count to 1.

        A confirmed lock is *not* terminal: if a different value is read
        ``n_override`` times in a row, the lock flips to that value. This is what
        lets a wrong early lock (e.g. 8005) self-correct once the bumper is read
        consistently as the true team (e.g. 4174), instead of staying frozen until
        an end-of-match audit.

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

        if value_str == self.current_value:
            self.consecutive_count += 1
        else:
            self.current_value = value_str
            self.consecutive_count = 1

        # Overriding a confirmed lock with a different value needs only n_override
        # consecutive reads; establishing/reaffirming a lock needs n_consecutive.
        overriding = (
            self.confirmed
            and self.locked_value is not None
            and value_str != self.locked_value
        )
        threshold = self.n_override if overriding else self.n_consecutive

        if self.consecutive_count >= threshold:
            self.locked_value = value_str
            self.confirmed = True

        return self.locked_value, self.confirmed

    def unlock(self) -> None:
        """Unlock the confirmed status and reset the tracking state."""
        self.current_value = None
        self.consecutive_count = 0
        self.locked_value = None
        self.confirmed = False
