"""2026 REBUILT hub shifts.

Teleop countdown 140 -> 0 (WPILib 2026 game-data reference implementation):
  > 130        TRANSITION  both hubs active
  130 .. 105   SHIFT 1     auto-fuel winner INACTIVE
  105 .. 80    SHIFT 2     auto-fuel winner active, other alliance inactive
  80  .. 55    SHIFT 3     auto-fuel winner INACTIVE
  55  .. 30    SHIFT 4     auto-fuel winner active, other alliance inactive
  <= 30        ENDGAME     both hubs active

"Auto-fuel winner" = the alliance that scored more FUEL in auto (TBA hub auto
totals). A tie is broken randomly by the FMS and is not in our data, so shift
activity is then UNKNOWN — never guessed.
"""
from __future__ import annotations

from typing import Optional

AUTO = "auto"
TRANSITION = "transition"
ACTIVE = "active"
INACTIVE = "inactive"
ENDGAME = "endgame"
UNKNOWN = "unknown"
BUCKETS = (AUTO, TRANSITION, ACTIVE, INACTIVE, ENDGAME, UNKNOWN)

# (name, remaining_hi, remaining_lo]: a teleop countdown value r falls in a
# shift when lo < r <= hi (transition: r > 130).
SHIFTS = (("transition", 140.0, 130.0), ("shift1", 130.0, 105.0),
          ("shift2", 105.0, 80.0), ("shift3", 80.0, 55.0),
          ("shift4", 55.0, 30.0), ("endgame", 30.0, -1.0))

# Physical order of driver stations along the alliance wall in ALLIANCE
# PERSPECTIVE y (own wall at x=0, y=0 = the alliance's right-hand side as the
# drivers face the field). Used only to DRAW station markers; the data is split
# by TBA station number regardless. TODO(verify against the 2026 manual).
STATION_Y_ORDER = {1: "high", 2: "mid", 3: "low"}


def teleop_shift(remaining: float) -> str:
    r = float(remaining)
    if r > 130.0:
        return "transition"
    for name, hi, lo in SHIFTS[1:]:
        if lo < r <= hi:
            return name
    return "endgame"


def auto_winner(hub_auto: dict) -> Optional[str]:
    red, blue = (hub_auto or {}).get("red"), (hub_auto or {}).get("blue")
    if red is None or blue is None or red == blue:
        return None
    return "red" if red > blue else "blue"


def bucket(period: str, remaining: Optional[float], alliance: str,
           hub_auto: dict) -> str:
    """Shift bucket for one robot sample of ``alliance``."""
    if period == "auto":
        return AUTO
    shift = teleop_shift(remaining if remaining is not None else 0.0)
    if shift == "transition":
        return TRANSITION
    if shift == "endgame":
        return ENDGAME
    winner = auto_winner(hub_auto)
    if winner is None:
        return UNKNOWN
    winner_inactive = shift in ("shift1", "shift3")
    is_winner = alliance == winner
    return INACTIVE if is_winner == winner_inactive else ACTIVE
