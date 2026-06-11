"""Match-order parsing and calibration-epoch selection (pure, no IO).

Match order is the FRC convention parsed from the match key — qual before playoff,
then ascending set/match number — NOT wall-clock time. Calibrations are anchored to
a start match and apply forward in this order until the next epoch.
"""
from __future__ import annotations
import re

# comp_level rank (lower = earlier in the event)
_LEVEL_RANK = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
_KEY_RE = re.compile(r".*_(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$")


def match_sort_key(match_key: str) -> tuple:
    """Sortable (level_rank, set_number, match_number) for an FRC match key like
    '2026hop_qm72' or '2026hop_sf1m1'. Unknown/garbage keys sort last."""
    m = _KEY_RE.match(match_key or "")
    if not m:
        return (99, 0, 0)
    level, n1, n2 = m.group(1), int(m.group(2)), m.group(3)
    if n2 is None:                      # quals: '_qm72' -> n1 is the match number
        return (_LEVEL_RANK[level], 0, n1)
    return (_LEVEL_RANK[level], n1, int(n2))   # playoffs: setN matchM


def select_epoch(versions, match_key):
    """Return the calibration version whose epoch covers ``match_key``.

    versions: list of dicts with at least 'start_match' (a match key or None) and
    'version_id'; 'created_at' used as a tiebreak / for the None-key fallback.
    Rule: pick the greatest start_match that sorts <= match_key. A None start_match
    (legacy) sorts before all matches (covers from the start). If every version
    starts AFTER match_key, the earliest still wins (earliest extends back to the
    start). Returns None only if versions is empty. If match_key is None, return the
    most recently created version.
    """
    if not versions:
        return None
    if match_key is None:
        return max(versions, key=lambda v: v.get("created_at") or "")
    target = match_sort_key(match_key)

    def start_key(v):
        sm = v.get("start_match")
        return (-1, 0, 0) if sm is None else match_sort_key(sm)

    covering = [v for v in versions if start_key(v) <= target]
    pool = covering if covering else versions   # else: earliest extends backward
    return max(pool, key=start_key)
