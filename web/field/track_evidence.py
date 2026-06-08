"""Per-track whole-match evidence (motion, confirmations, person overlap).

Pure data layer — no torch, no app imports — so it is unit-testable in isolation
and reusable by the classifier (slice 2) and backward pass (slice 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


@dataclass
class TrackEvidence:
    track_id: int
    frames_seen: list = field(default_factory=list)
    field_positions: list = field(default_factory=list)    # (frame, x, y) meters
    person_overlap_frames: int = 0
    ever_team_read: bool = False
    ever_appearance_match: bool = False
    ever_bumper_color: bool = False
    first_confident_frame: "int | None" = None

    def add_position(self, frame: int, x: float, y: float) -> None:
        self.field_positions.append((int(frame), float(x), float(y)))

    def net_displacement_m(self, window: int = 5) -> float:
        """Robust whole-match displacement: median (x, y) over consecutive windows
        of `window` samples, then the max pairwise distance between window medians.
        Per-window median rejects frame jitter and single outliers; max-pairwise
        catches movement anywhere in the match (even out-and-back). 0.0 if fewer
        than two full windows of positioned samples."""
        pts = self.field_positions
        if len(pts) < 2 * window:
            return 0.0
        centers = []
        for i in range(0, len(pts) - window + 1, window):
            chunk = pts[i:i + window]
            centers.append((_median([p[1] for p in chunk]),
                            _median([p[2] for p in chunk])))
        best = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = ((centers[i][0] - centers[j][0]) ** 2 +
                     (centers[i][1] - centers[j][1]) ** 2) ** 0.5
                if d > best:
                    best = d
        return best
