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

    def observe_frame(self, frame: int, overlapped_person: bool) -> None:
        self.frames_seen.append(int(frame))
        if overlapped_person:
            self.person_overlap_frames += 1

    def person_overlap_frac(self) -> float:
        n = len(self.frames_seen)
        return (self.person_overlap_frames / n) if n else 0.0

    def _mark_confident(self, frame: int) -> None:
        if self.first_confident_frame is None or frame < self.first_confident_frame:
            self.first_confident_frame = int(frame)

    def confirm(self, frame: int, *, team_read: bool = False,
                appearance: bool = False, bumper_color: bool = False) -> None:
        # team_read / appearance are strong robot evidence -> set first_confident_frame.
        # bumper_color is weak (red/blue field structure fires it) -> recorded only.
        if team_read:
            self.ever_team_read = True
            self._mark_confident(frame)
        if appearance:
            self.ever_appearance_match = True
            self._mark_confident(frame)
        if bumper_color:
            self.ever_bumper_color = True

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


class MatchEvidence:
    """Per-job registry of TrackEvidence keyed by fused track_id."""

    def __init__(self) -> None:
        self.by_track: dict = {}

    def _track(self, track_id: int) -> TrackEvidence:
        te = self.by_track.get(track_id)
        if te is None:
            te = TrackEvidence(track_id=int(track_id))
            self.by_track[track_id] = te
        return te

    def observe(self, track_id, frame, field_xy, *, overlapped_person) -> None:
        te = self._track(track_id)
        te.observe_frame(frame, overlapped_person)
        if field_xy is not None:
            te.add_position(frame, field_xy[0], field_xy[1])

    def confirm(self, track_id, frame, *, team_read=False, appearance=False,
                bumper_color=False) -> None:
        self._track(track_id).confirm(
            frame, team_read=team_read, appearance=appearance,
            bumper_color=bumper_color)
