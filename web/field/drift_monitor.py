"""Mid-match calibration drift monitor (pure bookkeeping; no cv2).

The match-start check (app._verify_calibration_alignment) catches a camera
that moved BEFORE the match; a bump at t=60s silently poisons everything
after it. This class throttles tag sampling to a low cadence and requires
the drift to be sustained (median over a window) so one occluded/misread
sample — a robot parked on a tag — cannot trip it."""
from __future__ import annotations
from statistics import median


class DriftMonitor:
    def __init__(self, interval_s: float, max_m: float,
                 min_tags: int = 2, window: int = 3):
        self.interval_s = float(interval_s)
        self.max_m = float(max_m)
        self.min_tags = int(min_tags)
        self.window = int(window)
        self._last_sample_t: float | None = None
        self._errs: list[float] = []

    def due(self, video_t: float) -> bool:
        return (self._last_sample_t is None
                or video_t - self._last_sample_t >= self.interval_s)

    def record(self, video_t: float, err_m: float, n_tags: int) -> None:
        self._last_sample_t = float(video_t)
        if n_tags >= self.min_tags:
            self._errs.append(float(err_m))
            if len(self._errs) > self.window:
                self._errs.pop(0)

    @property
    def drifted(self) -> bool:
        if len(self._errs) < self.window:
            return False
        return median(self._errs) > self.max_m
