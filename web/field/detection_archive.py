"""Per-boundary detection archive.

Every re-prompt boundary's YOLO detections (all views) are currently computed
and thrown away after seeding/corroboration. Persisting them (full-res view
coords, color, corrob-only flag) enables detection-vs-track agreement
telemetry, detection-gated VLM reads, retro-seed analysis, and hard-negative
mining — at zero extra GPU cost. Writes are best-effort (OSError swallowed),
buffered, and the in-memory index serves same-job queries."""
from __future__ import annotations

import bisect
import json


def _ios_boxes(a, b) -> float:
    """Intersection over the smaller box's area (pixel boxes, xyxy)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    sa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    sb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    smaller = min(sa, sb)
    return inter / smaller if smaller > 0 else 0.0


class DetectionArchive:
    FLUSH_EVERY = 16

    def __init__(self, path: str | None):
        self._path = path
        self._fh = None
        self._pending = 0
        # view -> (sorted source_frames, parallel list of det lists)
        self._index: dict[str, tuple[list, list]] = {}

    def record(self, view, local_idx, source_frame, dets):
        entry = [
            {"box": [float(v) for v in box], "color": color,
             "corrob_only": bool(co)}
            for box, color, co in dets
        ]
        frames, rows = self._index.setdefault(view, ([], []))
        i = bisect.bisect(frames, int(source_frame))
        frames.insert(i, int(source_frame))
        rows.insert(i, entry)
        if self._path is None:
            return
        try:
            if self._fh is None:
                self._fh = open(self._path, "a", encoding="utf-8")
            self._fh.write(json.dumps({
                "view": view, "local_idx": int(local_idx),
                "source_frame": int(source_frame), "dets": entry,
            }) + "\n")
            self._pending += 1
            if self._pending >= self.FLUSH_EVERY:
                self._fh.flush()
                self._pending = 0
        except OSError:
            self._fh = None  # disk trouble: keep the in-memory index alive

    def corroborates(self, view, source_frame, box, max_gap_frames,
                     ios: float = 0.3):
        frames, rows = self._index.get(view, ([], []))
        if not frames:
            return None
        i = bisect.bisect(frames, int(source_frame))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(frames) and abs(frames[j] - source_frame) <= max_gap_frames:
                if best is None or abs(frames[j] - source_frame) < abs(frames[best] - source_frame):
                    best = j
        if best is None:
            return None
        return any(_ios_boxes(d["box"], box) >= ios for d in rows[best])

    def close(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None
