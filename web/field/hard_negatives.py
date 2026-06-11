"""Persist frames where the robot/bumper detector fired on likely non-robots, so
they can be folded back into training as hard negatives (background images).

Disabled by default; enable with HARD_NEG_LOG=1. The detection hot path stays
I/O-free — `to_field_detections` only *collects* rejected boxes into a list, and
this logger owns all disk writes, rate-limiting, gating, and disk-budget capping.
Writes are best-effort: any OSError is swallowed so logging never breaks inference.

For each kept frame it writes three files under `out_dir`:
  <view>_<frame>.jpg          raw frame, ready to add to a YOLO dataset
  <view>_<frame>.json         manifest: accepted boxes + rejected boxes w/ reasons
  <view>_<frame>_preview.jpg  annotated (accepted=green, rejected=red) for review

The raw frame is the training artifact: label the real robots (the manifest's
`accepted` boxes are model pseudo-labels to start from) and leave the clutter
unlabeled — that turns each fired-on confuser into a background negative.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


class HardNegativeLogger:
    def __init__(self, out_dir, *, enabled=None, log_every=None, max_files=None):
        self.out_dir = Path(out_dir)
        self.enabled = _env_flag("HARD_NEG_LOG", False) if enabled is None else enabled
        self.log_every = (int(os.environ.get("HARD_NEG_LOG_EVERY", "30"))
                          if log_every is None else log_every)
        self.max_files = (int(os.environ.get("HARD_NEG_MAX", "2000"))
                          if max_files is None else max_files)
        self._written = 0
        self._last_frame = None

    def _should_log(self, accepted_boxes, rejected, frame_num, require_rejected) -> bool:
        has_content = bool(rejected) or (not require_rejected and bool(accepted_boxes))
        if not self.enabled or not has_content:
            return False
        if self._written >= self.max_files:
            return False
        if (self._last_frame is not None and frame_num is not None
                and (frame_num - self._last_frame) < self.log_every):
            return False
        return True

    def maybe_log(self, frame, accepted_boxes, rejected, *,
                  view_name="main", frame_num=0, require_rejected=True):
        """Persist this frame if it passes the gate/rate-limit/cap. By default only
        frames with rejected detections are kept; pass require_rejected=False (review
        mode) to also keep frames whose accepted detections need human FP triage."""
        if not self._should_log(accepted_boxes, rejected, frame_num, require_rejected):
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{view_name}_{int(frame_num):08d}"
            img_path = self.out_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)

            preview = frame.copy()
            for b in accepted_boxes:
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for r in rejected:
                x1, y1, x2, y2 = (int(v) for v in r["box"])
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(preview, r.get("reason", "?"), (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(self.out_dir / f"{stem}_preview.jpg"), preview)

            manifest = {
                "view": view_name,
                "frame_num": int(frame_num),
                "frame_shape": [int(v) for v in frame.shape],
                "accepted": [[float(v) for v in b] for b in accepted_boxes],
                "rejected": [
                    {"box": [float(v) for v in r["box"]],
                     "reason": r.get("reason"),
                     "field_xy": (list(r["field_xy"]) if r.get("field_xy") else None)}
                    for r in rejected
                ],
            }
            (self.out_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2))

            self._written += 1
            self._last_frame = frame_num
            return img_path
        except OSError:
            return None
