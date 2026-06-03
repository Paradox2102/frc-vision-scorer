"""Records the composited HUD frame stream + per-frame metadata for match replay.

Knows nothing about FastAPI, jobs, or the database: it consumes composited frames
and plain-dict metadata records and produces hud.mp4 + meta.json + thumb.jpg in a
given output directory.  The caller owns DB insertion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ReplayRecorder:
    def __init__(self) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._out_dir: Optional[Path] = None
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.codec: str = ""
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._mid_frame: Optional[np.ndarray] = None

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def open_lazy(self, first_frame: np.ndarray, fps: float, out_dir) -> None:
        if self._writer is not None:
            return
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 8.0
        self.height, self.width = first_frame.shape[:2]
        path = str(self._out_dir / "hud.mp4")
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height))
            if writer.isOpened():
                self._writer = writer
                self.codec = codec
                return
            writer.release()
        raise RuntimeError("Could not open a VideoWriter with avc1 or mp4v")

    def add_frame(self, frame: np.ndarray, record: dict) -> None:
        if self._writer is None:
            raise RuntimeError("add_frame called before open_lazy")
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)
        i = len(self.frames)
        rec = dict(record)
        rec["i"] = i
        rec["t"] = round(i / self.fps, 4)
        self.frames.append(rec)
        if self._mid_frame is None or i == len(self.frames) // 2:
            self._mid_frame = frame.copy()

    def add_event(self, event: dict) -> None:
        self.events.append(dict(event))

    def finalize(self, status: str, summary: dict) -> dict:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        frame_count = len(self.frames)
        duration = round(frame_count / self.fps, 4) if self.fps else 0.0

        thumb = self._mid_frame
        if thumb is None:
            thumb = np.zeros((max(1, self.height), max(1, self.width), 3), np.uint8)
        cv2.imwrite(str(self._out_dir / "thumb.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

        self._stamp_event_times()  # Task 3 implements this; no-op for now

        full_summary = dict(summary)
        full_summary.update({
            "fps": self.fps, "frame_count": frame_count, "duration_sec": duration,
        })
        meta = {"summary": full_summary, "frames": self.frames, "events": self.events}
        (self._out_dir / "meta.json").write_text(json.dumps(meta))

        return {
            "video_path": str(self._out_dir / "hud.mp4"),
            "meta_path": str(self._out_dir / "meta.json"),
            "thumb_path": str(self._out_dir / "thumb.jpg"),
            "fps": self.fps, "frame_count": frame_count, "duration_sec": duration,
            "width": self.width, "height": self.height,
            "codec": self.codec, "status": status,
        }

    def _stamp_event_times(self) -> None:
        """Map each event's source frame_num to the recorded-frame index, then t=i/fps.

        Uses the largest recorded frame whose frame_num <= event frame_num (the frame
        on screen when the event fired). Falls back to index 0 if none precede it.
        """
        # (recorded_index, frame_num) pairs in recorded order (ascending by frame_num)
        indexed = [(f["i"], f.get("frame_num")) for f in self.frames if f.get("frame_num") is not None]
        for ev in self.events:
            fn = ev.get("frame_num")
            idx = 0
            if fn is not None and indexed:
                for i, frame_num in indexed:
                    if frame_num <= fn:
                        idx = i
                    else:
                        break
            ev["t"] = round(idx / self.fps, 4) if self.fps else 0.0
