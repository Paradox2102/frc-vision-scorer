"""Records the composited HUD frame stream + per-frame metadata for match replay.

Knows nothing about FastAPI, jobs, or the database: it consumes composited frames
and plain-dict metadata records and produces hud.mp4 + meta.json + thumb.jpg in a
given output directory.  The caller owns DB insertion.

``record_video=False`` (the hud_post_render knob) is metadata-only: no ffmpeg,
the same meta.json, a provisional thumb.jpg, and the per-frame draw list
(``draw.jsonl.gz``) that web/hud_render.py renders the video from later.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ReplayRecorder:
    """Encodes frames to a browser-playable H.264 mp4.

    Prefers an ``ffmpeg`` subprocess (libx264 / yuv420p / +faststart) so the
    result plays in any HTML5 ``<video>``.  OpenCV's ``VideoWriter`` is only a
    fallback: its ``avc1`` path commonly fails to open and silently degrades to
    ``mp4v`` (MPEG-4 Part 2), which browsers cannot decode.
    """

    def __init__(self, record_video: bool = True, video_name: str = "hud.mp4") -> None:
        self.record_video = bool(record_video)
        self.video_name = video_name
        self._meta_open = False          # record_video=False: opened, no encoder
        self._draw_fh = None             # draw.jsonl.gz, opened on the first draw line
        self._writer: Optional[cv2.VideoWriter] = None
        self._proc: Optional[subprocess.Popen] = None
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
        return self._writer is not None or self._proc is not None or self._meta_open

    def open_lazy(self, first_frame: np.ndarray, fps: float, out_dir) -> None:
        if self.is_open:
            return
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 8.0
        h, w = first_frame.shape[:2]
        # libx264 + yuv420p require even dimensions; lock to even and resize frames to match.
        self.height, self.width = h - (h % 2), w - (w % 2)
        if not self.record_video:
            self._meta_open = True
            return
        path = str(self._out_dir / self.video_name)

        if shutil.which("ffmpeg"):
            self._proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{self.width}x{self.height}", "-r", f"{self.fps}",
                    "-i", "-", "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "20",
                    "-movflags", "+faststart", path,
                ],
                stdin=subprocess.PIPE,
            )
            self.codec = "h264"
            return

        # Fallback: no ffmpeg on PATH — use OpenCV (may produce non-browser-playable mp4v).
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height))
            if writer.isOpened():
                self._writer = writer
                self.codec = codec
                return
            writer.release()
        raise RuntimeError("Could not open ffmpeg or an OpenCV VideoWriter (avc1/mp4v)")

    def add_frame(self, frame: np.ndarray, record: dict, draw: Optional[dict] = None) -> None:
        """Encode ``frame`` (unless metadata-only), append ``record`` to the
        meta frames and, when given, ``draw`` as this frame's draw-list line
        (its ``"i"`` is set to the replay frame index)."""
        if not self.is_open:
            raise RuntimeError("add_frame called before open_lazy")
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        if not self.record_video:
            pass
        elif self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, ValueError):
                pass  # ffmpeg died; analysis must not crash. finalize() will surface it.
        else:
            self._writer.write(frame)
        i = len(self.frames)
        rec = dict(record)
        rec["i"] = i
        rec["t"] = round(i / self.fps, 4)
        self.frames.append(rec)
        if self._mid_frame is None or i == len(self.frames) // 2:
            self._mid_frame = frame.copy()
        if draw is not None:
            if self._draw_fh is None:
                self._draw_fh = gzip.open(self._out_dir / "draw.jsonl.gz", "wt",
                                          encoding="utf-8")
            self._draw_fh.write(json.dumps({**draw, "i": i}, separators=(",", ":")) + "\n")

    def add_event(self, event: dict) -> None:
        self.events.append(dict(event))

    def close_video(self) -> None:
        """Finish the encoder (and the draw list); safe to call twice."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=180)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._draw_fh is not None:
            self._draw_fh.close()
            self._draw_fh = None

    def finalize(self, status: str, summary: dict, diagnostics: Optional[Any] = None) -> dict:
        self.close_video()
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
        if diagnostics is not None:
            diag_path = self._out_dir / "diagnostics.jsonl"
            diagnostics.flush_jsonl(str(diag_path))
            full_summary["diagnostics_path"] = str(diag_path)
            full_summary["diagnostics_summary"] = diagnostics.summary()

        meta = {"summary": full_summary, "frames": self.frames, "events": self.events}
        (self._out_dir / "meta.json").write_text(json.dumps(meta))

        res = {
            "video_path": str(self._out_dir / "hud.mp4"),
            "meta_path": str(self._out_dir / "meta.json"),
            "thumb_path": str(self._out_dir / "thumb.jpg"),
            "fps": self.fps, "frame_count": frame_count, "duration_sec": duration,
            "width": self.width, "height": self.height,
            "codec": self.codec if self.record_video else None, "status": status,
        }
        if diagnostics is not None:
            res["diagnostics_path"] = str(self._out_dir / "diagnostics.jsonl")
        return res

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
