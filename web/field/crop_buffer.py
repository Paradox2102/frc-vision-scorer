"""Per-identity rolling buffer of raw bumper-band crops for multi-frame reads.

A Clef request accepts up to 4 images. On the labelled tracklet bench, sending
the newest crop plus the sharpest earlier crops of the SAME robot in one
request lifted clef-flash from 31/48 to 39-41/48 legible bumpers with 0 wrong
reads. This module keeps, per identity, the last ``window_s`` seconds of
source time of band crops (every processed frame, every camera view that saw
the robot) and picks which earlier frames to send next to the newest one.

Entries are RAW band crops (pre-upscale, independent ``.copy()``s of the view
frame), never lazy thunks, so the buffer never pins a full view frame.

Selection (``select``):
  * the newest entry is the crop at the read's own frame (the caller's
    single-crop choice; ``newest_cam`` picks the camera) — no entry there
    means the buffer does not vouch for that identity at that frame and the
    caller falls back to a single-crop read;
  * the earlier part of the window ``[newest - window, newest)`` is cut into
    ``n - 1`` equal time segments and the best fresh entry of each is taken;
    empty segments (e.g. right after a flush) are backfilled with the
    next-best fresh frames; at most one entry per source frame;
  * "fresh" = not already SENT in an earlier read of this identity, and with
    a non-zero legibility score (zero = no bumper colour in the crop);
  * "best": ``legibility_score`` is only comparable within ONE camera (it is
    computed pre-upscale on that camera's pixels), so each entry is ranked
    against its own camera's entries in the window (fraction of that camera's
    entries scoring <= it; 0 for a zero score). Ties — common, the sharpness
    term clamps at 1.0 — break on crop area, then recency.
Results are newest-first.

Identity continuity is the caller's job: whenever the physical object behind
an identity may have changed (binding events, rebinds, fused-id relinks...)
the caller ``flush``es it. The buffer itself flushes on a field-position jump
larger than ``max_jump_m`` between consecutive pushes.

Pure: numpy only (no cv2 / torch / app imports).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class CropEntry:
    frame: int                       # source-video frame the crop was cut from
    cam: str                         # camera view name ("main", "corner_left", ...)
    crop: np.ndarray                 # raw BGR band crop (independent copy)
    legibility: float                # legibility_score(crop), pre-upscale
    pos: Optional[tuple]             # fused field position (m) at that frame
    slot: Optional[int]              # tracker slot the identity held at that frame
    area: int                        # crop pixel area (tie-break)


class CropBuffer:
    """Rolling per-identity crop buffer. Not thread-safe: the app mutates it
    from the event loop only (push in the frame loop, select/mark/flush in the
    read pass before it hands off to a worker thread)."""

    def __init__(self, fps: float, window_s: float = 3.0, max_frames: int = 4,
                 max_entries: int = 96, max_jump_m: Optional[float] = None):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.window_s = float(window_s)
        self.window_frames = self.window_s * self.fps
        self.max_frames = max(1, int(max_frames))
        self.max_entries = max(1, int(max_entries))
        self.max_jump_m = None if max_jump_m is None else float(max_jump_m)
        self._buf: dict[int, list[CropEntry]] = {}
        self._sent: dict[int, set[int]] = {}
        self._flushes: dict[str, int] = defaultdict(int)
        self._pushes = 0
        self._entries_pushed = 0

    # ---- mutation ----

    def push(self, identity_id, frame: int, crops: Iterable, *, pos=None,
             slot: Optional[int] = None) -> int:
        """Add this frame's crops of one identity: ``crops`` is an iterable of
        ``(cam, crop_ndarray, legibility)``. Each crop is copied. Returns the
        number of entries added."""
        frame = int(frame)
        ents = self._buf.get(identity_id)
        if ents:
            last = ents[-1]
            if frame < last.frame:
                # Time went backwards (a new processing range): nothing older
                # is comparable any more.
                self.flush(identity_id, "time_reversal")
                ents = None
        if ents:
            self._evict(identity_id, frame)
            ents = self._buf.get(identity_id)
        if ents and self.max_jump_m is not None and pos is not None:
            prev = ents[-1].pos
            if prev is not None and math.hypot(float(pos[0]) - float(prev[0]),
                                               float(pos[1]) - float(prev[1])) > self.max_jump_m:
                self.flush(identity_id, "jump")
        p = None if pos is None else (float(pos[0]), float(pos[1]))
        added = 0
        for cam, crop, leg in crops:
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            c = np.array(crop, copy=True)
            self._buf.setdefault(identity_id, []).append(CropEntry(
                frame=frame, cam=str(cam), crop=c, legibility=float(leg or 0.0),
                pos=p, slot=slot, area=int(c.shape[0] * c.shape[1])))
            added += 1
        if added:
            ents = self._buf[identity_id]
            if len(ents) > self.max_entries:
                del ents[:len(ents) - self.max_entries]
            self._pushes += 1
            self._entries_pushed += added
        return added

    def _evict(self, identity_id, newest_frame: int) -> None:
        lo = newest_frame - self.window_frames
        ents = self._buf.get(identity_id)
        if ents:
            keep = [e for e in ents if e.frame >= lo]
            if keep:
                self._buf[identity_id] = keep
            else:
                self._buf.pop(identity_id, None)
        sent = self._sent.get(identity_id)
        if sent:
            sent.difference_update({f for f in sent if f < lo})

    def flush(self, identity_id, reason: str = "flush") -> bool:
        """Forget an identity's crops and sent-frame memory. Returns True when
        there was something to forget (only those flushes are counted)."""
        had = bool(self._buf.pop(identity_id, None))
        self._sent.pop(identity_id, None)
        if had:
            self._flushes[str(reason)] += 1
        return had

    def retain(self, live_ids, reason: str = "evicted") -> int:
        """Flush every identity not in ``live_ids``; returns how many."""
        live = set(live_ids)
        dead = [i for i in list(self._buf) if i not in live]
        for i in dead:
            self.flush(i, reason)
        for i in [i for i in list(self._sent) if i not in live]:
            self._sent.pop(i, None)
        return len(dead)

    def mark_sent(self, identity_id, frames: Iterable[int]) -> None:
        self._sent.setdefault(identity_id, set()).update(int(f) for f in frames)

    # ---- queries ----

    def entries(self, identity_id) -> list[CropEntry]:
        return list(self._buf.get(identity_id) or [])

    def sent_frames(self, identity_id) -> set:
        return set(self._sent.get(identity_id) or ())

    def select(self, identity_id, newest_frame: int, *, newest_cam: Optional[str] = None,
               slot: Optional[int] = None, n: Optional[int] = None) -> list[CropEntry]:
        """Newest-first entries for one multi-frame read (see module doc).

        Returns [] when the buffer holds no entry of this identity at
        ``newest_frame`` (for ``slot`` when given); a 1-element list means no
        fresh earlier frame is available."""
        ents = self._buf.get(identity_id)
        if not ents:
            return []
        newest_frame = int(newest_frame)
        n = self.max_frames if n is None else max(1, int(n))
        lo = newest_frame - self.window_frames
        window = [e for e in ents if lo <= e.frame <= newest_frame
                  and (slot is None or e.slot == slot)]
        at_newest = [e for e in window if e.frame == newest_frame]
        if not at_newest:
            return []
        by_cam: dict[str, list[float]] = defaultdict(list)
        for e in window:
            by_cam[e.cam].append(e.legibility)

        def rank(e: CropEntry) -> float:
            if e.legibility <= 0.0:
                return 0.0
            vals = by_cam[e.cam]
            return sum(1 for v in vals if v <= e.legibility) / len(vals)

        def key(e: CropEntry):
            return (rank(e), e.area, e.frame)

        pref = [e for e in at_newest if e.cam == newest_cam] if newest_cam else []
        newest = max(pref or at_newest, key=key)
        k = n - 1
        if k <= 0:
            return [newest]
        sent = self._sent.get(identity_id) or set()
        # A zero score means no bumper colour at all (or a sub-MIN_CROP_PX
        # crop): it adds nothing to a read and could only trip same_robot.
        pool = [e for e in window if e.frame < newest_frame and e.frame not in sent
                and e.legibility > 0.0]
        if not pool:
            return [newest]
        seg_len = (newest_frame - lo) / k
        segs: dict[int, list[CropEntry]] = defaultdict(list)
        for e in pool:
            segs[min(k - 1, max(0, int((e.frame - lo) / seg_len)))].append(e)
        picks: list[CropEntry] = []
        used: set[int] = set()
        for s in range(k):
            if segs.get(s):
                best = max(segs[s], key=key)
                picks.append(best)
                used.add(best.frame)
        if len(picks) < k:
            for e in sorted(pool, key=key, reverse=True):
                if len(picks) >= k:
                    break
                if e.frame in used:
                    continue
                picks.append(e)
                used.add(e.frame)
        picks.sort(key=lambda e: e.frame, reverse=True)
        return [newest] + picks

    def stats(self) -> dict:
        return {
            "identities": len(self._buf),
            "entries": sum(len(v) for v in self._buf.values()),
            "pushes": self._pushes,
            "entries_pushed": self._entries_pushed,
            "flushes": dict(self._flushes),
        }
