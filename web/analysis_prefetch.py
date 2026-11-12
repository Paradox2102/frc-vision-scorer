"""Analysis-loop prefetch for the ``stream_analysis`` knob.

The analysis loop can't start before SAM propagation ends: late-seed backfill
(default on) and the tracker's retroactive color backfill mutate a view's
results until that view finishes, so no frame of the last view is final
before then. What CAN overlap is the loop's per-frame work that never reads
SAM output:

* scoreboard OCR (``ocr_score``: a tesseract subprocess per alliance on every
  OCR_FRAME_INTERVAL-th processed frame, CPU only, ~30 s/match);
* the main 3-class YOLO pass on the overhead crop (``loop.yolo_main``);
* the per-view person detector (``loop.person_det``).

``AnalysisPrefetcher.run`` (a worker thread, started before the per-view
propagation) decodes the analysis frames with its own capture, exactly as the
loop will, computes those results and parks them in per-frame caches. The
loop ``take_*``s each one; anything not ready (or never computed) returns
``MISS`` and the loop computes it inline exactly as before. GPU work uses
private model instances built by the caller's ``setup_gpu`` inside the worker
thread — ultralytics predictors are not thread-safe and the SAM thread's
detect_fn shares the job's ``person_model``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event
from typing import Callable, Iterable, Iterator, Optional

MISS = object()  # cache-miss sentinel (None is a legitimate OCR result)

OCR_ALLIANCES = ("blue", "red")


def ocr_frames_for(analysis_frames: Iterable[int], interval: int,
                   min_index: int = 10) -> set[int]:
    """Source frames the analysis loop OCRs: it runs ``ocr_score`` when
    ``processed_frame_num >= min_index and processed_frame_num % interval ==
    0``, where ``processed_frame_num`` is the frame's index in the sorted
    analysis-frame list."""
    interval = max(1, int(interval))
    return {
        f for i, f in enumerate(sorted(analysis_frames))
        if i >= min_index and i % interval == 0
    }


@dataclass
class StreamPlan:
    """Which prefetch parts run for a job; ``notes`` explains every part that
    was switched off (logged at job start)."""
    ocr: bool = False
    gpu: bool = False
    notes: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.ocr or self.gpu


def stream_analysis_plan(*, enabled: bool, use_sam2: bool, device: str,
                         needs_vllm: bool, vram_fit: bool) -> StreamPlan:
    """Resolve the knob against the job: OFF -> nothing at all. No SAM pass
    -> nothing to overlap (sequential path). GPU detectors only on CUDA, and
    not when the vLLM reader must reclaim the card after tracking or the
    VRAM-fit mode has already sized SAM to the whole budget."""
    plan = StreamPlan()
    if not enabled:
        return plan
    if not use_sam2:
        plan.notes.append("no SAM pass to overlap -> sequential")
        return plan
    plan.ocr = True
    plan.gpu = True
    if device != "cuda":
        plan.gpu = False
        plan.notes.append(f"device={device} -> OCR prefetch only")
    if needs_vllm:
        plan.gpu = False
        plan.notes.append("vLLM reader reclaims VRAM after tracking -> OCR prefetch only")
    if vram_fit:
        plan.gpu = False
        plan.notes.append("sam2_max_gpu_frames VRAM-fit -> OCR prefetch only")
    return plan


class AnalysisPrefetcher:
    """Precompute SAM-independent loop work for a list of analysis frames.

    Args:
        read_frames: zero-arg callable returning an iterator of
            ``(source_frame, bgr_frame)`` for the analysis frames in order
            (decoded exactly as the loop decodes them).
        overhead_height: rows of the composite frame the loop calls the
            overhead crop (``frame[:overhead_height]``).
        views: ``[(view_name, extract_fn)]`` or ``[(view_name, extract_fn,
            active_until_frame)]`` in loop order; a view is skipped from its
            ``active_until_frame`` on (None: never), as the loop skips it.
        ocr_frames: source frames the loop will OCR (``ocr_frames_for``).
        ocr_fn: ``fn(overhead_frame, alliance) -> int | None`` (thread-safe).
        setup_gpu: optional zero-arg callable run in the worker thread,
            returning ``(main_det_fn, person_fn)``: ``main_det_fn(overhead)``
            -> the loop's ``detections``; ``person_fn(view_frame)`` -> list of
            person boxes. ``None`` (or a failing setup) = OCR only.
        should_stop: optional poll (the job's STOP flag).
    """

    def __init__(self, read_frames: Callable[[], Iterator], *,
                 overhead_height: int, views: list, ocr_frames: set,
                 ocr_fn: Optional[Callable] = None,
                 setup_gpu: Optional[Callable] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 log: Callable[[str], None] = print):
        self._read_frames = read_frames
        self._overhead_height = overhead_height
        self._views = list(views)
        self._ocr_frames = set(ocr_frames) if ocr_fn is not None else set()
        self._ocr_fn = ocr_fn
        self._setup_gpu = setup_gpu
        self._should_stop = should_stop
        self._log = log
        self._cancel = Event()
        self._loop_frame = -1
        self._main: dict = {}
        self._person: dict = {}
        self._ocr: dict = {}
        self.gpu_active = False
        self.done = False
        self.frames_done = 0
        self.wall_s = 0.0
        self.error: Optional[str] = None
        self.hits = {"main": 0, "person": 0, "ocr": 0}
        self.misses = {"main": 0, "person": 0, "ocr": 0}

    # -- loop side --------------------------------------------------------
    def note_loop_frame(self, frame_num: int) -> None:
        """The loop is processing ``frame_num``: the worker skips it and
        everything before it (the loop computes those inline)."""
        self._loop_frame = frame_num

    def _take(self, cache: dict, key, kind: str):
        val = cache.pop(key, MISS)
        if val is MISS:
            self.misses[kind] += 1
        else:
            self.hits[kind] += 1
        return val

    def take_main(self, frame_num: int):
        return self._take(self._main, frame_num, "main")

    def take_person(self, frame_num: int, view_name: str):
        return self._take(self._person, (frame_num, view_name), "person")

    def take_ocr(self, frame_num: int, alliance: str):
        return self._take(self._ocr, (frame_num, alliance), "ocr")

    def cancel(self) -> None:
        self._cancel.set()

    def clear(self) -> None:
        self._main.clear()
        self._person.clear()
        self._ocr.clear()

    def stats(self) -> dict:
        """Flat numeric timings keys (``stream.*``) for the replay."""
        out = {"stream.prefetch_s": round(self.wall_s, 2),
               "stream.prefetch_frames": self.frames_done}
        for kind in ("main", "person", "ocr"):
            out[f"stream.hit.{kind}"] = self.hits[kind]
            out[f"stream.miss.{kind}"] = self.misses[kind]
        return out

    # -- worker side ------------------------------------------------------
    def _stopping(self) -> bool:
        if self._cancel.is_set():
            return True
        if self._should_stop is not None:
            try:
                return bool(self._should_stop())
            except Exception:
                return False
        return False

    def run(self) -> None:
        """Blocking; run on a worker thread. Never raises: any failure just
        leaves the rest of the frames to the loop's inline path."""
        t0 = time.perf_counter()
        main_det_fn = person_fn = None
        try:
            if self._setup_gpu is not None:
                try:
                    main_det_fn, person_fn = self._setup_gpu()
                    self.gpu_active = True
                except Exception as e:
                    self._log(f"  [stream] GPU prefetch setup failed ({e}); OCR only")
            if main_det_fn is None and person_fn is None and not self._ocr_frames:
                return
            for frame_num, frame in self._read_frames():
                if self._stopping():
                    break
                if frame_num <= self._loop_frame:
                    continue  # the loop got here first
                overhead = frame[:self._overhead_height]
                if main_det_fn is not None:
                    self._main[frame_num] = main_det_fn(overhead)
                if person_fn is not None:
                    for name, extract, *until in self._views:
                        if until and until[0] is not None and frame_num >= until[0]:
                            continue
                        self._person[(frame_num, name)] = person_fn(extract(frame))
                if frame_num in self._ocr_frames:
                    for alliance in OCR_ALLIANCES:
                        self._ocr[(frame_num, alliance)] = self._ocr_fn(
                            overhead, alliance)
                self.frames_done += 1
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self._log(f"  [stream] prefetch stopped early ({self.error}); "
                      "the loop computes the remaining frames inline")
        finally:
            self.wall_s = time.perf_counter() - t0
            self.done = True
