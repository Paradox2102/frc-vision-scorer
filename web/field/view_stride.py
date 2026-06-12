"""Per-view frame decimation for the multi-camera tracking pass.

SAM2/SAM3 propagation cost scales with frame count per view (the model
resizes every frame to a fixed square, so export resolution is not a speed
lever). Corner cameras carry less identity/geometry signal than the main
broadcast view, so they can be tracked at a fraction of the main view's
frame rate: keep every ``stride``-th kept frame, phase-aligned so the SAM
prompt frame is always exported (seeding boxes are detected on that exact
frame and propagated backward from it).

The plan is computed up front from the shared kept-frame list so the export
loop, the tracker's ``detect_fn`` (local index -> source frame), and the
analysis loop (source frame -> local index) all agree on the mapping.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence


class ViewFramePlan(NamedTuple):
    """Frame bookkeeping for one camera view's tracking pass."""

    source_frames: List[int]        # local idx -> source video frame number
    source_to_local: Dict[int, int]  # source video frame number -> local idx
    prompt_idx: int                  # local idx of the SAM prompt frame


def build_view_frame_plan(
    all_frames: Sequence[int],
    prompt_local_idx: int,
    stride: int,
    dense_frames: int = 0,
) -> ViewFramePlan:
    """Decimate the shared kept-frame list for one view.

    ``all_frames`` is the pipeline-wide kept-frame list (already decimated to
    the target tracking FPS); ``prompt_local_idx`` indexes into it. The kept
    subset is phase-aligned to the prompt frame so it always survives
    decimation; frames before the first aligned index (at most ``stride - 1``
    kept frames at the very start) are dropped for this view.

    ``dense_frames`` exempts the first N kept indices from decimation: team
    locks form in the early part of a match (seeding + first legible bumper
    reads), and the corner-stride A/B showed full-time decimation starves
    confirmation. A dense head keeps the lock-formation window at full rate
    while the long tail pays the stride.
    """
    stride = max(1, int(stride))
    dense_frames = max(0, int(dense_frames))
    if not (0 <= prompt_local_idx < len(all_frames)):
        raise ValueError(
            f"prompt_local_idx {prompt_local_idx} outside kept-frame list "
            f"of length {len(all_frames)}")
    phase = prompt_local_idx % stride
    kept = [i for i in range(len(all_frames))
            if i < dense_frames or i % stride == phase]
    source_frames = [all_frames[i] for i in kept]
    source_to_local = {sf: vi for vi, sf in enumerate(source_frames)}
    return ViewFramePlan(
        source_frames=source_frames,
        source_to_local=source_to_local,
        prompt_idx=kept.index(prompt_local_idx),
    )


def scale_frame_knob(frames: int, stride: float) -> int:
    """Rescale a tracked-frame-denominated knob to a decimated view.

    Knobs like ``evict_dark_frames`` mean wall-clock durations ("dark for
    ~1s") expressed in tracked frames; a view tracked at 1/stride rate needs
    the threshold divided by stride to keep the same wall-clock meaning.
    ``stride`` may be fractional (the effective average of a mixed-rate
    timeline with a dense head).
    """
    return max(1, int(round(frames / max(1.0, float(stride)))))
