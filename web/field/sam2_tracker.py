"""SAM2 offline video-predictor tracker helpers and wrapper class.

The module is split into two tiers:

1. Pure, GPU-free helpers (``mask_to_box``, ``bottom_center``, ``_ios``,
   ``match_new_detections``) — importable with no torch/sam2 on the path.

2. ``Sam2FieldTracker`` — a stateful wrapper around SAM2's offline video
   predictor.  torch/sam2 are imported *lazily* inside the class so that
   the pure helpers remain torch-free.
"""
from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np

# Mask-shape plausibility knobs. A robot's SAM2 mask is a compact blob that
# stands on the carpet — roughly as wide as it is tall and widest near its
# base. Field fixtures (radio/antenna on a stand) segment as a blob-on-a-stick:
# tall, thin, and *top*-heavy. We only reject a mask when it is BOTH tall-thin
# AND top-heavy, so a legitimately tall-but-bottom-heavy robot (e.g. a raised
# climber hook) is never dropped.
MASK_MAX_ASPECT = float(os.environ.get("SAM2_MASK_MAX_ASPECT", "2.0"))
MASK_MIN_BASE_RATIO = float(os.environ.get("SAM2_MASK_MIN_BASE_RATIO", "0.35"))
MASK_BASE_BAND_FRAC = float(os.environ.get("SAM2_MASK_BASE_BAND_FRAC", "0.2"))

# Hijack-cull knobs (detector-absence + stationary). A SAM2 object whose mask has
# drifted onto a static field structure stays "visible" forever, so the existing
# absence-based retirement never fires. These drive a second retirement trigger.
# See docs/superpowers/specs/2026-06-08-sam2-mask-hijack-cull-design.md.
#
# DEFAULT ON since the 2026-06-10 qm72 gate: the corner_right trench mask was
# culled in ~12s (vs ~80s of slot pollution without), with identical scores and
# confirmed-team count vs the cull-off baseline — no false-cull signal. The
# main-view culls hit the same static junk eviction was already killing, just
# earlier (evictions 6 -> 0 because the cull preempts them).
CULL_HIJACK_ENABLED = os.environ.get("CULL_HIJACK_ENABLED", "1") not in ("0", "false", "False")
# Absences required *within the evidence window* before a stationary object is culled.
CULL_UNCORROBORATED_REPROMPTS = int(os.environ.get("CULL_UNCORROBORATED_REPROMPTS", "3"))
# Sliding-window length of corroboration outcomes kept per object. Must be >= K;
# the slack (W - K) is how many intermittent corroborations the cull tolerates
# without resetting suspicion.
CULL_EVIDENCE_WINDOW = int(os.environ.get("CULL_EVIDENCE_WINDOW", "5"))
# Min IoS of a fresh detection vs the mask to count as corroboration (lenient ->
# conservative culling).
CULL_CORROB_IOS = float(os.environ.get("CULL_CORROB_IOS", "0.3"))
# Max field-position span over the window to count as stationary.
CULL_STATIONARY_M = float(os.environ.get("CULL_STATIONARY_M", "0.5"))

# Capped-roster eviction. When the active roster is full and a fresh detection
# matches no tracked mask, the least-corroborated *stationary* active object may
# be evicted to make room. Uses the same evidence bar as the hijack cull
# (>= CULL_UNCORROBORATED_REPROMPTS absences in the window AND stationary), but
# is triggered on demand by concrete new evidence rather than spontaneously, so
# it is safe to default on while the spontaneous cull stays opt-in.
EVICT_WHEN_CAPPED = os.environ.get("SAM2_EVICT_WHEN_CAPPED", "1") not in ("0", "false", "False")

# A tracked mask may only claim (suppress) a fresh detection when its area is at
# least this fraction of the detection box's area. IoS-over-smaller scores 1.0
# for a degenerate sliver mask sitting inside a full-size detection, which let a
# collapsed mask block its own replacement.
SEED_MIN_MASK_AREA_FRAC = float(os.environ.get("SAM2_SEED_MIN_MASK_AREA_FRAC", "0.1"))

# Re-anchoring: a tracked mask that has collapsed to a sliver (area below
# SEED_MIN_MASK_AREA_FRAC of an overlapping fresh detection's box) is
# re-prompted under its SAME object id with that detection box, instead of the
# detection seeding a churned replacement id. Attacks propagation decay at the
# source (SAM2 re-conditions its memory on the fresh prompt) where the sliver
# guard / cull / eviction only treat the symptom. Color-conflicting detections
# never re-anchor. DEFAULT OFF pending A/B.
REANCHOR_ENABLED = os.environ.get("SAM2_REANCHOR_ENABLED", "0") not in ("0", "false", "False")
# Min IoS (over smaller) of the degraded mask vs the detection box to re-anchor.
# A sliver inside the box scores ~1.0; this floor exists so a sliver that has
# drifted AWAY from the detection (different robot) is not captured.
REANCHOR_IOS = float(os.environ.get("SAM2_REANCHOR_IOS", "0.5"))
# v2 guards (qm72 v1 failure: SAM2 never deletes retired objects, so a dead
# object's mask can flicker back as a noise sliver ANYWHERE and capture a live
# robot's detection — all 3 v1 re-anchors were such zombie identity transfers).
# Recency: the object must have had a stored mask within this many boundaries
# (a genuine collapse is fresh; a zombie has been sliver/absent for many).
REANCHOR_MAX_BOUNDARY_GAP = int(os.environ.get("SAM2_REANCHOR_MAX_BOUNDARY_GAP", "1"))
# Locality: the detection must be near the object's last recorded extent —
# field meters when a projector is available, else tracker pixels. Over one
# 2s boundary gap a fast robot can outrun 2 m; those churn (safe) by design.
REANCHOR_MAX_JUMP_M = float(os.environ.get("SAM2_REANCHOR_MAX_JUMP_M", "2.0"))
REANCHOR_MAX_JUMP_PX = float(os.environ.get("SAM2_REANCHOR_MAX_JUMP_PX", "100"))

# Color backfill: an object whose metadata color is unknown (seeded from a
# colorless whole-robot gap-fill) inherits the color of the bumper detections
# that corroborate its mask on this many CONSECUTIVE colored boundaries.
# Mirrors the team-lock n_override pattern: one read can be a neighbor's
# bumper clipping the mask; two agreeing in a row is deliberate evidence.
COLOR_BACKFILL_VOTES = int(os.environ.get("SAM2_COLOR_BACKFILL_VOTES", "2"))


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bounding box of nonzero pixels as (x1, y1, x2, y2).

    x2 and y2 are exclusive maxima (one past the last pixel), consistent with
    Python slice notation.  Returns None if the mask contains no True pixels.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def plausible_robot_mask(
    mask: np.ndarray,
    max_aspect: float = MASK_MAX_ASPECT,
    min_base_ratio: float = MASK_MIN_BASE_RATIO,
    base_band_frac: float = MASK_BASE_BAND_FRAC,
) -> bool:
    """Return False for a mask whose shape is geometrically impossible for a robot.

    A robot stands on the carpet: its mask is a solid blob, roughly as wide as
    it is tall, and widest near its base. The field-side radio/antenna segments
    as a blob-on-a-stick — tall, thin, and *top*-heavy (the wide part is up high,
    only a thin stalk reaches the ground).

    We reject only when BOTH hold, so that a tall-but-bottom-heavy robot (e.g. a
    raised climber hook) is preserved:

    1. The bounding box is tall and thin (height/width > ``max_aspect``).
    2. The mask is top-heavy: the widest row in its base band is a small
       fraction (< ``min_base_ratio``) of the widest row overall.
    """
    box = mask_to_box(mask)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return False
    if h / w <= max_aspect:
        return True  # not tall-and-thin -> shape is robot-plausible

    # Tall and thin: distinguish a bottom-heavy robot from a top-heavy stalk by
    # comparing the width of the base band against the widest row overall.
    row_widths = mask[y1:y2, x1:x2].sum(axis=1)
    widest = int(row_widths.max())
    if widest == 0:
        return False
    band = max(1, int(round(h * base_band_frac)))
    base_width = int(row_widths[-band:].max())
    return (base_width / widest) >= min_base_ratio


def bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the bottom-center point of a box as (cx, y_bottom).

    y2 is already the exclusive-max bottom from mask_to_box, so this is
    consistent with that convention.
    """
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, float(y2))


def _ios(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Smaller-area (IoS) between two boolean masks.

    Returns 0.0 if either mask is empty.
    """
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    smaller = min(area_a, area_b)
    if smaller == 0:
        return 0.0
    intersection = int((mask_a & mask_b).sum())
    return intersection / smaller


def match_new_detections(
    existing_masks: dict[int, np.ndarray],
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.5,
    existing_colors: dict[int, Any] | None = None,
    new_colors: list[Any] | None = None,
    min_mask_area_frac: float = SEED_MIN_MASK_AREA_FRAC,
) -> list[int]:
    """Return indices of new_boxes that are genuinely new robots (not already tracked).

    For each box in new_boxes the box is rasterised to a boolean mask of shape
    (height, width) and the max IoS against every existing mask is computed.
    Boxes whose best IoS is below ios_thresh are considered new and their
    indices (preserving input order) are returned.

    Args:
        existing_masks: mapping of object-id -> bool ndarray (h, w).
        new_boxes: list of (x1, y1, x2, y2) detections to test.
        wh: (width, height) of the frame.
        ios_thresh: minimum IoS to consider a detection already tracked.
        existing_colors: mapping of object-id -> color string.
        new_colors: list of color strings for each new box.
        min_mask_area_frac: an existing mask smaller than this fraction of a
            box's area cannot claim that box (a degenerate sliver scores IoS
            1.0 inside any box and would otherwise block its own replacement).

    Returns:
        List of integer indices into new_boxes for unmatched (new) robots.
    """
    w, h = wh
    mask_areas = {oid: int(m.sum()) for oid, m in existing_masks.items()}
    unmatched: list[int] = []
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        box_mask = np.zeros((h, w), bool)
        box_mask[y1:y2, x1:x2] = True
        box_area = int(box_mask.sum())
        best_ios = 0.0
        for oid, existing in existing_masks.items():
            if existing_colors and new_colors and idx < len(new_colors):
                e_col = existing_colors.get(oid)
                n_col = new_colors[idx]
                if e_col in {"red", "blue"} and n_col in {"red", "blue"} and e_col != n_col:
                    continue  # conflicting colors -> treat as non-overlapping
            if mask_areas[oid] < min_mask_area_frac * box_area:
                continue  # degenerate sliver mask cannot claim a full-size box
            score = _ios(existing, box_mask)
            if score > best_ios:
                best_ios = score
        if best_ios < ios_thresh:
            unmatched.append(idx)
    return unmatched


def mask_corroborated(
    mask: np.ndarray,
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.3,
    mask_color: Any | None = None,
    new_colors: list[Any] | None = None,
) -> bool:
    """Return True if any fresh detector box backs ``mask`` (IoS >= ios_thresh).

    The per-object complement of :func:`match_new_detections`. Used as the
    per-boundary corroboration test for the hijack cull. Color-conflicting
    detections are ignored (a blue box cannot corroborate a red mask).
    """
    if not new_boxes:
        return False
    w, h = wh
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        if mask_color in {"red", "blue"} and new_colors and idx < len(new_colors):
            n_col = new_colors[idx]
            if n_col in {"red", "blue"} and n_col != mask_color:
                continue
        box_mask = np.zeros((h, w), bool)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = True
        if _ios(mask, box_mask) >= ios_thresh:
            return True
    return False


def corroborating_colors(
    mask: np.ndarray,
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.3,
    new_colors: list[Any] | None = None,
) -> set:
    """Definite colors ('red'/'blue') of fresh detections corroborating ``mask``."""
    out: set = set()
    if not new_boxes or not new_colors:
        return out
    w, h = wh
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        if idx >= len(new_colors) or new_colors[idx] not in ("red", "blue"):
            continue
        box_mask = np.zeros((h, w), bool)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = True
        if _ios(mask, box_mask) >= ios_thresh:
            out.add(new_colors[idx])
    return out


def is_stationary(positions, max_span_m: float) -> bool:
    """Return True if all known field positions lie within ``max_span_m``.

    ``positions`` is a list of (x, y) field-meter points, possibly containing
    ``None`` (unprojectable box). ``None`` entries are ignored. With fewer than
    two known positions we cannot confirm stationarity, so we return False
    (conservative — refuse to cull when motion is unmeasurable).
    """
    pts = [p for p in positions if p is not None]
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    return span < max_span_m


def absent_enough(history, k: int) -> bool:
    """Return True if a bounded corroboration ``history`` shows >= ``k`` absences.

    ``history`` is an iterable of booleans over the last N *available* re-prompt
    boundaries (True = the mask was corroborated by a fresh detection that
    boundary, False = absent). Counting absences within a bounded window — rather
    than resetting a counter to zero on any single corroboration — means one
    intermittent overlap (a robot crossing the trench, or a whole-robot false
    positive on it) cannot indefinitely preserve a hijacked mask. The caller sizes
    the window (maxlen) with slack over ``k`` so a genuinely-present robot,
    corroborated on most boundaries, never reaches ``k`` absences.
    """
    absent = sum(1 for c in history if not c)
    return absent >= k


def pick_evictable(active_objects, corroboration_history, field_pos_window,
                   k: int, stationary_m: float) -> int | None:
    """Return the active object best justified for eviction, or None.

    An object is evictable only when its bounded corroboration history shows
    >= ``k`` detector absences AND its recent field positions are stationary —
    the same evidence bar as the spontaneous hijack cull, applied on demand
    when a fresh unmatched detection needs a roster slot. A moving object is
    never evictable (it could be a real robot in a detector blind spot), and
    neither is one without history yet. Among evictable objects the one with
    the most absences wins; ties go to the smaller (older) object id.
    """
    best = None
    best_absent = -1
    for oid in active_objects:
        hist = corroboration_history.get(oid)
        if not hist or not absent_enough(hist, k):
            continue
        if not is_stationary(field_pos_window.get(oid, []), stationary_m):
            continue
        absent = sum(1 for c in hist if not c)
        if absent > best_absent:
            best = oid
            best_absent = absent
    return best


# ---------------------------------------------------------------------------
# Stateful SAM2 wrapper
# ---------------------------------------------------------------------------

#: Default SAM2 config (relative to the project root, used by SAM2 itself).
_DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
#: Default SAM2 checkpoint path.
_DEFAULT_SAM2_CKPT = "checkpoints/sam2.1_hiera_large.pt"


@contextmanager
def _maybe_autocast(device: str):
    """Enter a torch.autocast context only when torch is available.

    Falls back to a null context-manager when torch is absent (stub path).
    """
    try:
        import torch
        ctx = torch.autocast(device if device.startswith("cuda") else "cpu",
                             dtype=torch.bfloat16)
        with ctx:
            yield
    except ImportError:
        # torch not installed — stub / pure-Python path; no-op
        yield


class Sam2FieldTracker:
    """Offline SAM2 mask-propagation tracker.

    Wraps SAM2's ``build_sam2_video_predictor`` flow:

    1. ``init_state`` — point the predictor at a directory of numbered JPEG
       frames.
    2. ``add_new_points_or_box`` — provide a bounding-box prompt per robot on
       the prompt frame.
    3. ``propagate_in_video`` — obtain per-frame masks for every prompted
       object.

    The class returns a nested dict keyed by ``{frame_idx: {obj_id: {"box":
    (x1,y1,x2,y2), "mask": ndarray}}}``.

    Dependency injection
    --------------------
    Pass a *predictor* instance directly to skip the SAM2 build step.  This is
    used for unit tests with a stub predictor and avoids requiring a GPU.

    Mid-stream reprompting
    ----------------------
    The ``reprompt_every`` and ``detect_fn`` parameters are accepted as a hook
    for future occlusion-recovery reprompting but are **not yet implemented**.
    Single-prompt offline propagation is the only supported mode.
    """

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        config: str = _DEFAULT_SAM2_CONFIG,
        checkpoint: str = _DEFAULT_SAM2_CKPT,
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        vos_optimized: bool = False,
    ) -> None:
        """Initialise the tracker.

        Args:
            predictor: An already-built SAM2 predictor (or stub).  When
                ``None``, ``build_sam2_video_predictor`` is called lazily.
            config: SAM2 model config file (passed to
                ``build_sam2_video_predictor``).
            checkpoint: Path to the SAM2 checkpoint file.
            device: torch device string (``"cuda"`` or ``"cpu"``).
            offload_video_to_cpu: Offload video frames to CPU to save GPU VRAM.
            offload_state_to_cpu: Offload tracker state to CPU to save GPU VRAM.
            vos_optimized: Build the ``SAM2VideoPredictorVOS`` variant with a
                ``torch.compile``d image encoder.  The image encoder is the
                dominant per-frame cost, so compiling it gives a large speedup
                (~2x) at the price of a one-time warmup on the first chunk.
                Only meaningful on CUDA.
        """
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

        if predictor is not None:
            self._predictor = predictor
            self._device = device
            self._real_torch = False  # stub path — skip autocast
        else:
            # Lazy import: only touch torch/sam2 when actually building a real
            # predictor.  This keeps `from web.field.sam2_tracker import
            # mask_to_box` fast and torch-free.
            import torch  # noqa: F401 — imported for side-effects (device check)
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]

            # SAM2's connected-components CUDA kernel (`sam2._C`) is an optional
            # native extension our pip install doesn't compile.  Without it,
            # SAM2 skips mask hole-filling (fill_hole_area=8) and warns once per
            # frame.  We track robots by bounding box (mask_to_box uses the
            # mask's outer extent), so filling ≤8px interior holes can't change
            # any result — silence just this message to keep analysis logs clean.
            import warnings

            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*Skipping the post-processing step due to the error above.*",
                category=UserWarning,
            )

            self._predictor = build_sam2_video_predictor(
                config, checkpoint, device=device, vos_optimized=vos_optimized
            )
            self._device = device
            self._real_torch = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        frames_dir: str,
        prompt_idx: int,
        prompt_boxes: list[tuple],
        *,
        reprompt_every: int | None = None,
        detect_fn=None,
        object_metadata=None,
        max_objects: int | None = None,
        max_active_objects: int = 6,
        max_lifetime_objects: int = 20,
        retirement_frames: int = 15,
        new_obj_ios_thresh: float = 0.5,
        propagate_backward: bool = False,
        cull_enabled: bool = CULL_HIJACK_ENABLED,
        cull_uncorroborated_reprompts: int = CULL_UNCORROBORATED_REPROMPTS,
        cull_evidence_window: int = CULL_EVIDENCE_WINDOW,
        cull_corrob_ios: float = CULL_CORROB_IOS,
        cull_stationary_m: float = CULL_STATIONARY_M,
        evict_when_capped: bool = EVICT_WHEN_CAPPED,
        reanchor_enabled: bool = REANCHOR_ENABLED,
        reanchor_ios: float = REANCHOR_IOS,
        reanchor_max_boundary_gap: int = REANCHOR_MAX_BOUNDARY_GAP,
        reanchor_max_jump_m: float = REANCHOR_MAX_JUMP_M,
        reanchor_max_jump_px: float = REANCHOR_MAX_JUMP_PX,
        seed_gate_fn=None,
        project_fn=None,
        progress_cb=None,
        diagnostics: Optional[Any] = None,
    ) -> dict[int, dict[int, dict]]:
        """Run offline SAM2 propagation over a directory of frames.

        Args:
            frames_dir: Path to a directory of zero-padded JPEG frames (e.g.
                ``00000.jpg``, ``00001.jpg`` …).  Passed verbatim to
                ``predictor.init_state``.
            prompt_idx: Frame index (local, 0-based) at which box prompts are
                given.
            prompt_boxes: Sequence of ``(x1, y1, x2, y2)`` boxes (float or
                int) defining one robot per entry.  Object IDs are assigned as
                ``1, 2, 3, …`` in list order.
            reprompt_every: When set together with ``detect_fn``, propagation is
                run in chunks of this many frames; at each chunk boundary
                ``detect_fn`` is run and any detection that does not already
                overlap a tracked object is added as a *new* object and tracked
                forward.  ``None`` (the default) runs a single forward pass —
                the original single-prompt behavior.
            detect_fn: Callable returning boxes or ``(box, metadata)`` pairs.
            object_metadata: Optional mutable ``{obj_id: metadata}`` map. Initial
                prompt metadata may be pre-populated; re-prompted objects are added.
            max_objects: Hard cap on the total number of tracked objects. If passed,
                acts as max_active_objects.
            max_active_objects: Cap on the number of concurrently active (visible
                or recently seen) objects.
            max_lifetime_objects: Safety cap on the total number of unique object IDs.
            retirement_frames: Number of absent frames before an object is retired
                from active cap count.
            new_obj_ios_thresh: A detection whose best intersection-over-smaller
                against any tracked mask is below this threshold is treated as a
                new robot.  Passed to :func:`match_new_detections`.
            propagate_backward: When True and the initial prompt/bootstrap frame
                is after frame zero, first propagate those objects backward to
                recover the beginning of the exported clip. Re-prompted objects
                remain forward-only.
            cull_enabled: When True, a stationary object whose mask goes
                detector-uncorroborated for enough boundaries is culled
                spontaneously. The detector itself ALWAYS runs at every chunk
                boundary regardless of this flag (corroboration evidence is
                always collected). Detector contract: ``detect_fn`` returning
                ``None`` (or raising) is treated as detector-unavailable for
                that boundary — it contributes no absence evidence. An empty
                list means the detector ran and found nothing (counts as
                absence).
            cull_uncorroborated_reprompts: Absences within the evidence window
                required before a stationary object is culled (cull groundwork).
            cull_evidence_window: Sliding-window length of corroboration outcomes
                kept per object (cull groundwork).
            cull_corrob_ios: Min IoS of a fresh detection vs the mask to count as
                corroboration (cull groundwork).
            cull_stationary_m: Max field-position span over the window to count as
                stationary (cull groundwork).
            evict_when_capped: When True (default) and the active roster is
                full, a fresh detection that matches no tracked mask may evict
                the least-corroborated *stationary* active object (same
                evidence bar as the cull) to claim its slot. Without this, one
                stale mask pinning the roster makes the tracker permanently
                blind to new robots.
            reanchor_enabled: When True, an unmatched detection that strongly
                overlaps a color-consistent SLIVER mask (area below
                ``SEED_MIN_MASK_AREA_FRAC`` of the box) re-prompts that mask's
                existing object id with the detection box instead of seeding a
                new id — preserving identity across propagation collapse.
            reanchor_ios: Min IoS (over smaller) of the sliver mask vs the
                detection box required to re-anchor.
            seed_gate_fn: Optional ``(box) -> bool``; a detection for which it
                returns False is never seeded as a NEW object (rejection reason
                ``static_structure``) but still corroborates existing masks.
                Used for static-structure zones: a real robot is never *born*
                inside a field element, but one parked there must keep counting
                as detector-present.
            project_fn: Callable mapping a mask box (tracker pixel coords) to field
                meters ``(x, y)``, or None when unprojectable; measures stationarity
                in field space so camera pan can't disguise a static structure as
                moving. When None, falls back to pixel bottom-center.
            progress_cb: Optional callback taking (processed_count, total_frames).
            diagnostics: Optional IdentityDiagnosticRecorder-like object.

        Returns:
            Nested dict ``{frame_idx: {obj_id: {"box": (x1,y1,x2,y2),
            "mask": ndarray_2d, "evidence": dict}}}``.  Objects with empty masks on a given
            frame are omitted from that frame's dict.
        """
        import os
        if os.path.isdir(frames_dir):
            frame_files = sorted(
                f for f in os.listdir(frames_dir)
                if f.endswith(".jpg") or f.endswith(".jpeg")
            )
            frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
        else:
            frame_paths = []
        total_frames = max(1, len(frame_paths))

        if max_objects is not None:
            max_active_objects = max_objects

        predictor = self._predictor

        state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
        )

        first_prompted_frame: dict[int, int] = {}
        last_visible_frame: dict[int, int] = {}
        birth_detector_overlap: dict[int, float] = {}
        retired_set: set[int] = set()
        culled_set: set[int] = set()
        corroboration_history: dict[int, deque] = {}
        field_pos_window: dict[int, list] = {}
        # oid -> (pending color, consecutive colored-corroboration count)
        color_backfill_votes: dict[int, tuple] = {}
        # Re-anchor v2 bookkeeping. boundary_no counts processed re-prompt
        # boundaries; boundary_mask_record[oid] = (boundary_no, area_px, box)
        # is the object's extent at the most recent boundary where its mask was
        # actually fresh (visible within the current chunk) — prompt boxes seed
        # it. Recency + locality guards read it before any re-anchor.
        boundary_no = 0
        boundary_mask_record: dict[int, tuple] = {}

        def _box_area(box) -> float:
            return max(0.0, float(box[2]) - float(box[0])) * \
                   max(0.0, float(box[3]) - float(box[1]))

        # Prompt each robot box on the prompt frame.  Object IDs are 1-based.
        next_obj_id = 1
        for box in prompt_boxes:
            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_idx,
                obj_id=next_obj_id,
                box=np.asarray(box, dtype=np.float32),
            )
            first_prompted_frame[next_obj_id] = prompt_idx
            birth_detector_overlap[next_obj_id] = 0.0
            boundary_mask_record[next_obj_id] = (
                0, _box_area(box), tuple(int(v) for v in box))
            if diagnostics:
                diagnostics.record(
                    "sam2.object_created",
                    details={"object_id": next_obj_id, "frame_idx": prompt_idx, "box": list(box)}
                )
            next_obj_id += 1

        results: dict[int, dict[int, dict]] = {}
        self._processed_count = 0

        reprompting = bool(reprompt_every) and detect_fn is not None
        detected = []
        detect_fn_error_logged = False  # first-error print guard (one line per track())

        def _normalize_detections(items):
            boxes = []
            metadata = []
            for item in items or []:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and hasattr(item[0], "__len__")
                    and len(item[0]) == 4
                ):
                    boxes.append(item[0])
                    metadata.append(item[1])
                else:
                    boxes.append(item)
                    metadata.append(None)
            return boxes, metadata

        # Real SAM2 cannot propagate an empty state. If the prompt frame contains
        # no robot, search the exported video until the detector finds one, prompt
        # there, and begin propagation from that frame.
        bootstrap_idx = prompt_idx
        if not prompt_boxes and reprompting:
            for candidate_idx in range(prompt_idx, len(frame_paths)):
                try:
                    detected = detect_fn(
                        candidate_idx, frame_paths[candidate_idx]
                    ) or []
                except Exception:
                    detected = []
                bootstrap_boxes, bootstrap_metadata = _normalize_detections(detected)
                if seed_gate_fn is not None:
                    gated = [
                        (b, m) for b, m in zip(bootstrap_boxes, bootstrap_metadata)
                        if seed_gate_fn(b)
                    ]
                    bootstrap_boxes = [b for b, _m in gated]
                    bootstrap_metadata = [m for _b, m in gated]
                if not bootstrap_boxes:
                    continue
                for box, metadata in zip(
                    bootstrap_boxes[:max_active_objects],
                    bootstrap_metadata[:max_active_objects],
                ):
                    if next_obj_id > max_lifetime_objects:
                        break
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=candidate_idx,
                        obj_id=next_obj_id,
                        box=np.asarray(box, dtype=np.float32),
                    )
                    first_prompted_frame[next_obj_id] = candidate_idx
                    birth_detector_overlap[next_obj_id] = 0.0
                    boundary_mask_record[next_obj_id] = (
                        0, _box_area(box), tuple(int(v) for v in box))
                    if object_metadata is not None:
                        object_metadata[next_obj_id] = metadata
                    if diagnostics:
                        diagnostics.record(
                            "sam2.object_created",
                            details={
                                "object_id": next_obj_id,
                                "frame_idx": candidate_idx,
                                "box": list(box),
                                "bootstrap": True,
                            },
                        )
                    next_obj_id += 1
                bootstrap_idx = candidate_idx
                break

            if next_obj_id == 1:
                if hasattr(predictor, "reset_state"):
                    predictor.reset_state(state)
                return results

        def _propagate(
            start_frame_idx=None,
            max_frame_num_to_track=None,
            *,
            reverse=False,
        ):
            # Pass chunk kwargs only when re-prompting, so single-pass callers
            # (and the simple test stub) keep the original zero-arg signature.
            if reprompting or reverse:
                kwargs = {
                    "start_frame_idx": start_frame_idx,
                    "max_frame_num_to_track": max_frame_num_to_track,
                }
                if reverse:
                    kwargs["reverse"] = True
            else:
                kwargs = {}
            if self._real_torch:
                import torch
                with torch.inference_mode(), _maybe_autocast(self._device):
                    yield from predictor.propagate_in_video(state, **kwargs)
            else:
                yield from predictor.propagate_in_video(state, **kwargs)

        def _record_chunk(start_frame_idx, max_frame_num_to_track):
            """Propagate one chunk; record masks; return the last frame index seen."""
            last_idx = None
            for f_idx, obj_ids, mask_logits in _propagate(
                start_frame_idx, max_frame_num_to_track
            ):
                f_idx_int = int(f_idx)
                # mask_logits may be a real torch tensor (N,1,H,W) or a stub.
                bool_arr = (mask_logits > 0).cpu().numpy()
                frame_entry: dict[int, dict] = {}
                for i, obj_id in enumerate(obj_ids):
                    oid_int = int(obj_id)
                    if oid_int in culled_set:
                        continue  # culled hijack mask — stop emitting it
                    if bool_arr.ndim == 4:
                        mask2d = bool_arr[i, 0].astype(bool)
                    elif bool_arr.ndim == 3:
                        mask2d = bool_arr[i].astype(bool)
                    else:
                        mask2d = bool_arr.astype(bool)
                    box = mask_to_box(mask2d)
                    if box is None:
                        if diagnostics:
                            if last_visible_frame.get(oid_int) == f_idx_int - 1:
                                diagnostics.record(
                                    "sam2.mask_gap",
                                    details={"object_id": oid_int, "frame_idx": f_idx_int}
                                )
                        continue  # object not visible this frame

                    if not plausible_robot_mask(mask2d):
                        # Shape impossible for a robot (e.g. field radio on a stand).
                        # Skip storing it; the slot coasts and retires like a gap.
                        if diagnostics:
                            diagnostics.record(
                                "sam2.mask_rejected",
                                reason="mask_implausible",
                                details={"object_id": oid_int, "frame_idx": f_idx_int,
                                         "box": [int(v) for v in box]},
                            )
                        continue

                    prev_lv = last_visible_frame.get(oid_int)
                    last_visible_frame[oid_int] = f_idx_int
                    if oid_int not in first_prompted_frame:
                        first_prompted_frame[oid_int] = f_idx_int
                        
                    overlap = birth_detector_overlap.get(oid_int, 0.0)
                    if prev_lv is None:
                        if first_prompted_frame[oid_int] == prompt_idx:
                            ev_type = "initial"
                        else:
                            ev_type = "reprompted_new_id"
                        gap_len = 0
                    elif prev_lv < f_idx_int - 1:
                        ev_type = "reappeared_same_id"
                        gap_len = f_idx_int - prev_lv - 1
                    else:
                        ev_type = "propagated"
                        gap_len = 0
                        
                    evidence = {
                        "type": ev_type,
                        "gap_len": gap_len,
                        "detector_overlap": overlap,
                    }
                    frame_entry[oid_int] = {
                        "box": box,
                        "mask": mask2d,
                        "evidence": evidence,
                    }
                results[f_idx_int] = frame_entry
                last_idx = f_idx_int
                self._processed_count += 1
                if progress_cb is not None:
                    try:
                        progress_cb(self._processed_count, total_work_frames)
                    except Exception:
                        pass
            if diagnostics and last_idx is not None:
                diagnostics.record(
                    "sam2.chunk_complete",
                    details={"start_frame": start_frame_idx, "end_frame": last_idx}
                )
            return last_idx

        total_work_frames = total_frames
        if propagate_backward and bootstrap_idx > 0:
            # Both directions emit the prompt frame, so it is counted twice.
            total_work_frames += bootstrap_idx + 1

        try:
            if propagate_backward and bootstrap_idx > 0:
                for f_idx, obj_ids, mask_logits in _propagate(
                    bootstrap_idx,
                    bootstrap_idx,
                    reverse=True,
                ):
                    f_idx_int = int(f_idx)
                    bool_arr = (mask_logits > 0).cpu().numpy()
                    frame_entry: dict[int, dict] = {}
                    for i, obj_id in enumerate(obj_ids):
                        oid_int = int(obj_id)
                        if bool_arr.ndim == 4:
                            mask2d = bool_arr[i, 0].astype(bool)
                        elif bool_arr.ndim == 3:
                            mask2d = bool_arr[i].astype(bool)
                        else:
                            mask2d = bool_arr.astype(bool)
                        box = mask_to_box(mask2d)
                        if box is None or not plausible_robot_mask(mask2d):
                            continue
                        frame_entry[oid_int] = {
                            "box": box,
                            "mask": mask2d,
                            "evidence": {
                                "type": "propagated",
                                "gap_len": 0,
                                "detector_overlap": birth_detector_overlap.get(
                                    oid_int, 0.0
                                ),
                            },
                        }
                    results[f_idx_int] = frame_entry
                    self._processed_count += 1
                    if progress_cb is not None:
                        try:
                            progress_cb(
                                self._processed_count,
                                total_work_frames,
                            )
                        except Exception:
                            pass

            if not reprompting:
                _record_chunk(None, None)
            else:
                cursor = bootstrap_idx
                last_end = bootstrap_idx - 1
                while True:
                    last_idx = _record_chunk(cursor, reprompt_every)
                    if last_idx is None or last_idx <= last_end:
                        break  # no forward progress -> end of video
                    last_end = last_idx
                    boundary_no += 1

                    # Re-prompt at the chunk boundary. Recompute the active set
                    # (excluding culled), retiring absent objects.
                    active_objects = []
                    for oid in range(1, next_obj_id):
                        if oid in culled_set:
                            continue
                        lv = last_visible_frame.get(oid)
                        if lv is not None and (last_idx - lv) <= retirement_frames:
                            active_objects.append(oid)
                        else:
                            if oid not in retired_set and lv is not None:
                                retired_set.add(oid)
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.object_retired",
                                        details={"object_id": oid, "frame_idx": last_idx},
                                    )

                    # Latest mask + color per active object (for corroboration & seeding).
                    existing = {}
                    existing_colors = {}
                    for oid in active_objects:
                        lv = last_visible_frame.get(oid)
                        if lv is not None and oid in results.get(lv, {}):
                            existing[oid] = results[lv][oid]["mask"]
                            if object_metadata is not None:
                                existing_colors[oid] = object_metadata.get(oid)

                    # Tri-state detector. detector_available is False when the
                    # detector could not run (detect_fn raised or returned None);
                    # such boundaries contribute NO absence evidence. The detector
                    # ALWAYS runs at a boundary: one YOLO pass per chunk is
                    # negligible next to SAM2 propagation, corroboration evidence
                    # must accumulate even while the roster is capped, and a
                    # capped boundary must log what it rejects. (Skipping it while
                    # capped once hid a match-long seeding blackout: stale masks
                    # pinned the roster, so a robot crossing the open field was
                    # never even looked for.)
                    run_detector = detect_fn is not None
                    detector_available = False
                    new_boxes, new_metadata = [], []
                    if run_detector:
                        frame_path = (
                            frame_paths[last_idx] if last_idx < len(frame_paths) else None
                        )
                        try:
                            detected = detect_fn(last_idx, frame_path)
                        except Exception as exc:
                            # Unavailable for this boundary, but never silently:
                            # a persistent detect_fn bug would otherwise disable
                            # re-prompting for the whole video with no trace.
                            detected = None
                            if not detect_fn_error_logged:
                                detect_fn_error_logged = True
                                print(f"  [SAM2] detect_fn failed at frame {last_idx} "
                                      f"(further errors suppressed): {exc!r}")
                            if diagnostics:
                                diagnostics.increment("sam2_detector_errors")
                                diagnostics.record(
                                    "sam2.detector_error",
                                    details={"frame_idx": last_idx, "error": repr(exc)},
                                )
                        if detected is None:
                            detector_available = False
                        else:
                            detector_available = True
                            new_boxes, new_metadata = _normalize_detections(detected)
                            if diagnostics:
                                for box in new_boxes:
                                    diagnostics.record(
                                        "sam2.reprompt_candidate",
                                        details={"frame_idx": last_idx, "box": list(box)},
                                    )

                    # ---- Corroboration bookkeeping (windowed detector-absence +
                    # stationarity), maintained whenever the detector ran. The
                    # spontaneous hijack cull stays gated behind cull_enabled;
                    # capped-roster eviction below reuses the same evidence on
                    # demand. ----
                    if detector_available and existing:
                        h, w = next(iter(existing.values())).shape
                        for oid, mask in list(existing.items()):
                            corroborated = mask_corroborated(
                                mask, new_boxes, (w, h),
                                ios_thresh=cull_corrob_ios,
                                mask_color=(existing_colors.get(oid)
                                            if existing_colors else None),
                                new_colors=new_metadata,
                            )
                            hist = corroboration_history.setdefault(
                                oid, deque(maxlen=cull_evidence_window))
                            hist.append(bool(corroborated))

                            # Color backfill for unknown-colored objects: commit
                            # the corroborating bumper color after enough
                            # CONSECUTIVE agreeing boundaries; mixed colors (two
                            # robots overlapping the mask) reset the votes. A
                            # definite color is never overridden.
                            if (object_metadata is not None
                                    and object_metadata.get(oid) not in ("red", "blue")):
                                definite = corroborating_colors(
                                    mask, new_boxes, (w, h),
                                    ios_thresh=cull_corrob_ios,
                                    new_colors=new_metadata,
                                )
                                if len(definite) == 1:
                                    c = next(iter(definite))
                                    prev_c, votes = color_backfill_votes.get(oid, (None, 0))
                                    votes = votes + 1 if prev_c == c else 1
                                    color_backfill_votes[oid] = (c, votes)
                                    if votes >= COLOR_BACKFILL_VOTES:
                                        object_metadata[oid] = c
                                        existing_colors[oid] = c
                                        if diagnostics:
                                            diagnostics.record(
                                                "sam2.color_backfilled",
                                                details={"object_id": oid,
                                                         "color": c,
                                                         "frame_idx": last_idx},
                                            )
                                elif len(definite) > 1:
                                    color_backfill_votes.pop(oid, None)

                            box = results[last_visible_frame[oid]][oid]["box"]
                            pos = project_fn(box) if project_fn is not None else bottom_center(box)
                            win = field_pos_window.setdefault(oid, [])
                            win.append(pos)
                            if len(win) > cull_evidence_window:
                                del win[0]

                            if (cull_enabled
                                    and absent_enough(hist, cull_uncorroborated_reprompts)
                                    and is_stationary(win, cull_stationary_m)):
                                culled_set.add(oid)
                                if diagnostics:
                                    # Uncapped counter so the cull count survives
                                    # event-list truncation (max_events) over a full
                                    # match — the at-a-glance on/off comparison signal.
                                    diagnostics.increment("sam2_objects_culled")
                                    diagnostics.record(
                                        "sam2.object_culled",
                                        reason="detector_absent_stationary",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "absences": sum(1 for c in hist if not c),
                                            "window": len(hist),
                                        },
                                    )

                        if culled_set:
                            active_objects = [o for o in active_objects if o not in culled_set]
                            for o in list(culled_set):
                                existing.pop(o, None)
                                existing_colors.pop(o, None)

                    # ---- New-object seeding (caps re-checked per seed; a capped
                    # roster may evict an evidently-stale object to make room) ----
                    if detector_available and new_boxes:
                        if existing:
                            h, w = next(iter(existing.values())).shape
                        else:
                            w = max(int(b[2]) for b in new_boxes) + 1
                            h = max(int(b[3]) for b in new_boxes) + 1
                        int_boxes = [
                            (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in new_boxes
                        ]
                        new_indices = match_new_detections(
                            existing, int_boxes, (w, h),
                            ios_thresh=new_obj_ios_thresh,
                            existing_colors=existing_colors,
                            new_colors=new_metadata,
                        )
                        if diagnostics:
                            for idx, box_val in enumerate(new_boxes):
                                if idx not in new_indices:
                                    diagnostics.record(
                                        "sam2.reprompt_rejected",
                                        reason="already_tracked",
                                        details={"frame_idx": last_idx, "box": list(box_val)},
                                    )
                        for j in new_indices:
                            if reanchor_enabled and existing:
                                # A detection is only in new_indices because no
                                # healthy mask claimed it — but a color-consistent
                                # SLIVER underneath it is the same robot whose
                                # propagation collapsed. Re-prompt that object id
                                # with the fresh box instead of churning a new id.
                                bx1, by1, bx2, by2 = int_boxes[j]
                                reanchor_box_mask = np.zeros((h, w), bool)
                                reanchor_box_mask[by1:by2, bx1:bx2] = True
                                reanchor_box_area = int(reanchor_box_mask.sum())
                                best_oid = None
                                best_score = 0.0
                                for oid, existing_mask in existing.items():
                                    if existing_colors and new_metadata:
                                        e_col = existing_colors.get(oid)
                                        n_col = new_metadata[j]
                                        if (e_col in {"red", "blue"}
                                                and n_col in {"red", "blue"}
                                                and e_col != n_col):
                                            continue
                                    if (int(existing_mask.sum())
                                            >= SEED_MIN_MASK_AREA_FRAC * reanchor_box_area):
                                        continue  # healthy mask -> not degraded
                                    # Recency guard: the object must have had a
                                    # fresh stored mask within the last
                                    # reanchor_max_boundary_gap boundaries. A
                                    # zombie (retired mask flickering back as
                                    # noise) has a much older snapshot.
                                    rec = boundary_mask_record.get(oid)
                                    if rec is None:
                                        continue
                                    rec_no, rec_area, rec_box = rec
                                    if boundary_no - rec_no > reanchor_max_boundary_gap:
                                        continue
                                    # The snapshot itself must have been healthy
                                    # (a sliver persisting across boundaries is
                                    # not a fresh collapse).
                                    if rec_area < SEED_MIN_MASK_AREA_FRAC * reanchor_box_area:
                                        continue
                                    # Locality guard: the detection must be near
                                    # the last healthy extent — field meters when
                                    # projectable, else tracker pixels.
                                    p_prev = p_det = None
                                    if project_fn is not None:
                                        p_prev = project_fn(rec_box)
                                        p_det = project_fn(int_boxes[j])
                                    if p_prev is not None and p_det is not None:
                                        jump = ((p_prev[0] - p_det[0]) ** 2
                                                + (p_prev[1] - p_det[1]) ** 2) ** 0.5
                                        if jump > reanchor_max_jump_m:
                                            continue
                                    else:
                                        cxp = (rec_box[0] + rec_box[2]) / 2.0
                                        cyp = (rec_box[1] + rec_box[3]) / 2.0
                                        cxd = (int_boxes[j][0] + int_boxes[j][2]) / 2.0
                                        cyd = (int_boxes[j][1] + int_boxes[j][3]) / 2.0
                                        jump = ((cxp - cxd) ** 2 + (cyp - cyd) ** 2) ** 0.5
                                        if jump > reanchor_max_jump_px:
                                            continue
                                    score = _ios(existing_mask, reanchor_box_mask)
                                    if score >= reanchor_ios and score > best_score:
                                        best_oid = oid
                                        best_score = score
                                if best_oid is not None:
                                    predictor.add_new_points_or_box(
                                        state, frame_idx=last_idx, obj_id=best_oid,
                                        box=np.asarray(new_boxes[j], dtype=np.float32),
                                    )
                                    # The fresh prompt re-conditions the object;
                                    # sliver-era absence evidence is stale, and the
                                    # restored extent must be visible to the rest
                                    # of this boundary (so a second detection over
                                    # the same robot can't also re-anchor it).
                                    corroboration_history.pop(best_oid, None)
                                    existing[best_oid] = reanchor_box_mask
                                    boundary_mask_record[best_oid] = (
                                        boundary_no, float(reanchor_box_area),
                                        tuple(int_boxes[j]))
                                    if diagnostics:
                                        diagnostics.increment("sam2_objects_reanchored")
                                        diagnostics.record(
                                            "sam2.object_reanchored",
                                            details={
                                                "object_id": best_oid,
                                                "frame_idx": last_idx,
                                                "box": list(new_boxes[j]),
                                                "ios": round(best_score, 3),
                                            },
                                        )
                                    continue
                            if seed_gate_fn is not None and not seed_gate_fn(new_boxes[j]):
                                # Zone-gated: corroboration already counted this
                                # box above; it just can't found a new object.
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.reprompt_rejected",
                                        reason="static_structure",
                                        details={"frame_idx": last_idx,
                                                 "box": list(new_boxes[j])},
                                    )
                                continue
                            active_objects = [
                                oid for oid in range(1, next_obj_id)
                                if oid not in culled_set
                                and last_visible_frame.get(oid) is not None
                                and (last_idx - last_visible_frame[oid]) <= retirement_frames
                            ]
                            roster_capped = len(active_objects) >= max_active_objects
                            lifetime_capped = (next_obj_id - 1) >= max_lifetime_objects
                            if roster_capped and not lifetime_capped and evict_when_capped:
                                # An unmatched detection is concrete evidence a
                                # roster slot is wrongly held. Evict the least-
                                # corroborated stationary object (same evidence
                                # bar as the cull) rather than going blind.
                                victim = pick_evictable(
                                    active_objects, corroboration_history,
                                    field_pos_window,
                                    cull_uncorroborated_reprompts,
                                    cull_stationary_m,
                                )
                                if victim is not None:
                                    culled_set.add(victim)
                                    existing.pop(victim, None)
                                    existing_colors.pop(victim, None)
                                    roster_capped = False
                                    if diagnostics:
                                        diagnostics.increment("sam2_objects_culled")
                                        diagnostics.record(
                                            "sam2.object_culled",
                                            reason="evicted_for_new_detection",
                                            details={
                                                "object_id": victim,
                                                "frame_idx": last_idx,
                                                "for_box": list(new_boxes[j]),
                                            },
                                        )
                            if roster_capped or lifetime_capped:
                                reason = "roster_cap" if roster_capped else "lifetime_cap"
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.reprompt_rejected",
                                        reason=reason,
                                        details={"frame_idx": last_idx, "box": list(new_boxes[j])},
                                    )
                                continue
                            box_mask = np.zeros((h, w), bool)
                            bx1, by1, bx2, by2 = int_boxes[j]
                            box_mask[by1:by2, bx1:bx2] = True
                            best_ios = 0.0
                            for oid, existing_mask in existing.items():
                                if existing_colors and new_metadata:
                                    e_col = existing_colors.get(oid)
                                    n_col = new_metadata[j]
                                    if (e_col in {"red", "blue"} and n_col in {"red", "blue"}
                                            and e_col != n_col):
                                        continue
                                score = _ios(existing_mask, box_mask)
                                if score > best_ios:
                                    best_ios = score
                            predictor.add_new_points_or_box(
                                state, frame_idx=last_idx, obj_id=next_obj_id,
                                box=np.asarray(new_boxes[j], dtype=np.float32),
                            )
                            first_prompted_frame[next_obj_id] = last_idx
                            birth_detector_overlap[next_obj_id] = best_ios
                            boundary_mask_record[next_obj_id] = (
                                boundary_no, _box_area(new_boxes[j]),
                                tuple(int_boxes[j]))
                            if diagnostics:
                                diagnostics.record(
                                    "sam2.object_created",
                                    details={"object_id": next_obj_id, "frame_idx": last_idx,
                                             "box": list(new_boxes[j])},
                                )
                            if object_metadata is not None:
                                object_metadata[next_obj_id] = new_metadata[j]
                            next_obj_id += 1

                    # Refresh re-anchor snapshots — only for objects whose mask
                    # was actually emitted within the chunk that just ended. A
                    # stale mask (absent object) must let its snapshot age out
                    # so the recency guard can refuse a later zombie sliver.
                    for oid, m in existing.items():
                        lv = last_visible_frame.get(oid)
                        if lv is None or (last_idx - lv) >= reprompt_every:
                            continue
                        mb = mask_to_box(m)
                        if mb is not None:
                            boundary_mask_record[oid] = (
                                boundary_no, float(int(m.sum())), mb)

                    cursor = last_idx + 1
        finally:
            if hasattr(predictor, "reset_state"):
                predictor.reset_state(state)

        return results
