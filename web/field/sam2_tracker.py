"""SAM2 offline video-predictor tracker helpers and wrapper class.

The module is split into two tiers:

1. Pure, GPU-free helpers (``mask_to_box``, ``bottom_center``, ``_ios``,
   ``match_new_detections``) — importable with no torch/sam2 on the path.

2. ``Sam2FieldTracker`` — a stateful wrapper around SAM2's offline video
   predictor.  torch/sam2 are imported *lazily* inside the class so that
   the pure helpers remain torch-free.
"""
from __future__ import annotations

import inspect
import os
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Optional

import numpy as np


class TrackStopped(Exception):
    """Raised by ``Sam2FieldTracker.track`` when ``should_stop`` returns True.

    Propagation runs on a worker thread (via ``asyncio.to_thread`` in the web
    app) and cannot be killed from outside, so the loop polls a caller-supplied
    predicate once per propagated frame and raises this to unwind cooperatively.
    The web app treats it as a user stop, NOT as a tracker failure.
    """

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
# DEFAULT OFF — reverted 2026-06-10 after the qm24 gate FAILED. The qm72 gate
# passed (trench culled ~12s vs ~80s, outcomes neutral), but on qm24 the cull
# killed corner-view objects at the FIRST evidence opportunity (~f80) that the
# eviction path had let live for 500+ frames of identity evidence — confirmed
# teams collapsed 4 -> 1. This is the design doc's predicted failure: absence +
# stationarity cannot distinguish static junk from a real robot parked in a
# corner camera's detector blind spot. Needs a positive hijack signal
# (appearance discontinuity at the transfer frame) or a corner-blind-spot
# exemption before default-on. Per-job override: cull_hijack_enabled on
# /api/infer (used by the qm72 trench workaround until then).
CULL_HIJACK_ENABLED = os.environ.get("CULL_HIJACK_ENABLED", "0") not in ("0", "false", "False")
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

# tracker_prob gates (SAM3 only — SAM2 emits no per-object confidence, so
# both gates are inert there). The absence cull's qm24 failure was that
# absence+stationarity cannot tell static junk from a robot parked in a
# corner detector blind spot; the tracker's OWN confidence is the missing
# positive discriminator — junk masks hold low prob, real robots high.
CULL_LOW_PROB_MAX = float(os.environ.get("SAM2_CULL_LOW_PROB_MAX", "0"))     # 0 disables
EVICT_PROB_IMMUNE = float(os.environ.get("SAM2_EVICT_PROB_IMMUNE", "0"))    # 0 disables

# Person-overlap veto: a stationary mask that sits mostly inside a person
# detection for consecutive boundaries is a hijacked track — a robot is never
# a person, so this is the POSITIVE hijack signal the absence-based cull
# lacks (qm24's false culls were absence-only; qm11's R3 spent all of auto
# parked on a referee's orange vest). Enabled by passing person_fn to track().
PERSON_VETO_COVERAGE = float(os.environ.get("SAM2_PERSON_VETO_COVERAGE", "0.5"))
PERSON_VETO_BOUNDARIES = int(os.environ.get("SAM2_PERSON_VETO_BOUNDARIES", "2"))

# Size-prior hijack cull: a tracked mask whose pixel width, at its projected
# field position, is far wider than a robot could appear (or far narrower) for
# sustained boundaries has latched onto structure/a pile/a sliver. Equipment
# hijacks are YOLO-CORROBORATED — the detector false-fires on the equipment
# itself — so the absence-based cull structurally cannot catch them. Geometry
# is the one channel a detector false-fire cannot corroborate, so this cull
# fires INDEPENDENT of cull_enabled / corroboration. Enabled by passing
# expected_px_fn to track() with size_prior_cull=True. DEFAULT OFF pending A/B.
SIZE_PRIOR_CULL_ENABLED = os.environ.get(
    "SIZE_PRIOR_CULL_ENABLED", "0") not in ("0", "false", "False")
SIZE_PRIOR_HI = float(os.environ.get("SIZE_PRIOR_HI", "3.0"))
SIZE_PRIOR_LO = float(os.environ.get("SIZE_PRIOR_LO", "0.25"))
SIZE_PRIOR_BOUNDARIES = int(os.environ.get("SIZE_PRIOR_BOUNDARIES", "3"))

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

# Dark-mask re-anchor: a fresh detection claimed by an object whose mask has
# not been emitted for >= SAM2_DARK_REANCHOR_FRAMES is a robot the tracker has
# effectively lost — the stale mask blocks seeding ("already_tracked") while
# producing no output. Re-prompt that SAME object id at the detection box
# instead of dropping the detection. Unlike the sliver re-anchor above this
# acts only on detections the matcher itself attributed to the object, so the
# identity-transfer risk that kept REANCHOR_ENABLED off does not apply.
DARK_REANCHOR_ENABLED = os.environ.get(
    "SAM2_DARK_REANCHOR_ENABLED", "1") not in ("0", "false", "False")
DARK_REANCHOR_FRAMES = int(os.environ.get("SAM2_DARK_REANCHOR_FRAMES", "2"))
# Consecutive re-anchor attempts allowed while an object stays dark. Every
# prompt accumulates per-object session state in the predictor; re-prompting
# a never-reviving object at every boundary grew SAM3 video sessions until
# they OOM'd a full 24GB card. After the cap, the detection seeds a NEW id
# (identity machinery re-binds it) and the zombie ages into retirement.
DARK_REANCHOR_MAX_ATTEMPTS = int(
    os.environ.get("SAM2_DARK_REANCHOR_MAX_ATTEMPTS", "3"))

# Misaligned-mask re-anchor: dark re-anchor needs total mask loss and the
# sliver re-anchor needs area collapse, but the COMMON drift case is a
# healthy-size mask that slid off its robot until detection overlap fell
# below the 0.5 match threshold — the unmatched detection then churns a new
# id or hits the roster cap. When a healthy, color-consistent mask partially
# overlaps an unmatched detection for misalign_boundaries CONSECUTIVE
# boundaries, re-prompt that SAME id at the detection box. Consecutiveness +
# the [ios_min, match-threshold) band keep one-boundary flybys and true new
# robots out. DEFAULT OFF pending A/B.
MISALIGN_REANCHOR = os.environ.get(
    "SAM2_MISALIGN_REANCHOR", "0") not in ("0", "false", "False")
MISALIGN_IOS_MIN = float(os.environ.get("SAM2_MISALIGN_IOS_MIN", "0.2"))
MISALIGN_BOUNDARIES = int(os.environ.get("SAM2_MISALIGN_BOUNDARIES", "2"))

# Capped-roster eviction, dark tier: an object whose mask has been absent for
# >= this many frames (yet not retired) may be evicted for a fresh unmatched
# detection even without the full stationary-absence evidence bar.
EVICT_DARK_FRAMES = int(os.environ.get("SAM2_EVICT_DARK_FRAMES", "8"))
# Capped-roster eviction, least-corroborated tier: minimum windowed detector
# absences before ANY object (moving included) may be evicted for a fresh
# detection. The roster should hold the objects most likely to be robots; an
# object the bumper detector repeatedly fails to back (referee, field clutter)
# loses its slot to a detector-backed newcomer.
EVICT_MIN_ABSENCES = int(os.environ.get("SAM2_EVICT_MIN_ABSENCES", "2"))

# Color backfill: an object whose metadata color is unknown (seeded from a
# colorless whole-robot gap-fill) inherits the color of the bumper detections
# that corroborate its mask on this many CONSECUTIVE colored boundaries.
# Mirrors the team-lock n_override pattern: one read can be a neighbor's
# bumper clipping the mask; two agreeing in a row is deliberate evidence.
COLOR_BACKFILL_VOTES = int(os.environ.get("SAM2_COLOR_BACKFILL_VOTES", "2"))

# Seed-box aspect gate + motion-required eviction (qm12 static-sliver fix).
# In match 86d6e154 qm12 a static vertical sliver detection on the blue hub
# face — box [320.8, 145.0, 334.5, 206.1], 13.7w x 61.1h, aspect h/w = 4.47 —
# recurred at every chunk boundary. Once it became unmatched on a capped
# roster it EVICTED a real robot object to seed itself, then emitted
# mask_implausible masks for 7 straight frames. Two independent gates close
# the two enabling holes:
#   1. A candidate seed BOX that is geometrically impossible for a robot
#      (too tall-and-thin) is rejected before it can re-anchor, seed, OR
#      evict — the aspect gate previously only guarded SAM OUTPUT masks.
#   2. Eviction may require the candidate to show real pixel motion, so a
#      STATIC detection can never displace a tracked robot.
# Both DEFAULT OFF pending an A/B gate — the existing behavior is unchanged
# until a per-job override or env var turns them on.
SEED_BOX_MAX_ASPECT = float(os.environ.get("SAM2_SEED_BOX_MAX_ASPECT", "0"))   # <=0 disables
EVICT_REQUIRES_MOTION = os.environ.get("SAM2_EVICT_REQUIRES_MOTION", "0") not in ("0", "false", "False")

# Opening-window dense reprompt (qm12 0:20-0:16 fix). Robots stream into the
# center during the first couple seconds after the active period opens, but the
# tracker only re-seeds new objects at chunk boundaries spaced reprompt_every
# frames apart (default ~2s), so a robot arriving mid-opening waits up to a full
# boundary to be tracked. When both are > 0, chunks that START within the first
# OPENING_WINDOW_FRAMES of tracking use the smaller OPENING_REPROMPT_EVERY step
# so those robots seed within a fraction of a second; after the window the
# normal reprompt_every resumes. Both 0 (default) leaves behavior byte-identical.
OPENING_REPROMPT_EVERY = int(os.environ.get("SAM2_OPENING_REPROMPT_EVERY", "0"))   # 0 disables
OPENING_WINDOW_FRAMES = int(os.environ.get("SAM2_OPENING_WINDOW_FRAMES", "0"))

# Largest-connected-component mask reduction (qm11 stretched-box fix). A robot
# is one blob; when SAM leaks the mask onto adjacent structure (a bump/ramp
# beside the robot) mask_to_box returns the union bbox spanning both, so the
# rendered box stretches and the projected field position (box bottom-center)
# lands between the robot and the leak. Dropping ONLY components that are BOTH
# small AND detached removes the leak before the box / plausibility / stored
# mask are computed while preserving a legitimate secondary component (a robot
# arm, or a robot split by hub/ball occlusion into two blobs). DEFAULT OFF
# pending re-validation. Unconditional largest-component (the earlier attempt)
# regressed qm10/qm12 by dropping those legitimate halves.
MASK_LEAK_CLEANUP = os.environ.get("SAM2_MASK_LEAK_CLEANUP", "0") not in ("0", "false", "False")
# Gating thresholds for the leak cleaner. A component is dropped only when it is
# BOTH small (area < MASK_LEAK_MAX_FRAC * largest_area) AND detached (Chebyshev
# bbox gap from the largest component > MASK_LEAK_GAP_PX px).
MASK_LEAK_MAX_FRAC = float(os.environ.get("SAM2_MASK_LEAK_MAX_FRAC", "0.15"))
MASK_LEAK_GAP_PX = float(os.environ.get("SAM2_MASK_LEAK_GAP_PX", "10"))

# Late-seed backward backfill. YOLO seeds an object only at the prompt frame
# or a later reprompt boundary; a robot first detected at +6s is forward-only
# today, so its first ~6s of history is permanently lost even though SAM2/SAM3
# can propagate backward from the seed frame exactly as well as forward. After
# the forward tracking loop finishes, each surviving late-seeded object is
# re-prompted at its OWN seed frame and propagated backward a bounded span;
# recovered masks are merged into `results` (never overwriting a frame an
# earlier/original object already occupies). DEFAULT OFF pending A/B.
LATE_SEED_BACKFILL = os.environ.get(
    "SAM2_LATE_SEED_BACKFILL", "0") not in ("0", "false", "False")
# Max frames to propagate backward from an object's seed frame.
BACKFILL_MAX_FRAMES = int(os.environ.get("SAM2_BACKFILL_MAX_FRAMES", "32"))
# Minimum lived span (last_visible_frame - first_prompted_frame) required
# before an object is worth backfilling — a short-lived seed is frequently
# roster pollution from a static structure (qm12 seed/roster pollution), and
# spending a reverse propagation pass on it is wasted compute.
BACKFILL_MIN_LIVED_FRAMES = int(os.environ.get("SAM2_BACKFILL_MIN_LIVED", "8"))
# Consecutive frames with NO target mask recovered during backward propagation
# before giving up on the group — the object either didn't exist yet at that
# point in the video, or backward propagation lost it; stopping early keeps it
# from latching onto background.
BACKFILL_DARK_STOP = int(os.environ.get("SAM2_BACKFILL_DARK_STOP", "4"))
# Max IoS (over smaller) a recovered mask may score against another object's
# mask already stored in that frame. Above this the "late seed" is most likely
# a re-detection of an already-tracked robot under a new id — storing it would
# double-count the same robot in fusion.
BACKFILL_OVERLAP_IOS = float(os.environ.get("SAM2_BACKFILL_OVERLAP_IOS", "0.5"))


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bounding box of nonzero pixels as (x1, y1, x2, y2).

    x2 and y2 are exclusive maxima (one past the last pixel), consistent with
    Python slice notation.  Returns None if the mask contains no True pixels.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return a copy of `mask` with only its largest 8-connected True region.

    A robot is one blob; when a SAM mask leaks onto adjacent structure (a bump,
    a ramp) the union bbox stretches across both and the projected field
    position (box bottom-center) lands between them. Keeping the largest
    component drops the leak. Single-component masks are returned unchanged.
    Empty masks are returned unchanged.
    """
    if not mask.any():
        return mask
    mask_bool = mask.astype(bool)
    # scipy/cv2 imported locally so the module stays torch/cv2-light at import
    # time (mirrors make_frame_diff_motion_fn's local `import cv2`).
    try:
        from scipy import ndimage
        # 8-connectivity: full 3x3 structuring element.
        structure = np.ones((3, 3), dtype=bool)
        labels, n = ndimage.label(mask_bool, structure=structure)
        if n <= 1:
            return mask
        # Count pixels per label (index 0 is background) and keep the biggest.
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        keep = int(counts.argmax())
        return labels == keep
    except ImportError:
        import cv2
        n, labels = cv2.connectedComponents(
            mask_bool.astype(np.uint8), connectivity=8)
        if n <= 2:  # background + at most one component
            return mask
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        keep = int(counts.argmax())
        return labels == keep


def clean_disconnected_leaks(
    mask: np.ndarray,
    min_frac: float = MASK_LEAK_MAX_FRAC,
    gap_px: float = MASK_LEAK_GAP_PX,
) -> np.ndarray:
    """Drop a mask component only if it is BOTH small AND detached from the body.

    Keeps the largest 8-connected component, plus every other component that is
    either large (area >= min_frac * largest_area) OR spatially near the largest
    (bbox gap <= gap_px px). Removes only components that are small AND separated
    by a clear gap -- a SAM mask that leaked onto separate structure (a bump).
    Preserves an arm touching the body and an occlusion-split second half, which
    unconditional largest-component wrongly dropped (regressed qm10/qm12).
    Single-component and empty masks are returned unchanged.
    """
    if not mask.any():
        return mask
    mask_bool = mask.astype(bool)
    # scipy/cv2 imported locally so the module stays torch/cv2-light at import
    # time (mirrors make_frame_diff_motion_fn's local `import cv2`).
    try:
        from scipy import ndimage
        # 8-connectivity: full 3x3 structuring element.
        structure = np.ones((3, 3), dtype=bool)
        labels, n = ndimage.label(mask_bool, structure=structure)
    except ImportError:
        import cv2
        n_cc, labels = cv2.connectedComponents(
            mask_bool.astype(np.uint8), connectivity=8)
        n = n_cc - 1  # cv2 counts background as label 0; n = # foreground labels
    if n <= 1:
        return mask  # single component (or none) — nothing to gate

    # Per-component area (pixel count) and tight bbox. Label 0 is background.
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(counts.argmax())
    largest_area = int(counts[largest])

    def _bbox(comp_label):
        ys, xs = np.where(labels == comp_label)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _bbox_gap(a, b):
        # Chebyshev gap between two rectangles (x1,y1,x2,y2); 0 if they overlap
        # or touch, else the larger of the horizontal/vertical separations.
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(0, max(ax1 - bx2, bx1 - ax2))
        dy = max(0, max(ay1 - by2, by1 - ay2))
        return max(dx, dy)

    largest_bbox = _bbox(largest)
    keep = labels == largest
    for lbl in range(1, len(counts)):
        if lbl == largest or counts[lbl] == 0:
            continue
        area = int(counts[lbl])
        is_large = area >= min_frac * largest_area
        is_near = _bbox_gap(_bbox(lbl), largest_bbox) <= gap_px
        if is_large or is_near:
            keep |= labels == lbl
        # else: small AND detached -> drop (a leak onto separate structure).
    return keep


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


def plausible_seed_box(box, max_aspect: float) -> bool:
    """Return False for a detection box too tall-and-thin to be a robot.

    A robot's detection box is roughly as wide as it is tall; a field fixture
    (the qm12 hub-face sliver was 13.7w x 61.1h, aspect 4.47) is a narrow
    vertical stripe. This is a cheap pre-check on the *candidate box* — the
    complement of :func:`plausible_robot_mask`, which only ever ran on SAM
    output masks. Applied before a box may re-anchor, seed a new object, or
    evict a tracked one, so a geometrically impossible detection does nothing.

    Rejects when height/width > ``max_aspect``. A degenerate box (width or
    height <= 0) is also rejected. Callers gate on ``max_aspect > 0``; this
    helper does not check that, so passing a non-positive ``max_aspect`` here
    would reject everything (never done — the call sites bypass it when off).
    """
    x1, y1, x2, y2 = box
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w <= 0 or h <= 0:
        return False
    return (h / w) <= max_aspect


def size_prior_ratio(box, expected_px):
    """Ratio of a mask box's pixel width to the geometrically expected robot
    width at its field position, or None when unmeasurable (no calibration
    pose, degenerate box, or no expectation)."""
    if expected_px is None or expected_px <= 0:
        return None
    w = float(box[2]) - float(box[0])
    if w <= 0:
        return None
    return w / expected_px


def size_prior_violation(ratio, hi, lo):
    """A tracked mask far wider than a robot could appear at its position has
    latched onto structure/a pile; far narrower is a sliver/fragment. Geometry
    is the one channel a detector false-fire cannot corroborate — this is the
    hijack signal that works even when YOLO fires on the equipment itself."""
    return ratio is not None and (ratio > hi or ratio < lo)


def bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the bottom-center point of a box as (cx, y_bottom).

    y2 is already the exclusive-max bottom from mask_to_box, so this is
    consistent with that convention.
    """
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, float(y2))


def mask_bottom_center(
    mask: np.ndarray, rows: int = 3,
) -> tuple[float, float] | None:
    """Return the (x, y) footprint of a mask's carpet-contact band.

    The tracked box bottom-center is the tight-bbox midpoint, so an extended
    intake/arm at a different column than the base shifts the ray origin off the
    robot's actual carpet contact. This finds the lowest nonzero row ``ymax``,
    takes the nonzero pixels within the bottom ``rows`` rows (a small band up
    from ``ymax``), and returns ``(mean_x, ymax)`` — x is the mean column of
    those bottom-band pixels, y is the true lowest contact row. The band clamps
    at the mask top, so a mask shorter than ``rows`` uses all its rows.

    Returns None for an empty mask.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    ymax = int(ys.max())
    band_lo = ymax - (rows - 1)  # inclusive lowest row of the band
    in_band = ys >= band_lo
    return (float(xs[in_band].mean()), float(ymax))


def box_motion_energy(prev_gray: np.ndarray, cur_gray: np.ndarray,
                      box: tuple[float, float, float, float]) -> float:
    """Mean absolute intensity change inside ``box`` between two grayscale
    frames, normalized to [0, 1].

    The cameras are fixed, so frame differencing is a direct motion probe: a
    robot sweeping through its detection box scores high (motion blur only
    amplifies the signal), static equipment scores ~0 (JPEG noise). Callers
    threshold the result; a degenerate or fully-clipped box scores 0.0.
    """
    h, w = cur_gray.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = prev_gray[y1:y2, x1:x2].astype(np.int16)
    b = cur_gray[y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(b - a).mean()) / 255.0


def make_frame_diff_motion_fn(frames_dir: str, energy_thresh: float,
                              filename_fmt: str = "{:05d}.jpg"):
    """Build a ``motion_fn(local_idx, box) -> bool`` over exported frames.

    Compares the boundary frame against its immediate predecessor (adjacent
    exported frames, so one track_fps step apart) with
    :func:`box_motion_energy`. Frames are cached one pair at a time —
    boundaries are processed in order and every candidate at a boundary
    shares the same pair. Missing/unreadable frames yield False, never an
    exception (motion is only ever positive evidence).
    """
    import cv2

    cache: dict[int, np.ndarray | None] = {}

    def _gray(idx: int):
        if idx not in cache:
            if len(cache) > 4:
                cache.clear()
            img = cv2.imread(
                os.path.join(frames_dir, filename_fmt.format(idx)),
                cv2.IMREAD_GRAYSCALE)
            cache[idx] = img
        return cache[idx]

    def motion_fn(local_idx: int, box) -> bool:
        if local_idx <= 0:
            return False
        prev, cur = _gray(local_idx - 1), _gray(local_idx)
        if prev is None or cur is None or prev.shape != cur.shape:
            return False
        return box_motion_energy(prev, cur, box) >= energy_thresh

    return motion_fn


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


def best_partial_claimant(existing, existing_colors, mask_areas, box_mask,
                          box_area, color, ios_lo, ios_hi, min_area_frac):
    """The healthy (area >= min_area_frac * box_area), color-consistent mask
    with the highest IoS against ``box_mask`` inside [ios_lo, ios_hi) — the
    drifted-claimant candidate for the misaligned-mask re-anchor. Returns
    (oid, ios) or (None, 0.0)."""
    best_oid, best = None, 0.0
    for oid, mask in existing.items():
        e_col = (existing_colors or {}).get(oid)
        if (e_col in {"red", "blue"} and color in {"red", "blue"}
                and e_col != color):
            continue
        if mask_areas.get(oid, 0) < min_area_frac * box_area:
            continue  # sliver -> the sliver re-anchor's jurisdiction
        s = _ios(mask, box_mask)
        if ios_lo <= s < ios_hi and s > best:
            best_oid, best = oid, s
    return best_oid, best


def match_new_detections(
    existing_masks: dict[int, np.ndarray],
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.5,
    existing_colors: dict[int, Any] | None = None,
    new_colors: list[Any] | None = None,
    min_mask_area_frac: float = SEED_MIN_MASK_AREA_FRAC,
    with_claims: bool = False,
) -> list[int] | tuple[list[int], dict[int, int]]:
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
        with_claims: when True, also return a mapping of matched box index ->
            claiming object id, so the caller can act on WHO blocked a
            detection (e.g. re-anchor a dark object instead of dropping it).

    Returns:
        List of integer indices into new_boxes for unmatched (new) robots;
        with ``with_claims`` a ``(unmatched, claims)`` tuple.
    """
    w, h = wh
    mask_areas = {oid: int(m.sum()) for oid, m in existing_masks.items()}
    unmatched: list[int] = []
    claims: dict[int, int] = {}
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        box_mask = np.zeros((h, w), bool)
        box_mask[y1:y2, x1:x2] = True
        box_area = int(box_mask.sum())
        best_ios = 0.0
        best_oid = None
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
                best_oid = oid
        if best_ios < ios_thresh:
            unmatched.append(idx)
        elif best_oid is not None:
            claims[idx] = best_oid
    if with_claims:
        return unmatched, claims
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


def normalize_detections(items):
    """Normalize detect_fn output into (boxes, metadata, corrob_only) lists.

    Accepted item shapes: a bare 4-box, ``(box, metadata)``, or
    ``(box, metadata, corrob_only)``. corrob_only detections count as
    presence evidence (corroboration, color backfill) but may never seed,
    match, re-anchor, or evict — the sub-seed-confidence band exists so a
    weakly-detected parked robot does not accrue absences (see
    corrob_conf_floor in job_config.py)."""
    boxes, metadata, corrob_only = [], [], []
    for item in items or []:
        if (isinstance(item, tuple) and len(item) in (2, 3)
                and hasattr(item[0], "__len__") and len(item[0]) == 4):
            boxes.append(item[0])
            metadata.append(item[1])
            corrob_only.append(bool(item[2]) if len(item) == 3 else False)
        else:
            boxes.append(item)
            metadata.append(None)
            corrob_only.append(False)
    return boxes, metadata, corrob_only


# Negative-point prompting: seed/re-anchor prompts carry label-0 points at
# the centers of OTHER concurrent detections near the prompt box, so SAM
# conditions against bleeding onto a neighboring robot / adjacent structure
# from birth — prevention at the prompt, where the culls above only treat
# symptoms. DEFAULT OFF pending A/B.
NEG_POINT_PROMPTS = os.environ.get(
    "SAM2_NEG_POINT_PROMPTS", "0") not in ("0", "false", "False")
NEG_POINT_MAX = int(os.environ.get("SAM2_NEG_POINT_MAX", "4"))
NEG_POINT_INFLATE = float(os.environ.get("SAM2_NEG_POINT_INFLATE", "1.5"))


def negative_points_for(target_box, other_boxes, max_points: int = None,
                        inflate: float = None):
    """Centers of ``other_boxes`` that fall inside ``target_box`` inflated by
    ``inflate`` — nearest first, capped at ``max_points``. A neighbor's center
    is the strongest single-pixel statement of "not this object"."""
    if max_points is None:
        max_points = NEG_POINT_MAX
    if inflate is None:
        inflate = NEG_POINT_INFLATE
    x1, y1, x2, y2 = (float(v) for v in target_box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Half-extent of the inflated window is the FULL box dimension times
    # inflate (not half-width times inflate) — inflate=1.5 on a 10px box
    # reaches a neighbor centered 9px away, one box-width beyond the edge.
    hw, hh = (x2 - x1) * inflate, (y2 - y1) * inflate
    out = []
    for b in other_boxes:
        ox = (float(b[0]) + float(b[2])) / 2.0
        oy = (float(b[1]) + float(b[3])) / 2.0
        if abs(ox - cx) <= hw and abs(oy - cy) <= hh:
            out.append(((ox - cx) ** 2 + (oy - cy) ** 2, (ox, oy)))
    out.sort(key=lambda t: t[0])
    return [p for _d, p in out[:max_points]]


def _predictor_accepts_points(predictor) -> bool:
    """Whether ``predictor.add_new_points_or_box`` supports a ``points`` kwarg.

    Uses signature introspection rather than a try/except TypeError probe so
    that a genuine internal error from a supporting predictor (bad dtype,
    bad shape, etc.) propagates instead of being mistaken for "unsupported
    kwarg" and silently degraded to a box-only call.
    """
    try:
        sig = inspect.signature(predictor.add_new_points_or_box)
    except (TypeError, ValueError):
        # Not introspectable (e.g. some C callables) -- assume unsupported,
        # the safe/conservative default.
        return False
    for param in sig.parameters.values():
        if param.name == "points" or param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _prompt_with_negatives(predictor, state, frame_idx, obj_id, box, neg_points):
    """add_new_points_or_box with optional negative points; degrades to a
    box-only prompt when the predictor doesn't take point kwargs (stubs,
    legacy adapters)."""
    if neg_points and _predictor_accepts_points(predictor):
        predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=obj_id,
            box=np.asarray(box, dtype=np.float32),
            points=np.asarray(neg_points, dtype=np.float32),
            labels=np.zeros(len(neg_points), dtype=np.int32),
        )
        return
    predictor.add_new_points_or_box(
        state, frame_idx=frame_idx, obj_id=obj_id,
        box=np.asarray(box, dtype=np.float32))


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


def mean_or_none(vals):
    """Mean of the non-None entries, or None when empty/all-None. Used for the
    per-boundary tracker_prob window (SAM3 confidence; SAM2 yields no probs,
    so None keeps every prob-conditioned gate inert)."""
    xs = [float(v) for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def cull_enabled_for_view(requested, view_name: str, main_only: bool):
    """Scope the absence-based hijack cull to the main view.

    Mid-field equipment (feeder rig, OVERFLOW bins) is fully visible to
    main-view YOLO, so repeated detector absence there is strong evidence.
    Corner views are where qm24's false culls happened — a robot parked in a
    corner camera's detector blind spot is indistinguishable from junk by
    absence+stationarity alone — so with ``main_only`` they never cull.

    ``requested`` may be None (defer to the tracker's env-knob default); the
    tri-state is preserved for the main view, but a scoped-out corner view is
    forced to False (never "defer", which could re-enable via env).
    """
    if not main_only or view_name == "main":
        return requested
    return False


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
                   k: int, stationary_m: float,
                   last_visible_frame=None, current_idx=None,
                   dark_frames: int | None = None,
                   min_absences: int = EVICT_MIN_ABSENCES,
                   moving_candidate: bool = False,
                   prob_means: dict | None = None,
                   prob_immune: float = 0.0) -> int | None:
    """Return the active object best justified for eviction, or None.

    The roster should hold the objects MOST LIKELY to be robots; a fresh
    unmatched detection (detector-backed by construction) outranks a roster
    member with weak evidence. Candidates are ranked in tiers, strongest
    eviction evidence first:

    1. Stationary + >= ``k`` windowed detector absences — the spontaneous
       hijack-cull bar (static field clutter). Most absences wins.
    2. Dark — the object's mask has not been emitted for >= ``dark_frames``
       (tracked in name only, not yet retired). Longest darkness wins.
    2.5. Motion-proven candidate (``moving_candidate=True``) — a stationary
       object with ANY windowed absence yields to a detection that proved
       pixel motion. Moving almost always means robot; a stationary object
       the detector failed to back at least once is the weaker claim.
    3. Least corroborated — >= ``min_absences`` windowed absences regardless
       of motion (a moving hijack such as a referee is exactly the object the
       bumper detector keeps failing to back). Most absences wins.

    Universal guard: an object the detector corroborated at the most recent
    boundary is never evictable, and neither is one without history yet (a
    just-seeded object has had no chance to prove itself). Ties go to the
    smaller (older) object id.

    ``prob_means``/``prob_immune``: when a candidate's mean tracker_prob (SAM3
    confidence; see ``mean_or_none``) is at or above ``prob_immune`` (> 0), it
    is spared from tiers 1, 2.5, and 3 — the tracker's own confidence is
    positive evidence the eviction-worthiness tiers below can't see. Tier 2
    (dark) is EXEMPT: a dark object's prob window is stale (no fresh frames
    to average), so a high value there is not live evidence.
    """
    def _evidence(oid):
        hist = corroboration_history.get(oid)
        if not hist or hist[-1]:
            return None  # no history yet, or detector just backed it
        return hist

    def _immune(oid):
        if prob_immune <= 0 or not prob_means:
            return False
        p = prob_means.get(oid)
        return p is not None and p >= prob_immune

    # Tier 1: stationary + absent_enough(k).
    best, best_absent = None, -1
    for oid in active_objects:
        if _immune(oid):
            continue
        hist = _evidence(oid)
        if not hist or not absent_enough(hist, k):
            continue
        if not is_stationary(field_pos_window.get(oid, []), stationary_m):
            continue
        absent = sum(1 for c in hist if not c)
        if absent > best_absent:
            best, best_absent = oid, absent
    if best is not None:
        return best

    # Tier 2: dark mask (no emission for >= dark_frames). Exempt from
    # immunity — a dark object's prob window is stale, not live evidence.
    if (last_visible_frame is not None and current_idx is not None
            and dark_frames is not None):
        best, best_dark = None, -1
        for oid in active_objects:
            if _evidence(oid) is None:
                continue
            lv = last_visible_frame.get(oid)
            if lv is None:
                continue
            gap = current_idx - lv
            if gap >= dark_frames and gap > best_dark:
                best, best_dark = oid, gap
        if best is not None:
            return best

    # Tier 2.5: motion-proven candidate vs stationary absentee. YOLO
    # priority: an object the detector backed at EITHER of the last two
    # boundaries is immune — a parked robot missing one boundary read must
    # not lose its seat to motion-only evidence (ball cluster, misread).
    if moving_candidate:
        best, best_absent = None, -1
        for oid in active_objects:
            if _immune(oid):
                continue
            hist = _evidence(oid)
            if not hist:
                continue
            if len(hist) < 2 or hist[-2]:
                continue  # corroborated too recently (or unproven)
            absent = sum(1 for c in hist if not c)
            if not is_stationary(field_pos_window.get(oid, []), stationary_m):
                continue
            if absent > best_absent:
                best, best_absent = oid, absent
        if best is not None:
            return best

    # Tier 3: least corroborated, motion irrelevant.
    best, best_absent = None, -1
    for oid in active_objects:
        if _immune(oid):
            continue
        hist = _evidence(oid)
        if not hist:
            continue
        absent = sum(1 for c in hist if not c)
        if absent >= min_absences and absent > best_absent:
            best, best_absent = oid, absent
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

    #: Tags for log lines and diagnostics event names; the SAM3 subclass
    #: overrides both so runs label themselves with the backend in use.
    LOG_TAG = "SAM2"
    DIAG_PREFIX = "sam2"

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
        evict_dark_frames: int = EVICT_DARK_FRAMES,
        evict_min_absences: int = EVICT_MIN_ABSENCES,
        dark_reanchor_enabled: bool = DARK_REANCHOR_ENABLED,
        dark_reanchor_frames: int = DARK_REANCHOR_FRAMES,
        dark_reanchor_max_attempts: int = DARK_REANCHOR_MAX_ATTEMPTS,
        reanchor_enabled: bool = REANCHOR_ENABLED,
        reanchor_ios: float = REANCHOR_IOS,
        reanchor_max_boundary_gap: int = REANCHOR_MAX_BOUNDARY_GAP,
        reanchor_max_jump_m: float = REANCHOR_MAX_JUMP_M,
        reanchor_max_jump_px: float = REANCHOR_MAX_JUMP_PX,
        misalign_reanchor: bool = MISALIGN_REANCHOR,
        misalign_ios_min: float = MISALIGN_IOS_MIN,
        misalign_boundaries: int = MISALIGN_BOUNDARIES,
        seed_box_max_aspect: float = SEED_BOX_MAX_ASPECT,
        evict_requires_motion: bool = EVICT_REQUIRES_MOTION,
        mask_leak_cleanup: bool = MASK_LEAK_CLEANUP,
        late_seed_backfill: bool = LATE_SEED_BACKFILL,
        backfill_max_frames: int = BACKFILL_MAX_FRAMES,
        backfill_min_lived_frames: int = BACKFILL_MIN_LIVED_FRAMES,
        opening_reprompt_every: int | None = None,
        opening_window_frames: int | None = None,
        seed_gate_fn=None,
        corrob_gate_fn=None,
        project_fn=None,
        motion_fn=None,
        person_fn=None,
        person_veto_coverage: float = PERSON_VETO_COVERAGE,
        person_veto_boundaries: int = PERSON_VETO_BOUNDARIES,
        expected_px_fn=None,
        size_prior_cull: bool = SIZE_PRIOR_CULL_ENABLED,
        size_prior_hi: float = SIZE_PRIOR_HI,
        size_prior_lo: float = SIZE_PRIOR_LO,
        size_prior_boundaries: int = SIZE_PRIOR_BOUNDARIES,
        cull_low_prob_max: float = CULL_LOW_PROB_MAX,
        evict_prob_immune: float = EVICT_PROB_IMMUNE,
        neg_point_prompts: bool = NEG_POINT_PROMPTS,
        progress_cb=None,
        diagnostics: Optional[Any] = None,
        should_stop: Optional[Callable[[], bool]] = None,
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
            misalign_reanchor: When True, a healthy, color-consistent mask
                that partially overlaps an unmatched detection (IoS in
                ``[misalign_ios_min, new_obj_ios_thresh)``) for
                ``misalign_boundaries`` CONSECUTIVE boundaries re-prompts that
                SAME object id at the detection box — the drift case between
                dark and sliver re-anchor. DEFAULT OFF.
            misalign_ios_min: Lower bound of the partial-overlap band used to
                find a misaligned-mask claimant.
            misalign_boundaries: Consecutive boundaries a partial claimant
                must persist before it is re-anchored.
            seed_box_max_aspect: When > 0, a candidate detection box whose
                height/width exceeds this ratio (a tall-thin field sliver, not
                a robot) is rejected at every chunk boundary before it can
                re-anchor, seed a new object, or evict a tracked one, and at
                bootstrap. ``0`` (the default) disables the gate — behavior is
                unchanged. See :func:`plausible_seed_box`. Rejections record a
                ``reprompt_rejected`` diagnostic with reason
                ``implausible_seed_box``.
            evict_requires_motion: When True AND a ``motion_fn`` is supplied, a
                capped-roster candidate that shows no pixel motion may not evict
                any tracked object — the eviction is skipped (the candidate then
                falls through to the normal ``roster_cap`` rejection) and an
                ``eviction_skipped``/``static_candidate`` diagnostic is
                recorded. When ``motion_fn`` is None this gate is inert (motion
                evidence is unavailable, so eviction proceeds exactly as today).
                Default False.
            mask_leak_cleanup: qm11/qm10/qm12 gated leak cleanup. When True,
                each per-frame SAM mask is passed through
                :func:`clean_disconnected_leaks` BEFORE the box / plausibility /
                stored mask are computed. It drops a mask component ONLY when it
                is BOTH small (area below MASK_LEAK_MAX_FRAC of the largest) AND
                detached (bbox gap above MASK_LEAK_GAP_PX) — a leak onto adjacent
                structure (a bump/ramp) that would otherwise stretch the box and
                drag the projected field position between the robot and the leak.
                A legitimate secondary component (a robot arm, or a robot split
                by hub/ball occlusion into two blobs) is PRESERVED — unconditional
                largest-component wrongly dropped those and regressed qm10/qm12.
                The cleaned mask is what gets stored, so downstream
                IoS/corroboration also use it. Default False (behavior
                byte-identical) pending re-validation.
            late_seed_backfill: When True, after the forward tracking loop
                finishes, each surviving NEW object seeded at a late reprompt
                boundary (not a dark/sliver/misalign re-anchor, which reuse an
                existing id) is re-prompted at its own seed frame and
                propagated BACKWARD up to ``backfill_max_frames``, recovering
                the history YOLO missed by seeding late. Recovered masks are
                merged into ``results`` — a frame already holding that object
                is left untouched, and a recovered mask that heavily overlaps
                another object's mask already stored in that frame is dropped
                (see :data:`BACKFILL_OVERLAP_IOS`) rather than double-counting
                the same robot under two ids. Backward propagation for one
                object stops early after :data:`BACKFILL_DARK_STOP` consecutive
                frames with no recovered mask. DEFAULT OFF pending A/B.
            backfill_max_frames: Max frames to propagate backward from an
                object's seed frame. ``None``-like knobs resolve to
                :data:`BACKFILL_MAX_FRAMES`.
            backfill_min_lived_frames: An object must have lived at least this
                many frames (``last_visible_frame - first_prompted_frame``)
                before backfill is attempted — a short-lived seed is
                frequently roster pollution from a static structure, and
                backfilling it wastes a reverse propagation pass.
            opening_reprompt_every: qm12 opening-window dense reprompt. When this
                and ``opening_window_frames`` are both > 0, chunks that START
                within the first ``opening_window_frames`` of tracking propagate
                ``min(reprompt_every, opening_reprompt_every)`` frames instead of
                the full ``reprompt_every`` — new robots streaming into the
                center during the period opening (clock 0:20-0:16) seed within a
                fraction of a second instead of waiting a full ~2s boundary.
                After the window the normal ``reprompt_every`` resumes. ``None``
                resolves to ``OPENING_REPROMPT_EVERY``; 0 (default) disables the
                dense step, leaving behavior byte-identical.
            opening_window_frames: Length of the dense-reprompt window, in
                processed frames measured from ``bootstrap_idx`` (the cursor
                origin). ``None`` resolves to ``OPENING_WINDOW_FRAMES``; 0
                (default) disables. Only meaningful together with
                ``opening_reprompt_every``.
            motion_fn: Optional ``(local_idx, box) -> bool``; True means the
                detection box shows real pixel motion at this boundary (e.g.
                frame-diff energy above threshold). Motion is positive
                evidence of robot-ness and unblocks two rejection gates: a
                moving detection claimed by a STATIONARY mask seeds a new
                object anyway (the claimant is parked equipment, not the
                mover), and a moving unmatched detection may evict a
                stationary once-absent object from a capped roster
                (``pick_evictable`` tier 2.5). False/None/raising never
                penalizes a detection — absence of motion is not evidence of
                absence of a robot.
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
            expected_px_fn: Callable mapping a field ``(x, y)`` to the expected
                robot pixel width there (already in this tracker's export pixel
                space), or None when unmeasurable. Drives the size-prior cull.
            size_prior_cull: When True, an object whose mask box violates the
                size prior (too wide/too narrow vs ``expected_px_fn``) at every
                one of ``size_prior_boundaries`` consecutive boundaries is
                culled. INDEPENDENT of ``cull_enabled``/corroboration: equipment
                hijacks are detector-corroborated, so geometry is the only clean
                signal. A None expectation is unmeasurable, never a violation.
            size_prior_hi: Ratio (mask width / expected width) above which the
                mask is implausibly wide (latched onto structure/a pile).
            size_prior_lo: Ratio below which the mask is implausibly narrow (a
                sliver/fragment).
            size_prior_boundaries: Consecutive violating boundaries required to
                cull.
            cull_low_prob_max: When > 0, the absence cull (above) additionally
                requires the object's mean tracker_prob (SAM3 per-object
                confidence, windowed like corroboration) to be below this
                value. This is the positive discriminator the qm24 false-cull
                failure lacked: absence+stationarity alone cannot tell static
                junk from a robot parked in a corner detector blind spot, but
                junk holds low tracker confidence while a real robot holds
                high. A None mean (SAM2, or no evidence yet) always allows the
                cull — unmeasurable is not evidence of high confidence. 0 (the
                default) disables the gate — behavior is unchanged.
            evict_prob_immune: When > 0, a capped-roster eviction candidate
                (``pick_evictable``) whose mean tracker_prob is at or above
                this value is spared from the weak eviction tiers (stationary-
                absent, motion-proven-vs-absentee, least-corroborated) — high
                tracker confidence is positive evidence the tiers can't see.
                The dark tier is exempt (a dark object's window is stale, not
                live evidence). 0 (the default) disables the gate — behavior
                is unchanged.
            progress_cb: Optional callback taking (processed_count, total_frames).
            diagnostics: Optional IdentityDiagnosticRecorder-like object.
            should_stop: Optional predicate polled once per propagated frame.
                When it returns True the loop raises ``TrackStopped`` after the
                predictor state is reset (the ``finally`` below) — a cooperative
                abort for a user STOP, since the propagation thread can't be
                killed from outside. Callers treat this as a stop, not a
                tracker failure.

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
        # tracker_prob bookkeeping (SAM3 only — SAM2 emits None, so both
        # prob-conditioned gates stay inert). prob_accum collects the
        # per-frame tracker_prob for an object between boundaries;
        # prob_window holds the per-boundary mean (mean_or_none of a flushed
        # prob_accum), sized like corroboration_history.
        prob_accum: dict[int, list] = {}
        prob_window: dict[int, deque] = {}
        # Dark re-anchor budget: consecutive attempts per object and the
        # boundary frame of the last attempt (for success detection).
        dark_reanchor_attempts: dict[int, int] = {}
        dark_reanchor_last: dict[int, int] = {}
        field_pos_window: dict[int, list] = {}
        # oid -> last `size_prior_boundaries` size-prior violation outcomes
        # (True == violated). Sustained all-True culls the object independent
        # of corroboration (equipment hijacks are detector-corroborated).
        size_violation_history: dict[int, deque] = {}
        # oid -> consecutive boundaries with >= person_veto_coverage overlap
        person_overlap_counts: dict[int, int] = {}
        # oid -> (pending color, consecutive colored-corroboration count)
        color_backfill_votes: dict[int, tuple] = {}
        # Re-anchor v2 bookkeeping. boundary_no counts processed re-prompt
        # boundaries; boundary_mask_record[oid] = (boundary_no, area_px, box)
        # is the object's extent at the most recent boundary where its mask was
        # actually fresh (visible within the current chunk) — prompt boxes seed
        # it. Recency + locality guards read it before any re-anchor.
        boundary_no = 0
        boundary_mask_record: dict[int, tuple] = {}
        # Misaligned-mask re-anchor: consecutive-boundary streak of a partial
        # (drifted) claim per object id. Reset (pruned away) for any object
        # that gets no partial-claimant hit at a boundary -- consecutiveness
        # is the whole point (a one-boundary flyby must not re-anchor).
        misalign_streak: dict[int, int] = {}
        # Late-seed backward backfill: (seed_frame, obj_id, box) collected at
        # the moment each NEW object is seeded (not re-anchored) at a chunk
        # boundary, so the post-loop backfill phase knows where to re-prompt
        # and how far it can propagate backward.
        backfill_requests: list = []

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
                    f"{self.DIAG_PREFIX}.object_created",
                    details={"object_id": next_obj_id, "frame_idx": prompt_idx, "box": list(box)}
                )
            next_obj_id += 1

        results: dict[int, dict[int, dict]] = {}
        self._processed_count = 0

        reprompting = bool(reprompt_every) and detect_fn is not None
        # qm12 opening-window dense reprompt knobs (None -> module default).
        oe = opening_reprompt_every if opening_reprompt_every is not None else OPENING_REPROMPT_EVERY
        ow = opening_window_frames if opening_window_frames is not None else OPENING_WINDOW_FRAMES
        detected = []
        detect_fn_error_logged = False  # first-error print guard (one line per track())
        motion_fn_error_logged = False  # same guard for motion_fn
        person_fn_error_logged = False  # same guard for person_fn
        corrob_gate_error_logged = False  # same guard for corrob_gate_fn

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
                bootstrap_boxes, bootstrap_metadata, _bs_corrob = normalize_detections(detected)
                gated = [
                    (b, m) for b, m, c in zip(bootstrap_boxes, bootstrap_metadata, _bs_corrob)
                    if not c
                ]
                bootstrap_boxes = [b for b, _m in gated]
                bootstrap_metadata = [m for _b, m in gated]
                if seed_box_max_aspect > 0:
                    # A geometrically implausible box (tall-thin field sliver)
                    # can never *found* an object, so drop it before seeding.
                    gated = [
                        (b, m) for b, m in zip(bootstrap_boxes, bootstrap_metadata)
                        if plausible_seed_box(b, seed_box_max_aspect)
                    ]
                    bootstrap_boxes = [b for b, _m in gated]
                    bootstrap_metadata = [m for _b, m in gated]
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
                            f"{self.DIAG_PREFIX}.object_created",
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
            for out in _propagate(start_frame_idx, max_frame_num_to_track):
                # SAM2 / legacy stubs yield 3-tuples; the SAM3 adapter appends
                # per-object tracker confidence as a 4th element (may be None).
                f_idx, obj_ids, mask_logits, *rest = out
                tracker_probs = rest[0] if rest else None
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
                    if mask_leak_cleanup:
                        # qm11/qm10/qm12: drop a mask component only if BOTH
                        # small AND detached (a leak onto adjacent structure)
                        # before the box / plausibility / stored mask; a robot
                        # arm or occlusion-split half is preserved.
                        mask2d = clean_disconnected_leaks(mask2d)
                    box = mask_to_box(mask2d)
                    if box is None:
                        if diagnostics:
                            if last_visible_frame.get(oid_int) == f_idx_int - 1:
                                diagnostics.record(
                                    f"{self.DIAG_PREFIX}.mask_gap",
                                    details={"object_id": oid_int, "frame_idx": f_idx_int}
                                )
                        continue  # object not visible this frame

                    if not plausible_robot_mask(mask2d):
                        # Shape impossible for a robot (e.g. field radio on a stand).
                        # Skip storing it; the slot coasts and retires like a gap.
                        if diagnostics:
                            diagnostics.record(
                                f"{self.DIAG_PREFIX}.mask_rejected",
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
                        "tracker_prob": (
                            float(tracker_probs[i])
                            if tracker_probs is not None
                            and i < len(tracker_probs)
                            else None
                        ),
                    }
                    if evidence["tracker_prob"] is not None:
                        prob_accum.setdefault(oid_int, []).append(
                            evidence["tracker_prob"])
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
                # Cooperative stop: checked once per propagated frame so a user
                # STOP takes effect within a frame's propagation time rather
                # than after the whole (multi-minute) chunk/view completes.
                if should_stop is not None and should_stop():
                    raise TrackStopped()
            if diagnostics and last_idx is not None:
                diagnostics.record(
                    f"{self.DIAG_PREFIX}.chunk_complete",
                    details={"start_frame": start_frame_idx, "end_frame": last_idx}
                )
            return last_idx

        total_work_frames = total_frames
        if propagate_backward and bootstrap_idx > 0:
            # Both directions emit the prompt frame, so it is counted twice.
            total_work_frames += bootstrap_idx + 1

        try:
            if propagate_backward and bootstrap_idx > 0:
                for out in _propagate(
                    bootstrap_idx,
                    bootstrap_idx,
                    reverse=True,
                ):
                    f_idx, obj_ids, mask_logits, *rest = out
                    tracker_probs = rest[0] if rest else None
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
                        if mask_leak_cleanup:
                            # qm11/qm10/qm12: drop a mask component only if BOTH
                            # small AND detached (a leak onto adjacent structure)
                            # before the box / plausibility / stored mask.
                            mask2d = clean_disconnected_leaks(mask2d)
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
                                "tracker_prob": (
                                    float(tracker_probs[i])
                                    if tracker_probs is not None
                                    and i < len(tracker_probs)
                                    else None
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
                    # Cooperative stop, per propagated frame (backward pass).
                    if should_stop is not None and should_stop():
                        raise TrackStopped()

            if not reprompting:
                _record_chunk(None, None)
            else:
                cursor = bootstrap_idx
                last_end = bootstrap_idx - 1
                while True:
                    # qm12 opening-window: a chunk that STARTS within the first
                    # `ow` frames of tracking (measured from bootstrap_idx, the
                    # cursor origin) propagates a smaller step so robots
                    # streaming into center at the period opening (0:20-0:16)
                    # seed at the next dense boundary instead of waiting a full
                    # reprompt_every. After the window the normal step resumes.
                    step = reprompt_every
                    if oe > 0 and ow > 0 and (cursor - bootstrap_idx) < ow:
                        step = min(step, oe)
                    last_idx = _record_chunk(cursor, step)
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
                                        f"{self.DIAG_PREFIX}.object_retired",
                                        details={"object_id": oid, "frame_idx": last_idx},
                                    )

                    # A retired/culled object can no longer accrue a misalign
                    # streak; prune to the active set computed above.
                    if misalign_streak:
                        active_set = set(active_objects)
                        misalign_streak = {
                            o: s for o, s in misalign_streak.items()
                            if o in active_set}

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
                    corrob_extra_boxes, corrob_extra_meta = [], []
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
                                print(f"  [{self.LOG_TAG}] detect_fn failed at frame {last_idx} "
                                      f"(further errors suppressed): {exc!r}")
                            if diagnostics:
                                diagnostics.increment(f"{self.DIAG_PREFIX}_detector_errors")
                                diagnostics.record(
                                    f"{self.DIAG_PREFIX}.detector_error",
                                    details={"frame_idx": last_idx, "error": repr(exc)},
                                )
                        if detected is None:
                            detector_available = False
                        else:
                            detector_available = True
                            all_boxes, all_metadata, all_corrob = normalize_detections(detected)
                            # Seedable detections drive matching/seeding/eviction;
                            # corrob-only detections join them ONLY for
                            # corroboration bookkeeping below.
                            new_boxes = [b for b, c in zip(all_boxes, all_corrob) if not c]
                            new_metadata = [m for m, c in zip(all_metadata, all_corrob) if not c]
                            corrob_extra_boxes = [
                                b for b, c in zip(all_boxes, all_corrob) if c]
                            corrob_extra_meta = [
                                m for m, c in zip(all_metadata, all_corrob) if c]
                            if diagnostics:
                                for box, c in zip(all_boxes, all_corrob):
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.reprompt_candidate",
                                        details={
                                            "frame_idx": last_idx,
                                            "box": list(box),
                                            "corrob_only": bool(c),
                                        },
                                    )

                    # Lazy per-boundary motion probe: True means candidate j
                    # shows real pixel motion at this boundary. A crashing
                    # motion_fn degrades to False (no override) — motion is
                    # only ever positive evidence.
                    motion_cache: dict[int, bool] = {}

                    def _candidate_moving(j: int) -> bool:
                        nonlocal motion_fn_error_logged
                        if motion_fn is None:
                            return False
                        if j not in motion_cache:
                            try:
                                motion_cache[j] = bool(motion_fn(last_idx, new_boxes[j]))
                            except Exception as exc:
                                motion_cache[j] = False
                                if not motion_fn_error_logged:
                                    motion_fn_error_logged = True
                                    print(f"  [{self.LOG_TAG}] motion_fn failed at frame "
                                          f"{last_idx} (further errors suppressed): {exc!r}")
                        return motion_cache[j]

                    # ---- Corroboration bookkeeping (windowed detector-absence +
                    # stationarity), maintained whenever the detector ran. The
                    # spontaneous hijack cull stays gated behind cull_enabled;
                    # capped-roster eviction below reuses the same evidence on
                    # demand. ----
                    if detector_available and existing:
                        h, w = next(iter(existing.values())).shape
                        # Zone-blind corroboration: a detection whose box the
                        # gate rejects (ground contact inside a static-structure
                        # zone) is invisible to corroboration/color bookkeeping.
                        # The OVERFLOW bin's bumper-lookalike false positive
                        # otherwise vouches for its own hijacked mask forever,
                        # so the absence cull can never fire. Seeding below
                        # still sees the full detection list (the seed gate has
                        # its own zone test). Fails OPEN per box: a broken
                        # projector must not blind the whole roster.
                        corrob_boxes = list(new_boxes) + corrob_extra_boxes
                        corrob_metadata = list(new_metadata) + corrob_extra_meta
                        if corrob_gate_fn is not None and corrob_boxes:
                            # Gate the COMBINED evidence list (seedable + corrob-
                            # only) — a corrob-only detection is presence evidence
                            # too, so a zone gate must vet it identically or the
                            # band silently vanishes whenever zone-blind
                            # corroboration is configured.
                            kept_boxes, kept_meta = [], []
                            for _j, _b in enumerate(corrob_boxes):
                                try:
                                    _ok = bool(corrob_gate_fn(_b))
                                except Exception as exc:
                                    _ok = True
                                    if not corrob_gate_error_logged:
                                        corrob_gate_error_logged = True
                                        print(f"  [{self.LOG_TAG}] corrob_gate_fn failed at "
                                              f"frame {last_idx} (further errors "
                                              f"suppressed): {exc!r}")
                                if _ok:
                                    kept_boxes.append(_b)
                                    kept_meta.append(
                                        corrob_metadata[_j]
                                        if _j < len(corrob_metadata) else None)
                            corrob_boxes, corrob_metadata = kept_boxes, kept_meta
                        for oid, mask in list(existing.items()):
                            corroborated = mask_corroborated(
                                mask, corrob_boxes, (w, h),
                                ios_thresh=cull_corrob_ios,
                                mask_color=(existing_colors.get(oid)
                                            if existing_colors else None),
                                new_colors=corrob_metadata,
                            )
                            hist = corroboration_history.setdefault(
                                oid, deque(maxlen=cull_evidence_window))
                            hist.append(bool(corroborated))

                            pw = prob_window.setdefault(
                                oid, deque(maxlen=cull_evidence_window))
                            pw.append(mean_or_none(prob_accum.pop(oid, [])))

                            # Color backfill for unknown-colored objects: commit
                            # the corroborating bumper color after enough
                            # CONSECUTIVE agreeing boundaries; mixed colors (two
                            # robots overlapping the mask) reset the votes. A
                            # definite color is never overridden.
                            if (object_metadata is not None
                                    and object_metadata.get(oid) not in ("red", "blue")):
                                definite = corroborating_colors(
                                    mask, corrob_boxes, (w, h),
                                    ios_thresh=cull_corrob_ios,
                                    new_colors=corrob_metadata,
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
                                                f"{self.DIAG_PREFIX}.color_backfilled",
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

                            prob_mean = mean_or_none(prob_window.get(oid, []))
                            prob_allows_cull = (
                                cull_low_prob_max <= 0
                                or prob_mean is None
                                or prob_mean < cull_low_prob_max)
                            if (cull_enabled and prob_allows_cull
                                    and absent_enough(hist, cull_uncorroborated_reprompts)
                                    and is_stationary(win, cull_stationary_m)):
                                culled_set.add(oid)
                                if diagnostics:
                                    # Uncapped counter so the cull count survives
                                    # event-list truncation (max_events) over a full
                                    # match — the at-a-glance on/off comparison signal.
                                    diagnostics.increment(f"{self.DIAG_PREFIX}_objects_culled")
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.object_culled",
                                        reason="detector_absent_stationary",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "absences": sum(1 for c in hist if not c),
                                            "window": len(hist),
                                            "prob_mean": prob_mean,
                                        },
                                    )

                            # ---- Size-prior violation bookkeeping + cull.
                            # Reuse `pos` (already projected above). Independent
                            # of corroboration: a detector-corroborated mask 4x
                            # too wide has latched onto structure the detector
                            # itself false-fires on. A None ratio (no pose /
                            # degenerate box / no expectation) is unmeasurable,
                            # never a violation. ----
                            svh = size_violation_history.setdefault(
                                oid, deque(maxlen=size_prior_boundaries))
                            ratio = size_prior_ratio(
                                box,
                                expected_px_fn(pos)
                                if (expected_px_fn is not None and pos is not None)
                                else None,
                            )
                            svh.append(size_prior_violation(
                                ratio, size_prior_hi, size_prior_lo))
                            if (size_prior_cull
                                    and len(svh) == size_prior_boundaries
                                    and all(svh)):
                                culled_set.add(oid)
                                if diagnostics:
                                    diagnostics.increment(f"{self.DIAG_PREFIX}_objects_culled")
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.object_culled",
                                        reason="size_prior_violation",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "ratio": ratio,
                                            "boundaries": len(svh),
                                        },
                                    )

                        if culled_set:
                            active_objects = [o for o in active_objects if o not in culled_set]
                            for o in list(culled_set):
                                existing.pop(o, None)
                                existing_colors.pop(o, None)

                    # ---- Person-overlap veto (positive hijack signal). A
                    # stationary mask spending consecutive boundaries mostly
                    # inside a person detection is a hijacked track — a robot
                    # is never a person — so unlike the absence-based cull
                    # above this fires independent of cull_enabled and needs
                    # no detector-absence evidence. Moving masks are spared:
                    # motion is the strongest robot-not-junk signal we have,
                    # and a robot legitimately drives past wall-side refs. ----
                    if person_fn is not None and existing:
                        pv_frame_path = (
                            frame_paths[last_idx] if last_idx < len(frame_paths) else None
                        )
                        try:
                            person_boxes = list(person_fn(last_idx, pv_frame_path) or [])
                        except Exception as exc:
                            person_boxes = []
                            if not person_fn_error_logged:
                                person_fn_error_logged = True
                                print(f"  [{self.LOG_TAG}] person_fn failed at frame "
                                      f"{last_idx} (further errors suppressed): {exc!r}")
                        from .detect_schema import box_coverage
                        veto_culled: set[int] = set()
                        for oid in list(existing):
                            box = results[last_visible_frame[oid]][oid]["box"]
                            covered = any(
                                box_coverage(box, pb) >= person_veto_coverage
                                for pb in person_boxes
                            )
                            if covered:
                                person_overlap_counts[oid] = (
                                    person_overlap_counts.get(oid, 0) + 1)
                            else:
                                person_overlap_counts.pop(oid, None)
                            if (person_overlap_counts.get(oid, 0) >= person_veto_boundaries
                                    and is_stationary(
                                        field_pos_window.get(oid, []),
                                        cull_stationary_m)):
                                veto_culled.add(oid)
                                if diagnostics:
                                    diagnostics.increment(
                                        f"{self.DIAG_PREFIX}_objects_culled")
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.object_culled",
                                        reason="person_overlap",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "consecutive_boundaries":
                                                person_overlap_counts.get(oid, 0),
                                        },
                                    )
                        if veto_culled:
                            culled_set |= veto_culled
                            active_objects = [
                                o for o in active_objects if o not in veto_culled]
                            for o in veto_culled:
                                existing.pop(o, None)
                                existing_colors.pop(o, None)
                                person_overlap_counts.pop(o, None)

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
                        new_indices, matched_claims = match_new_detections(
                            existing, int_boxes, (w, h),
                            ios_thresh=new_obj_ios_thresh,
                            existing_colors=existing_colors,
                            new_colors=new_metadata,
                            with_claims=True,
                        )
                        # Dark-mask re-anchor: a matched detection whose
                        # claimant has emitted no mask for dark_reanchor_frames
                        # is the lost robot itself — re-prompt the SAME obj id
                        # at the fresh box instead of dropping the detection as
                        # redundant (a stale mask must not both block seeding
                        # AND produce no output). Attempts are capped: each
                        # prompt accumulates predictor session state, so a
                        # never-reviving object gets dark_reanchor_max_attempts
                        # tries and then its detection seeds a NEW id.
                        dark_reanchored: set[int] = set()
                        for idx, box_val in enumerate(new_boxes):
                            if idx in new_indices:
                                continue
                            oid = matched_claims.get(idx)
                            lv = (last_visible_frame.get(oid)
                                  if oid is not None else None)
                            if oid is not None and lv is not None:
                                # A mask emitted since the last re-anchor means
                                # the attempt worked — reset the budget.
                                if lv > dark_reanchor_last.get(oid, -1):
                                    dark_reanchor_attempts[oid] = 0
                            if (dark_reanchor_enabled and lv is not None
                                    and (last_idx - lv) >= dark_reanchor_frames
                                    and dark_reanchor_attempts.get(oid, 0)
                                        >= dark_reanchor_max_attempts):
                                # Budget exhausted: promote the detection to
                                # the new-object seeding path below.
                                new_indices.append(idx)
                                if diagnostics:
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.reprompt_rejected",
                                        reason="dark_reanchor_exhausted",
                                        details={"frame_idx": last_idx,
                                                 "object_id": oid,
                                                 "box": list(box_val)},
                                    )
                                continue
                            if (dark_reanchor_enabled and lv is not None
                                    and (last_idx - lv) >= dark_reanchor_frames
                                    and oid not in dark_reanchored):
                                negs = negative_points_for(
                                    box_val,
                                    [b for q, b in enumerate(int_boxes) if q != idx]
                                ) if neg_point_prompts else []
                                _prompt_with_negatives(
                                    predictor, state, last_idx, oid, box_val, negs)
                                dark_reanchored.add(oid)
                                dark_reanchor_attempts[oid] = (
                                    dark_reanchor_attempts.get(oid, 0) + 1)
                                dark_reanchor_last[oid] = last_idx
                                # Fresh prompt re-conditions the object: stale
                                # absence evidence no longer applies, and the
                                # restored extent must be visible to the rest
                                # of this boundary (a second detection over the
                                # same robot cannot also re-anchor it).
                                corroboration_history.pop(oid, None)
                                size_violation_history.pop(oid, None)
                                prob_window.pop(oid, None)
                                prob_accum.pop(oid, None)
                                bx1, by1, bx2, by2 = int_boxes[idx]
                                refreshed = np.zeros((h, w), bool)
                                refreshed[by1:by2, bx1:bx2] = True
                                existing[oid] = refreshed
                                boundary_mask_record[oid] = (
                                    boundary_no, float(refreshed.sum()),
                                    tuple(int_boxes[idx]))
                                if diagnostics:
                                    diagnostics.increment(f"{self.DIAG_PREFIX}_objects_reanchored")
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.object_reanchored",
                                        reason="dark_mask",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "box": list(box_val),
                                            "dark_frames": last_idx - lv,
                                        },
                                    )
                            elif (oid is not None and _candidate_moving(idx)
                                    and is_stationary(
                                        field_pos_window.get(oid, []),
                                        cull_stationary_m)):
                                # Motion override: a MOVING detection claimed
                                # by a mask that has sat still all window is a
                                # robot passing static equipment/hijack, not a
                                # redundant sighting — promote it to the
                                # new-object seeding path. A claimant that is
                                # itself moving is a robot following its own
                                # detection and keeps suppressing it.
                                new_indices.append(idx)
                                if diagnostics:
                                    diagnostics.increment(
                                        f"{self.DIAG_PREFIX}_motion_overrides")
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.motion_override",
                                        details={"frame_idx": last_idx,
                                                 "box": list(box_val),
                                                 "claimant": oid},
                                    )
                            elif diagnostics:
                                diagnostics.record(
                                    f"{self.DIAG_PREFIX}.reprompt_rejected",
                                    reason="already_tracked",
                                    details={"frame_idx": last_idx, "box": list(box_val)},
                                )
                        mis_hits: set[int] = set()
                        for j in new_indices:
                            if seed_box_max_aspect > 0 and not plausible_seed_box(
                                    new_boxes[j], seed_box_max_aspect):
                                # A geometrically impossible detection (tall-thin
                                # field sliver, qm12 hub face) must do NOTHING:
                                # not re-anchor, not seed, not evict a real
                                # track. Gate at the loop top covers the
                                # motion-override additions to new_indices too.
                                if diagnostics:
                                    bx1, by1, bx2, by2 = int_boxes[j]
                                    aspect = (round((by2 - by1) / (bx2 - bx1), 2)
                                              if bx2 > bx1 else None)
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.reprompt_rejected",
                                        reason="implausible_seed_box",
                                        details={"frame_idx": last_idx,
                                                 "box": list(new_boxes[j]),
                                                 "aspect": aspect},
                                    )
                                continue
                            if misalign_reanchor and existing:
                                bx1, by1, bx2, by2 = int_boxes[j]
                                mis_box_mask = np.zeros((h, w), bool)
                                mis_box_mask[by1:by2, bx1:bx2] = True
                                mis_areas = {o: int(m.sum())
                                             for o, m in existing.items()}
                                cand, cand_ios = best_partial_claimant(
                                    existing, existing_colors, mis_areas,
                                    mis_box_mask, int(mis_box_mask.sum()),
                                    new_metadata[j] if j < len(new_metadata) else None,
                                    misalign_ios_min, new_obj_ios_thresh,
                                    SEED_MIN_MASK_AREA_FRAC)
                                if cand is not None:
                                    mis_hits.add(cand)
                                    streak = misalign_streak.get(cand, 0) + 1
                                    misalign_streak[cand] = streak
                                    if streak >= misalign_boundaries:
                                        negs = (negative_points_for(
                                            new_boxes[j],
                                            [b for q, b in enumerate(int_boxes)
                                             if q != j])
                                            if neg_point_prompts else [])
                                        _prompt_with_negatives(
                                            predictor, state, last_idx, cand,
                                            new_boxes[j], negs)
                                        misalign_streak.pop(cand, None)
                                        corroboration_history.pop(cand, None)
                                        size_violation_history.pop(cand, None)
                                        prob_window.pop(cand, None)
                                        prob_accum.pop(cand, None)
                                        existing[cand] = mis_box_mask
                                        boundary_mask_record[cand] = (
                                            boundary_no,
                                            float(mis_box_mask.sum()),
                                            tuple(int_boxes[j]))
                                        if diagnostics:
                                            diagnostics.increment(
                                                f"{self.DIAG_PREFIX}_objects_reanchored")
                                            diagnostics.record(
                                                f"{self.DIAG_PREFIX}.object_reanchored",
                                                reason="misaligned_mask",
                                                details={
                                                    "object_id": cand,
                                                    "frame_idx": last_idx,
                                                    "box": list(new_boxes[j]),
                                                    "ios": round(cand_ios, 3),
                                                    "streak": streak,
                                                },
                                            )
                                        continue
                                    # Streak building but not yet met: this
                                    # detection is provisionally attributed to
                                    # `cand` (its drift is being watched) and
                                    # must NOT seed a churned replacement id
                                    # while under evaluation — that would
                                    # defeat the whole point of waiting for
                                    # consecutive confirmation.
                                    if diagnostics:
                                        diagnostics.record(
                                            f"{self.DIAG_PREFIX}.reprompt_rejected",
                                            reason="misaligned_mask_pending",
                                            details={
                                                "frame_idx": last_idx,
                                                "object_id": cand,
                                                "box": list(new_boxes[j]),
                                                "ios": round(cand_ios, 3),
                                                "streak": streak,
                                            },
                                        )
                                    continue
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
                                    negs = negative_points_for(
                                        new_boxes[j],
                                        [b for q, b in enumerate(int_boxes) if q != j]
                                    ) if neg_point_prompts else []
                                    _prompt_with_negatives(
                                        predictor, state, last_idx, best_oid,
                                        new_boxes[j], negs)
                                    # The fresh prompt re-conditions the object;
                                    # sliver-era absence evidence is stale, and the
                                    # restored extent must be visible to the rest
                                    # of this boundary (so a second detection over
                                    # the same robot can't also re-anchor it).
                                    corroboration_history.pop(best_oid, None)
                                    size_violation_history.pop(best_oid, None)
                                    prob_window.pop(best_oid, None)
                                    prob_accum.pop(best_oid, None)
                                    existing[best_oid] = reanchor_box_mask
                                    boundary_mask_record[best_oid] = (
                                        boundary_no, float(reanchor_box_area),
                                        tuple(int_boxes[j]))
                                    if diagnostics:
                                        diagnostics.increment(f"{self.DIAG_PREFIX}_objects_reanchored")
                                        diagnostics.record(
                                            f"{self.DIAG_PREFIX}.object_reanchored",
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
                                        f"{self.DIAG_PREFIX}.reprompt_rejected",
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
                            # Motion-required eviction: a STATIC candidate must
                            # not displace a tracked robot (qm12 — a static hub
                            # sliver evicted a real object). Only blocks when
                            # motion evidence is actually available; with
                            # motion_fn None, _candidate_moving is always False
                            # and this gate stays inert so eviction is unchanged.
                            evict_blocked_static = (
                                evict_requires_motion
                                and motion_fn is not None
                                and not _candidate_moving(j))
                            if (roster_capped and not lifetime_capped
                                    and evict_when_capped and evict_blocked_static):
                                if diagnostics:
                                    diagnostics.record(
                                        f"{self.DIAG_PREFIX}.eviction_skipped",
                                        reason="static_candidate",
                                        details={"frame_idx": last_idx,
                                                 "box": list(new_boxes[j])},
                                    )
                            elif roster_capped and not lifetime_capped and evict_when_capped:
                                # An unmatched detection is concrete evidence a
                                # roster slot is wrongly held. The roster keeps
                                # the objects most likely to be robots: evict
                                # the weakest-evidence member (stationary-
                                # absent, then dark, then least-corroborated)
                                # rather than going blind.
                                victim = pick_evictable(
                                    active_objects, corroboration_history,
                                    field_pos_window,
                                    cull_uncorroborated_reprompts,
                                    cull_stationary_m,
                                    last_visible_frame=last_visible_frame,
                                    current_idx=last_idx,
                                    dark_frames=evict_dark_frames,
                                    min_absences=evict_min_absences,
                                    moving_candidate=_candidate_moving(j),
                                    prob_means={
                                        o: mean_or_none(prob_window.get(o, []))
                                        for o in active_objects},
                                    prob_immune=evict_prob_immune,
                                )
                                if victim is not None:
                                    culled_set.add(victim)
                                    existing.pop(victim, None)
                                    existing_colors.pop(victim, None)
                                    roster_capped = False
                                    if diagnostics:
                                        diagnostics.increment(f"{self.DIAG_PREFIX}_objects_culled")
                                        diagnostics.record(
                                            f"{self.DIAG_PREFIX}.object_culled",
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
                                        f"{self.DIAG_PREFIX}.reprompt_rejected",
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
                            negs = negative_points_for(
                                new_boxes[j],
                                [b for q, b in enumerate(int_boxes) if q != j]
                            ) if neg_point_prompts else []
                            _prompt_with_negatives(
                                predictor, state, last_idx, next_obj_id,
                                new_boxes[j], negs)
                            first_prompted_frame[next_obj_id] = last_idx
                            birth_detector_overlap[next_obj_id] = best_ios
                            boundary_mask_record[next_obj_id] = (
                                boundary_no, _box_area(new_boxes[j]),
                                tuple(int_boxes[j]))
                            if diagnostics:
                                diagnostics.record(
                                    f"{self.DIAG_PREFIX}.object_created",
                                    details={"object_id": next_obj_id, "frame_idx": last_idx,
                                             "box": list(new_boxes[j])},
                                )
                            if object_metadata is not None:
                                object_metadata[next_obj_id] = new_metadata[j]
                            if late_seed_backfill:
                                # Only a genuinely NEW id qualifies — this site
                                # is reached only when nothing re-anchored the
                                # detection under an existing id (every
                                # re-anchor path above `continue`s before here).
                                backfill_requests.append(
                                    (last_idx, next_obj_id, tuple(new_boxes[j])))
                            next_obj_id += 1

                        # Consecutiveness: an object with no partial-claimant
                        # hit THIS boundary had its drift streak broken (a
                        # one-boundary flyby, or the mask re-aligned) -- drop it
                        # so a later hit starts a fresh streak.
                        misalign_streak = {
                            o: s for o, s in misalign_streak.items()
                            if o in mis_hits}

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

            # ---- Late-seed backward backfill ----
            # The forward loop is done: every surviving late-seeded object's
            # full forward life is already in `results`. Re-prompt each one at
            # its OWN seed frame and propagate backward a bounded span,
            # recovering the pre-seed history YOLO missed by seeding late.
            if late_seed_backfill and backfill_requests:
                # Filter to objects worth the compute: culled objects have
                # nothing to recover, and a short-lived seed is frequently
                # roster pollution from a static structure (qm12) rather than
                # a real robot whose early history is worth chasing.
                surviving = []
                for seed_frame, oid, box in backfill_requests:
                    if oid in culled_set:
                        continue
                    lv = last_visible_frame.get(oid)
                    if lv is None:
                        continue
                    if lv - seed_frame < backfill_min_lived_frames:
                        continue
                    surviving.append((seed_frame, oid, box))

                # Group by seed frame and walk DESCENDING — minimizes SAM3
                # reverse-chunk churn (a later seed's reverse span is a subset
                # of an earlier seed's, so processing late-to-early reuses
                # freshly-opened chunks instead of reopening them).
                groups: dict[int, list] = {}
                for seed_frame, oid, box in surviving:
                    groups.setdefault(seed_frame, []).append((oid, box))

                total_objects_backfilled = 0
                total_frames_recovered = 0
                for seed_frame in sorted(groups, reverse=True):
                    targets = groups[seed_frame]
                    target_ids = {oid for oid, _box in targets}
                    if isinstance(state, dict):
                        # Adapter scoping (SAM3): confine the reverse pass to
                        # ONLY these ids so the carried-roster re-seed loops
                        # don't drag every other live object backward too.
                        # Harmless for real SAM2 / plain stubs.
                        state["solo_ids"] = target_ids
                    for oid, box in targets:
                        predictor.add_new_points_or_box(
                            state, frame_idx=seed_frame, obj_id=oid,
                            box=np.asarray(box, dtype=np.float32))

                    span = min(backfill_max_frames, seed_frame)
                    per_object_frames = {oid: 0 for oid, _box in targets}
                    dark_streak = 0
                    dark_stopped = False
                    if span > 0:
                        for out in _propagate(seed_frame, span, reverse=True):
                            f_idx, obj_ids, mask_logits, *rest = out
                            tracker_probs = rest[0] if rest else None
                            f_idx_int = int(f_idx)
                            bool_arr = (mask_logits > 0).cpu().numpy()
                            stored_this_frame = False
                            for i, obj_id in enumerate(obj_ids):
                                oid_int = int(obj_id)
                                if oid_int not in target_ids:
                                    continue  # this group backfills ONLY its own targets
                                if bool_arr.ndim == 4:
                                    mask2d = bool_arr[i, 0].astype(bool)
                                elif bool_arr.ndim == 3:
                                    mask2d = bool_arr[i].astype(bool)
                                else:
                                    mask2d = bool_arr.astype(bool)
                                if mask_leak_cleanup:
                                    mask2d = clean_disconnected_leaks(mask2d)
                                box_rec = mask_to_box(mask2d)
                                if box_rec is None or not plausible_robot_mask(mask2d):
                                    continue
                                entry = results.setdefault(f_idx_int, {})
                                if oid_int in entry:
                                    continue  # merge, never clobber — already tracked
                                # Overlap guard: a recovered mask that heavily
                                # overlaps another object's mask ALREADY in this
                                # frame is most likely a re-detection of an
                                # already-tracked robot under this new id —
                                # storing it would double-count the robot.
                                overlaps_other = any(
                                    _ios(mask2d, other["mask"]) > BACKFILL_OVERLAP_IOS
                                    for other_oid, other in entry.items()
                                    if other_oid != oid_int
                                )
                                if overlaps_other:
                                    continue
                                entry[oid_int] = {
                                    "box": box_rec,
                                    "mask": mask2d,
                                    "evidence": {
                                        "type": "backfilled",
                                        "gap_len": 0,
                                        "detector_overlap": birth_detector_overlap.get(
                                            oid_int, 0.0),
                                        "tracker_prob": (
                                            float(tracker_probs[i])
                                            if tracker_probs is not None
                                            and i < len(tracker_probs)
                                            else None
                                        ),
                                    },
                                }
                                per_object_frames[oid_int] += 1
                                total_frames_recovered += 1
                                stored_this_frame = True
                            if stored_this_frame:
                                dark_streak = 0
                            else:
                                dark_streak += 1
                                if dark_streak >= BACKFILL_DARK_STOP:
                                    dark_stopped = True
                                    break
                            # Cooperative stop, same contract as the other loops.
                            if should_stop is not None and should_stop():
                                raise TrackStopped()

                    if isinstance(state, dict):
                        state["solo_ids"] = None

                    if dark_stopped:
                        stop_reason = "dark"
                    elif seed_frame - span <= 0:
                        stop_reason = "video_start"
                    else:
                        stop_reason = "span"
                    for oid, _box in targets:
                        n = per_object_frames.get(oid, 0)
                        if diagnostics:
                            diagnostics.record(
                                f"{self.DIAG_PREFIX}.backfill",
                                details={
                                    "object_id": oid,
                                    "seed_frame": seed_frame,
                                    "frames_recovered": n,
                                    "stop_reason": stop_reason,
                                },
                            )
                        if n > 0:
                            total_objects_backfilled += 1
                            if diagnostics:
                                diagnostics.increment(
                                    f"{self.DIAG_PREFIX}_backfilled_objects")

                print(f"  [{self.LOG_TAG}] late-seed backfill: "
                      f"{total_objects_backfilled} objects, "
                      f"{total_frames_recovered} frames recovered")
        finally:
            if hasattr(predictor, "reset_state"):
                predictor.reset_state(state)

        return results
