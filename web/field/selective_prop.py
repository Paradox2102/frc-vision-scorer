"""Image-space tracking-by-detection (TBD) tracker with assignment margins.

Re-implementation of the cheap per-frame tracker from arXiv 2606.13033. Every
frame we run a Hungarian match between constant-velocity track predictions and
raw detections using an expanded-IoU cost (Deep-EIoU style). The *margin* of
each assignment -- how much cheaper the winning track is than the next-best
competitor for the same detection (paper Eq. 1) -- is the ambiguity signal that
Task 3's window finder consumes to decide where expensive SAM3 mask propagation
is actually needed.

Pure logic: numpy + scipy.optimize.linear_sum_assignment only. No torch, no GPU.
Determinism: tracks are iterated in oid order (cost-matrix rows), detections in
input order (columns); nothing depends on RNG or wall-clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost assigned to a track/detection pair whose expanded boxes do not overlap.
# Large but finite so linear_sum_assignment stays well-conditioned; any such
# match is discarded afterwards and never counts as a margin competitor.
FORBIDDEN = 1e6
# Additive penalty when a detection's known color contradicts the track's.
COLOR_PENALTY = 0.3


@dataclass
class SelectiveConfig:
    """Knobs for the selective-mask-propagation pipeline.

    Mirrors the ``selective_*`` knobs registered in web/job_config.py (Task 1).
    Only the TBD-tracker-relevant fields are consumed by this module; the rest
    are carried for Tasks 3-4 (window finder / propagation gate).
    """
    margin_entry: float = 0.05
    margin_exit: float = 0.10      # also SEED_MARGIN
    min_streak: int = 5            # SEED_CONSECUTIVE and EXIT_CONSECUTIVE
    gap_trigger: int = 7
    window_max_frames: int = 120
    seed_clean_iou: float = 0.10   # not knob-exposed (paper constant)
    ioma_exit: float = 0.80        # not knob-exposed
    mask_overlap_exit: float = 0.90
    area_degradation: float = 0.25
    border_margin_px: int = 20
    max_missed: int = 30           # TBD track death (die when missed > this)
    iou_expand: float = 0.25       # expanded-IoU box inflation factor
    sam_budget_frac: float = 1.0   # SAM-frame budget as a fraction of n_frames
    kill_coast: bool = False       # coast killed objects instead of raw TBD
    kill_coast_frames: int = 24    # max frames to coast a killed object
    app_weight: float = 0.0        # appearance cost weight; 0 = OFF (exact no-op)
    app_ema: float = 0.9           # track-embedding EMA retention on confident matches


@dataclass
class FrameObs:
    """One (frame, oid) TBD observation.

    ``box`` is the matched (or spawning) detection's box in export coords;
    ``margin`` is this assignment's margin (inf when unambiguous / new / solo);
    ``color`` is the detection's raw color for this frame (None if unknown).
    """
    box: tuple[int, int, int, int]
    margin: float
    color: str | None


def _expand_box(box, expand: float):
    """Inflate a box by ``expand`` fraction of its own width/height per side."""
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * expand
    dy = (y2 - y1) * expand
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def expanded_iou(a, b, expand: float) -> float:
    """IoU of two boxes after each is inflated by ``expand`` per side.

    Deep-EIoU-style expansion helps fast movers whose raw boxes barely touch.
    Returns 0.0 when the expanded boxes do not overlap.
    """
    ax1, ay1, ax2, ay2 = _expand_box(a, expand)
    bx1, by1, bx2, by2 = _expand_box(b, expand)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class _Track:
    """Internal per-object state (never leaked across the public interface)."""

    __slots__ = ("oid", "box", "color", "_centers", "missed", "last_seen", "emb")

    def __init__(self, oid: int, box, color, frame_idx: int, emb=None):
        self.oid = oid
        self.box = tuple(int(v) for v in box)
        self.color = color            # sticky: first *known* color wins
        self._centers = [_center(box)]
        self.missed = 0
        self.last_seen = frame_idx
        self.emb = None if emb is None else np.asarray(emb, dtype=np.float32)

    def velocity(self):
        """Average of the last <=3 center deltas; (0, 0) for new tracks."""
        c = self._centers[-4:]
        if len(c) < 2:
            return (0.0, 0.0)
        dxs = [c[i][0] - c[i - 1][0] for i in range(1, len(c))]
        dys = [c[i][1] - c[i - 1][1] for i in range(1, len(c))]
        return (sum(dxs) / len(dxs), sum(dys) / len(dys))

    def predicted_box(self):
        """Last box translated by the constant-velocity estimate."""
        vx, vy = self.velocity()
        x1, y1, x2, y2 = self.box
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def observe(self, box, color, frame_idx: int):
        self.box = tuple(int(v) for v in box)
        self._centers.append(_center(box))
        if self.color is None and color is not None:
            self.color = color        # first known color sticks
        self.missed = 0
        self.last_seen = frame_idx

    def update_emb(self, det_emb, ema: float):
        """EMA toward ``det_emb`` (re-L2-normalized). Caller gates on margin."""
        if self.emb is None:
            self.emb = np.asarray(det_emb, dtype=np.float32)
            return
        mixed = ema * self.emb + (1.0 - ema) * np.asarray(det_emb, dtype=np.float32)
        norm = float(np.linalg.norm(mixed))
        self.emb = (mixed / norm).astype(np.float32) if norm > 0.0 \
            else np.asarray(det_emb, dtype=np.float32)


class TbdTracker:
    """Deterministic tracking-by-detection with per-assignment margins."""

    def __init__(self, cfg: SelectiveConfig, first_oid: int = 1):
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_oid = first_oid
        self._history: dict[int, dict[int, FrameObs]] = {}
        self._gap_reappearances: list[tuple[int, int, int]] = []

    # -- public read-only interface (consumed by Tasks 3-4) ----------------
    @property
    def history(self) -> dict[int, dict[int, FrameObs]]:
        return self._history

    @property
    def gap_reappearances(self) -> list[tuple[int, int, int]]:
        return self._gap_reappearances

    @property
    def next_oid(self) -> int:
        return self._next_oid

    # -- core step ---------------------------------------------------------
    def _pair_cost(self, track: _Track, det_box, det_color, det_emb=None) -> float:
        eiou = expanded_iou(track.predicted_box(), det_box, self.cfg.iou_expand)
        if eiou <= 0.0:
            return FORBIDDEN
        cost = 1.0 - eiou
        if (det_color is not None and track.color is not None
                and det_color != track.color):
            cost += COLOR_PENALTY
        # Appearance (blue-gap fix #3): same-color pairs only -- cross-color
        # is already penalized above; this term separates same-alliance
        # robots that motion+IoU cannot. 1 - cosine in [0, 2].
        if (self.cfg.app_weight > 0.0 and det_emb is not None
                and track.emb is not None and det_color is not None
                and det_color == track.color):
            cost += self.cfg.app_weight * (1.0 - float(np.dot(track.emb, det_emb)))
        return cost

    def emb_of(self, oid: int):
        """Read-only view of a live track's EMA embedding (None if absent)."""
        trk = self._tracks.get(oid)
        return None if trk is None else trk.emb

    def step(self, frame_idx: int,
             dets: list[tuple]) -> dict[int, FrameObs]:
        """Advance one frame.

        ``dets = [(box_xyxy, color), ...]`` or ``[(box_xyxy, color, emb), ...]``
        where ``emb`` is an ``np.ndarray|None`` appearance vector. Returns this
        frame's matched and newly-spawned observations keyed by oid. Also
        appends to ``history``.
        """
        dets = [(d[0], d[1], d[2] if len(d) > 2 else None) for d in dets]
        live = sorted(self._tracks.values(), key=lambda t: t.oid)  # rows by oid
        obs: dict[int, FrameObs] = {}
        matched_oids: set[int] = set()
        matched_dets: set[int] = set()

        if live and dets:
            n, m = len(live), len(dets)
            cost = np.empty((n, m), dtype=float)
            for i, trk in enumerate(live):
                for j, (dbox, dcolor, demb) in enumerate(dets):
                    cost[i, j] = self._pair_cost(trk, dbox, dcolor, demb)

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                mcost = cost[r, c]
                if mcost >= FORBIDDEN:
                    continue                       # discard forbidden matches
                trk = live[r]
                dbox, dcolor, demb = dets[c]
                # Margin (paper Eq. 1): cheapest finite competitor in this
                # detection's column minus the winning cost. No finite
                # competitor -> unambiguous -> inf.
                competitors = [cost[i, c] for i in range(n)
                               if i != r and cost[i, c] < FORBIDDEN]
                margin = (min(competitors) - mcost) if competitors else float("inf")
                # Gap reappearance: a still-alive track that went dark for
                # >= gap_trigger frames and now matches again.
                if trk.missed >= self.cfg.gap_trigger:
                    self._gap_reappearances.append(
                        (trk.oid, trk.last_seen, frame_idx))
                trk.observe(dbox, dcolor, frame_idx)
                if (self.cfg.app_weight > 0.0 and demb is not None
                        and margin >= self.cfg.margin_exit):
                    trk.update_emb(demb, self.cfg.app_ema)
                matched_oids.add(trk.oid)
                matched_dets.add(c)
                obs[trk.oid] = FrameObs(
                    box=tuple(int(v) for v in dbox),
                    margin=float(margin),
                    color=dcolor,
                )

        # Unmatched live tracks coast (miss counter climbs).
        for trk in live:
            if trk.oid not in matched_oids:
                trk.missed += 1

        # Unmatched detections spawn fresh tracks (input order); margin inf.
        for j, (dbox, dcolor, demb) in enumerate(dets):
            if j in matched_dets:
                continue
            oid = self._next_oid
            self._next_oid += 1
            self._tracks[oid] = _Track(
                oid, dbox, dcolor, frame_idx,
                emb=demb if self.cfg.app_weight > 0.0 else None)
            obs[oid] = FrameObs(
                box=tuple(int(v) for v in dbox),
                margin=float("inf"),
                color=dcolor,
            )

        # Reap dead tracks: die when missed exceeds max_missed (never revive).
        dead = [oid for oid, trk in self._tracks.items()
                if trk.missed > self.cfg.max_missed]
        for oid in dead:
            del self._tracks[oid]

        # Emit in oid order for stable, deterministic output.
        obs = {oid: obs[oid] for oid in sorted(obs)}
        self._history[frame_idx] = obs
        return obs


# ==========================================================================
# Task 3: offline window construction, exit state machine, sam2_raw merge.
#
# All three surfaces are pure logic over the completed TBD ``history`` /
# ``gap_reappearances`` produced above -- no predictor, no torch. Task 4
# orchestrates ``ExitMachine`` against the real SAM predictor and then calls
# ``merge_selective_results``. Everything here iterates in sorted order and
# uses no RNG, so a fixed history yields a byte-identical result.
# ==========================================================================

# Healthy exit states contribute masks to the merged stream; abort states
# leave the pure-TBD output standing (paper: "accept only on confident
# contradiction").
_HEALTHY_STATES = frozenset({"clean", "swap", "end"})


@dataclass
class WindowSpec:
    """One selective-propagation window: seed SAM at ``seed_frame`` for ``oid``
    and evaluate the exit machine from ``entry_frame`` onward."""
    oid: int
    seed_frame: int
    entry_frame: int
    trigger: str                    # "margin" | "gap" | "witness"


@dataclass
class WindowOutcome:
    """Result of running a window through :class:`ExitMachine`.

    ``masks`` (and the parallel ``probs``) are populated ONLY for healthy
    states, and only for frames in ``[entry_frame, close_frame]``. ``probs`` is
    an internal extension carrying the SAM ``tracker_prob`` per masked frame so
    :func:`merge_selective_results` can stamp it into each ``selective_sam``
    evidence dict; Task 4 passes outcomes straight through and need not read it.
    """
    spec: WindowSpec
    state: str                      # clean|swap|stale|edge|degraded|killed|end
    swap_to: int | None
    close_frame: int
    masks: dict[int, "np.ndarray"] = field(default_factory=dict)
    probs: dict[int, float | None] = field(default_factory=dict)
    # Kill-coast seed (Task 2, selective_kill_coast): the frame of the last
    # confident SAM box before the convergence streak, and the short trailing
    # run of SAM boxes (<=4, ascending) used to coast it at constant velocity.
    # Only populated for ``killed`` outcomes when ``cfg.kill_coast`` is on;
    # ``merge_selective_results`` consumes them. Default None/[] leaves every
    # existing outcome (and the knob-off kill path) byte-identical.
    coast_seed_frame: int | None = None
    coast_boxes: list = field(default_factory=list)


@dataclass
class RenameEvent:
    """Pairwise oid swap (a<->b) effective from ``frame`` (= a SWAP entry)."""
    frame: int
    a: int
    b: int


# -- geometry / mask helpers (plain numpy, box coords are export-space) ------
def _iou(a, b) -> float:
    """Plain IoU of two ``(x1, y1, x2, y2)`` boxes; 0.0 when disjoint."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _pixels_in_box(mask: np.ndarray, box) -> int:
    """Count True pixels of ``mask`` falling inside ``box`` (x2/y2 exclusive)."""
    h, w = mask.shape
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(mask[y1:y2, x1:x2].sum())


def _mask_bbox(mask: np.ndarray):
    """Tight ``(x1, y1, x2, y2)`` bbox (x2/y2 exclusive) or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _near_border(mask: np.ndarray, frame_hw, margin_px: int) -> bool:
    """True when the mask's bbox comes within ``margin_px`` of any frame edge."""
    bbox = _mask_bbox(mask)
    if bbox is None:
        return False
    h, w = frame_hw
    x1, y1, x2, y2 = bbox
    return x1 < margin_px or y1 < margin_px or (w - x2) < margin_px or (h - y2) < margin_px


def _pixel_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two bool masks; 0.0 when the union is empty."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


# -- window finder -----------------------------------------------------------
def _seed_walk(history, oid: int, entry_frame: int, cfg: SelectiveConfig):
    """Backward seed walk. Return the earliest frame of the closest run of
    ``min_streak`` consecutive *clean* frames strictly before ``entry_frame``,
    or None when no such run exists.

    A frame is clean for ``oid`` when the oid is present, its assignment margin
    is >= ``margin_exit`` (unambiguous), and no other oid's box overlaps it with
    IoU > ``seed_clean_iou``. The walk starts at ``entry_frame - 1`` so the seed
    run lies wholly before the ambiguity; the ``min_streak`` warm-up frames
    between seed and entry are exactly this run.
    """
    streak = 0
    for f in range(entry_frame - 1, -1, -1):
        obs = history.get(f, {})
        clean = oid in obs and obs[oid].margin >= cfg.margin_exit
        if clean:
            box = obs[oid].box
            for j in obs:
                if j != oid and _iou(box, obs[j].box) > cfg.seed_clean_iou:
                    clean = False
                    break
        if clean:
            streak += 1
            if streak >= cfg.min_streak:
                return f
        else:
            streak = 0
    return None


def find_windows(history, gap_reappearances, n_frames, cfg: SelectiveConfig):
    """Offline window construction over the completed TBD history.

    Margin triggers (any matched oid with ``margin < margin_entry``) and gap
    triggers (each recorded ``(oid, last_seen, reappear)``) open candidates;
    every other oid whose box overlaps a trigger oid's box at the entry frame
    becomes a "witness" candidate. Each candidate gets its own backward seed
    walk and is dropped if no clean seed exists. Candidates are de-duplicated on
    ``(oid, entry_frame)`` (margin > gap > witness priority) and the surviving
    :class:`WindowSpec`s are sorted by ``(seed_frame, entry_frame, oid)``.
    """
    # (oid, entry) -> trigger, first insert wins (priority: margin, gap, witness).
    candidates: dict[tuple[int, int], str] = {}

    def _add(oid, entry, trigger):
        candidates.setdefault((int(oid), int(entry)), trigger)

    # (1) margin triggers.
    for f in sorted(history):
        for oid in sorted(history[f]):
            if history[f][oid].margin < cfg.margin_entry:
                _add(oid, f, "margin")
    # (2) gap triggers (entry = last_seen).
    for oid, last_seen, _reappear in sorted(gap_reappearances):
        _add(oid, last_seen, "gap")

    # (3) witness expansion off the primary (margin/gap) triggers only.
    for (oid, entry), _trig in list(candidates.items()):
        frame_obs = history.get(entry, {})
        if oid not in frame_obs:
            continue
        obox = frame_obs[oid].box
        for other in sorted(frame_obs):
            if other != oid and _iou(obox, frame_obs[other].box) > 0.0:
                _add(other, entry, "witness")

    specs = []
    for (oid, entry), trigger in candidates.items():
        seed = _seed_walk(history, oid, entry, cfg)
        if seed is None:
            continue
        specs.append(WindowSpec(oid=oid, seed_frame=seed,
                                entry_frame=entry, trigger=trigger))
    specs.sort(key=lambda w: (w.seed_frame, w.entry_frame, w.oid))
    return specs


def cluster_windows(specs: list["WindowSpec"],
                    cfg: SelectiveConfig) -> list[list["WindowSpec"]]:
    """Merge windows into episodes by padded-interval overlap (Design 1).

    Each spec covers a padded interval ``[seed_frame - min_streak,
    entry_frame + min_streak]``; specs whose padded intervals overlap OR merely
    touch (``lo <= running_hi``) share one episode, so a single SAM session can
    host them. Standard sort-and-sweep interval merge.

    The boundary is pinned by the touch rule: two point windows (seed == entry)
    ``min_streak`` apart on each side -- i.e. ``gap == 2*min_streak`` -- touch
    and merge, while ``gap == 2*min_streak + 1`` stays separate.

    Deterministic: specs are sorted internally by ``(seed_frame, entry_frame,
    oid)`` before sweeping, so a shuffled input yields an identical partition.
    Returns episodes ordered by episode start (min seed); within an episode the
    specs keep that same ``(seed_frame, entry_frame, oid)`` order.
    """
    if not specs:
        return []
    k = cfg.min_streak
    ordered = sorted(specs, key=lambda w: (w.seed_frame, w.entry_frame, w.oid))
    episodes: list[list[WindowSpec]] = [[ordered[0]]]
    cur_hi = ordered[0].entry_frame + k          # running max padded high
    for spec in ordered[1:]:
        lo = spec.seed_frame - k
        if lo <= cur_hi:                         # overlap or touch -> same episode
            episodes[-1].append(spec)
            cur_hi = max(cur_hi, spec.entry_frame + k)
        else:
            episodes.append([spec])
            cur_hi = spec.entry_frame + k
    return episodes


# -- exit state machine ------------------------------------------------------
class _WinState:
    """Mutable per-window bookkeeping used inside :class:`ExitMachine`."""

    __slots__ = ("spec", "state", "swap_to", "close_frame", "seed_area",
                 "dark_streak", "settle_oid", "settle_streak", "last_near_border",
                 "masks", "probs", "coast_seed_frame", "coast_boxes")

    def __init__(self, spec: WindowSpec):
        self.spec = spec
        self.state: str | None = None       # None == still open
        self.swap_to: int | None = None
        self.close_frame: int | None = None
        self.seed_area = 0
        self.dark_streak = 0
        self.settle_oid: int | None = None
        self.settle_streak = 0
        self.last_near_border = False
        self.masks: dict[int, np.ndarray] = {}
        self.probs: dict[int, float | None] = {}
        # Kill-coast seed, captured at kill time before _close wipes ``masks``.
        self.coast_seed_frame: int | None = None
        self.coast_boxes: list = []


class ExitMachine:
    """Runs a seed-frame group of windows against propagated masks/probs.

    Fed one propagated frame at a time in ascending order from ``seed_frame``.
    Frames between seed and each window's ``entry_frame`` only warm the mask;
    at ``entry_frame`` the seeded mask is checked for STALE, and from there on
    the window is watched for DEGRADED / EDGE / killed aborts and for a
    sustained settle (CLEAN / SWAP). A window still open at
    ``seed_frame + window_max_frames`` (or at ``outcomes()`` time, i.e. sequence
    end) closes healthy as END.
    """

    def __init__(self, group, seed_boxes, frame_hw, history, cfg: SelectiveConfig):
        self.cfg = cfg
        self.history = history
        self.frame_hw = frame_hw
        # Stored but not read here: carried for interface compatibility with
        # Task 4's call site (the orchestrator prompts SAM with these boxes).
        self.seed_boxes = seed_boxes
        self._wins = [_WinState(s) for s in group]
        self._pair_streak: dict[tuple[int, int], int] = {}
        self._last_frame: int | None = None

    # -- per-frame settle candidate -----------------------------------------
    def _settle_candidate(self, frame_idx: int, mask: np.ndarray):
        """The single oid whose isolated, unambiguous TBD box contains the mask
        (IoMA >= ioma_exit), or None when the frame does not cleanly settle."""
        frame_obs = self.history.get(frame_idx, {})
        total = int(mask.sum())
        if total == 0:
            return None
        qualifying = [j for j in sorted(frame_obs)
                      if _pixels_in_box(mask, frame_obs[j].box) / total >= self.cfg.ioma_exit]
        if len(qualifying) != 1:
            return None
        cand = qualifying[0]
        if frame_obs[cand].margin < self.cfg.margin_exit:
            return None
        cbox = frame_obs[cand].box
        for j in frame_obs:
            if j != cand and _iou(cbox, frame_obs[j].box) > 0.0:
                return None                 # box not isolated
        return cand

    def _close(self, w: _WinState, state: str, frame_idx: int, swap_to=None):
        w.state = state
        w.close_frame = frame_idx
        w.swap_to = swap_to
        if state not in _HEALTHY_STATES:
            w.masks = {}                    # aborts contribute no masks
            w.probs = {}

    def _eval_window(self, w: _WinState, frame_idx: int, mask, prob):
        if w.state is not None:
            return
        spec = w.spec
        # Seed gate (Design 3): a window is COMPLETELY inert for any frame
        # before its own seed, so one episode session can host windows with
        # different seed frames. No seed-area capture, no dark counting, no
        # settle -- the window simply does not exist yet this frame.
        if frame_idx < spec.seed_frame:
            return
        dark = mask is None or not bool(mask.any())
        if frame_idx == spec.seed_frame:
            w.seed_area = 0 if dark else int(mask.sum())
        if frame_idx < spec.entry_frame:
            return                          # warm-up only (no settle evaluation)

        # STALE is evaluated once, at the entry frame.
        if frame_idx == spec.entry_frame:
            obs = self.history.get(frame_idx, {})
            box = obs[spec.oid].box if spec.oid in obs else None
            if dark or box is None or _pixels_in_box(mask, box) == 0:
                self._close(w, "stale", frame_idx)
                return

        if not dark:
            w.masks[frame_idx] = mask.astype(bool)
            w.probs[frame_idx] = prob

        if dark:
            if w.last_near_border:
                self._close(w, "edge", frame_idx)
                return
            w.dark_streak += 1
            w.settle_oid = None
            w.settle_streak = 0
            if w.dark_streak >= self.cfg.min_streak:
                self._close(w, "degraded", frame_idx)
            return

        w.dark_streak = 0
        area = int(mask.sum())
        if w.seed_area > 0 and area < self.cfg.area_degradation * w.seed_area:
            self._close(w, "degraded", frame_idx)
            return
        w.last_near_border = _near_border(mask, self.frame_hw, self.cfg.border_margin_px)

        cand = self._settle_candidate(frame_idx, mask)
        if cand is None:
            w.settle_oid = None
            w.settle_streak = 0
            return
        if cand == w.settle_oid:
            w.settle_streak += 1
        else:
            w.settle_oid = cand
            w.settle_streak = 1
        if w.settle_streak >= self.cfg.min_streak:
            if cand == spec.oid:
                self._close(w, "clean", frame_idx)
            else:
                self._close(w, "swap", frame_idx, swap_to=cand)

    def _capture_coast(self, w: _WinState, close_frame: int):
        """Record ``w``'s last confident SAM boxes before the convergence streak
        so :func:`merge_selective_results` can coast the killed object instead of
        dumping it to raw TBD (Task 2, ``selective_kill_coast``).

        The kill fires on ``min_streak`` consecutive mask-overlap frames ending
        at ``close_frame``, so the last pre-convergence frame is at or before
        ``close_frame - min_streak``. We take the trailing run (<=4) of stored
        live-mask frames up to that cutoff and their tight bboxes; the last is
        the coast seed box, the run gives a constant-velocity estimate. No-op
        when the knob is off or no such confident frame exists (coast then
        degrades to today's raw-TBD fallback). Must run before :meth:`_close`
        wipes ``w.masks``, and captures only once (the first kill pairing wins).
        """
        if not self.cfg.kill_coast or w.coast_seed_frame is not None:
            return
        cutoff = close_frame - self.cfg.min_streak
        frames = sorted(f for f in w.masks if f <= cutoff)
        if not frames:
            return
        fb = [(f, _mask_bbox(w.masks[f])) for f in frames[-4:]]
        fb = [(f, b) for f, b in fb if b is not None]
        if not fb:
            return
        w.coast_seed_frame = fb[-1][0]
        w.coast_boxes = [b for _, b in fb]

    def _check_killed(self, frame_idx: int, masks_by_oid):
        """Two windows whose masks overlap (pixel-IoU >= mask_overlap_exit) for
        ``min_streak`` consecutive frames have collapsed onto one object -> both
        die as ``killed``."""
        # Seed gate (Design 3): a pre-seed window is excluded from the kill set
        # on BOTH sides -- it can neither kill nor be killed nor perturb another
        # window's pairwise isolation streak until its own seed frame arrives.
        active = [w for w in self._wins
                  if w.state is None
                  and frame_idx >= w.spec.seed_frame
                  and frame_idx >= w.spec.entry_frame]
        active.sort(key=lambda w: w.spec.oid)
        for i, wa in enumerate(active):
            for wb in active[i + 1:]:
                key = (wa.spec.oid, wb.spec.oid)
                ma = masks_by_oid.get(wa.spec.oid)
                mb = masks_by_oid.get(wb.spec.oid)
                if (ma is None or mb is None or not ma.any() or not mb.any()
                        or _pixel_iou(ma, mb) < self.cfg.mask_overlap_exit):
                    self._pair_streak[key] = 0
                    continue
                self._pair_streak[key] = self._pair_streak.get(key, 0) + 1
                if self._pair_streak[key] >= self.cfg.min_streak:
                    # Capture coast seeds BEFORE _close wipes each window's masks.
                    self._capture_coast(wa, frame_idx)
                    self._capture_coast(wb, frame_idx)
                    self._close(wa, "killed", frame_idx)
                    self._close(wb, "killed", frame_idx)

    def step(self, frame_idx: int, masks_by_oid, probs_by_oid) -> bool:
        """Feed one propagated frame. Returns True once every window is closed."""
        self._last_frame = frame_idx
        probs_by_oid = probs_by_oid or {}
        for w in self._wins:
            self._eval_window(w, frame_idx,
                              masks_by_oid.get(w.spec.oid),
                              probs_by_oid.get(w.spec.oid))
        self._check_killed(frame_idx, masks_by_oid)
        # window budget exhausted -> healthy END.
        for w in self._wins:
            if w.state is None and frame_idx >= w.spec.seed_frame + self.cfg.window_max_frames:
                self._close(w, "end", frame_idx)
        return all(w.state is not None for w in self._wins)

    def outcomes(self):
        """Finalize any still-open windows (sequence ended) as healthy END, then
        return one :class:`WindowOutcome` per window."""
        for w in self._wins:
            if w.state is None:
                self._close(w, "end",
                            self._last_frame if self._last_frame is not None
                            else w.spec.seed_frame)
        return [WindowOutcome(spec=w.spec, state=w.state, swap_to=w.swap_to,
                              close_frame=w.close_frame, masks=w.masks, probs=w.probs,
                              coast_seed_frame=w.coast_seed_frame,
                              coast_boxes=w.coast_boxes)
                for w in self._wins]


# -- merge into the sam2_raw schema ------------------------------------------
def _apply_renames(results, events):
    """Relabel oids in-place per the SWAP rename events.

    For frame f, ALL events with ``event.frame <= f`` compose in ascending
    frame order: each event is a transposition (a<->b) applied to the running
    label permutation. Composition keeps every earlier correction in effect
    when a later, independent swap lands (1<->2 at f=3 and 3<->4 at f=7 both
    apply from f=7 on) and makes a same-pair re-swap cancel out (swap twice =
    identity from the second event onward -- SAM detected a swap back).

    Events live in RAW history-label space (the ExitMachine settles against
    the never-renamed TBD history), so each transposition composes on the
    INPUT (raw) side of the running raw->corrected mapping: raw streams a and
    b trade their corrected destinations. Chains sharing an oid depend on
    this direction: (1,2)@f3 then (2,3)@f7 must yield raw1->2, raw2->3,
    raw3->1 (physical robot i keeps output label i), not the inverse cycle.
    """
    if not events:
        return
    # Chronological order; pair order breaks ties deterministically when two
    # distinct transpositions share a frame.
    ordered = sorted(events, key=lambda e: (e.frame, min(e.a, e.b), max(e.a, e.b)))
    perm: dict[int, int] = {}          # composed relabel: raw oid -> corrected
    ei = 0
    for f in sorted(results):
        while ei < len(ordered) and ordered[ei].frame <= f:
            ev = ordered[ei]
            ei += 1
            # Input-side composition: raw a and b swap corrected destinations.
            va, vb = perm.get(ev.a, ev.a), perm.get(ev.b, ev.b)
            perm[ev.a], perm[ev.b] = vb, va
            perm = {k: v for k, v in perm.items() if k != v}
        if not perm:
            continue
        relabeled = {perm.get(oid, oid): entry
                     for oid, entry in results[f].items()}
        results[f] = {oid: relabeled[oid] for oid in sorted(relabeled)}


def _coast_velocity(boxes):
    """Constant-velocity ``(dx, dy)`` from the last <=3 center deltas of
    ``boxes`` (mirrors :meth:`_Track.velocity`; ``(0, 0)`` for a single box)."""
    if len(boxes) < 2:
        return (0.0, 0.0)
    centers = [_center(b) for b in boxes]
    dxs = [centers[i][0] - centers[i - 1][0] for i in range(1, len(centers))]
    dys = [centers[i][1] - centers[i - 1][1] for i in range(1, len(centers))]
    return (sum(dxs) / len(dxs), sum(dys) / len(dys))


def _coast_reclaim_margin(coasted_box, frame_obs, cfg: SelectiveConfig):
    """Assignment margin (paper Eq. 1) of matching ``coasted_box`` against the
    TBD detections in ``frame_obs`` by expanded-IoU cost.

    Returns the gap between the cheapest and second-cheapest overlapping
    detection (``inf`` when exactly one overlaps), or None when the coasted box
    overlaps no detection at all. A large margin means one detection
    unambiguously owns the coasted box -- the signal that the same-alliance
    ambiguity has resolved and normal TBD can be trusted again.

    Note: this margin is deliberately **pure expanded-IoU** and does not
    include the appearance term (``selective_app_weight``) that
    ``FrameObs.margin`` carries elsewhere (find_windows, settle, EMA gate)
    when ``app_weight > 0`` -- kill-coast reclaim asks a geometry question,
    not an appearance one, so the two features' margin units intentionally
    diverge when both knobs are on.
    """
    costs = []
    for obs in frame_obs.values():
        eiou = expanded_iou(coasted_box, obs.box, cfg.iou_expand)
        if eiou > 0.0:
            costs.append(1.0 - eiou)
    if not costs:
        return None
    costs.sort()
    if len(costs) == 1:
        return float("inf")
    return costs[1] - costs[0]


# Diagnostics from the most recent _apply_kill_coast call. Single-view
# pipeline: sam2_tracker reads this right after merge_selective_results and
# folds it into selective_stats, so the per-view log line shows whether the
# coast actually wrote entries (kills, no_seed, entries written, and a
# per-kill (oid, seed_frame, written, end_reason) breakdown).
last_coast_stats: dict | None = None


def _apply_kill_coast(results, history, outcomes, cfg: SelectiveConfig):
    """Coast each killed object from its last confident SAM box (Task 2).

    For every ``killed`` outcome carrying a coast seed, overwrite that oid's
    raw-TBD entries from the frame after the seed with a constant-velocity
    extrapolation of the seed box, for up to ``cfg.kill_coast_frames`` frames.
    This pins the killed object's identity to a physically-plausible position
    instead of the appearance-blind TBD association, which mis-links close
    same-alliance robots (the qm11 blue gap, docs/selective-mask-prop-blue-gap.md).

    The coast ends early the first frame a TBD detection unambiguously re-claims
    the coasted box (its expanded-IoU assignment margin >= ``cfg.margin_exit``);
    that frame and every later one keep the raw-TBD stream. Coasted entries
    carry ``evidence.type == "kill_coast"`` and no mask.

    Mutates ``results`` in place in RAW history-label space, so it must run
    BEFORE :func:`_apply_renames` -- the coasted entries are then relabeled by
    any SWAP exactly like the rest of that oid's stream.
    """
    global last_coast_stats
    stats = {"kills": 0, "no_seed": 0, "entries": 0, "per_kill": []}
    for oc in outcomes:
        if oc.state != "killed":
            continue
        stats["kills"] += 1
        if oc.coast_seed_frame is None or not oc.coast_boxes:
            stats["no_seed"] += 1
            stats["per_kill"].append((oc.spec.oid, None, 0, "no_seed"))
            continue
        oid = oc.spec.oid
        seed_frame = oc.coast_seed_frame
        seed_box = oc.coast_boxes[-1]
        vx, vy = _coast_velocity(oc.coast_boxes)
        written, end = 0, "budget"
        for step in range(1, cfg.kill_coast_frames + 1):
            f = seed_frame + step
            if f not in results:
                end = "range"               # past the tracked range
                break
            coasted = (seed_box[0] + vx * step, seed_box[1] + vy * step,
                       seed_box[2] + vx * step, seed_box[3] + vy * step)
            margin = _coast_reclaim_margin(coasted, history.get(f, {}), cfg)
            if margin is not None and margin >= cfg.margin_exit:
                end = "reclaim"             # unambiguous re-claim -> TBD resumes
                break
            results[f][oid] = {
                "box": tuple(int(v) for v in coasted),
                "mask": None,
                "evidence": {"type": "kill_coast", "gap_len": 0,
                             "detector_overlap": 1.0, "tracker_prob": None},
            }
            written += 1
        stats["entries"] += written
        stats["per_kill"].append((oid, seed_frame, written, end))
    last_coast_stats = stats


def merge_selective_results(history, outcomes, cfg: SelectiveConfig):
    """Build the sam2_raw-schema result stream from the TBD history plus the
    window outcomes.

    Returns ``(results, renames)`` where ``results`` is
    ``{frame: {oid: {"box", "mask", "evidence"}}}``. Outside healthy windows
    every entry is a pure-TBD box (``mask=None``, evidence type ``"tbd"``). SWAP
    outcomes relabel the whole stream pairwise from their entry frame; then each
    healthy outcome overwrites its oid's entry across ``[entry, close]`` with the
    SAM mask, ``mask_to_box(mask)``, and ``selective_sam`` evidence carrying the
    per-frame ``tracker_prob``.

    When ``cfg.kill_coast`` is on, each ``killed`` outcome additionally coasts
    its object from the last confident SAM box (``kill_coast`` evidence,
    mask=None) instead of leaving it on raw TBD -- see :func:`_apply_kill_coast`.
    Off (the default) this is a no-op and the killed path is byte-identical.
    """
    # mask_to_box lives in sam2_tracker; import lazily so this module stays
    # torch-free at import time (sam2_tracker's pure helpers are torch-free, but
    # a function-level import keeps that guarantee independent of that module).
    from web.field.sam2_tracker import mask_to_box

    results: dict[int, dict[int, dict]] = {}
    for f in sorted(history):
        frame = {}
        for oid in sorted(history[f]):
            obs = history[f][oid]
            frame[oid] = {
                "box": tuple(int(v) for v in obs.box),
                "mask": None,
                "evidence": {"type": "tbd", "gap_len": 0,
                             "detector_overlap": 1.0, "tracker_prob": None},
            }
        results[f] = frame

    # SWAP rename events (dedup identical (frame, {a, b}) pairs).
    events: list[RenameEvent] = []
    seen: set[tuple[int, frozenset]] = set()
    for oc in outcomes:
        if oc.state != "swap" or oc.swap_to is None:
            continue
        a, b = oc.spec.oid, oc.swap_to
        key = (oc.spec.entry_frame, frozenset((a, b)))
        if key in seen:
            continue
        seen.add(key)
        events.append(RenameEvent(frame=oc.spec.entry_frame, a=a, b=b))

    # Kill-coast fallback (Task 2): pin killed objects to their coasted SAM box
    # before renames, so coasted entries are relabeled with the rest of the
    # stream. Knob off -> no-op, leaving the killed path byte-identical.
    if cfg.kill_coast:
        _apply_kill_coast(results, history, outcomes, cfg)

    _apply_renames(results, events)

    # Overlay masks for healthy outcomes (under the seeded oid, post-rename).
    for oc in outcomes:
        if oc.state not in _HEALTHY_STATES:
            continue
        oid = oc.spec.oid
        for f, mask in sorted(oc.masks.items()):
            if f < oc.spec.entry_frame or f > oc.close_frame:
                continue
            box = mask_to_box(mask)
            if box is None:
                continue
            results.setdefault(f, {})[oid] = {
                "box": tuple(int(v) for v in box),
                "mask": mask,
                "evidence": {"type": "selective_sam", "gap_len": 0,
                             "detector_overlap": 1.0,
                             "tracker_prob": oc.probs.get(f)},
            }

    return results, events
