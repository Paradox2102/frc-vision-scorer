"""Detect where each physical camera sits inside a broadcast composite frame.

FRC broadcasts tile several cameras (an overhead "main" view and, at most
events, two lower "corner" views) into one 1920x1080 frame, with static
broadcast graphics (separator strips, pillars, a scoreboard band, centre logo
panels, event-name banners) around and on top of them. The tiling differs per
event, so a single global layout (data/field/composite_layout.json) breaks
calibration and tracking elsewhere.

Approach (no templates, no model, no knowledge of the graphics package):

1. Frame selection. Sample ~40 frames from the in-play window. Two candidate
   frame sets are analysed (steps 2-6) and the one giving the most camera
   views wins (ties keep the first):
   - all samples less gross outliers (full-screen graphics, replays) that do
     not share the dominant frame structure (the composite fills the window
     at most official events);
   - every sizeable group of one shot type: greedy medoid clustering of
     64x36 thumbnails (mean |diff| <= GROUP_TAU). Offseason streams can
     spend most of the window on cutaways (full-screen wide shots,
     commentators with lower-thirds, results), which would otherwise pollute
     the median and activity maps; their groups are single full-frame views
     and lose to the composite group.
   When the chosen set holds under ~3/4 of the samples, detect_layout_video
   samples 2 more frames between every two consecutive used samples and
   runs again (the decorrelation cue below needs ~30+ frames); the topped-up
   result is kept unless it loses a camera or merely confirms the first.
2. Per pixel: the temporal median frame M (colour), an activity map
   (mean |frame - median| on a blurred half-res copy; camera views carry
   motion, broadcast graphics do not), a bit-identical map, and temporal
   decorrelation maps: across a row (column) boundary b, the two pixels
   DECOR_D apart straddling it are "decorrelated" when their temporal
   correlation is < DECOR_CORR while each side correlates > DECOR_WITHIN
   with its own neighbour at the same distance. Inside one camera nearby
   pixels vary together (same objects, shake, exposure); across a boundary
   between two cameras, or a camera and an animated graphic, they do not.
   The within-side test drops busy textures (crowds) that decorrelate on
   their own; low-variance pixels are ignored.
3. Composite seams are long, perfectly straight horizontal/vertical edges in
   M. For every row (column) boundary, the "seam score" is the fraction of the
   span where a local-max step edge sits on that boundary (+-1 px); the
   "decor score" is the decorrelated share of the informative pixels.
4. Recursive XY-cut, alternating axes. Along one axis of a region:
   - a band at either end (scoreboard band, letterbox) needs a near-full-span
     seam (>= SEAM_BAND) with flat, bit-identical graphics on every cross
     position outside it; only across the whole frame (inside a view, a still
     stretch reaching the edge is camera content);
   - an interior still run (separator strip, pillar, logo panel) must leave
     room for a view on both sides, have busy camera content on both sides
     and be far stiller than it, and be bounded by a strong seam or have a
     perfectly still core. Horizontal strips often sit above a quiet camera
     area (dark arena ceiling) that extends the still run, so their far edge
     is the first seam whose span-mean colour jump is about the largest in a
     strip-sized reach (the strip's own texture seams barely move the mean).
     Vertical strips take the still run itself less the activity blur bleed,
     snapped to a strong seam nearby;
   - two busy pieces tiled with no strip between them (camera | camera, or
     camera | animated sponsor panel) split on a very strong seam
     (>= SEAM_CAMCAM, cue "edge"), or on a fairly strong one
     (>= SEAM_DECOR_EDGE) whose decor score is >= SEAM_DECOR (cue "decor":
     abutting cameras whose median-frame seam is weak because the two
     scenes have similar colours there, or the seam is blended).
   The cut log in diag["cuts"] lists each region's seams as (boundary, seam
   score, decor score) and each cam<->cam split with the cue that made it.
5. Leaves are camera views: main = largest, corner_left/right = the outermost
   views of the row below main (a centre logo/sponsor panel between them,
   still or animated, is ignored). Views that do not exist are omitted.
   Confidence = the weakest seam score of the view's edges, scaled down for
   low motion.
6. Overlay masks: inside each view, closed + hole-filled components of pixels
   that are bit-for-bit identical across samples, kept only when they look
   like a graphic box (mostly filled bounding box whose sides sit on crisp
   straight edges, with some texture). Still, dark camera areas can be
   identical too but are ragged. A flat, featureless box is kept only when
   the view edge cuts it (the plain end of a caption bar laid across two
   views), after growing it back to its outline. Conservative on purpose: a
   missed mask costs less than masking field.

Everything here is pure numpy/OpenCV. detect_layout_video() / detect_layout()
work on a video file; detect_layout_from_frames() is the entry point for
tests that build synthetic composites (no top-up there).
"""
from __future__ import annotations

import time
from typing import Sequence

import cv2
import numpy as np

# --- tunables (fractions are of the region being split) ---------------------
ACT_THR = 6.0            # blurred abs deviation (grey levels) counted as motion
STATIC_AF = 0.06         # a row/col with < this fraction of active px is still
SEAM_BAND = 0.8          # a static run touching the frame edge is its own band
SEAM_SEP = 0.65          # a seam that bounds an interior static run (separator)
SEAM_WEAK = 0.4          # weakest seam used to place a band edge
GRAPHIC_FRAC = 0.6       # rows/cols with this share of bit-identical px are graphic
SEAM_CAMCAM = 0.85       # split two active regions with no separator
EDGE_THR = 12            # grey-level step across a seam (max over channels)
MIN_VIEW_FRAC = 0.12     # min camera size along the split axis (of frame dim)
IDENT_DIFF = 2           # |d| <= this between samples counts as "identical"
IDENT_FRAC = 0.85        # fraction of sample pairs identical -> graphic pixel
BLEED = 3                # px the blurred half-res activity map bleeds into stills
# temporal decorrelation (camera<->camera seams with no strip, weak median edge)
DECOR_D = 5              # px between the two pixels compared across a boundary
DECOR_CORR = 0.4         # temporal correlation below this = decorrelated
DECOR_WITHIN = 0.5       # both sides must correlate this well at the same distance
DECOR_STD = 4.0          # min temporal std (grey levels) of a usable pixel
DECOR_MIN_VALID = 0.08   # min share of the span with usable pixels
SEAM_DECOR = 0.8         # decorrelated share of the span for a cam<->cam split ...
SEAM_DECOR_EDGE = 0.5    # ... on a median-frame seam at least this strong
# shot grouping (non-match cutaways in the sampled window)
GROUP_TAU = 20.0         # thumbnail mean |diff| (grey levels) within one shot type
GROUP_MIN_FRAC = 0.2     # min share of the samples for a group to be analysed
GROUP_MIN = 6


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _video_meta(cap) -> tuple[float, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    return fps, (n / fps if fps > 0 else 0.0)


def _window(dur: float, t_window) -> tuple[float, float]:
    t0, t1 = t_window if t_window else (20, None)
    t0 = float(t0 if t0 is not None else 20)
    if dur > 0:
        if t1 is None:
            # match videos run ~215 s with ~35 s of results/slides at the end
            t1 = dur - 35 if dur > 120 else dur * 0.9
        t1 = min(float(t1), dur - 0.5)
        if t1 - t0 < 10:      # short clip: use what there is
            t0, t1 = dur * 0.1, dur * 0.9
    else:
        t1 = t0 + 150 if t1 is None else float(t1)
    return t0, t1


def sample_frames(video_path, t_window=(20, None), n_frames: int = 40):
    """BGR frames evenly spaced over the window, their times and the duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    try:
        fps, dur = _video_meta(cap)
        t0, t1 = _window(dur, t_window)
        frames, times = [], []
        for t in np.linspace(t0, t1, max(3, int(n_frames))):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
                times.append(float(t))
    finally:
        cap.release()
    return frames, times, dur


def _inlier_mask(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Drop frames that do not share the dominant composite (full-screen
    graphics, replays, transitions): their mean deviation from the median
    thumbnail is far above the typical in-play frame's."""
    th = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (192, 108),
                              interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    keep = np.ones(len(frames), bool)
    for _ in range(2):
        med = np.median(th[keep], axis=0)
        d = np.abs(th - med).mean(axis=(1, 2))
        ref = float(np.median(d[keep]))
        keep = d <= max(2.0 * ref, ref + 6.0)
        if keep.sum() < max(3, len(frames) // 3):   # degenerate: keep all
            keep[:] = True
            break
    return keep


def _thumbs(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36),
                                interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)


def _shot_groups(frames: Sequence[np.ndarray], tau: float = GROUP_TAU) -> list[np.ndarray]:
    """Partition the samples into groups of one shot type (the match composite,
    a full-screen wide shot, the commentators, results graphics), largest
    first. Greedy medoid clustering on 64x36 thumbnails: the frame with the
    most neighbours within tau takes them, repeat. Graphics-agnostic: needs no
    knowledge of the broadcast package."""
    th = _thumbs(frames)
    d = np.abs(th[:, None] - th[None]).mean(axis=(2, 3))
    left = np.ones(len(frames), bool)
    groups = []
    while left.any():
        near = (d <= tau) & left[None, :] & left[:, None]
        cnt = near.sum(axis=1)
        m = int(np.argmax(cnt))
        members = np.where(near[m])[0]
        if len(members) < 2:
            break
        groups.append(members)
        left[members] = False
    groups.sort(key=lambda g: -len(g))
    return groups


# --------------------------------------------------------------------------
# temporal decorrelation
# --------------------------------------------------------------------------

def _decor_maps(gray: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """For every row boundary b (between rows b-1 and b) and every step-th
    column: low[b, x] = the pixels DECOR_D apart straddling b vary
    independently over time (temporal correlation < DECOR_CORR) while each
    side correlates well with its own neighbour at the same distance;
    valid[b, x] = that test was informative (enough temporal variance and
    coherent sides).

    Inside one camera, nearby pixels vary together (the same objects, shake,
    exposure); across a boundary between two cameras, or a camera and an
    animated graphic, they do not. Busy textures (crowds) decorrelate on
    their own, so a pixel only counts where both sides are coherent."""
    D = DECOR_D
    X = gray[:, :, ::step]
    N, H, W = X.shape
    low = np.zeros((H, W), bool)
    valid = np.zeros((H, W), bool)
    if N < 3 or H <= 3 * D:
        return low, valid
    s = np.empty((H, W), np.float32)
    c = np.empty((H - D, W), np.float32)            # c[i] = corr(row i, row i + D)
    for r0 in range(0, H - D, 128):                 # in row chunks: bounded memory
        r1 = min(H - D, r0 + 128)
        Xc = X[:, r0:r1 + D].astype(np.float32)
        sc = Xc.std(axis=0)
        s[r0:r1 + D] = sc
        Z = (Xc - Xc.mean(axis=0)) / np.maximum(sc, 1e-3)
        c[r0:r1] = (Z[:, :-D] * Z[:, D:]).mean(axis=0)
    ok = (s[:-D] > DECOR_STD) & (s[D:] > DECOR_STD)
    n = len(c)
    wa = np.full_like(c, -1.0)
    wb = np.full_like(c, -1.0)
    wa[D:] = c[:n - D]                                # rows i-D .. i (above)
    wb[:n - D] = c[D:]                                # rows i+D .. i+2D (below)
    v = ok & (wa > DECOR_WITHIN) & (wb > DECOR_WITHIN)
    k = (D + 1) // 2                                  # rows i, i+D straddle b = i+k
    low[k:k + n] = v & (c < DECOR_CORR)
    valid[k:k + n] = v
    return low, valid


def _decor_profile(low: np.ndarray, valid: np.ndarray, lo: int, hi: int,
                   c0: int, c1: int, step: int = 2) -> np.ndarray:
    """Decorrelated share of the informative pixels of the cross span [c0, c1)
    at each boundary in (lo, hi), max over +-1 px (0 where too few pixels are
    informative)."""
    n = low.shape[0]
    prof = np.zeros(n, np.float32)
    a, b = c0 // step, max(c0 // step + 1, c1 // step)
    lo, hi = max(1, lo - 1), min(n, hi + 1)
    if hi <= lo or low.size == 0:
        return prof
    L = low[lo:hi, a:b].sum(axis=1).astype(np.float32)
    V = valid[lo:hi, a:b].sum(axis=1).astype(np.float32)
    fr = L / np.maximum(V, 1.0)
    fr[(V < DECOR_MIN_VALID * (b - a)) | (V < 12)] = 0.0
    prof[lo:hi] = fr
    out = prof.copy()
    out[1:] = np.maximum(out[1:], prof[:-1])
    out[:-1] = np.maximum(out[:-1], prof[1:])
    return out


# --------------------------------------------------------------------------
# per-pixel statistics
# --------------------------------------------------------------------------

def compute_stats(frames: Sequence[np.ndarray]) -> dict:
    """Median frame, activity and identical-pixel maps (all full resolution)."""
    H, W = frames[0].shape[:2]
    k = len(frames) // 2
    M = np.empty_like(frames[0])                                   # colour median
    for r0 in range(0, H, 120):                                    # row chunks: bounded memory
        F = np.stack([f[r0:r0 + 120] for f in frames])
        M[r0:r0 + 120] = np.partition(F, k, axis=0)[k]
    del F
    hw, hh = max(1, W // 2), max(1, H // 2)
    small = np.stack([cv2.blur(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (hw, hh),
                                          interpolation=cv2.INTER_AREA), (3, 3))
                      for f in frames]).astype(np.int16)
    med = np.partition(small, len(small) // 2, axis=0)[len(small) // 2]
    act_s = np.abs(small - med).mean(axis=0).astype(np.float32)
    act = cv2.resize(act_s, (W, H), interpolation=cv2.INTER_NEAREST)
    # identical-pixel map from an unfiltered subsample (keeps the noise that
    # distinguishes a static camera area from a flat graphic)
    sub = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[::2, ::2] for f in frames]).astype(np.int16)
    if len(sub) >= 2:
        ident_s = (np.abs(np.diff(sub, axis=0)) <= IDENT_DIFF).mean(axis=0).astype(np.float32)
    else:
        ident_s = np.zeros(sub.shape[1:], np.float32)
    ident = cv2.resize(ident_s, (W, H), interpolation=cv2.INTER_NEAREST)
    # temporal decorrelation across row boundaries (every 2nd column) and
    # across column boundaries (every 2nd row)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    dlow_r, dval_r = _decor_maps(gray, 2)
    dlow_c, dval_c = _decor_maps(np.ascontiguousarray(gray[:, ::2, :].transpose(0, 2, 1)), 1)
    return {"M": M, "act": act, "ident": ident,
            "decor_rows": (dlow_r, dval_r, 2), "decor_cols": (dlow_c, dval_c, 2)}


def _edge_map(M: np.ndarray) -> np.ndarray:
    """e[b, x]: a local-max step edge sits on the row boundary b (between rows
    b-1 and b) at column x."""
    Mi = M.astype(np.int16)
    if Mi.ndim == 2:
        Mi = Mi[..., None]
    H = Mi.shape[0]
    st = np.zeros(Mi.shape[:2], np.int16)
    if H > 4:
        st[2:H - 1] = np.abs(Mi[3:] - Mi[:-3]).max(axis=-1)
    e = st > EDGE_THR
    for k in (1, 2):
        e[k:] &= st[k:] >= st[:-k]
        e[:-k] &= st[:-k] >= st[k:]
    return e


# --------------------------------------------------------------------------
# recursive XY-cut
# --------------------------------------------------------------------------

def _seam_profile(e: np.ndarray, lo: int, hi: int, c0: int, c1: int) -> np.ndarray:
    """Coverage of a seam at each boundary b in (lo, hi) over the cross span
    [c0, c1), tolerating +-1 px of blur."""
    n = e.shape[0]
    a, b = max(1, lo - 1), min(n - 1, hi + 1)
    prof = np.zeros(n, np.float32)
    if b - a < 1 or c1 <= c0:
        return prof
    sub = e[a - 1:b + 1, c0:c1]
    tol = sub[1:-1] | sub[:-2] | sub[2:]
    prof[a:b] = tol.mean(axis=1)
    return prof


def _seam_pos(e: np.ndarray, b: int, c0: int, c1: int) -> int:
    """Exact boundary among b-1, b, b+1 with the most raw edge pixels."""
    best, bv = b, -1
    for k in (b, b - 1, b + 1):
        if 0 < k < e.shape[0]:
            v = int(e[k, c0:c1].sum())
            if v > bv:
                best, bv = k, v
    return best


def _peaks(prof: np.ndarray, lo: int, hi: int, thr: float) -> list[int]:
    """Seam candidates in (lo, hi): local maxima >= thr, non-max suppressed."""
    idx = [b for b in range(lo + 2, hi - 1) if prof[b] >= thr]
    idx.sort(key=lambda b: -prof[b])
    picked: list[int] = []
    for b in idx:
        if all(abs(b - p) > 3 for p in picked):
            picked.append(b)
    return sorted(picked)


def _runs(mask: np.ndarray, off: int, min_len: int) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((off + i, off + j))
            i = j
        else:
            i += 1
    return out


def _mean_jump(Mspan: np.ndarray, i: int, w: int = 4) -> float:
    """L1 change of the span-mean colour across boundary i (between i-1, i),
    comparing w-row bands one px away from it."""
    a = Mspan[max(0, i - 1 - w):max(0, i - 1)]
    b = Mspan[i + 1:i + 1 + w]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.abs(a.mean(axis=0) - b.mean(axis=0)).sum())


def _split_axis(e, A, G, lo, hi, c0, c1, full, full_cross=True, Mean=None, trust_run=False,
                decor=None):
    """Split [lo, hi) along one axis (the rows of e/A/G) over the cross span
    [c0, c1).

    Static runs (no motion, or flat bit-identical graphics) become graphic
    bands when a strong seam bounds them: a band at either end of the span
    needs a near-full-span seam on its inner side (a scoreboard band, a
    pillarbox); an interior run (separator strip, pillar, logo panel) needs a
    seam on at least one side, or a wide, perfectly still core. Active gaps
    between bands are camera views; two views tiled with no separator are
    split on a very strong seam, or on a fairly strong one across which the
    pixels decorrelate over time (``decor`` = (low, valid, step) maps from
    _decor_maps for this axis).

    Returns {"cams": [(a, b, conf_a, conf_b)], "statics": [(a, b)], "seams":
    [...], "splits": [...]}, conf_* None where the piece edge is the span edge.
    """
    prof = _seam_profile(e, lo, hi, c0, c1)
    dprof = (_decor_profile(decor[0], decor[1], lo, hi, c0, c1, decor[2])
             if decor is not None else np.zeros_like(prof))
    Mspan = Mean[lo:hi, c0:c1].mean(axis=1)           # (n, channels) mean colour
    Af = A[lo:hi, c0:c1].mean(axis=1)
    Gf = G[lo:hi, c0:c1].mean(axis=1)
    peaks = _peaks(prof, lo, hi, SEAM_WEAK)
    pos = lambda b: _seam_pos(e, b, c0, c1)

    def weak_edge(x, side):
        near = [c for c in peaks if abs(c - x) <= 8]
        if near:
            c = max(near, key=lambda c: prof[c])
            return pos(c), float(prof[c])
        return (x - 2 if side == "lo" else x + 2), 0.3

    statics = []
    # 1. bands at either end of the span (scoreboard band, letter/pillarbox):
    #    a near-full-span seam with graphics on every cross position outside it
    ilo, ihi = lo, hi
    span = hi - lo

    def band_ok(a, b):
        if b - a < 2:
            return False
        gcol = G[a:b, c0:c1].mean(axis=0)
        if float((gcol >= 0.25).mean()) >= 0.9:
            return True
        acol = A[a:b, c0:c1].mean(axis=0)     # thin textured strip, no motion
        return (b - a) <= 0.06 * span and float((acol < 0.01).mean()) >= 0.95

    # only across the whole frame: inside a view, a static stretch reaching
    # the frame edge (a wall, the crowd) is camera content
    strong = [c for c in peaks if prof[c] >= SEAM_BAND] if full_cross else []
    for c in sorted([c for c in strong if c - lo <= 0.45 * span], reverse=True):
        p = pos(c)
        if band_ok(lo, p):
            statics.append((lo, p, None, float(prof[c])))
            ilo = p
            break
    for c in sorted([c for c in strong if hi - c <= 0.45 * span]):
        p = pos(c)
        if band_ok(p, hi):
            statics.append((p, hi, float(prof[c]), None))
            ihi = p
            break
    # 2. interior static runs: separator strips, pillars, logo panels
    static_mask = (Af < STATIC_AF) | (Gf >= GRAPHIC_FRAC)
    if ihi - ilo >= 8 and float(static_mask[ilo - lo:ihi - lo].mean()) > 0.97:
        statics.append((ilo, ihi, None, None))               # nothing moves here
        static_mask[:] = False

    def side_activity(a, b):
        """Mean activity of a camera-like window (None if graphic/too thin)."""
        a, b = max(lo, a), min(hi, b)
        if b - a < 4 or float(Gf[a - lo:b - lo].mean()) >= 0.45:
            return None
        return float(Af[a - lo:b - lo].mean())

    runs = []
    for s, t in _runs(static_mask[ilo - lo:ihi - lo], ilo, 3):
        if runs and s - runs[-1][1] <= 6:
            runs[-1] = (runs[-1][0], t)              # shading splits one strip
        else:
            runs.append((s, t))
    min_len = max(8, int(MIN_VIEW_FRAC * full))
    for s, t in runs:
        if s - ilo < min_len or ihi - t < min_len:
            continue                       # no room for a camera view beyond it:
                                           # a still area of one view, not a strip
        # a separator has camera views on both sides, and is far stiller
        # than either of them (a quiet strip of one camera is not)
        sa, sb = side_activity(s - 40, s), side_activity(t, t + 40)
        if sa is None or sb is None or min(sa, sb) < 0.03:
            continue
        if float(Af[s - lo:t - lo].mean()) > 0.35 * min(sa, sb):
            continue
        mid = (s + t) / 2.0
        cands = [c for c in peaks if s - 16 <= c <= t + 16 and prof[c] >= SEAM_SEP]
        # the stillest part of the run: a strip of graphic is perfectly still
        cores = _runs(Af[s - lo:t - lo] <= 0.012, s, 8)
        core = max(cores, key=lambda r: r[1] - r[0]) if cores else None
        still = core is not None and float(Gf[core[0] - lo:core[1] - lo].mean()) < 0.35
        if not cands and not still:
            continue
        # The strip's own texture makes seams inside it and a quiet camera
        # area can extend the still run past it, so pick its edges by how much
        # the span's mean colour jumps across each seam: the strip's outer
        # edges change the whole span's colour, texture edges inside barely do.
        jumps = {c: _mean_jump(Mspan, c - lo) for c in cands}
        def snap(x):
            """x, or the strongest strong seam within 6 px of it. A still-run
            edge with busy views beside it is fairly trustworthy (0.6)."""
            near = [c for c in cands if abs(c - x) <= 6]
            if near:
                c = max(near, key=lambda c: prof[c])
                return pos(c), float(prof[c])
            return x, 0.6

        if trust_run:
            # vertical strips (pillar, logo panel) usually have busy views on
            # both sides: the still run, less the activity blur bleed, is the
            # strip including its soft shaded edges. Where a quiet area of a
            # view widens the run, fall back to the perfectly still core.
            if core is not None and core[0] - s > 12:
                strong = [c for c in cands if c <= core[0]]
                if strong:
                    c = max(strong, key=lambda c: prof[c])
                    ea, ca = pos(c), float(prof[c])
                else:
                    ea, ca = weak_edge(core[0], "lo")
            else:
                ea, ca = snap(s - BLEED)
            if core is not None and t - core[1] > 12:
                strong = [c for c in cands if c >= core[1]]
                if strong:
                    c = max(strong, key=lambda c: prof[c])
                    eb, cb = pos(c), float(prof[c])
                else:
                    eb, cb = weak_edge(core[1], "hi")
            else:
                eb, cb = snap(t + BLEED)
        else:
            # horizontal strips often sit above a quiet camera area (dark
            # arena ceiling) that extends the still run: near edge = the
            # strongest seam in the near half; far edge = the first seam
            # after it whose colour jump is about the largest in a
            # strip-sized reach (texture seams inside the strip barely move
            # the span's mean colour)
            lo_c = [c for c in cands if c <= mid]
            if lo_c:
                c = max(lo_c, key=lambda c: prof[c])
                ea, ca = pos(c), float(prof[c])
            else:
                ea, ca = weak_edge(s, "lo")
                if s - BLEED < ea and ea - (s - BLEED) <= 10:
                    ea = s - BLEED
            after = [c for c in cands if ea + 3 < c <= ea + max(16, 0.1 * full)]
            ref = max((jumps[c] for c in after), default=0.0)
            hi_c = [c for c in after if jumps[c] >= 0.8 * ref and jumps[c] > 0]
            if hi_c:
                c = min(hi_c)
                eb, cb = pos(c), float(prof[c])
            else:
                eb, cb = weak_edge(t, "hi")
                if t + BLEED > eb and (t + BLEED) - eb <= 10:
                    eb = t + BLEED
        if eb - ea >= 3:
            statics.append((max(lo, ea), min(hi, eb), ca, cb))
    statics.sort()
    merged: list[list] = []
    for a, b, ca, cb in statics:
        if merged and a <= merged[-1][1]:
            if b > merged[-1][1]:
                merged[-1][1], merged[-1][3] = b, cb
        else:
            merged.append([a, b, ca, cb])
    # camera gaps between static bands
    gaps, cur, cconf = [], lo, None
    for a, b, ca, cb in merged:
        if a - cur >= 8:
            gaps.append((cur, a, cconf, ca))
        cur, cconf = b, cb
    if hi - cur >= 8:
        gaps.append((cur, hi, cconf, None))
    min_len = max(8, int(MIN_VIEW_FRAC * full))
    cams, splits = [], []

    def camcam(c):
        """Cue that splits two abutting busy views at c, or None."""
        if prof[c] >= SEAM_CAMCAM:
            return "edge"
        if prof[c] >= SEAM_DECOR_EDGE and dprof[c] >= SEAM_DECOR:
            return "decor"
        return None

    for a, b, ca, cb in gaps:
        inner = [c for c in peaks if a + min_len <= c <= b - min_len and camcam(c)]
        start, sconf = a, ca
        for c in inner:
            p = pos(c)
            if p - start >= min_len and b - p >= min_len:
                conf = max(float(prof[c]), float(dprof[c]) if camcam(c) == "decor" else 0.0)
                cams.append((start, p, sconf, conf))
                splits.append({"at": int(p), "cue": camcam(c), "edge": round(float(prof[c]), 3),
                               "decor": round(float(dprof[c]), 3)})
                start, sconf = p, conf
        cams.append((start, b, sconf, cb))
    return {"cams": cams, "statics": [(int(a), int(b)) for a, b, _, _ in merged],
            "seams": [(int(b), round(float(prof[b]), 3), round(float(dprof[b]), 3))
                      for b in peaks if prof[b] >= SEAM_SEP or
                      (prof[b] >= SEAM_DECOR_EDGE and dprof[b] >= 0.6)],
            "splits": splits}


def _xycut(ctx, region, axis, tried_other, depth, leaves, log):
    eH, eV, A, AT, G, GT, H, W, Mh, Mv, dR, dC = ctx
    y0, y1, x0, x1, conf = region
    if depth > 5 or (y1 - y0) < MIN_VIEW_FRAC * H or (x1 - x0) < MIN_VIEW_FRAC * W:
        leaves.append(region)
        return
    if axis == 0:   # split rows over columns [x0, x1)
        r = _split_axis(eH, A, G, y0, y1, x0, x1, H, full_cross=(x0 == 0 and x1 == W), Mean=Mh,
                        decor=dR)
        keys = ("top", "bottom")
        pieces = [(a, b, x0, x1, ca, cb) for a, b, ca, cb in r["cams"]]
    else:           # split columns over rows [y0, y1)
        r = _split_axis(eV, AT, GT, x0, x1, y0, y1, W, full_cross=(y0 == 0 and y1 == H), Mean=Mv,
                        trust_run=True, decor=dC)
        keys = ("left", "right")
        pieces = [(y0, y1, a, b, ca, cb) for a, b, ca, cb in r["cams"]]
    out = []
    for py0, py1, px0, px1, ca, cb in pieces:
        c = dict(conf)
        if ca is not None:
            c[keys[0]] = ca
        if cb is not None:
            c[keys[1]] = cb
        out.append((py0, py1, px0, px1, c))
    log.append({"region": [int(y0), int(y1), int(x0), int(x1)], "axis": "rows" if axis == 0 else "cols",
                "seams": r["seams"], "statics": [list(s) for s in r["statics"]],
                "splits": r["splits"],
                "cams": [[int(p[0]), int(p[1]), int(p[2]), int(p[3])] for p in out]})
    if len(out) == 1 and tuple(out[0][:4]) == (y0, y1, x0, x1):
        if tried_other:
            leaves.append(out[0])
        else:
            _xycut(ctx, out[0], 1 - axis, True, depth + 1, leaves, log)
        return
    for p in out:
        _xycut(ctx, p, 1 - axis, False, depth + 1, leaves, log)


# --------------------------------------------------------------------------
# overlay masks
# --------------------------------------------------------------------------

def _side_edge_frac(g: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Mean over the four sides of the rect [x0, x1) x [y0, y1) of the fraction
    of the side that sits on a crisp step edge (+-2 px). A side on the image
    border counts as edged."""
    gi = g.astype(np.int16)
    H, W = g.shape
    fr = []

    def hside(yy):
        if yy <= 1 or yy >= H - 1:
            return 1.0
        best = np.zeros(x1 - x0, bool)
        for d in range(-2, 3):
            r = yy + d
            if 2 <= r < H - 1:
                best |= np.abs(gi[r + 1, x0:x1] - gi[r - 2, x0:x1]) > 20
        return float(best.mean()) if x1 > x0 else 0.0

    def vside(xx):
        if xx <= 1 or xx >= W - 1:
            return 1.0
        best = np.zeros(y1 - y0, bool)
        for d in range(-2, 3):
            c = xx + d
            if 2 <= c < W - 1:
                best |= np.abs(gi[y0:y1, c + 1] - gi[y0:y1, c - 2]) > 20
        return float(best.mean()) if y1 > y0 else 0.0

    fr = [hside(y0), hside(y1), vside(x0), vside(x1)]
    return float(np.mean(fr))


def _grow_flat_box(g: np.ndarray, bx: int, by: int, bw: int, bh: int, v: float,
                   tol: float = 8.0, reach: int = 8) -> tuple[int, int, int, int]:
    """Grow each side of a flat box (median grey v) by up to reach px while
    the next row/column outside it is still the box colour."""
    H, W = g.shape
    x0, y0, x1, y1 = bx, by, bx + bw, by + bh
    gf = g.astype(np.float32)
    for _ in range(reach):
        if y0 > 0 and abs(gf[y0 - 1, x0:x1].mean() - v) < tol: y0 -= 1
        if y1 < H and abs(gf[y1, x0:x1].mean() - v) < tol: y1 += 1
        if x0 > 0 and abs(gf[y0:y1, x0 - 1].mean() - v) < tol: x0 -= 1
        if x1 < W and abs(gf[y0:y1, x1].mean() - v) < tol: x1 += 1
    return x0, y0, x1 - x0, y1 - y0


def _overlay_masks(stats: dict, crop: tuple, max_masks: int = 8) -> list[list[int]]:
    """Crop-local [x, y, w, h] rects over static broadcast graphics drawn on
    top of a view (scoreboard boxes, event banner).

    Seeds are bit-identical pixels (ident >= IDENT_FRAC); text/digits inside a
    box change, so components are closed and hole-filled. A component is kept
    only when it looks like a graphic box: mostly filled bbox whose sides sit
    on crisp straight edges. Static dark camera areas can also be identical
    but are ragged, so they fail the edge test. Conservative on purpose."""
    x, y, w, h = crop
    M = stats["M"][y:y + h, x:x + w]
    ident = stats["ident"][y:y + h, x:x + w]
    if M.size == 0:
        return []
    g = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY) if M.ndim == 3 else M
    seed = (ident >= IDENT_FRAC).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(5, int(round(min(w, h) * 0.015)) | 1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(w * h)
    out = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 16 or bh < 10 or bw * bh < 0.002 * area or bw * bh > 0.35 * area:
            continue
        filled = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(filled, [cnt - np.array([bx, by])], -1, 1, -1)
        # trim ragged fringes (still crowd patches the closing glued on) until
        # every side row/column is mostly covered
        t0, t1, l0, l1 = 0, bh, 0, bw
        while t1 - t0 >= 10 and l1 - l0 >= 16:
            sub = filled[t0:t1, l0:l1]
            dens = [sub[0].mean(), sub[-1].mean(), sub[:, 0].mean(), sub[:, -1].mean()]
            i = int(np.argmin(dens))
            if dens[i] >= 0.6:
                break
            if i == 0: t0 += 1
            elif i == 1: t1 -= 1
            elif i == 2: l0 += 1
            else: l1 -= 1
        filled = filled[t0:t1, l0:l1]
        bx, by, bw, bh = bx + l0, by + t0, l1 - l0, t1 - t0
        if bw < 16 or bh < 10 or bw * bh < 0.002 * area or filled.mean() < 0.7:
            continue                                   # ragged: not a box
        gv = g[by:by + bh, bx:bx + bw][filled > 0]
        if gv.std() < 12:
            # flat and featureless: not a graphic box, unless it is a crisp
            # bar cut by the view edge (the plain end of a caption bar that
            # spans two views, its text in the other view or between them).
            # The seed erodes a flat box's rim: grow it back to its outline.
            cut = bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2
            if not cut or bh > 0.3 * h:
                continue
            bx, by, bw, bh = _grow_flat_box(g, bx, by, bw, bh, float(np.median(gv)))
            if _side_edge_frac(g, bx, by, bx + bw, by + bh) < 0.7:
                continue
        elif _side_edge_frac(g, bx, by, bx + bw, by + bh) < 0.6:
            continue                                   # no crisp outline
        pad = 2
        rx, ry = max(0, bx - pad), max(0, by - pad)
        out.append([rx, ry, min(w, bx + bw + pad) - rx, min(h, by + bh + pad) - ry])
    # merge rects that overlap or touch
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if (a[0] <= b[0] + b[2] + 2 and b[0] <= a[0] + a[2] + 2 and
                        a[1] <= b[1] + b[3] + 2 and b[1] <= a[1] + a[3] + 2):
                    nx, ny = min(a[0], b[0]), min(a[1], b[1])
                    out[i] = [nx, ny, max(a[0] + a[2], b[0] + b[2]) - nx,
                              max(a[1] + a[3], b[1] + b[3]) - ny]
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    out.sort(key=lambda r: -r[2] * r[3])
    return [[int(v) for v in r] for r in out[:max_masks]]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _name_views(leaves: list, H: int, W: int, A: np.ndarray) -> dict:
    views = []
    for (y0, y1, x0, x1, conf) in leaves:
        if y1 - y0 < 0.1 * H or x1 - x0 < 0.1 * W:
            continue
        af = float(A[y0:y1, x0:x1].mean())
        if af < STATIC_AF:
            continue
        views.append({"crop": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                      "seams": conf, "active_frac": af})
    if not views:
        return {}
    views.sort(key=lambda v: -v["crop"][2] * v["crop"][3])
    main = views[0]
    out = {"main": main}
    mx, my, mw, mh = main["crop"]
    below = [v for v in views[1:] if v["crop"][1] >= my + mh - 4]
    below.sort(key=lambda v: v["crop"][0])
    if len(below) >= 2:
        out["corner_left"], out["corner_right"] = below[0], below[-1]
    elif len(below) == 1:
        v = below[0]
        cx = v["crop"][0] + v["crop"][2] / 2
        out["corner_left" if cx < W / 2 else "corner_right"] = v
    return out


def _confidence(v: dict) -> float:
    s = v["seams"]
    edge = min(s.values()) if s else 1.0
    act = min(1.0, v["active_frac"] / 0.3)
    return round(float(max(0.0, min(1.0, edge)) * (0.5 + 0.5 * act)), 3)


def layout_from_stats(stats: dict) -> dict:
    M, act = stats["M"], stats["act"]
    H, W = M.shape[:2]
    A = act > ACT_THR
    G = stats["ident"] >= IDENT_FRAC
    eH = _edge_map(M)
    eV = _edge_map(np.ascontiguousarray(M.transpose(1, 0, 2) if M.ndim == 3 else M.T))
    Mf = M.astype(np.float32) if M.ndim == 3 else M.astype(np.float32)[..., None]
    ctx = (eH, eV, A, np.ascontiguousarray(A.T), G, np.ascontiguousarray(G.T), H, W,
           Mf, np.ascontiguousarray(Mf.transpose(1, 0, 2)),
           stats.get("decor_rows"), stats.get("decor_cols"))
    leaves: list = []
    log: list = []
    _xycut(ctx, (0, H, 0, W, {}), 0, False, 0, leaves, log)
    named = _name_views(leaves, H, W, A)
    cameras, confidence = {}, {}
    for name, v in named.items():
        cameras[name] = {"crop": v["crop"], "masks": _overlay_masks(stats, tuple(v["crop"]))}
        confidence[name] = _confidence(v)
    return {
        "frame_size": [int(W), int(H)],
        "cameras": cameras,
        "confidence": confidence,
        "diag": {"leaves": [[int(l[2]), int(l[0]), int(l[3] - l[2]), int(l[1] - l[0])] for l in leaves],
                 "views": {k: {"seams": {kk: round(float(vv), 3) for kk, vv in v["seams"].items()},
                               "active_frac": round(v["active_frac"], 3)} for k, v in named.items()},
                 "cuts": log},
    }


def detect_layout_from_frames(frames: Sequence[np.ndarray]) -> dict:
    """detect_layout on already-decoded BGR frames (same size)."""
    if len(frames) < 3:
        raise ValueError("need at least 3 frames")
    t = time.time()
    n = len(frames)
    # candidate frame sets: all samples less gross outliers (the composite
    # dominates the window at most events), plus every sizeable group of one
    # shot type (an offseason stream can spend most of the window on
    # cutaways: wide shots, commentators, results)
    keep = _inlier_mask(frames)
    cands = [("inliers", np.where(keep)[0])]
    groups = _shot_groups(frames)
    for k, g in enumerate(groups):
        if len(g) < max(3, GROUP_MIN, GROUP_MIN_FRAC * n):
            continue
        if any(len(set(g.tolist()) ^ set(c.tolist())) <= 0.1 * n for _, c in cands):
            continue                                   # same frames as a candidate
        cands.append((f"group{k}", g))
    best, best_key, tried = None, None, []
    for i, (label, idx) in enumerate(cands):
        out = layout_from_stats(compute_stats([frames[j] for j in idx]))
        ncam = len(out["cameras"])
        tried.append({"label": label, "n": int(len(idx)), "n_cams": ncam,
                      "crops": {k: v["crop"] for k, v in out["cameras"].items()}})
        # the most camera views wins (a composite over a single full-screen
        # shot); ties keep the earlier candidate (plain inliers first)
        if best is None or ncam > best_key:
            best, best_key, used_idx, chosen = out, ncam, idx, label
    out = best
    used_set = set(int(j) for j in used_idx)
    out["diag"].update({"n_frames": n, "n_used": len(used_idx),
                        "used_idx": sorted(used_set),
                        "rejected": [j for j in range(n) if j not in used_set],
                        "selection": {"chosen": chosen, "candidates": tried,
                                      "groups": [len(g) for g in groups]},
                        "t_analyse_s": round(time.time() - t, 2)})
    return out


def read_frames_at(video_path, times) -> tuple[list, list]:
    """BGR frames at the given times (s); unreadable ones are skipped."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    frames, got = [], []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
                got.append(float(t))
    finally:
        cap.release()
    return frames, got


def _topup_times(times: Sequence[float], used_idx: Sequence[int], per_gap: int = 2) -> list[float]:
    """Extra sample times inside the stretches the chosen shot group covers:
    per_gap evenly spaced times between every two consecutive samples that
    were both used (a gap between two composite samples is almost always
    composite too)."""
    used = set(int(i) for i in used_idx)
    out = []
    for i in range(len(times) - 1):
        if i in used and i + 1 in used:
            a, b = times[i], times[i + 1]
            out += [a + (b - a) * k / (per_gap + 1) for k in range(1, per_gap + 1)]
    return out


TOPUP_BELOW = 0.75       # chosen group under this share of the samples -> top up
TOPUP_MIN_USED = 24      # ... or under this many frames


def detect_layout_video(video_path, t_window=(20, None), n_frames: int = 40):
    """Sample, detect, and when the composite holds only part of the window
    (cutaways, a dead feed), sample more frames between the composite
    samples and detect again: the decorrelation cue needs ~30+ frames to
    separate camera seams from natural lines. Returns (detect_layout dict,
    the frames the chosen result used)."""
    t = time.time()
    frames, times, dur = sample_frames(video_path, t_window, n_frames)
    if len(frames) < 3:
        raise ValueError(f"could not read frames from {video_path}")
    out = detect_layout_from_frames(frames)
    d = out["diag"]
    used = [frames[i] for i in d["used_idx"]]
    topup = None
    if d["n_used"] < max(TOPUP_BELOW * len(frames), min(TOPUP_MIN_USED, len(frames))):
        extra_t = _topup_times(times, d["used_idx"])
        if extra_t:
            extra, got_t = read_frames_at(video_path, extra_t)
            if extra:
                base = used
                out2 = detect_layout_from_frames(base + extra)
                topup = {"n_extra": len(extra), "first": {
                    "n_used": d["n_used"], "n_cams": len(out["cameras"]),
                    "crops": {k: v["crop"] for k, v in out["cameras"].items()}}}
                # more frames give the sharper cues: keep the topped-up result
                # unless it lost a camera, or it only confirms the first one
                # (then the evenly spaced samples' exact edges stand)
                c1, c2 = out["cameras"], out2["cameras"]
                same = set(c1) == set(c2) and all(
                    max(abs(a - b) for a, b in zip(c1[k]["crop"], c2[k]["crop"])) <= 6 for k in c1)
                if same:
                    topup["kept"] = False
                    topup["agrees"] = True
                elif len(c2) >= len(c1):
                    allf = base + extra
                    used = [allf[i] for i in out2["diag"]["used_idx"]]
                    out = out2
                    topup["kept"] = True
                else:
                    topup["kept"] = False
    out["diag"].update({"video": str(video_path), "duration_s": round(dur, 2),
                        "t_window": [round(times[0], 2), round(times[-1], 2)],
                        "t_total_s": round(time.time() - t, 2)})
    if topup is not None:
        out["diag"]["topup"] = topup
    return out, used


def detect_layout(video_path, *, t_window=(20, None), n_frames: int = 40) -> dict:
    """Detect the composite layout of a broadcast match video.

    Returns {"frame_size": [w, h], "cameras": {name: {"crop": [x, y, w, h],
    "masks": [[x, y, w, h], ...]}}, "confidence": {name: 0..1}, "diag": {...}}.
    Masks are crop-local. Cameras that do not exist are omitted.
    """
    return detect_layout_video(video_path, t_window, n_frames)[0]
