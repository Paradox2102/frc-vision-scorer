"""Multi-frame registration + fusion of a bumper tracklet into one sharper image.

Several consecutive native-resolution frames of the same bumper (a tracklet,
e.g. scripts/build_tracklet_bench.py output: fixed-size boxes that follow the
bumper at integer-pixel precision) are upsampled, aligned to an anchor frame
with ECC, and fused into ONE image that any single-image reader can read.

Classical registration + fusion only: every output pixel is a robust average
of real observations of that point, so nothing is invented (no generative or
learned super-resolution, zero hallucination risk). A frame that does not
align well is DROPPED, never fused: a bad alignment would print ghost or
doubled digits, which is worse than no fusion at all.

Pipeline (``register_and_fuse``)
--------------------------------
0. Interlace: the FRC match video this was built for is interlaced (woven
   fields 1/2 frame apart; moving robots and digits comb). With
   ``deinterlace="auto"`` every frame is split into its two fields (each
   resampled to full height at its true row positions) when any nearby frame
   combs (``comb_ratio``); each field is then a separate, clean sample, and the
   anchor is the sharper field of the anchor frame.
1. Selection: samples nearest the anchor in time (``times``), skipping blurry
   ones (Laplacian variance below ``min_rel_sharpness`` x the anchor's) and
   weak tracker matches (``match_scores`` below ``min_match``), up to
   ``max_samples``.
2. Registration of each sample to the anchor on a small grid (``reg_scale``:
   about 72 px tall; the warp is mapped back to native coordinates):
   * translation init from Hann-windowed phase correlation;
   * ``cv2.findTransformECC`` (MOTION_AFFINE by default, or HOMOGRAPHY /
     TRANSLATION / EUCLIDEAN) on Gaussian-smoothed gray images;
   * rejection: ECC did not converge, ECC correlation < ``min_ecc``,
     implausible warp (scale outside 1 +- ``max_scale_dev``, shear above
     ``max_shear``, rotation above ``max_rot_deg``, box-centre shift above
     ``max_shift_frac`` of the box), the warped sample covers < ``min_overlap``
     of the anchor, or post-warp NCC with the anchor < ``min_ncc``.
3. Fusion on the ``upscale`` x native grid (each sample is resampled from
   native straight onto that grid through its warp: one cubic interpolation).
   ``robust`` (default): weighted mean with per-pixel anti-ghost weights
   (``ghost_weights``: a pixel that disagrees with the anchor after a small
   blur drops out, so a part that moved differently or a residual misalignment
   cannot print doubled digits); or ``median``, ``mean``, ``weighted``
   (per-sample sharpness weights) or ``sas`` (shift-and-add splatting +
   normalised Gaussian convolution). Pixels a warped sample does not cover fall
   back to the anchor.
4. Mild unsharp mask (``unsharp_amount``, sigma 0.5 native px).

With fewer than ``min_used`` aligned samples (anchor included) the result is
the anchor sample alone through the same upscale + unsharp path, and
``diagnostics["fused"]`` is False.

Typical cost on CPU: ~20 ms per tracklet (p90 ~33 ms) on the bumper bench.

Pure numpy / OpenCV, no app imports.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import cv2
import numpy as np

MOTIONS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}
FUSE_MODES = ("robust", "median", "mean", "weighted", "sas")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def to_gray_f32(img: np.ndarray) -> np.ndarray:
    """uint8/float BGR or gray -> float32 gray in [0, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = img.astype(np.float32)
    if img.dtype == np.uint8:
        g /= 255.0
    return g


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian of the gray image (same measure as the
    tracklet builder's ``sharpness``)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def ncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Zero-mean normalised cross-correlation of two same-size gray images."""
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.size < 4:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 1e-12 else 0.0


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(peak * peak / mse)


def _scale_warp(warp: np.ndarray, s: float) -> np.ndarray:
    """Map a warp estimated on images resized by ``s`` (pixel-centre convention
    x_s = s*x + (s-1)/2, as cv2.resize uses) from native coordinates to the
    s-times grid. ``warp`` is 2x3 (affine family) or 3x3 (homography)."""
    c = (s - 1.0) / 2.0
    S = np.array([[s, 0, c], [0, s, c], [0, 0, 1]], dtype=np.float64)
    W = np.eye(3)
    W[: warp.shape[0]] = warp
    out = S @ W @ np.linalg.inv(S)
    return out[: warp.shape[0]].astype(np.float32)


def _to3x3(warp: np.ndarray) -> np.ndarray:
    W = np.eye(3)
    W[: warp.shape[0]] = warp
    return W


def warp_plausibility(warp: np.ndarray, w: int, h: int, max_scale_dev: float,
                      max_shear: float, max_rot_deg: float,
                      max_shift_frac: float, max_persp: float = 0.002) -> Optional[str]:
    """None when the warp (native coords, anchor -> frame) is plausible, else a
    reason. The 2x2 linear part is decomposed into rotation x scale x shear
    (QR): scales and shear bounded, rotation bounded; the box-centre
    displacement must stay within ``max_shift_frac`` of the box size."""
    W = _to3x3(np.asarray(warp, dtype=np.float64))
    if not np.all(np.isfinite(W)):
        return "non_finite"
    if abs(W[2, 0]) > max_persp or abs(W[2, 1]) > max_persp:
        return "perspective"
    A = W[:2, :2]
    if np.linalg.det(A) <= 0:
        return "flip"
    # A = R @ [[sx, k], [0, sy]]
    sx = math.hypot(A[0, 0], A[1, 0])
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    sy = np.linalg.det(A) / sx
    k = (A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / sx
    shear = k / sy
    for sc in (sx, sy):
        if not (1.0 - max_scale_dev <= sc <= 1.0 + max_scale_dev):
            return f"scale {sc:.3f}"
    if abs(shear) > max_shear:
        return f"shear {shear:.3f}"
    if abs(rot) > max_rot_deg:
        return f"rotation {rot:.1f}"
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    p = W @ np.array([cx, cy, 1.0])
    p = p[:2] / p[2]
    if abs(p[0] - cx) > max_shift_frac * w or abs(p[1] - cy) > max_shift_frac * h:
        return f"shift ({p[0] - cx:.1f},{p[1] - cy:.1f})"
    return None


def phase_init(tmpl_g: np.ndarray, img_g: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) such that img(x + d) ~ tmpl(x), plus the phase
    correlation response, from Hann-windowed phase correlation."""
    h, w = tmpl_g.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(tmpl_g.astype(np.float32), img_g.astype(np.float32), win)
    return float(dx), float(dy), float(resp)


def unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """float image -> img + amount * (img - blur(img))."""
    if amount <= 0 or sigma <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# --------------------------------------------------------------------------- #
# Frame selection
# --------------------------------------------------------------------------- #

def select_frames(n: int, anchor_idx: int, times: Optional[Sequence[float]] = None,
                  sharpness: Optional[Sequence[float]] = None,
                  match_scores: Optional[Sequence[float]] = None, max_frames: int = 7,
                  min_rel_sharpness: float = 0.5, min_match: float = 0.6,
                  max_dt: Optional[float] = None) -> tuple[list[int], list[tuple[int, str]]]:
    """Indices (anchor first, then nearest-in-time) of up to ``max_frames``
    frames to register, and [(idx, reason)] for the ones skipped before
    registration. Time ties go to the sharper frame."""
    times = list(times) if times is not None else list(range(n))
    a_t = times[anchor_idx]
    a_sh = sharpness[anchor_idx] if sharpness is not None else None
    skipped: list[tuple[int, str]] = []
    cands = []
    for i in range(n):
        if i == anchor_idx:
            continue
        dt = abs(times[i] - a_t)
        if max_dt is not None and dt > max_dt:
            skipped.append((i, "far_in_time"))
            continue
        if match_scores is not None and match_scores[i] is not None and match_scores[i] < min_match:
            skipped.append((i, f"match {match_scores[i]:.2f}"))
            continue
        if (sharpness is not None and a_sh and sharpness[i] is not None
                and sharpness[i] < min_rel_sharpness * a_sh):
            skipped.append((i, f"blurry {sharpness[i] / a_sh:.2f}x"))
            continue
        sh = sharpness[i] if sharpness is not None and sharpness[i] is not None else 0.0
        cands.append((dt, -sh, i))
    cands.sort()
    picked = [anchor_idx] + [i for _dt, _s, i in cands[: max(0, max_frames - 1)]]
    skipped += [(i, "beyond_max_frames") for _dt, _s, i in cands[max(0, max_frames - 1):]]
    return picked, skipped


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def register_pair(anchor: np.ndarray, frame: np.ndarray, motion: str = "affine",
                  reg_scale: float = 2.0, iters: int = 60, eps: float = 1e-4,
                  gauss: int = 3, max_init_shift_frac: float = 0.35) -> dict:
    """ECC-align ``frame`` to ``anchor`` (same size). Returns {"ok", "warp"
    (native coords, anchor -> frame, 2x3 or 3x3), "ecc", "init", "reason"}."""
    h, w = anchor.shape[:2]
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    s = float(reg_scale)
    size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
    ta = cv2.resize(to_gray_f32(anchor), size, interpolation=cv2.INTER_CUBIC)
    tf = cv2.resize(to_gray_f32(frame), size, interpolation=cv2.INTER_CUBIC)
    dx, dy, resp = phase_init(ta, tf)
    out = {"ok": False, "warp": None, "ecc": None,
           "init": (dx / s, dy / s, resp), "reason": None}
    if abs(dx) > max_init_shift_frac * size[0] or abs(dy) > max_init_shift_frac * size[1]:
        dx = dy = 0.0  # implausible phase peak: start ECC from identity instead
    mtype = MOTIONS[motion]
    warp = np.eye(3, dtype=np.float32) if mtype == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    warp[0, 2], warp[1, 2] = dx, dy
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        ecc, warp = cv2.findTransformECC(ta, tf, warp, mtype, crit, None, gauss)
    except cv2.error:
        out["reason"] = "ecc_no_converge"
        return out
    out["ecc"] = float(ecc)
    out["warp"] = _scale_warp(warp, 1.0 / s)  # reg grid -> native coords
    out["ok"] = True
    return out


def _warp_image(img: np.ndarray, warp: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``img`` (frame, on the output grid) into anchor coordinates.
    Returns (warped float32 image, valid mask)."""
    flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    ones = np.ones(img.shape[:2], np.float32)
    if warp.shape[0] == 3:
        out = cv2.warpPerspective(img, warp, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpPerspective(ones, warp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        out = cv2.warpAffine(img, warp, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(ones, warp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, m > 0.5


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def _shift_and_add(native: list[np.ndarray], warps: list[np.ndarray], weights: list[float],
                   up: int, out_hw: tuple[int, int], sigma: float) -> np.ndarray:
    """Splat each native pixel (bilinear) at its position on the fine grid
    (anchor coords, via the inverse of anchor->frame), then normalised Gaussian
    convolution. Frames: float32 HxWxC native images."""
    H, W = out_hw
    C = native[0].shape[2] if native[0].ndim == 3 else 1
    acc = np.zeros((H * W, C), np.float64)
    wsum = np.zeros(H * W, np.float64)
    for img, warp, wt in zip(native, warps, weights):
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], 0).astype(np.float64)
        inv = np.linalg.inv(_to3x3(warp))   # frame -> anchor (native coords)
        q = inv @ pts
        q = q[:2] / q[2]
        fx = q[0] * up + (up - 1) / 2.0     # native -> fine grid (pixel centres)
        fy = q[1] * up + (up - 1) / 2.0
        vals = img.reshape(h * w, C).astype(np.float64)
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        ax, ay = fx - x0, fy - y0
        for oy, ox, bw in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)),
                           (1, 0, (1 - ax) * ay), (1, 1, ax * ay)):
            xx, yy = x0 + ox, y0 + oy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            idx = yy[ok] * W + xx[ok]
            bwk = bw[ok] * wt
            np.add.at(wsum, idx, bwk)
            for c in range(C):
                np.add.at(acc[:, c], idx, vals[ok, c] * bwk)
    acc = acc.reshape(H, W, C).astype(np.float32)
    wsum = wsum.reshape(H, W).astype(np.float32)
    acc_b = cv2.GaussianBlur(acc, (0, 0), sigma)
    if acc_b.ndim == 2:
        acc_b = acc_b[:, :, None]
    w_b = cv2.GaussianBlur(wsum, (0, 0), sigma)
    return acc_b / np.maximum(w_b, 1e-6)[:, :, None]




def ghost_weights(nat_anchor: np.ndarray, nat_aligned: list[np.ndarray], tau: float,
                  sigma: float = 0.5) -> list[np.ndarray]:
    """Per-pixel anti-ghost weights on the NATIVE grid (HDR+-style robust
    merge): exp(-D / (2 tau^2)), D = channel-mean squared difference between
    the aligned sample and the anchor after a Gaussian blur of ``sigma`` px (so
    pixel noise does not count as mismatch). Where a sample disagrees - a part
    that moved differently, a residual misalignment, an occluder - its weight
    drops to ~0 and the anchor shows through."""
    refb = cv2.GaussianBlur(nat_anchor, (0, 0), sigma)
    out = []
    for x in nat_aligned:
        d = cv2.GaussianBlur(x, (0, 0), sigma) - refb
        D = (d * d).mean(axis=2) if d.ndim == 3 else d * d
        out.append(np.exp(-D / (2.0 * tau * tau)).astype(np.float32))
    return out


def robust_merge(stack: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Weighted mean of the output-grid stack; stack[0] is the anchor (weight
    1), ``weights`` are per-pixel maps on the output grid for stack[1:]."""
    num = stack[0].astype(np.float32).copy()
    den = np.ones(stack[0].shape[:2], np.float32)
    for x, wp in zip(stack[1:], weights):
        num += (wp[:, :, None] * x) if x.ndim == 3 else wp * x
        den += wp
    return (num / den[:, :, None]) if num.ndim == 3 else num / den


# --------------------------------------------------------------------------- #
# Interlace handling
# --------------------------------------------------------------------------- #

def comb_ratio(img: np.ndarray, q: float = 0.98) -> float:
    """Interlace-combing score: the ``q`` quantile over 9x9 neighbourhoods of
    mean |x[r] - (x[r-1] + x[r+1]) / 2| / (mean |x[r-1] - x[r+1]| / 2 + 4),
    on rows first averaged over 5 columns (comb teeth run horizontally; pixel
    noise does not). Local, because combing only shows where something moved
    between the two field times (often just the robot, or just the digits).
    ~0.4-0.7 on progressive content, > 1 where the woven fields disagree."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.blur(g.astype(np.float32), (5, 1))
    if g.shape[0] < 4:
        return 0.0
    c = np.abs(g[1:-1] - (g[:-2] + g[2:]) * 0.5)
    d = np.abs(g[:-2] - g[2:]) * 0.5
    r = cv2.blur(c, (9, 9)) / (cv2.blur(d, (9, 9)) + 4.0)
    return float(np.quantile(r, q))


def near_frames(times: Sequence[float], anchor_idx: int, k: int) -> list[int]:
    """The ``k`` frame indices nearest the anchor in time (anchor incl.), in
    index order."""
    order = sorted(range(len(times)), key=lambda i: (abs(times[i] - times[anchor_idx]), i))
    return sorted(order[:k])


def wants_fields(frames: Sequence[np.ndarray], anchor_idx: int, comb_thr: float = 0.9,
                 times: Optional[Sequence[float]] = None, max_samples: int = 8) -> bool:
    """deinterlace="auto" rule, as register_and_fuse applies it: split into
    fields when any of the frames it would consider (the ``max_samples + 3``
    nearest the anchor) combs - one tracklet comes from one source, so one
    combed frame means the video is interlaced."""
    times = list(times) if times is not None else list(range(len(frames)))
    return any(comb_ratio(frames[i]) > comb_thr
               for i in near_frames(times, anchor_idx, max_samples + 3))


def split_fields(img: np.ndarray) -> list[np.ndarray]:
    """The two fields of a woven interlaced frame, each resampled back to full
    height at its true row positions (field p holds frame rows p, p+2, ...;
    output row y samples field row (y - p) / 2, linear)."""
    h, w = img.shape[:2]
    out = []
    for p in (0, 1):
        f = np.ascontiguousarray(img[p::2])
        M = np.array([[1, 0, 0], [0, 0.5, -p / 2.0]], np.float32)
        out.append(cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


def deinterlace_crop(img: np.ndarray, comb_thr: float = 0.9, unsharp_amount: float = 0.6,
                     unsharp_sigma: float = 0.7) -> np.ndarray:
    """Single-crop deinterlace for the read path ("anchor" mode of
    ``register_and_fuse`` without the extra frames): when ``img`` combs
    (``comb_ratio`` > ``comb_thr``), return the sharper of its two fields
    resampled to full height, lightly unsharped; otherwise return ``img``
    itself, untouched. Native resolution in and out (upscale afterwards)."""
    if img is None or img.ndim < 2 or img.shape[0] < 8 or img.shape[1] < 4:
        return img
    if comb_ratio(img) <= comb_thr:
        return img
    fields = split_fields(img)
    best = max(fields, key=laplacian_sharpness)
    out = unsharp(best.astype(np.float32), unsharp_sigma, unsharp_amount)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def auto_upscale(h: int, target_h: int = 160, max_up: int = 4) -> int:
    """Output upscale so a box ``h`` px tall lands near ``target_h`` (1..max_up)."""
    return int(min(max_up, max(1, math.ceil(target_h / max(1, h)))))


def register_and_fuse(
    frames: Sequence[np.ndarray],
    anchor_idx: int,
    upscale: Optional[int] = None,
    *,
    times: Optional[Sequence[float]] = None,
    sharpness: Optional[Sequence[float]] = None,
    match_scores: Optional[Sequence[float]] = None,
    deinterlace: str = "auto",
    comb_thr: float = 0.9,
    max_samples: int = 8,
    max_dt: Optional[float] = None,
    min_rel_sharpness: float = 0.5,
    min_match: float = 0.6,
    motion: str = "affine",
    reg_scale: Optional[float] = None,
    ecc_iters: int = 60,
    min_ecc: float = 0.75,
    min_ncc: float = 0.70,
    min_overlap: float = 0.85,
    max_scale_dev: float = 0.12,
    max_shear: float = 0.10,
    max_rot_deg: float = 8.0,
    max_shift_frac: float = 0.30,
    fuse: str = "robust",
    robust_tau: float = 10.0,
    min_used: int = 2,
    unsharp_amount: float = 0.6,
    unsharp_sigma: Optional[float] = None,
    sas_sigma: float = 0.25,
    target_h: int = 160,
) -> tuple[np.ndarray, dict]:
    """Register ``frames`` (same-size native crops of one bumper; BGR uint8 or
    gray) to ``frames[anchor_idx]`` and fuse them into one image ``upscale``
    times larger (None: ``auto_upscale`` towards ``target_h``). Returns (fused
    uint8 image, diagnostics).

    Interlace (``deinterlace``): "auto" splits EVERY frame into its two fields
    when any frame combs (``comb_ratio`` > ``comb_thr``, see ``wants_fields``),
    "fields" always splits, "none" never. Each field is then a separate sample
    (half a frame apart in time); the anchor is the sharper field of the
    anchor frame.

    Selection: ``times`` (e.g. frame offsets) - nearest first;
    ``match_scores`` gate weak tracker matches; ``sharpness`` (per frame; None =
    Laplacian variance computed here on each sample, which is what you want
    for interlaced input since combing inflates it) gates blurry samples.

    Diagnostics: {"fused", "n_used" (distinct source frames used, anchor
    incl.), "n_samples", "used" [frame idx], "used_samples" [[frame, field]],
    "rejected" [(sample, reason)], "skipped", "ecc", "ncc", "merge_w" (robust
    mean weight per used sample), "split_fields", "comb", "upscale", "ms"}.
    """
    t0 = time.perf_counter()
    if fuse not in FUSE_MODES:
        raise ValueError(f"fuse must be one of {FUSE_MODES}")
    if motion not in MOTIONS:
        raise ValueError(f"motion must be one of {tuple(MOTIONS)}")
    if deinterlace not in ("auto", "fields", "none"):
        raise ValueError("deinterlace must be auto, fields or none")
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    h, w = frames[anchor_idx].shape[:2]
    frames = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
              for f in frames]
    up = int(upscale) if upscale else auto_upscale(h, target_h)
    H, W = h * up, w * up
    rs = float(reg_scale) if reg_scale else float(min(2.0, max(0.5, 72.0 / h)))
    times = list(times) if times is not None else list(range(len(frames)))

    # ---- samples (whole frames or fields), from the frames nearest in time ----
    near = near_frames(times, anchor_idx, max_samples + 3)  # spares for the blur/match gates
    combs = {i: comb_ratio(frames[i]) for i in near}
    split = deinterlace == "fields" or (
        deinterlace == "auto" and h >= 8 and max(combs.values()) > comb_thr)
    samples = []   # (frame idx, field or None, image, time)
    for i in near:
        f = frames[i]
        if split:
            for p, fi in enumerate(split_fields(f)):
                samples.append((i, p, fi, times[i] + 0.25 * p))
        else:
            samples.append((i, None, f, times[i]))
    sharp = [laplacian_sharpness(s[2]) if (sharpness is None or split) else sharpness[s[0]]
             for s in samples]
    a_cands = [k for k, s in enumerate(samples) if s[0] == anchor_idx]
    a_k = max(a_cands, key=lambda k: sharp[k])
    ms = [match_scores[s[0]] if match_scores is not None else None for s in samples]
    picked, skipped = select_frames(
        len(samples), a_k, [s[3] for s in samples], sharp, ms if match_scores is not None else None,
        max_samples, min_rel_sharpness, min_match, max_dt)
    # native -> output grid (pixel centres): X = up * x + (up - 1) / 2
    c0 = (up - 1) / 2.0
    S_inv = np.array([[1.0 / up, 0, -c0 / up], [0, 1.0 / up, -c0 / up], [0, 0, 1]])

    def to_grid(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
        """Sample native ``img`` through ``warp`` (anchor -> sample, native
        coords) straight onto the output grid: one cubic interpolation."""
        M = _to3x3(warp) @ S_inv
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
        if warp.shape[0] == 3:
            return cv2.warpPerspective(img, M.astype(np.float32), (W, H), flags=flags,
                                       borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(img, M[:2].astype(np.float32), (W, H), flags=flags,
                              borderMode=cv2.BORDER_REPLICATE)

    def gray(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x

    anchor = samples[a_k][2]
    a_nat = anchor.astype(np.float32)
    a_nat_g = gray(a_nat)
    ident = np.eye(2, 3, dtype=np.float32)
    a_up = to_grid(a_nat, ident)
    used_k = [a_k]
    nat_stack = []      # aligned samples on the native grid (anchor excluded)
    nat_valid = []
    warps_used = [ident]
    wts = [1.0]
    rejected: list[tuple[int, str]] = []
    ecc_d: dict = {}
    ncc_d: dict = {}
    for k in picked[1:]:
        r = register_pair(anchor, samples[k][2], motion, rs, ecc_iters)
        if not r["ok"]:
            rejected.append((k, r["reason"]))
            continue
        ecc_d[k] = round(r["ecc"], 4)
        if r["ecc"] < min_ecc:
            rejected.append((k, f"ecc {r['ecc']:.3f}"))
            continue
        why = warp_plausibility(r["warp"], w, h, max_scale_dev, max_shear, max_rot_deg,
                                max_shift_frac)
        if why:
            rejected.append((k, f"warp {why}"))
            continue
        al, valid = _warp_image(samples[k][2].astype(np.float32), r["warp"], (w, h))
        cover = float(valid.mean())
        if cover < min_overlap:
            rejected.append((k, f"overlap {cover:.2f}"))
            continue
        c = ncc(a_nat_g, gray(al), valid)
        ncc_d[k] = round(c, 4)
        if c < min_ncc:
            rejected.append((k, f"ncc {c:.3f}"))
            continue
        al[~valid] = a_nat[~valid]
        used_k.append(k)
        nat_stack.append(al)
        nat_valid.append(valid)
        warps_used.append(r["warp"])
        rel_sh = sharp[k] / sharp[a_k] if sharp[a_k] > 0 else 1.0
        wts.append(float(min(1.0, max(0.1, rel_sh))))

    def up_stack():
        out = [a_up]
        for k, v, wp in zip(used_k[1:], nat_valid, warps_used[1:]):
            x = to_grid(samples[k][2].astype(np.float32), wp)
            vu = cv2.resize(v.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            x[~vu] = a_up[~vu]
            out.append(x)
        return out

    used_frames = list(dict.fromkeys(samples[k][0] for k in used_k))
    fused_ok = len(used_k) >= max(2, min_used)
    merge_w = None
    if not fused_ok:
        out = a_up
    elif fuse == "robust":
        gw = ghost_weights(a_nat, nat_stack, robust_tau)
        merge_w = [1.0] + [float((g * fw).mean()) for g, fw in zip(gw, wts[1:])]
        gw_up = [cv2.resize(g * fw, (W, H), interpolation=cv2.INTER_LINEAR)
                 for g, fw in zip(gw, wts[1:])]
        out = robust_merge(up_stack(), gw_up)
    elif fuse == "median":
        out = np.median(np.stack(up_stack(), 0), axis=0)
    elif fuse == "mean":
        out = np.mean(np.stack(up_stack(), 0), axis=0)
    elif fuse == "weighted":
        wv = np.asarray(wts, np.float32)
        out = np.tensordot(wv / wv.sum(), np.stack(up_stack(), 0), axes=1)
    else:  # sas
        nat = []
        for k in used_k:
            f = samples[k][2].astype(np.float32)
            nat.append(f if f.ndim == 3 else f[:, :, None])
        out = _shift_and_add(nat, warps_used, wts, up, (H, W), sigma=sas_sigma * up)
        if anchor.ndim == 2:
            out = out[:, :, 0]
    sig = unsharp_sigma if unsharp_sigma is not None else 0.5 * up
    out = unsharp(out.astype(np.float32), sig, unsharp_amount)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def sid(k):
        return [samples[k][0], samples[k][1]]
    diag = {
        "fused": fused_ok, "n_used": len(used_frames) if fused_ok else 1,
        "n_samples": len(used_k) if fused_ok else 1,
        "used": used_frames if fused_ok else [anchor_idx],
        "used_samples": [sid(k) for k in used_k],
        "rejected": [(sid(k), why) for k, why in rejected],
        "skipped": [(sid(k), why) for k, why in skipped],
        "ecc": {f"{samples[k][0]}.{samples[k][1]}": v for k, v in ecc_d.items()},
        "ncc": {f"{samples[k][0]}.{samples[k][1]}": v for k, v in ncc_d.items()},
        "merge_w": [round(x, 3) for x in merge_w] if merge_w else None,
        "split_fields": split, "comb": round(combs[anchor_idx], 3),
        "fuse": fuse, "motion": motion, "upscale": up, "reg_scale": round(rs, 3),
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return out, diag
