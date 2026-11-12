"""Same-view check for a fixed camera: register two crops of one camera (typically
temporal medians from two matches) with ORB features + a RANSAC homography and
report how far the crop's corners and centre move.

Used by the per-match corner step (web/field/corner_per_match.py) when the corner
shows too few AprilTags to check its calibration (2025 corners see 1-2 tags): a
view that registers onto the calibration frame within a few pixels is the view the
calibration was solved on. Pure: images in, numbers out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class RegistrationConfig:
    n_features: int = 4000
    ransac_px: float = 3.0
    min_matches: int = 60
    min_inliers: int = 60
    min_inlier_ratio: float = 0.25
    # Inlier bounding box / crop area: a static broadcast graphic (logo, score bug)
    # inside the crop registers at identity whatever the camera did — require the
    # agreeing features to span the view.
    min_inlier_spread: float = 0.2


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def register_views(ref_bgr, img_bgr, *, n_features: Optional[int] = None,
                   ransac_px: Optional[float] = None,
                   cfg: Optional[RegistrationConfig] = None) -> dict:
    """Homography from ``ref_bgr`` to ``img_bgr`` (same camera crop). Returns
    {"ok", "disp_px" (90th percentile shift of the RANSAC inlier features),
    "corner_disp_px" (max shift of the 4 crop corners + centre under the homography;
    diagnostic), "inliers", "matches", "spread" (inlier bbox / crop area), "reason"
    (why not ok, else None)}."""
    cfg = cfg or RegistrationConfig()
    n_features = cfg.n_features if n_features is None else n_features
    ransac_px = cfg.ransac_px if ransac_px is None else ransac_px
    out = {"ok": False, "disp_px": None, "inliers": 0, "matches": 0, "spread": None,
           "reason": None}
    if ref_bgr is None or img_bgr is None:
        out["reason"] = "missing image"
        return out
    if ref_bgr.shape[:2] != img_bgr.shape[:2]:
        out["reason"] = f"shape mismatch {ref_bgr.shape[:2]} vs {img_bgr.shape[:2]}"
        return out
    orb = cv2.ORB_create(nfeatures=int(n_features))
    k1, d1 = orb.detectAndCompute(_gray(ref_bgr), None)
    k2, d2 = orb.detectAndCompute(_gray(img_bgr), None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        out["reason"] = "too few features"
        return out
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    out["matches"] = len(matches)
    if len(matches) < max(4, cfg.min_matches):
        out["reason"] = f"too few matches ({len(matches)} < {cfg.min_matches})"
        return out
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_px))
    if H is None or mask is None:
        out["reason"] = "no homography"
        return out
    inl = mask.ravel().astype(bool)
    out["inliers"] = n_in = int(inl.sum())
    h, w = ref_bgr.shape[:2]
    pts = src.reshape(-1, 2)[inl]
    if n_in:
        bw, bh = pts.max(axis=0) - pts.min(axis=0)
        out["spread"] = round(float(bw * bh) / float(w * h), 3)
    # disp_px: 90th percentile of the inlier features' own shift. The homography
    # extrapolated to the crop corners (corner_disp_px, diagnostic only) amplifies
    # perspective-term noise at the edges: the same unmoved 2025hop corner read
    # 2.8 px on one median and 8.2 px on another, while the inlier p90 held at
    # 2-4 px on qm2-qm4. A real re-aim / zoom shifts the edge features, which p90
    # still catches.
    if n_in:
        shift = np.linalg.norm(dst.reshape(-1, 2)[inl] - pts, axis=1)
        disp = float(np.percentile(shift, 90))
    else:
        disp = float("nan")
    out["disp_px"] = round(disp, 2) if np.isfinite(disp) else None
    probe = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
                        [(w - 1) / 2, (h - 1) / 2]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(probe, H)
    cdisp = float(np.max(np.linalg.norm((moved - probe).reshape(-1, 2), axis=1)))
    out["corner_disp_px"] = round(cdisp, 2) if np.isfinite(cdisp) else None
    if n_in < cfg.min_inliers:
        out["reason"] = f"too few inliers ({n_in} < {cfg.min_inliers})"
    elif n_in < cfg.min_inlier_ratio * len(matches):
        out["reason"] = (f"inlier ratio {n_in / len(matches):.2f} < "
                         f"{cfg.min_inlier_ratio:.2f}")
    elif out["spread"] is None or out["spread"] < cfg.min_inlier_spread:
        out["reason"] = (f"inlier spread {out['spread']} < {cfg.min_inlier_spread} "
                         f"(agreeing features cover too little of the view)")
    elif out["disp_px"] is None:
        out["reason"] = "degenerate homography"
    else:
        out["ok"] = True
    return out


def same_view(ref_bgr, img_bgr, max_px: float,
              cfg: Optional[RegistrationConfig] = None) -> dict:
    """``register_views`` plus the gate: ``same`` when it registered and no probe
    point moved more than ``max_px``. ``reason`` explains a False."""
    r = register_views(ref_bgr, img_bgr, cfg=cfg)
    r["gate_px"] = float(max_px)
    r["same"] = bool(r["ok"] and r["disp_px"] <= max_px)
    if r["ok"] and not r["same"]:
        r["reason"] = f"view moved {r['disp_px']:.1f} px > gate {max_px:.0f} px"
    elif not r["ok"]:
        r["reason"] = f"registration failed: {r['reason']}"
    return r
