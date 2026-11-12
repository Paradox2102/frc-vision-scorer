"""Borrow a sibling corner camera's lens and solve only the pose.

Broadcasts often mount identical cameras in pairs (2025 Champs: two wide-lens
cameras on the barge, one per reef). When one of them sees too few tags to pin down
its own lens, a free-lens solve trades focal length against distance and lands the
camera on the wrong side of the field (2025hop corner_right: fx 1168, x = 6.9 m).
Its sibling's lens, with only the 6-DOF pose solved, puts it where it is (x = 9.3 m,
the red face of the barge).

Rule (operator): corner cameras always have a different lens from the main camera.
Only corner cameras (any non-"main" layout camera) lend to corner cameras; main
never lends and never borrows, and a camera never "borrows" its own lens.

Nothing here does IO: callers pass the versions / records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from web.field.calibration import CalibrationError, CameraCalibration
from web.field.match_order import select_epoch

MAIN_CAMERA = "main"
# Auto records (this run) whose candidate may lend its lens. "verified" records carry
# no calibration of their own: their lens is the covering version, found as one.
LENDING_RECORD_STATUSES = ("ready",)


def borrow_rule_error(borrower: str, lender: str) -> Optional[str]:
    """Why ``borrower`` may not use ``lender``'s lens, or None when it may."""
    if borrower == MAIN_CAMERA or lender == MAIN_CAMERA:
        return ("the main camera never lends or borrows a lens (its lens differs from "
                "the corner cameras')")
    if borrower == lender:
        return f"{borrower} cannot borrow its own lens"
    return None


def transfer_lens(source: CameraCalibration, target_size) -> CameraCalibration:
    """A lens prior for a ``target_size`` (w, h) crop from ``source``'s lens.

    Camera panels are scaled copies of the camera image and only the vertical scale
    is reliable (panels are cropped left/right to fit the row), so fx and fy both
    scale by target_h / source_h. The principal point goes to the target crop centre
    (the convention the constrained pp_* models use) and the distortion coefficients
    carry over unchanged (they act on normalised coordinates). No pose: rvec/tvec are
    None. Pinhole only — a fisheye ("camera") or homography source raises."""
    if source is None or source.mode != "pinhole" or source.K is None or source.D is None:
        mode = None if source is None else source.mode
        raise CalibrationError(f"only a pinhole lens can be borrowed (source mode "
                               f"'{mode}')")
    sw, sh = (int(v) for v in source.image_size)
    tw, th = (int(v) for v in target_size)
    if sh <= 0 or th <= 0 or tw <= 0:
        raise CalibrationError(f"cannot scale a lens from {sw}x{sh} to {tw}x{th}")
    s = th / float(sh)
    K0 = np.asarray(source.K, float)
    K = np.array([[K0[0, 0] * s, 0.0, tw / 2.0],
                  [0.0, K0[1, 1] * s, th / 2.0],
                  [0.0, 0.0, 1.0]])
    return CameraCalibration(mode="pinhole", K=K,
                             D=np.asarray(source.D, float).reshape(-1).copy(),
                             image_size=(tw, th), crop_region=(0, 0, tw, th),
                             field_season=source.field_season)


def solve_pose_with_lens(corr: dict, image_size, season: str,
                         lens: CameraCalibration) -> CameraCalibration:
    """The 6-DOF pose of a camera with ``lens`` fixed (``tag_locator`` "pose" model:
    multi-start PnP + the plausibility gate). ``corr``: {key: (img (N,2), obj (N,3))},
    >= 2 tags. Raises CalibrationError."""
    from web.field import tag_locator as tl
    return tl.solve_model(corr, image_size, season, "pose", lens=lens)


def per_tag_residuals(cal: CameraCalibration, corr: dict) -> dict:
    """{key: [px residual per point]} of ``cal`` on ``corr``."""
    out = {}
    for k, (img, obj) in corr.items():
        errs = []
        for uv, p in zip(np.asarray(img, float).reshape(-1, 2),
                         np.asarray(obj, float).reshape(-1, 3)):
            u, v = cal.field_to_image(tuple(p))
            errs.append(round(float(np.hypot(u - uv[0], v - uv[1])), 2))
        out[k] = errs
    return out


@dataclass
class LensLender:
    camera: str
    calibration: CameraCalibration
    version_id: Optional[str] = None
    kind: str = "version"             # "version" (confirmed / operator) | "auto_record"

    def describe(self) -> str:
        if self.version_id:
            return f"{self.camera} (version {self.version_id})"
        return f"{self.camera} (this run's ready auto result)"

    def source_fields(self) -> dict:
        """What a record / version stores about the lens it borrowed."""
        return {"lens_from": self.camera, "lens_version_id": self.version_id,
                "lens_kind": self.kind}


def _pinhole(cal) -> bool:
    return cal is not None and cal.mode == "pinhole" and cal.K is not None


def find_lenders(camera: str, cameras: Iterable[str], versions_for: Callable,
                 match_key: Optional[str], records: Optional[dict] = None) -> list:
    """Sibling corner cameras that can lend ``camera`` a lens, in ``cameras`` order.
    Usable: the version covering ``match_key`` (confirmed / operator) when the
    sibling has versions, else its ``records[cam]`` auto record of THIS run when
    ready. Pinhole only. ``versions_for(cam)`` -> that camera's versions."""
    out = []
    for cam in dict.fromkeys(cameras):
        if borrow_rule_error(camera, cam) is not None:
            continue
        vs = versions_for(cam) or []
        if vs:
            ep = select_epoch(vs, match_key)
            if ep is not None and _pinhole(ep["calibration"]):
                out.append(LensLender(cam, ep["calibration"], ep["version_id"], "version"))
            continue
        rec = (records or {}).get(cam)
        if rec and rec.get("status") in LENDING_RECORD_STATUSES and rec.get("calibration"):
            try:
                cal = CameraCalibration.from_json(rec["calibration"])
            except (KeyError, TypeError, ValueError):
                continue
            if _pinhole(cal):
                out.append(LensLender(cam, cal, None, "auto_record"))
    return out


MIN_BORROW_TAGS = 2      # tag_locator.MODEL_MIN_TAGS["pose"] / calibration.MIN_POSE_TAGS


def should_retry(rec: Optional[dict], camera: str, lenders) -> bool:
    """A second auto-calibration pass for ``camera``: its full solve failed, it saw
    enough tags for a pose, and a lender exists that was not tried yet (a sibling
    became ready later in the same run)."""
    if not rec or not lenders or camera == MAIN_CAMERA:
        return False
    if rec.get("status") != "failed" or rec.get("method") != "full_solve":
        return False
    if int((rec.get("metrics") or {}).get("n_tags") or 0) < MIN_BORROW_TAGS:
        return False
    tried = set(rec.get("lens_lenders_tried") or ())
    return any(l.camera not in tried for l in lenders)
