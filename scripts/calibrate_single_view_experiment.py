"""EXPERIMENT (not part of the SP1 plan): single-view full camera calibration on
the static main-camera broadcast feed.

The match camera is static (one viewpoint), so cv2.fisheye.calibrate is degenerate.
This spike tries cv2.calibrateCamera (standard pinhole + distortion) on a single
view of the NON-coplanar field tags, accumulated across many frames. It also
resolves the 2D(detector)<->3D(field_model) corner-order alignment empirically by
searching the 8 dihedral orderings for minimum reprojection RMS.

Outputs (to data/test_footage/):
  - prints best corner-order permutation, RMS, K, distortion
  - calib_overlay.jpg : field outline + tag reprojection drawn on a sample frame
Run: .venv/bin/python scripts/calibrate_single_view_experiment.py
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.field.field_model import FieldModel, TAG_SIZE_M  # noqa: E402
from web.field.apriltag_detect import detect_tags  # noqa: E402

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
MAIN_MAX_Y_FRAC = 0.58   # tags with all corners above this belong to the main cam
N_SAMPLES = 200          # frames to scan for tag accumulation

# 8 dihedral orderings of corner indices (rotations + reflected rotations).
PERMS = [(0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2),
         (3,2,1,0),(0,3,2,1),(1,0,3,2),(2,1,0,3)]


def accumulate_main_tags(video, n_samples, max_y):
    """Average each main-cam tag's image corners over the frames it appears in."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_cut = h * max_y
    acc, cnt = {}, {}
    for fi in np.linspace(0, total - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        for tid, corners in detect_tags(frame).items():
            if corners[:, 1].max() >= y_cut:      # not fully in the main region
                continue
            acc[tid] = acc.get(tid, np.zeros((4, 2))) + corners
            cnt[tid] = cnt.get(tid, 0) + 1
    cap.release()
    return {t: acc[t] / cnt[t] for t in acc}, cnt


def main():
    fm = FieldModel.load("2026-rebuilt-welded")
    tags, counts = accumulate_main_tags(VIDEO, N_SAMPLES, MAIN_MAX_Y_FRAC)
    print(f"main-cam tags accumulated: {sorted(tags)}")
    print("frames seen per tag:", {t: counts[t] for t in sorted(counts)})
    if len(tags) < 4:
        print(f"ONLY {len(tags)} tags — single-view calibration needs more spread. "
              "Increase N_SAMPLES or relax filter.")
        return

    img_size = (1920, 1080)
    tag_ids = sorted(tags)
    corners_2d = {t: tags[t].astype(np.float32) for t in tag_ids}
    corners_3d = {t: fm.tag_corners_3d(t).astype(np.float32) for t in tag_ids}

    # Guessed intrinsics: wide fisheye, main view occupies the upper composite.
    # fx from ~100deg HFOV over 1920px; principal point near main-view center.
    K_guess = np.array([[900., 0., 960.], [0., 900., 360.], [0., 0., 1.]])

    def reproj_rms(perm, K, dist, rvec, tvec):
        e = []
        for t in tag_ids:
            p, _ = cv2.projectPoints(corners_3d[t][list(perm)], rvec, tvec, K, dist)
            e.append(((p.reshape(-1, 2) - corners_2d[t]) ** 2).sum(axis=1))
        return float(np.sqrt(np.concatenate(e).mean()))

    # --- Step 1: resolve corner order via PnP with guessed K (no distortion) ---
    print("\n[1] PnP corner-order search (guessed K, no distortion):")
    pnp_best = None
    for perm in PERMS:
        obj = np.vstack([corners_3d[t][list(perm)] for t in tag_ids]).astype(np.float32)
        img = np.vstack([corners_2d[t] for t in tag_ids]).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K_guess, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        r = reproj_rms(perm, K_guess, None, rvec, tvec)
        print(f"   perm {perm}: PnP reproj RMS = {r:7.1f} px")
        if pnp_best is None or r < pnp_best[0]:
            pnp_best = (r, perm, rvec, tvec)
    rms_pnp, perm, rvec, tvec = pnp_best
    print(f"  -> BEST perm {perm}: RMS {rms_pnp:.1f} px (guessed K, pinhole)")

    # --- Step 2: try to also refine intrinsics+distortion from the single view ---
    K, dist = K_guess, np.zeros(5)
    rms = rms_pnp
    obj = np.vstack([corners_3d[t][list(perm)] for t in tag_ids]).astype(np.float32)
    img = np.vstack([corners_2d[t] for t in tag_ids]).astype(np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    try:
        rms2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            [obj], [img], img_size, K_guess.copy(), np.zeros(5), flags=flags)
        print(f"\n[2] calibrateCamera (intrinsic-guess, fixed cx/cy): RMS {rms2:.2f} px")
        K, dist, rvec, tvec, rms = K2, dist2, rvecs2[0], tvecs2[0], rms2
    except cv2.error as e:
        print(f"\n[2] calibrateCamera failed ({str(e)[:70]}); using PnP+guessed-K result")

    print(f"\nRESULT: perm={perm}  RMS={rms:.2f}px  ({len(tag_ids)} tags, {len(tag_ids)*4} pts)")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"dist (k1,k2,p1,p2,k3): {np.asarray(dist).ravel()}")

    # image_to_field sanity: back-project each tag's image center to the ground,
    # compare to its known (x,y) (note tags are above ground, so expect some offset).
    R, _ = cv2.Rodrigues(rvec)
    cam_center = (-R.T @ tvec).reshape(3)
    print("\nimage_to_field check (tag image-center -> ground xy vs known tag xy):")
    for t in tag_ids:
        uv = corners_2d[t].mean(axis=0).reshape(1, 1, 2)
        norm = cv2.undistortPoints(uv, K, dist).reshape(2)
        ray = R.T @ np.array([norm[0], norm[1], 1.0])
        if abs(ray[2]) < 1e-9:
            continue
        s = -cam_center[2] / ray[2]
        gx, gy = (cam_center + s * ray)[:2]
        kx, ky, kz = fm.tag_translation(t)
        print(f"  tag {t:2d}: ground=({gx:6.2f},{gy:6.2f})  known_xy=({kx:6.2f},{ky:6.2f})  "
              f"z={kz:.2f}")

    # Overlay: reproject field perimeter + all tag positions onto a sample frame.
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.35))
    ok, frame = cap.read()
    cap.release()
    L, W = fm.length_m, fm.width_m
    perim = np.array([[0,0,0],[L,0,0],[L,W,0],[0,W,0]], np.float32)
    proj, _ = cv2.projectPoints(perim, rvec, tvec, K, dist)
    cv2.polylines(frame, [proj.astype(int).reshape(-1,2)], True, (0,255,0), 2)
    for t in fm.tags:
        p, _ = cv2.projectPoints(fm.tag_translation(t).reshape(1,3).astype(np.float32),
                                 rvec, tvec, K, dist)
        u, v = p.reshape(2)
        if 0 <= u < 1920 and 0 <= v < 1080:
            seen = t in tags
            cv2.circle(frame, (int(u), int(v)), 5, (0,255,0) if seen else (0,165,255), -1)
            cv2.putText(frame, str(t), (int(u)+4, int(v)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if seen else (0,165,255), 1)
    out = ROOT / "data/test_footage/calib_overlay.jpg"
    cv2.imwrite(str(out), frame)
    print(f"\noverlay written: {out}")
    print("GREEN = used tags / field outline; ORANGE = other tags reprojected.")


if __name__ == "__main__":
    main()
