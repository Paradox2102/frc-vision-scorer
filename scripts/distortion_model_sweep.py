"""Compare single-view distortion models on the user's 13-tag correspondence set.
Reconstructs the exact clicks (4 auto tags re-detected + 9 manual centers from the UI
table), sweeps calibrateCamera flag combos, and writes an overlay per model so we can
judge EDGE alignment (not just RMS, which only rewards overfitting).
Run: .venv/bin/python scripts/distortion_model_sweep.py"""
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.field.apriltag_detect import detect_tags
from web.field.field_model import FieldModel

V = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
TS = 5
CROP = 0.60

# Manual tag center clicks read from the UI correspondence table (id: (u, v)).
MANUAL = {1: (1337, 319), 3: (1367, 372), 5: (1471, 399), 14: (1698, 382),
          16: (1834, 475), 18: (521, 404), 20: (617, 384), 22: (662, 316),
          31: (147, 488)}

fm = FieldModel.load("2026-rebuilt-welded")
cap = cv2.VideoCapture(str(V)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(TS * fps)); ok, fr = cap.read(); cap.release()
main = fr[: int(fr.shape[0] * CROP), :]
H, W = main.shape[:2]
det = detect_tags(main)

obj, img = [], []
for tid in [6, 8, 17, 27]:
    obj.append(fm.tag_corners_3d(tid)); img.append(det[tid])
for tid, uv in MANUAL.items():
    obj.append(fm.tag_corners_3d(tid).mean(axis=0).reshape(1, 3))
    img.append(np.array([uv], float))
obj = np.vstack(obj).astype(np.float32)
img = np.vstack(img).astype(np.float32)
print(f"{len(img)} points, image {W}x{H}\n")

L, Wm = fm.length_m, fm.width_m
MODELS = [
    ("A current (k1,k2; fix pp@center; fix k3)",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
     cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, H / 2.0),
    ("B free k3 (k1,k2,k3; fix pp@center)",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
     cv2.CALIB_ZERO_TANGENT_DIST, H / 2.0),
    ("C free principal point (k1,k2; fix k3)",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST |
     cv2.CALIB_FIX_K3, H / 2.0),
    ("D free pp + k3",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST, H / 2.0),
    ("E rational (k1..k6) + free pp",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST |
     cv2.CALIB_RATIONAL_MODEL, H / 2.0),
    ("F pp seeded @ full-frame center (cy=540), fix pp, k1,k2,k3",
     cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
     cv2.CALIB_ZERO_TANGENT_DIST, 540.0),
]

def overlay(name, K, D, rvec, tvec):
    im = main.copy()
    def fp(x, y):
        p, _ = cv2.projectPoints(np.array([[[x, y, 0.0]]]), rvec, tvec, K, D)
        return tuple(p.reshape(2).astype(int))
    cv2.polylines(im, [np.array([fp(0, 0), fp(L, 0), fp(L, Wm), fp(0, Wm)], np.int32)],
                  True, (0, 255, 0), 2)
    x = 0.0
    while x <= L + 1e-6:
        cv2.line(im, fp(x, 0), fp(x, Wm), (0, 180, 0), 1); x += 2.0
    y = 0.0
    while y <= Wm + 1e-6:
        cv2.line(im, fp(0, y), fp(L, y), (0, 180, 0), 1); y += 2.0
    out = ROOT / f"data/test_footage/dist_{name}.jpg"
    cv2.imwrite(str(out), im)

for name, flags, cy in MODELS:
    K0 = np.array([[float(W), 0., W / 2.0], [0., float(W), cy], [0., 0., 1.]])
    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [obj.reshape(-1, 1, 3)], [img.reshape(-1, 1, 2)], (W, H),
            K0.copy(), np.zeros(8 if "rational" in name else 5), flags=flags)
    except cv2.error as e:
        print(f"{name:48s} FAILED {str(e)[:50]}"); continue
    fx, fy, cx, cyy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dd = np.asarray(D).ravel()
    print(f"{name:48s} RMS={rms:6.2f}  fx={fx:6.0f} cy={cyy:5.0f}  "
          f"k=[{', '.join(f'{v:.3f}' for v in dd[:3])}...]")
    tag = name.split()[0]
    overlay(tag, K, D, rvecs[0], tvecs[0])
print("\noverlays: data/test_footage/dist_{A..F}.jpg")
