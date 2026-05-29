"""Reproduce the composite-layout measurement in data/field/composite_layout.json.

The composite broadcast frame (1920x1080) stacks the main overhead camera on top
and the two corner cameras side-by-side on the bottom. This script dumps the
artifacts used to read off the geometry:

  - seam row between main and the corner row (strongest horizontal edge ~y=703,
    the top of the brown perimeter-wall band the low corner cams see edge-on)
  - the L/R split column (x=960)
  - the always-present lower-third graphic bar (straddles the seam) + molex
    watermark, isolated in a fine-gridded bottom strip for visual measurement

Run: .venv/bin/python scripts/measure_composite_layout.py
Then open the /tmp/*.jpg artifacts and read the rects, or trust the committed
data/field/composite_layout.json which was derived this way.
"""
import cv2
import numpy as np

VIDEO = "data/test_footage/qual1_first_champs_2026.mp4"
SEAM = 703   # main/corner boundary (top of perimeter-wall band)


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 1) seam row: strongest horizontal edge in the plausible band
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(20 * fps))
    ok, f = cap.read()
    if not ok:
        raise SystemExit("could not read frame")
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rowdiff = np.abs(np.diff(g, axis=0)).mean(axis=1)
    seam = 600 + int(np.argmax(rowdiff[600:760]))
    print(f"detected seam row ~{seam} (using {SEAM})")

    # 2) fine-gridded bottom strip for reading the bar / molex rects (corner-local y)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(60 * fps))
    ok, f = cap.read()
    cap.release()
    strip = f[SEAM + 260:SEAM + 377, :].copy()      # corner-local y 260..377
    sh, sw = strip.shape[:2]
    strip = cv2.resize(strip, (sw, sh * 2))
    for x in range(0, 1920, 50):
        cv2.line(strip, (x, 0), (x, strip.shape[0]), (0, 255, 0), 1)
        if x % 100 == 0:
            cv2.putText(strip, str(x), (x + 1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    cv2.line(strip, (960, 0), (960, strip.shape[0]), (0, 0, 255), 2)  # seam
    cv2.imwrite("/tmp/strip_grid.jpg", strip)
    print("wrote /tmp/strip_grid.jpg (x grid 50/label 100; red=seam x=960; y is corner-local-260)")


if __name__ == "__main__":
    main()
