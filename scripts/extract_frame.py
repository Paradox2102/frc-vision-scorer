#!/usr/bin/env python3
"""Save one full frame from a video as a PNG, for annotating in tools/box_annotator.html.

Usage:
    .venv/bin/python scripts/extract_frame.py --video web/uploads/<uuid>.mp4 \\
        --frame 2160 --out /tmp/frame_qm30.png
"""
import argparse
import sys

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read frame {args.frame} from {args.video}")
    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f"wrote {args.out}  ({w}x{h})  frame {args.frame} of {args.video}")


if __name__ == "__main__":
    main()
