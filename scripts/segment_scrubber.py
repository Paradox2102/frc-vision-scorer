"""Segment scrubber — pick the SAM2-spike hard segment and read anchor uv.

Two modes, both operating on the MAIN VIEW (top OVERHEAD_CROP_RATIO of the
composite, the space the calibration and the spike harness use), so any uv you
read here drops straight into the spike label sidecar's `anchors`.

  montage : tile every Nth frame across a range, each labelled with its frame
            number, so you can eyeball where the hub scrum / same-alliance
            crossing happens and bracket the segment.

  frame   : dump one full-resolution main-view frame, optionally with a coordinate
            grid (uv labelled every 100 px), so you can read off the (u, v) of each
            robot for an anchor record.

Examples:
  .venv/bin/python scripts/segment_scrubber.py montage --start 200 --end 1200 --step 20 --out /tmp/montage.png
  .venv/bin/python scripts/segment_scrubber.py frame --idx 640 --grid --out /tmp/frame_640.png

Default video is the footage the validated test/main calibration was built on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
# Main view = top fraction of the composite. Mirrors web.app.OVERHEAD_CROP_RATIO
# (kept as a literal so the scrubber stays import-light — no model/app load).
OVERHEAD_CROP_RATIO = 0.67

_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)


def _open(video: Path):
    if not video.exists():
        sys.exit(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"could not open: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, n, fps


def _main_view(frame: np.ndarray) -> np.ndarray:
    h = frame.shape[0]
    return frame[: int(h * OVERHEAD_CROP_RATIO), :]


def _read(cap, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    return frame if ok else None


def montage(args):
    cap, n, fps = _open(args.video)
    end = args.end if args.end > 0 else n - 1
    idxs = list(range(args.start, min(end, n - 1) + 1, args.step))
    if not idxs:
        sys.exit("empty frame range")
    tiles = []
    for i in idxs:
        frame = _read(cap, i)
        if frame is None:
            continue
        view = _main_view(frame)
        tile = cv2.resize(view, (args.tile_w, int(args.tile_w * view.shape[0] / view.shape[1])))
        secs = i / fps
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(tile, f"f{i}  {secs:0.1f}s", (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _YELLOW, 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    if not tiles:
        sys.exit("no frames read")
    th, tw = tiles[0].shape[:2]
    tiles = [t if t.shape[:2] == (th, tw) else cv2.resize(t, (tw, th)) for t in tiles]
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(str(args.out), sheet)
    print(f"montage: {len(tiles)} frames (f{idxs[0]}..f{idxs[-1]} step {args.step}) -> {args.out}")
    print(f"video: {args.video.name}  {n} frames @ {fps:.2f}fps  (~{n/fps:0.0f}s)")


def frame(args):
    cap, n, fps = _open(args.video)
    if not (0 <= args.idx < n):
        sys.exit(f"idx {args.idx} out of range 0..{n-1}")
    img = _read(cap, args.idx)
    cap.release()
    if img is None:
        sys.exit(f"could not read frame {args.idx}")
    view = _main_view(img).copy()
    h, w = view.shape[:2]
    if args.grid:
        step = args.grid_step
        for x in range(0, w, step):
            cv2.line(view, (x, 0), (x, h), _GREEN, 1)
            cv2.putText(view, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, _GREEN, 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(view, (0, y), (w, y), _GREEN, 1)
            cv2.putText(view, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, _GREEN, 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out), view)
    print(f"frame {args.idx} ({args.idx/fps:0.2f}s) main-view {w}x{h}"
          f"{' +grid('+str(args.grid_step)+'px)' if args.grid else ''} -> {args.out}")
    print("uv read from this image is in main-crop pixels — drop straight into anchors.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                   help=f"source video (default: {DEFAULT_VIDEO.name})")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("montage", help="tile sampled frames to find the segment")
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--end", type=int, default=-1, help="-1 = to end")
    m.add_argument("--step", type=int, default=20)
    m.add_argument("--cols", type=int, default=6)
    m.add_argument("--tile-w", type=int, default=320)
    m.add_argument("--out", type=Path, default=Path("/tmp/scrub_montage.png"))
    m.set_defaults(func=montage)

    f = sub.add_parser("frame", help="dump one main-view frame for reading anchor uv")
    f.add_argument("--idx", type=int, required=True)
    f.add_argument("--grid", action="store_true", help="overlay a uv coordinate grid")
    f.add_argument("--grid-step", type=int, default=100)
    f.add_argument("--out", type=Path, default=Path("/tmp/scrub_frame.png"))
    f.set_defaults(func=frame)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
