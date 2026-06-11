#!/usr/bin/env python3
"""Download a YouTube match video as OpenCV-decodable H.264 mp4, reusing the app's
working recipe (deno on PATH before importing yt_dlp + format_sort vcodec:h264, so
we avoid the AV1 streams OpenCV can't decode). For grabbing other-event 2026 footage
to mine hard negatives from.

Usage:
  .venv/bin/python scripts/download_match.py \\
      --url https://youtu.be/6JEQOB0o5jU --out data/test_footage/2026hiho_qm41.mp4
"""
import argparse
import os

# deno must be on PATH BEFORE importing yt_dlp (unlocks >360p formats)
_deno = os.path.expanduser("~/.deno/bin")
if os.path.isdir(_deno) and _deno not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _deno + os.pathsep + os.environ.get("PATH", "")

import yt_dlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    opts = {
        "format": ("bestvideo[height<=1080]+bestaudio[ext=m4a]/"
                   "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"),
        "format_sort": ["res:1080", "vcodec:h264"],
        "merge_output_format": "mp4",
        "outtmpl": args.out,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([args.url])
    print(f"downloaded -> {args.out}")


if __name__ == "__main__":
    main()
