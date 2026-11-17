#!/usr/bin/env python3
"""Get the blog post into Chief Delphi (Discourse) with every image, in two pastes.

1. python3 discourse.py prep
     Copies exactly the images the post uses into upload/ (nothing else).
2. In a NEW Chief Delphi composer, drag ALL files from upload/ in at once.
   Discourse uploads them and inserts one ![name|WxH](upload://...) line each.
   Select all in the composer (Ctrl+A) and copy (Ctrl+C).
3. python3 discourse.py merge
     Reads that copied text from the clipboard (or: merge FILE, or pipe it on stdin),
     swaps every img/... path in the post for its upload:// link, writes
     vision-scouting-post.discourse.md and puts it on the clipboard.
     It refuses if any image is missing, so nothing gets silently dropped.
4. Back in the composer: Ctrl+A, Ctrl+V. Done.

The upload:// links belong to your account, so they work in any post you write,
including when you paste the merged post into the same composer.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POST = HERE / "vision-scouting-post.md"
OUT = HERE / "vision-scouting-post.discourse.md"
UPLOAD_DIR = HERE / "upload"

# ![caption|690x414](img/chart.png)  -> path group
POST_IMG = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
# ![chart|690x414](upload://abc.png)  -> name, url
UPLOADED = re.compile(r"!\[([^\]|]*)(?:\|[^\]]*)?\]\((upload://[^)\s]+)\)")


def key(name: str) -> str:
    """Filename stem, normalised the way Discourse may mangle it."""
    stem = Path(name).stem if "." in Path(name).name else name
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def post_images(text: str) -> list[str]:
    return [m.group(2) for m in POST_IMG.finditer(text) if not m.group(2).startswith(("upload://", "http"))]


def prep() -> None:
    text = POST.read_text()
    paths = post_images(text)
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir()
    seen = {}
    for p in paths:
        src = (HERE / p).resolve()
        if not src.exists():
            sys.exit(f"post references a missing file: {p}")
        k = key(src.name)
        if k in seen and seen[k] != src:
            sys.exit(f"two different images share the name {src.stem}: {seen[k]} and {src}")
        seen[k] = src
        shutil.copy2(src, UPLOAD_DIR / src.name)
    big = [f.name for f in UPLOAD_DIR.iterdir() if f.stat().st_size > 4 * 1024 * 1024]
    print(f"{len(seen)} images copied to {UPLOAD_DIR}")
    if big:
        print("over 4 MB (Discourse may shrink or refuse these):", ", ".join(big))
    print("Next: drag every file in that folder into a new Chief Delphi composer, Ctrl+A, Ctrl+C,"
          " then run: python3 discourse.py merge")


def read_uploaded(arg: str | None) -> str:
    if arg:
        return Path(arg).read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    for cmd in (["wl-paste", "--no-newline"], ["xclip", "-o", "-selection", "clipboard"], ["pbpaste"]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    sys.exit("could not read the clipboard; save the copied text to a file and run: merge FILE")


def copy_to_clipboard(text: str) -> bool:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["pbcopy"]):
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def merge(arg: str | None) -> None:
    uploaded = read_uploaded(arg)
    links = {key(name): url for name, url in UPLOADED.findall(uploaded)}
    if not links:
        sys.exit("no ![name](upload://...) lines found in what you copied. "
                 "Copy the composer text after the uploads finish.")
    text = POST.read_text()
    missing = []

    def swap(m: re.Match) -> str:
        path = m.group(2)
        if path.startswith(("upload://", "http")):
            return m.group(0)
        url = links.get(key(Path(path).name))
        if url is None:
            missing.append(path)
            return m.group(0)
        return f"{m.group(1)}{url}{m.group(3)}"

    merged = POST_IMG.sub(swap, text)
    if missing:
        sys.exit("these images have no upload yet (drag them in too, then copy and merge again):\n  "
                 + "\n  ".join(missing))
    OUT.write_text(merged)
    n = len(post_images(text))
    extra = len(links) - len({key(Path(p).name) for p in post_images(text)})
    print(f"all {n} images linked -> {OUT.name}")
    if extra > 0:
        print(f"({extra} uploaded image(s) aren't used by the post; that's harmless)")
    print("copied to clipboard: paste over the composer contents" if copy_to_clipboard(merged)
          else f"clipboard unavailable: open {OUT} and copy it")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prep":
        prep()
    elif cmd == "merge":
        merge(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
