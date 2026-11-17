#!/usr/bin/env python3
"""Get the blog post into Chief Delphi (Discourse) with every image.

1. python3 discourse.py prep
     Copies the images the post still needs into upload/ (only ones that don't
     already have an upload:// link remembered from an earlier batch).
2. In a Chief Delphi composer, drag ALL files from upload/ in at once.
   Discourse uploads them and inserts one ![name|WxH](upload://...) line each.
   Select all in the composer (Ctrl+A) and copy (Ctrl+C).
3. python3 discourse.py merge
     Reads that copied text from the clipboard (or: merge FILE, or pipe it on stdin)
     and remembers every link in upload_links.tsv. If some uploads failed, it
     refills upload/ with just the missing images: wait a minute (Chief Delphi
     rate-limits uploads), drag those in, copy, and merge again.
     Once every image has a link it writes vision-scouting-post.discourse.md
     and puts it on the clipboard.
4. In the composer: Ctrl+A, Ctrl+V. Done.

Links are remembered per image along with a hash of the file, so an image you
re-render afterwards is treated as needing a new upload. `prep --fresh` forgets
every remembered link. The upload:// links belong to your account, so they work
in any post you write.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POST = HERE / "vision-scouting-post.md"
OUT = HERE / "vision-scouting-post.discourse.md"
UPLOAD_DIR = HERE / "upload"
LINKS = HERE / "upload_links.tsv"            # key \t url \t sha1-of-file-when-learned

# ![caption|690x414](img/chart.png)  -> path group
POST_IMG = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
# ![chart|690x414](upload://abc.png)  -> name, url
UPLOADED = re.compile(r"!\[([^\]|]*)(?:\|[^\]]*)?\]\((upload://[^)\s]+)\)")


def key(name: str) -> str:
    """Filename stem, normalised the way Discourse may mangle it."""
    stem = Path(name).stem if "." in Path(name).name else name
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def post_images() -> dict[str, Path]:
    """key -> image file, for every local image the post references."""
    out: dict[str, Path] = {}
    for m in POST_IMG.finditer(POST.read_text()):
        p = m.group(2)
        if p.startswith(("upload://", "http")):
            continue
        src = (HERE / p).resolve()
        if not src.exists():
            sys.exit(f"post references a missing file: {p}")
        k = key(src.name)
        if k in out and out[k] != src:
            sys.exit(f"two different images share the name {src.stem}: {out[k]} and {src}")
        out[k] = src
    return out


def load_links() -> dict[str, tuple[str, str]]:
    links = {}
    if LINKS.exists():
        for line in LINKS.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                links[parts[0]] = (parts[1], parts[2])
    return links


def save_links(links: dict[str, tuple[str, str]]) -> None:
    LINKS.write_text("".join(f"{k}\t{u}\t{h}\n" for k, (u, h) in sorted(links.items())))


def still_needed(images: dict[str, Path], links: dict[str, tuple[str, str]]) -> dict[str, Path]:
    return {k: p for k, p in images.items() if k not in links or links[k][1] != sha1(p)}


def fill_upload_dir(files: dict[str, Path]) -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir()
    for p in files.values():
        shutil.copy2(p, UPLOAD_DIR / p.name)


def prep(fresh: bool) -> None:
    if fresh and LINKS.exists():
        LINKS.unlink()
    images = post_images()
    need = still_needed(images, load_links())
    fill_upload_dir(need)
    have = len(images) - len(need)
    print(f"{len(images)} images in the post; {have} already uploaded, {len(need)} copied to {UPLOAD_DIR}")
    big = [p.name for p in need.values() if p.stat().st_size > 4 * 1024 * 1024]
    if big:
        print("over 4 MB (Discourse may shrink or refuse these):", ", ".join(big))
    if need:
        print("Next: drag every file in that folder into the Chief Delphi composer, Ctrl+A, Ctrl+C,"
              " then run: python3 discourse.py merge")
    else:
        print("Nothing to upload. Run: python3 discourse.py merge")


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
    images = post_images()
    links = load_links()
    found = UPLOADED.findall(read_uploaded(arg))
    learned = 0
    for name, url in found:
        k = key(name)
        if k in images:               # remember against the file as it is right now
            if links.get(k, ("", ""))[0] != url:
                learned += 1
            links[k] = (url, sha1(images[k]))
    save_links(links)
    if found:
        print(f"read {len(found)} upload link(s) ({learned} new)")

    need = still_needed(images, links)
    if need:
        fill_upload_dir(need)
        names = "\n  ".join(p.name for p in need.values())
        sys.exit(f"{len(images) - len(need)} of {len(images)} images have links. Still needed:\n  {names}\n"
                 f"Those are now the only files in {UPLOAD_DIR}. Wait a minute, drag them into the"
                 " composer, Ctrl+A, Ctrl+C, then run merge again.")

    def swap(m: re.Match) -> str:
        path = m.group(2)
        if path.startswith(("upload://", "http")):
            return m.group(0)
        return f"{m.group(1)}{links[key(Path(path).name)][0]}{m.group(3)}"

    merged = POST_IMG.sub(swap, POST.read_text())
    OUT.write_text(merged)
    print(f"all {len(images)} images linked -> {OUT.name}")
    print("copied to clipboard: in the composer, Ctrl+A then Ctrl+V" if copy_to_clipboard(merged)
          else f"clipboard unavailable: open {OUT} and copy it")


GITHUB_OUT = HERE / "vision-scouting.md"
GITHUB_HEADER = ("<!-- Generated from vision-scouting-post.md by `python3 discourse.py github`; "
                 "edit the post, not this file. -->\n\n")


def github() -> None:
    """Shareable GitHub-rendering copy: no [color] tags, no |WxH hints, captions under images."""
    text = POST.read_text()
    text = re.sub(r"\[color=#[0-9A-Fa-f]{3,8}\]", "", text).replace("[/color]", "")
    text = re.sub(r"^(#+ .*?)\s+-\s+", r"\1: ", text, flags=re.M)      # "Title  - Subtitle" headers

    def img(m: re.Match) -> str:
        cap, path = m.group(1), m.group(2)
        cap = cap.split("|")[0].strip()
        if len(cap.split()) <= 4:            # title-like alt text (the splash): no caption line
            return f"![{cap}]({path})"
        return f"![{cap}]({path})\n*{cap}*"

    text, n_images = re.subn(r"!\[([^\]]*)\]\(([^)\s]+)\)", img, text)
    GITHUB_OUT.write_text(GITHUB_HEADER + text)
    print(f"wrote {GITHUB_OUT.name} ({n_images} images)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "github":
        github()
    elif cmd == "prep":
        prep(fresh="--fresh" in sys.argv[2:])
    elif cmd == "merge":
        merge(next((a for a in sys.argv[2:] if not a.startswith("-")), None))
    else:
        print(__doc__)
