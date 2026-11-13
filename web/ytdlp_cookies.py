"""Optional YouTube cookies for every yt-dlp call.

Datacenter IPs (e.g. a rented RunPod GPU) often hit YouTube's "Sign in to
confirm you're not a bot" check. Exporting a logged-in browser's cookies.txt
and pointing ``YTDLP_COOKIES_FILE`` at it lets yt-dlp through. Unset (the
default, and the local setup) -> options are returned unchanged.
"""
from __future__ import annotations

import os

_warned = False


def with_cookies(opts: dict) -> dict:
    """``opts`` plus ``cookiefile`` when YTDLP_COOKIES_FILE names a readable file."""
    global _warned
    path = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    if not path:
        return opts
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        if not _warned:
            print(f"  [yt-dlp] YTDLP_COOKIES_FILE={path} not found; downloading without cookies")
            _warned = True
        return opts
    return {**opts, "cookiefile": path}
