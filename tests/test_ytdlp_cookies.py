"""YTDLP_COOKIES_FILE -> cookiefile on every yt-dlp call (no-op when unset)."""
from __future__ import annotations

import re
from pathlib import Path

from web import ytdlp_cookies

REPO = Path(__file__).resolve().parents[1]


def test_unset_is_noop(monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES_FILE", raising=False)
    opts = {"quiet": True}
    assert ytdlp_cookies.with_cookies(opts) is opts


def test_set_adds_cookiefile_without_mutating(monkeypatch, tmp_path):
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(f))
    opts = {"quiet": True}
    out = ytdlp_cookies.with_cookies(opts)
    assert out == {"quiet": True, "cookiefile": str(f)}
    assert "cookiefile" not in opts


def test_missing_file_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(tmp_path / "nope.txt"))
    opts = {"quiet": True}
    assert ytdlp_cookies.with_cookies(opts) == opts


def test_every_youtubedl_call_goes_through_the_helper():
    for rel in ("web/app.py", "web/field/auto_calibrate.py"):
        src = (REPO / rel).read_text()
        calls = re.findall(r"yt_dlp\.YoutubeDL\(([^)]*\))", src)
        assert calls, rel
        for c in calls:
            assert c.startswith(("with_ytdlp_cookies(", "with_cookies(")), (rel, c)
