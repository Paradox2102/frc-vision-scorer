"""Viewer wiring for the HUD render state (spec section 7): the new elements
exist in index.html, app.js uses them and the new endpoint, and app.js parses."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
IDS = ("replayRenderState", "replayRenderText", "replayRenderRetryBtn", "replayRenderBar",
       "replayRerenderBtn", "replayRerenderStatus", "replayRenderStats")


def test_elements_exist_and_are_used():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    for i in IDS:
        assert f'id="{i}"' in html, i
        assert f"'{i}'" in js, i
    assert "/render`, { method: 'POST' }" in js
    assert re.search(r"render_status === 'render_failed'", js)
    assert "frames relabelled" in js and "Rendering video" in js


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_app_js_parses():
    p = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def _method(js: str, name: str) -> str:
    """The body of an app.js class method (up to the next 4-space method)."""
    start = re.search(rf"\n    (async )?{name}\(", js).start()
    nxt = re.search(r"\n    (async )?[A-Za-z_]+\([^)]*\) \{\n", js[start + 5:])
    return js[start: start + 5 + nxt.start()] if nxt else js[start:]


def test_poll_keeps_polling_with_backoff_on_errors():
    js = (STATIC / "app.js").read_text()
    body = _method(js, "pollReplayRender")
    assert body.count("this.scheduleReplayRenderPoll(id") >= 2      # non-OK and network error
    sched = _method(js, "scheduleReplayRenderPoll")
    assert "_replayRenderBackoff" in sched and "Math.min(" in sched


def test_video_src_is_unloaded_and_cache_busted_by_the_render():
    js = (STATIC / "app.js").read_text()
    body = _method(js, "applyReplayRenderState")
    assert re.search(r"removeAttribute\('src'\);\s*this\.replayVideo\.load\(\);", body)
    assert "Date.now()" not in body
    assert "rendered_at" in body


def test_rerendering_a_live_encoded_replay_asks_first():
    js = (STATIC / "app.js").read_text()
    body = _method(js, "requestReplayRender")
    assert "confirm(" in body and "hud.live.mp4" in body and "without masks" in body
    assert "this._replayRenderStatus == null" in body


def test_a_correction_during_a_render_says_to_re_render_after():
    js = (STATIC / "app.js").read_text()
    msg = "a render is running — re-render again when it finishes"
    assert msg in _method(js, "saveIdentityCorrection")
    import web.app as app_mod
    assert app_mod.RENDER_BUSY_MESSAGE == msg
    body = _method(js, "requestReplayRender")
    assert "data.error" in body and "resp.status === 409" in body
