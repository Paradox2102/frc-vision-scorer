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
