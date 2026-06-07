#!/usr/bin/env python3
"""Tiny local server for the box annotator (tools/box_annotator.html). Lists the
extracted frames as a dropdown, serves them, and saves drawn boxes to
data/hard_negatives/boxes/<frame>.json on Submit (which auto-advances to the next
frame). Reopening a frame reloads its saved boxes.

Run:  .venv/bin/python scripts/annotator_server.py
Then open http://localhost:8765
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

FRAMES_DIR = ROOT / "tools"
BOXES_DIR = ROOT / "data" / "hard_negatives" / "boxes"
HTML_PATH = ROOT / "tools" / "box_annotator.html"

app = FastAPI()


def _safe_frame(name: str):
    p = (FRAMES_DIR / name).resolve()
    if p.parent != FRAMES_DIR.resolve() or not p.name.startswith("frame_") or p.suffix != ".png":
        return None
    return p


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text()


@app.get("/api/frames")
def frames():
    return sorted(p.name for p in FRAMES_DIR.glob("frame_*.png"))


@app.get("/api/frame/{name}")
def frame(name: str):
    p = _safe_frame(name)
    if p is None or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/api/boxes/{name}")
def get_boxes(name: str):
    p = BOXES_DIR / (Path(name).stem + ".json")
    return json.loads(p.read_text()) if p.exists() else []


@app.post("/api/save")
async def save(req: Request):
    body = await req.json()
    frame_name = Path(body["frame"]).stem
    boxes = body.get("boxes", [])
    BOXES_DIR.mkdir(parents=True, exist_ok=True)
    out = BOXES_DIR / (frame_name + ".json")
    out.write_text(json.dumps(boxes, indent=2))
    return {"saved": str(out), "n": len(boxes)}


if __name__ == "__main__":
    print("Annotator at http://localhost:8765  (frames from tools/, boxes -> data/hard_negatives/boxes/)")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
