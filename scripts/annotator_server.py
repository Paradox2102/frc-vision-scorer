#!/usr/bin/env python3
"""Tiny local server for the box annotator (tools/box_annotator.html). Lists the
extracted frames as a dropdown, serves them, and saves drawn boxes to
data/hard_negatives/boxes/<frame>.json on Submit (which auto-advances to the next
frame). Reopening a frame reloads its saved boxes.

Run:  .venv/bin/python scripts/annotator_server.py
Then open http://localhost:8765
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

FRAMES_DIR = ROOT / "tools"
BOXES_DIR = ROOT / "data" / "hard_negatives" / "boxes"
HTML_PATH = ROOT / "tools" / "box_annotator.html"
PRUNE_HTML_PATH = ROOT / "tools" / "negative_pruner.html"

app = FastAPI()


def _safe_frame(name: str):
    p = (FRAMES_DIR / name).resolve()
    if p.parent != FRAMES_DIR.resolve() or not p.name.startswith("frame_") or p.suffix != ".png":
        return None
    return p


def _safe_crop(relative_path: str):
    p = (BOXES_DIR / relative_path).resolve()
    try:
        p.relative_to(BOXES_DIR.resolve())
    except ValueError:
        return None
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.name.startswith("_"):
        return None
    return p


def _semantic_label(directory_name: str) -> str:
    return re.sub(r"\d+$", "", directory_name)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text()


@app.get("/prune", response_class=HTMLResponse)
def prune():
    return PRUNE_HTML_PATH.read_text()


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


@app.get("/api/prune/images")
def prune_images(batch: str = Query("equipment", pattern="^(equipment|all|legacy)$")):
    images = []
    for p in BOXES_DIR.glob("*/*/frame*"):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        event = p.parent.parent.name
        is_equipment = event.startswith("equipment_")
        if batch == "equipment" and not is_equipment:
            continue
        if batch == "legacy" and is_equipment:
            continue
        rel = p.relative_to(BOXES_DIR).as_posix()
        images.append({
            "path": rel,
            "event": event.removeprefix("equipment_"),
            "label": _semantic_label(p.parent.name),
            "box": p.parent.name,
            "frame": p.stem.removeprefix("frame"),
        })
    images.sort(key=lambda item: (item["event"], item["label"], item["box"], item["frame"]))
    return {"images": images, "count": len(images)}


@app.get("/api/prune/image")
def prune_image(path: str):
    p = _safe_crop(path)
    if p is None or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.delete("/api/prune/image")
def delete_prune_image(path: str):
    p = _safe_crop(path)
    if p is None or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    p.unlink()
    return {"deleted": path}


if __name__ == "__main__":
    print("Annotator listening on http://0.0.0.0:8765  "
          "(use this machine's Tailscale IP from another computer)")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
