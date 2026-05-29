"""End-to-end check of the tag-anchored calibration backend on real footage.
Grabs a frame (with auto-detected tags), solves from those tags, prints RMS, and
writes the overlay to data/test_footage/tag_calib_overlay.jpg.
Run: .venv/bin/python scripts/verify_tag_calibration.py"""
import base64, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from web.app import app

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
if not VIDEO.exists():
    print("test video not present — skipping"); sys.exit(0)

c = TestClient(app)
fr = c.post("/api/calibration/frame", json={"timestamp": 75}).json()
auto = fr.get("detected_tags", {})
print(f"auto-detected tags: {sorted(int(k) for k in auto)}  size={fr['width']}x{fr['height']}")
r = c.post("/api/calibration/solve-tags", json={
    "event": "verify", "camera": "main", "timestamp": 75,
    "image_size": [fr["width"], fr["height"]], "auto_tags": auto, "manual_tags": []})
j = r.json()
if j.get("error"):
    print("SOLVE ERROR:", j["error"]); sys.exit(1)
print(f"RMS = {j['rms_error']:.2f} px  ({j['n_tags']} tags, {j['n_points']} pts)")
if j.get("overlay"):
    out = ROOT / "data/test_footage/tag_calib_overlay.jpg"
    out.write_bytes(base64.b64decode(j["overlay"]))
    print("overlay written:", out)
