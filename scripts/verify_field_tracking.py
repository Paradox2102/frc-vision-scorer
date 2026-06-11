"""Smoke check for SP2 field-space tracking.

Drives the real field modules (robot YOLO -> detect_bridge -> FieldTracker ->
resolve_occlusion) over a window of the test video using the saved calibration —
independent of the full InferenceJob (which needs an event session + websocket).
Asserts the fixed-6 invariant (<=6 slots, <=3 per alliance) and that occlusion
resolves without error. Writes an annotated+radar sample for eyeballing.

Run: .venv/bin/python scripts/verify_field_tracking.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.tracker import FieldTracker
from web.field.occlusion import hub_prisms, resolve_occlusion
from web.field.detect_bridge import to_field_detections, field_adjacent_persons
from web.field.radar import Radar
from web.app import MODEL_PATH, CALIB_MAIN_CROP_FRAC

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
if not VIDEO.exists():
    print("test video not present — skipping"); sys.exit(0)

cal = field_store.load("test", "main")
if cal is None or cal.camera_center is None:
    print("no pose-bearing calibration for test/main — calibrate first; skipping")
    sys.exit(0)

fm = FieldModel.load(cal.field_season)
dims = (fm.length_m, fm.width_m)
prisms = hub_prisms(fm)
trk = FieldTracker(fm.length_m)
radar = Radar(fm.length_m, fm.width_m)
robot_model = YOLO(str(MODEL_PATH))
person_model = YOLO(str(ROOT / "yolo26n.pt"))

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
view_flags, seen_slots, max_blue, max_red = {}, set(), 0, 0
last_frame = None
for fi in range(200, 700, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, fr = cap.read()
    if not ok:
        break
    main = fr[: int(fr.shape[0] * CALIB_MAIN_CROP_FRAC), :]
    res = robot_model(main, conf=0.4, verbose=False)[0]
    rboxes = [res.boxes.xyxy[i].cpu().numpy() for i in range(len(res.boxes))
              if int(res.boxes.cls[i]) == 1]
    dets = to_field_detections(
        rboxes, cal, dims, colors=["unknown"] * len(rboxes)
    )
    pres = person_model(main, classes=[0], conf=0.35, verbose=False)[0]
    pboxes = [tuple(pres.boxes.xyxy[i].cpu().numpy()) for i in range(len(pres.boxes))]
    persons = field_adjacent_persons(pboxes, cal, dims)
    outs = trk.update(dets, view_flags, dt=5 / fps)
    occ = resolve_occlusion(cal, [{"slot_id": t.slot_id, "pos": t.pos, "box": t.box}
                                  for t in outs if t.box is not None], persons, prisms)
    view_flags = {s: {"view_occluded": v["view_occluded"]} for s, v in occ.items()}
    seen_slots.update(t.slot_id for t in outs)
    vis = [t for t in outs if t.state == "visible"]
    max_blue = max(max_blue, sum(t.alliance == "blue" for t in vis))
    max_red = max(max_red, sum(t.alliance == "red" for t in vis))
    last_frame = (main, outs)
cap.release()

print(f"distinct slot ids seen: {sorted(seen_slots)}")
print(f"max simultaneous visible: blue={max_blue} red={max_red}")
assert seen_slots <= set(range(6)), f"more than 6 slots! {seen_slots}"
assert max_blue <= 3 and max_red <= 3, "exceeded 3 per alliance"
print("OK: fixed-6 / 3-per-alliance invariants held; occlusion resolved cleanly.")

if last_frame is not None:
    main, outs = last_frame
    rad = radar.render([(t.pos, t.alliance) for t in outs if t.state != "empty"])
    out = ROOT / "data/test_footage/field_tracking_sample.jpg"
    cv2.imwrite(str(out), main)
    cv2.imwrite(str(out.with_name("field_tracking_radar.jpg")), rad)
    print(f"samples written: {out.name}, field_tracking_radar.jpg")
