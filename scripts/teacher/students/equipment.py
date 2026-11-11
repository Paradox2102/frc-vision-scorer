#!/usr/bin/env python3
"""Equipment hard negatives for S3, cut in the harvester's crop convention (design §5).

Source: the July detector hard-negative work (data/hard_negatives, curated by hand):
  boxes/<ev_qm>/<label>/frameNNNNNNNN.jpg         hub / outpost regions  (boxes/frame_<ev_qm>.json)
  boxes/equipment_<ev_qm>/<label>/frame*.jpg      driver-station tablets / field radios
                                                  (boxes/frame_equipment_<ev_qm>_f*.json)
      Regions were drawn once per match in tools/box_annotator.html and cut from frames spread
      over the match by scripts/crop_negatives_from_boxes.py; the per-label frame lists were then
      PRUNED by hand (tools/negative_pruner.html) to drop frames with a robot in front.
  2026cur/qm{8,10,11}/<cluster>/frame*.jpg        crops of the bin / field-edge detector false
                                                  fires that hijacked tracks (2026cur/equip_clusters.json)
  The YOLO sets bumper_hardneg_equipment / bumper_hardneg_2026cur hold the SAME crops as
  empty-label background images (no new images), and user_negs/ are 4 loose seed crops without a
  source frame, so neither adds anything in the harvester convention.

None of these crops is a harvester crop: they are fixed regions (or detector boxes cut without
padding). So for every kept (match, label, frame) the source video frame is re-read, the
production detectors run on it exactly like scripts/teacher/harvest.py (bumper model + robot class
of the hub_zone model, imgsz 640, detection on the even field when the video is woven), and every
detection lying mostly inside that label's region (IoA >= --ioa of the detection box, region grown
by 10%) is cut with the harvester padding (bumper 25%, robot 10%), deinterlaced like the harvester
and saved. The video is verified against the stored crop of the same frame first (mean abs diff).
For 2026cur, whose crops are detector boxes rather than fixed regions, the stored crop is located
in its frame by template matching and that box (grown by 25%) is the region.

Split by source match/video: the held-out EQUIPMENT TEST slice is 2026alhu_qm51 (b2 held-out
event), 2026bcvi_qm48 (event not trained on) and 2026cur_qm8 / qm11 (GT matches, never trained on,
design §3); everything else is training data.

Review: nothing is used until listed in review.json "keep" (load_manifest). Nov 10: 920 kept /
145 dropped (any visible robot/bumper -> drop); every crop actually used (428 train after the
per-region cap + 349 test) was re-checked on contact sheets Nov 11: no robots. The 2026cur_qm11
"OVERFLOW" crate with "1477" printed on its red base is field equipment (the known qm11 poison
read, docs/superpowers/plans/2026-07-07-vlm-quality-checks.md), not a robot.

Output (MAIN/data/teacher/equipment_negs):
  crops/<source>/<label>_<det>_f<frame>_<k>.jpg, manifest.jsonl, contact_<split>.jpg,
  features.npy + features.paths.json (DINOv3-S CLS, letterbox 224 = students/features.py)

  .venv/bin/python -m scripts.teacher.students.equipment extract
  .venv/bin/python -m scripts.teacher.students.equipment features
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.harvest import (BUMPER_CONF, MAIN, PAD_BUMPER, PAD_ROBOT, ROBOT_CONF,  # noqa: E402
                                     Detectors, contact_sheet, detect_interlaced, field_frame,
                                     pad_box, video_fps)

HN = MAIN / "data/hard_negatives"
OUT = MAIN / "data/teacher/equipment_negs"
FOOTAGE = MAIN / "data/test_footage"
VIDEOS = {  # sources whose video is not data/test_footage/<ev_qm>.mp4
    "2026hop_qm30": FOOTAGE / "2026hop_match.mp4",
    "2026cur_qm8": MAIN / "web/uploads/ff4be1d9-b8e5-49ff-ac78-4a6a0e77a697.mp4",
    "2026cur_qm10": MAIN / "web/uploads/e67173bf-170e-4865-8a19-bf088ff46105.mp4",
    "2026cur_qm11": FOOTAGE / "2026cur_qm11.mp4",
}
TEST_SOURCES = {"2026alhu_qm51", "2026bcvi_qm48", "2026cur_qm8", "2026cur_qm11"}
VERIFY_MAD = 12.0      # mean abs diff (0-255) between the stored crop and the re-read frame region


def _ioa(det, reg) -> float:
    ix = max(0.0, min(det[2], reg[2]) - max(det[0], reg[0]))
    iy = max(0.0, min(det[3], reg[3]) - max(det[1], reg[1]))
    a = max(1e-6, (det[2] - det[0]) * (det[3] - det[1]))
    return ix * iy / a


def _grow(box, frac, W, H):
    return pad_box(box, frac, W, H)


def _label_names(boxes: List[dict]) -> List[str]:
    """crop_negatives_from_boxes.py naming: first 'hub', then 'hub2', 'hub3', ..."""
    seen: Dict[str, int] = defaultdict(int)
    out = []
    for b in boxes:
        seen[b["label"]] += 1
        out.append(b["label"] if seen[b["label"]] == 1 else f"{b['label']}{seen[b['label']]}")
    return out


def box_sources() -> List[dict]:
    """[{source, kind, label, region, frames, stored: {frame: path}}] for boxes/ (fixed regions)."""
    out = []
    bdir = HN / "boxes"
    for d in sorted(p for p in bdir.iterdir() if p.is_dir()):
        tag = d.name
        ev_qm = tag.replace("equipment_", "")
        js = sorted(bdir.glob(f"frame_{tag}.json")) or sorted(bdir.glob(f"frame_{tag}_f*.json"))
        if not js:
            print(f"  skip {tag}: no box json")
            continue
        boxes = json.loads(js[0].read_text())
        for name, b in zip(_label_names(boxes), boxes):
            ld = d / name
            if not ld.is_dir():
                continue
            stored = {int(re.search(r"frame(\d+)", p.name).group(1)): p for p in sorted(ld.glob("frame*.jpg"))}
            out.append({"source": ev_qm, "kind": "equipment" if tag.startswith("equipment_") else "structure",
                        "label": b["label"], "label_dir": name,
                        "region": (b["x1"], b["y1"], b["x2"], b["y2"]), "frames": sorted(stored),
                        "stored": stored})
    return out


def cur_sources() -> List[dict]:
    """2026cur bin / field-edge clusters: per-frame regions found by template matching later."""
    out = []
    for qd in sorted((HN / "2026cur").iterdir()):
        if not qd.is_dir():
            continue
        for ld in sorted(p for p in qd.iterdir() if p.is_dir()):
            stored = {int(re.search(r"frame(\d+)", p.name).group(1)): p for p in sorted(ld.glob("frame*.jpg"))}
            out.append({"source": f"2026cur_{qd.name}", "kind": "bin_cluster", "label": ld.name,
                        "label_dir": ld.name, "region": None, "frames": sorted(stored), "stored": stored})
    return out


def video_for(source: str) -> Path:
    return VIDEOS.get(source, FOOTAGE / f"{source}.mp4")


def read_frame(cap, fi: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)   # same seek as crop_negatives_from_boxes.py
    ok, fr = cap.read()
    return fr if ok else None


def locate(frame: np.ndarray, crop: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
    """Best template-match box of ``crop`` in ``frame`` and its mean abs diff."""
    res = cv2.matchTemplate(frame, crop, cv2.TM_SQDIFF)
    _, _, mn, _ = cv2.minMaxLoc(res)
    x, y = mn
    h, w = crop.shape[:2]
    mad = float(np.abs(frame[y:y + h, x:x + w].astype(np.float32) - crop.astype(np.float32)).mean())
    return (x, y, x + w, y + h), mad


def cmd_extract(a) -> None:
    from web.bumper_register import deinterlace_crop
    srcs = box_sources() + cur_sources()
    by_src = defaultdict(list)
    for s in srcs:
        by_src[s["source"]].append(s)
    det = Detectors()
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    rows, sheets, stats = [], defaultdict(list), Counter()
    for source, items in sorted(by_src.items()):
        vid = video_for(source)
        if not vid.exists():
            print(f"{source}: no video {vid} -> skipped")
            stats["no_video_sources"] += 1
            continue
        vfps, nfr, W, H = video_fps(vid)
        interlaced, _ = detect_interlaced(vid, nfr / vfps if vfps else 0.0)
        do_deint = interlaced is not False   # harvest.py 'auto': unknown -> per-crop comb gate
        cap = cv2.VideoCapture(str(vid))
        split = "test" if source in TEST_SOURCES else "train"
        n_src, bad = 0, 0
        for it in items:
            for fi in it["frames"]:
                fr = read_frame(cap, fi)
                if fr is None:
                    continue
                stored = cv2.imread(str(it["stored"][fi]))
                if it["region"] is not None:
                    x0, y0, x1, y1 = [int(v) for v in it["region"]]
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                    reg = fr[y0:y1, x0:x1]
                    mad = (float(np.abs(reg.astype(np.float32) - stored.astype(np.float32)).mean())
                           if reg.shape == stored.shape else 999.0)
                    region = (x0, y0, x1, y1)
                    grow = 0.10
                else:
                    region, mad = locate(fr, stored)
                    grow = 0.25
                if mad > VERIFY_MAD:
                    bad += 1
                    stats["verify_fail"] += 1
                    continue
                stats["frames_ok"] += 1
                # the curated region itself as a pseudo detection (whole-object box -> robot
                # padding; a stored detector box keeps its detector's padding)
                rdet = "bumper" if it["kind"] == "bin_cluster" and "bumper" in it["label"] else "robot"
                px0, py0, px1, py1 = pad_box(region, PAD_BUMPER if rdet == "bumper" else PAD_ROBOT, W, H)
                raw = fr[py0:py1, px0:px1]
                img = deinterlace_crop(raw) if do_deint else raw
                rel = Path("regions") / source / f"{it['label_dir']}_f{fi}.jpg"
                (OUT / rel.parent).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"path": str(rel), "origin": "region", "source": source,
                             "event": source.split("_")[0], "split": split, "kind": it["kind"],
                             "label": it["label"], "label_dir": it["label_dir"], "detector": rdet,
                             "conf": None, "above_prod_conf": None, "box": [float(v) for v in region],
                             "frame": fi, "frame_wh": [W, H], "px_w": int(region[2] - region[0]),
                             "crop_wh": [int(img.shape[1]), int(img.shape[0])],
                             "interlaced_video": bool(interlaced), "verify_mad": round(mad, 2)})
                dframe = field_frame(fr) if interlaced else fr   # harvest: field_detect = bool(interlaced)
                greg = _grow(region, grow, W, H)
                for detector, dets, pad, conf_min in (
                        ("bumper", det.bumpers([dframe], conf=a.conf_floor)[0], PAD_BUMPER, BUMPER_CONF),
                        ("robot", det.robots([dframe], conf=a.conf_floor)[0], PAD_ROBOT, ROBOT_CONF)):
                    for k, (box, conf, _c) in enumerate(dets):
                        if _ioa(box, greg) < a.ioa:
                            continue
                        bw, bh = box[2] - box[0], box[3] - box[1]
                        if min(bw, bh) < 8:
                            continue
                        px0, py0, px1, py1 = pad_box(box, pad, W, H)
                        raw = fr[py0:py1, px0:px1]
                        img = deinterlace_crop(raw) if do_deint else raw
                        rel = Path("crops") / source / f"{it['label_dir']}_{detector}_f{fi}_{k}.jpg"
                        (OUT / rel.parent).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(OUT / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        row = {"path": str(rel), "origin": "detector", "source": source, "event": source.split("_")[0],
                               "split": split, "kind": it["kind"], "label": it["label"],
                               "label_dir": it["label_dir"], "detector": detector, "conf": float(conf),
                               "above_prod_conf": bool(conf >= conf_min), "box": [float(v) for v in box],
                               "frame": fi, "frame_wh": [W, H], "px_w": int(round(bw)),
                               "crop_wh": [int(img.shape[1]), int(img.shape[0])],
                               "interlaced_video": bool(interlaced), "verify_mad": round(mad, 2)}
                        rows.append(row)
                        n_src += 1
                        sheets[f"det_{split}"].append((img, f"{source[4:]} {it['label_dir'][:8]} {detector[0]}{conf:.2f}"))
        cap.release()
        print(f"{source} [{split}] interlaced={interlaced}: {n_src} crops, verify-fail frames {bad}", flush=True)
    with (OUT / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for split, tiles in sheets.items():
        for i in range(0, len(tiles), 96):
            cv2.imwrite(str(OUT / f"contact_{split}_{i // 96}.jpg"),
                        contact_sheet(tiles[i:i + 96], cols=12, tw=120, th=120,
                                      title=f"{split} {i}-{i + len(tiles[i:i + 96])}"))
    c = Counter((r["origin"], r["split"], r["kind"], r["detector"], r["above_prod_conf"]) for r in rows)
    print(dict(stats))
    for k in sorted(c):
        print("  ", k, c[k])
    print("->", OUT)


def load_manifest(out: Path = OUT, reviewed_only: bool = True) -> List[dict]:
    """Manifest rows; by default only crops a reviewer marked as equipment (review.json
    {"keep": [path, ...], "drop": [...]}: automated "inside the equipment region" extraction
    catches robots driving in front of it, so nothing is used unreviewed)."""
    p = out / "manifest.jsonl"
    rows = [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
    if not reviewed_only:
        return rows
    rv = out / "review.json"
    keep = set(json.loads(rv.read_text())["keep"]) if rv.exists() else set()
    return [r for r in rows if r["path"] in keep]


def cmd_features(a) -> None:
    from scripts.teacher.students import features as F
    rows = load_manifest()
    bb = F.Backbone()
    X = F.embed_paths(bb, OUT, [r["path"] for r in rows], F.DEFAULT_MODE, F.DEFAULT_SIZE, batch=64, workers=2)
    np.save(OUT / "features.npy", X)
    (OUT / "features.paths.json").write_text(json.dumps([r["path"] for r in rows]))
    print(f"{len(rows)} crops -> {OUT / 'features.npy'}")


def load_features(out: Path = OUT) -> Tuple[Dict[str, int], np.ndarray]:
    p = out / "features.npy"
    if not p.exists():
        return {}, np.zeros((0, 384), np.float32)
    paths = json.loads((out / "features.paths.json").read_text())
    return {f"equipment_negs/{q}": i for i, q in enumerate(paths)}, np.load(p).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--conf-floor", type=float, default=0.10,
                   help="detections kept down to this conf (production crop conf is recorded per row)")
    e.add_argument("--ioa", type=float, default=0.6)
    e.set_defaults(fn=cmd_extract)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
