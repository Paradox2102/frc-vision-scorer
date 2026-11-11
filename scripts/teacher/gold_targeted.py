#!/usr/bin/env python3
"""Targeted gold batch for the S2/S3 gaps (training events only; same blind item/key format as gold.py).

Pools (exclusive, crops already in earlier gold batches skipped):
  unlinked_robot : robot-detector crops whose robot tracklet has no linked bumper (equipment, people,
                   field elements, robots with hidden bumpers); half from short tracklets (<=10 dets)
  blurry_2026    : 2026 bumper crops from Jev-unsure tracklets, 40-100 px wide (S2 boundary)
  s3_ambiguous   : crops the S3 student scores between --lo and --hi (scored on a random subsample)

  python -m scripts.teacher.gold_targeted --events 2025cur,2026cur,2026hop --gold data/teacher/gold/b3
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.teacher.gold import _data_root, load_match, render


def previously_used(gold_root: Path) -> set:
    used = set()
    for k in list(gold_root.glob("*/key.jsonl")) + list(gold_root.glob("*.key.hidden.jsonl")):
        for l in k.open():
            used.add(json.loads(l)["path"])
    return used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="2025cur,2026cur,2026hop")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--n-unlinked", type=int, default=220)
    ap.add_argument("--n-blurry", type=int, default=150)
    ap.add_argument("--n-ambig", type=int, default=110)
    ap.add_argument("--score-sample", type=int, default=8000)
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = _data_root() / "harvest_v2"
    gold = Path(a.gold)
    used = previously_used(gold.parent)
    unl_short, unl_long, blurry, everything = [], [], [], []
    for ev in a.events.split(","):
        for mdir in sorted((root / ev).iterdir()):
            if not (mdir / "manifest.jsonl").exists():
                continue
            rows, labels = load_match(mdir)
            for r in rows:
                if r["path"] in used:
                    continue
                lab = labels.get((r["detector"], r["tracklet_id"]))
                everything.append((r, lab))
                if r["detector"] == "robot" and not r.get("linked_tracklet_id") and not r.get("linked_bumper_tracklets"):
                    (unl_short if (r.get("tracklet_len") or 0) <= 10 else unl_long).append((r, lab))
                elif (r["detector"] == "bumper" and r["year"] == 2026 and lab and lab["decision"] == "unsure"
                      and 40 <= r["px_w"] <= 100):
                    blurry.append((r, lab))
    print(f"pools: unlinked short {len(unl_short)} long {len(unl_long)}, blurry_2026 {len(blurry)}, all {len(everything)}")
    picked, seen = [], set()

    def take(pool, n, name):
        rng.shuffle(pool)
        got = 0
        for r, lab in pool:
            if got >= n:
                break
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            picked.append((name, r, lab))
            got += 1

    take(unl_short, a.n_unlinked // 2, "unlinked_robot_short")
    take(unl_long, a.n_unlinked - a.n_unlinked // 2, "unlinked_robot_long")
    take(blurry, a.n_blurry, "blurry_2026")
    # S3-ambiguous: score a random subsample with the S3 student
    from scripts.teacher.students.predict import Students
    st = Students()
    cand = [x for x in rng.sample(everything, min(a.score_sample, len(everything))) if x[0]["path"] not in seen]
    scores = []
    for i in range(0, len(cand), 256):
        chunk = cand[i:i + 256]
        imgs = [cv2.imread(str(root / r["path"])) for r, _ in chunk]
        s3 = np.asarray(st.predict(imgs)["s3"]).reshape(-1)
        scores.extend(s3.tolist())
    amb = [c for c, s in zip(cand, scores) if a.lo <= s <= a.hi]
    print(f"s3 scored {len(cand)}, ambiguous [{a.lo},{a.hi}]: {len(amb)}")
    take(amb, a.n_ambig, "s3_ambiguous")
    rng.shuffle(picked)
    (gold / "items").mkdir(parents=True, exist_ok=True)
    with (gold / "items.jsonl").open("w") as fi, (gold / "key.jsonl").open("w") as fk:
        for n, (name, r, lab) in enumerate(picked):
            gid = f"g{n:04d}"
            render(root / r["path"], gold / "items" / f"{gid}.jpg")
            fi.write(json.dumps({"id": gid, "image": f"items/{gid}.jpg", "roster": r["roster"]}) + "\n")
            fk.write(json.dumps({
                "id": gid, "stratum": [r["event"], r["detector"], name], "pool": name, "path": r["path"],
                "event": r["event"], "match_key": r["match_key"], "detector": r["detector"],
                "tracklet_id": r["tracklet_id"], "px_w": r["px_w"], "colour": r.get("alliance_colour"),
                "jev_decision": (lab or {}).get("decision"), "jev_team": (lab or {}).get("team"),
                "crop_was_jev_called": bool(lab) and any(c.get("path") == r["path"] for c in lab.get("calls", [])),
            }) + "\n")
    print(f"{len(picked)} items -> {gold}", Counter(p[0] for p in picked))


if __name__ == "__main__":
    main()
