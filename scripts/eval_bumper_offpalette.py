#!/usr/bin/env python3
"""Before/after eval of bumper YOLO weights on held-out matches, per bumper colour.

    .venv/bin/python scripts/eval_bumper_offpalette.py \\
        --arm before=runs/train/frc_bumpers_equipment_hardneg/weights/best.pt \\
        --arm after=runs/train/frc_bumpers_offpalette_v1/weights/best.pt

Every arm (weights file) runs on the SAME frames: the x-ray cache lines
(data/demo_reel/xray_cache/<replay>_*.jsonl.gz, made by scripts/xray_layers.py)
of each held-out match, every ``--stride``-th line (stride 3 of the 15 fps layers
= 5 fps). For each frame the main view (rows 0..overhead_height-1) of the source
video is run through every arm exactly as the pipeline does
(``model(crop, conf=conf, verbose=False, device=device)[0]``, model's own imgsz).

Ground truth is weak but shared by all arms:
- "robot in view" samples = the cache's SAM 3 masks of tracked slots;
- a sample is detected when a bumper box is matched to its mask (one-to-one,
  greedy by confidence; box centre in the mask bbox's lower 65 % (+10 % below));
- the bumper colour GROUP (teal / royal blue / navy / red) comes from HSV pixel
  fractions in the lower 40 % of the mask bbox (identical to the baseline
  snippet); expected YOLO colour = blue for teal/royal/navy, red for red;
- focus-team rows use the replay's slot labels (meta.json) for the team and the
  mask's alliance as the expected colour; they include every mask sample (no
  colour-group filter), like scripts/bumper_team_stats.py.

False boxes = detections matched to no mask, whose centre is in no mask bbox, and
that overlap no cached whole-robot box with IoS (intersection / det area) >= 0.5.
Detections matched to no mask but sitting on a robot (mask bbox or robot box) are
counted separately as "extra on-robot" boxes (duplicates / second boxes).

Raw per-frame detections of each arm are cached under ``--det-cache`` (keyed by
arm name, weights mtime, match, conf and device), as are the per-frame colour
groups, so a re-run with everything cached does not decode video at all.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Data, weights and videos live in the main checkout (a worktree may only hold
# LFS pointers); every relative data path below is resolved against this root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.students.paths import MAIN as REPO_DATA  # noqa: E402
PROD_WEIGHTS_REL = "runs/train/frc_bumpers_equipment_hardneg/weights/best.pt"
DEFAULT_OUT_REL = "data/eval/bumper_offpalette"

DEFAULT_MATCHES = {
    "2026cabl_qm5": "525ed57e-127f-4efe-ae4e-a37ef00656ec",
    "2026cabl_qm12": "2dcf7631-ccdf-4420-9c40-a7b16f6cfb5e",
    "2026cur_qm103": "669e605e-b989-4b03-afcb-401e496cf5f5",
    "2026cur_qm112": "14602f93-cc3b-4cfa-9f78-08128d196f89",
}

GROUPS = ("teal", "royal blue", "navy", "red")
GROUP_COLOUR = {"teal": "blue", "royal blue": "blue", "navy": "blue", "red": "red"}
# Masks no colour group claims (tiny, or washed-out venue colour — most cabl
# robots fail the strict royal/red saturation gates). Detection rate is still a
# label-free regression signal; colour is not scored (expected None).
UNCLASSIFIED = "unclassified"
REPORT_GROUPS = GROUPS + (UNCLASSIFIED,)
GROUP_COLOUR[UNCLASSIFIED] = None
CACHE_ARM = "cache"          # pseudo-arm: the x-ray cache's own "bumpers" field
DET_CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# Matching / labelling helpers (copied from scripts/bumper_team_stats.py so the
# numbers line up with the baseline; keep them in sync by hand).
# --------------------------------------------------------------------------- #

def bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def match_bumpers(masks, bumpers):
    """{slot: bumper or None}: bumper centre inside the mask bbox's lower 65 %
    (+10 % below), one-to-one, highest-confidence bumpers first."""
    boxes = {m["slot"]: bbox(m["poly"]) for m in masks if m.get("poly")}
    out = {s: None for s in boxes}
    for b in sorted(bumpers, key=lambda b: -b["conf"]):
        cx = (b["box"][0] + b["box"][2]) / 2
        cy = (b["box"][1] + b["box"][3]) / 2
        best, best_d = None, None
        for s, (x0, y0, x1, y1) in boxes.items():
            if out[s] is not None:
                continue
            h = y1 - y0
            if x0 <= cx <= x1 and y0 + 0.35 * h <= cy <= y1 + 0.10 * h:
                d = abs(cx - (x0 + x1) / 2)
                if best is None or d < best_d:
                    best, best_d = s, d
        if best is not None:
            out[best] = b
    return out


def team_lookup(meta_frames):
    fns = [f["frame_num"] for f in meta_frames]

    def at(frame, slot):
        i = bisect.bisect_left(fns, frame)
        cands = [j for j in (i - 1, i) if 0 <= j < len(fns)]
        j = min(cands, key=lambda j: abs(fns[j] - frame))
        for r in meta_frames[j]["robots"]:
            if r["slot"] == slot:
                return r.get("team_number"), r.get("alliance")
        return None, None
    return at


def colour_group(frame, poly):
    """Bumper colour group of one mask from the BGR full frame, or None (skip).

    Exactly the baseline snippet: HSV fractions in the lower 40 % of the mask
    bbox (+3 px below), argmax over (teal, royal blue, navy, red), skip tiny
    masks (< 600 px^2 bbox) and masks where no group reaches 8 %."""
    x0, y0, x1, y1 = map(int, bbox(poly))
    if (x1 - x0) * (y1 - y0) < 600:
        return None
    c = frame[int(y0 + 0.6 * (y1 - y0)):y1 + 3, x0:x1]
    if c.size == 0:          # degenerate slice (not hit by the baseline data)
        return None
    h = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    H, S, V = [h[..., k].astype(int) for k in range(3)]
    teal = ((H >= 85) & (H <= 100) & (S > 60) & (V > 60)).mean()
    royal = ((H > 100) & (H < 128) & (S > 110) & (V > 50)).mean()
    navy = ((H > 100) & (H < 128) & (S > 60) & (V > 15) & (V <= 50)).mean()
    red = (((H < 8) | (H > 170)) & (S > 110) & (V > 70)).mean()
    fr = (teal, royal, navy, red)
    k = int(np.argmax(fr))
    if fr[k] < 0.08:
        return None
    return GROUPS[k]


def ios(det, box):
    """Intersection over the DETECTION's area."""
    ix = max(0.0, min(det[2], box[2]) - max(det[0], box[0]))
    iy = max(0.0, min(det[3], box[3]) - max(det[1], box[1]))
    area = max(1e-9, (det[2] - det[0]) * (det[3] - det[1]))
    return ix * iy / area


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def resolve(root: Path, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else root / q


def parse_kv(items, what):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--{what} expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        if k in out:
            raise SystemExit(f"duplicate --{what} name {k!r}")
        out[k] = v
    return out


def xray_cache_path(root: Path, replay_id: str) -> Path:
    caches = sorted((root / "data/demo_reel/xray_cache").glob(f"{replay_id}_*.jsonl.gz"),
                    key=lambda p: -p.stat().st_size)
    if not caches:
        raise SystemExit(f"no x-ray cache for replay {replay_id} (run scripts/xray_layers.py)")
    return caches[0]


def load_xray(path: Path, stride: int, max_frames: int | None):
    """(header, [cache line]) keeping every stride-th layer line."""
    lines = []
    with gzip.open(path, "rt") as fh:
        header = json.loads(fh.readline())["header"]
        for k, line in enumerate(fh):
            if k % stride:
                continue
            d = json.loads(line)
            lines.append({"frame": int(d["frame"]), "masks": d.get("masks") or [],
                          "bumpers": d.get("bumpers") or [], "robots": d.get("robots") or []})
            if max_frames and len(lines) >= max_frames:
                break
    frames = [d["frame"] for d in lines]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise SystemExit(f"{path.name}: cache frames are not strictly increasing")
    return header, lines


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def read_gz_json(path: Path):
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_gz_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


@dataclass
class Arm:
    name: str
    weights: Path
    model: object = None
    names: dict = field(default_factory=dict)

    def cache_file(self, cache_dir: Path, match: str, conf: float, device: str) -> Path:
        mtime = int(self.weights.stat().st_mtime)
        return cache_dir / (f"dets__{safe(self.name)}__{mtime}__{safe(match)}"
                            f"__conf{conf:g}__dev{safe(device)}.json.gz")

    def detect(self, crop, conf: float, device: str) -> list:
        res = self.model(crop, conf=conf, verbose=False, device=device)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        out = []
        for k in range(len(cls)):
            name = self.names.get(int(cls[k]), "")
            colour = "blue" if "blue" in name else "red" if "red" in name else None
            # Same rounding as the x-ray cache so cached and fresh dets compare 1:1.
            out.append({"box": [round(float(v), 1) for v in xyxy[k]],
                        "conf": round(float(cf[k]), 3), "color": colour})
        return out


# --------------------------------------------------------------------------- #
# Per-match run
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    match: str
    frame: int
    slot: str
    alliance: str | None     # mask's (slot's) alliance
    team: str | None         # replay label at the nearest meta frame
    group: str | None        # colour group, None = skipped by the colour rule
    det: dict                # arm name -> matched bumper dict or None


def run_match(a, root: Path, match: str, replay_id: str, arms: list[Arm]):
    xpath = xray_cache_path(root, replay_id)
    header, lines = load_xray(xpath, a.stride, a.max_frames)
    oh = int(header["overhead_height"])
    meta = json.load(open(root / "web/replays" / replay_id / "meta.json"))
    team_at = team_lookup(meta["frames"])
    video = root / "data/videos" / f"{match}.mp4"
    print(f"\n[{match}] replay {replay_id}  cache {xpath.name}  "
          f"{len(lines)} frames (stride {a.stride} of {header['layer_fps']:g} fps layers, "
          f"main view rows 0..{oh - 1})", flush=True)

    # Load cached detections / colour groups; work out what still needs computing.
    cache_dir = Path(a.det_cache)
    dets = {}           # arm -> {frame(str) -> [det]}
    det_files = {}
    for arm in arms:
        f = arm.cache_file(cache_dir, match, a.conf, a.device)
        det_files[arm.name] = f
        c = read_gz_json(f)
        ok = c and c.get("version") == DET_CACHE_VERSION and c.get("weights") == str(arm.weights)
        dets[arm.name] = c["frames"] if ok else {}
    gfile = cache_dir / f"groups__{safe(match)}__{safe(xpath.name)}.json.gz"
    gc = read_gz_json(gfile)
    groups = gc["frames"] if gc and gc.get("version") == DET_CACHE_VERSION else {}

    need = [d for d in lines if str(d["frame"]) not in groups
            or any(str(d["frame"]) not in dets[arm.name] for arm in arms)]
    n_cached = len(lines) - len(need)
    if n_cached:
        print(f"[{match}] {n_cached}/{len(lines)} frames fully cached", flush=True)

    if need:
        if not video.is_file():
            raise SystemExit(f"source video missing: {video}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        # One seek to the first needed frame (as xray_layers.py's SourceReader
        # does), then decode forward; grab() skips frames without converting.
        pos = need[0]["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        t0 = time.perf_counter()
        new = defaultdict(int)
        try:
            for k, d in enumerate(need):
                fn = d["frame"]
                while pos < fn:
                    if not cap.grab():
                        raise SystemExit(f"{video.name} ended at frame {pos}")
                    pos += 1
                ok, img = cap.read()
                pos += 1
                if not ok:
                    raise SystemExit(f"{video.name} ended at frame {fn}")
                key = str(fn)
                if key not in groups:
                    groups[key] = {m["slot"]: colour_group(img, m["poly"])
                                   for m in d["masks"] if m.get("poly")}
                    new["groups"] += 1
                crop = np.ascontiguousarray(img[:oh])
                for arm in arms:     # every arm on this frame before moving on
                    if key not in dets[arm.name]:
                        dets[arm.name][key] = arm.detect(crop, a.conf, a.device)
                        new[arm.name] += 1
                if (k + 1) % 50 == 0 or k == len(need) - 1:
                    el = time.perf_counter() - t0
                    eta = el / (k + 1) * (len(need) - k - 1)
                    print(f"[{match}] {k + 1}/{len(need)} frames decoded (frame {fn})  "
                          f"{el:.0f} s, eta {eta:.0f} s", flush=True)
        finally:
            cap.release()
        for arm in arms:
            if new[arm.name]:
                write_gz_json(det_files[arm.name], {
                    "version": DET_CACHE_VERSION, "weights": str(arm.weights),
                    "match": match, "replay_id": replay_id, "conf": a.conf,
                    "device": a.device, "frames": dets[arm.name]})
        if new["groups"]:
            write_gz_json(gfile, {"version": DET_CACHE_VERSION, "xray_cache": xpath.name,
                                  "frames": groups})

    # Build samples + box accounting from the (now complete) per-frame data.
    samples = []
    boxes = {arm.name: {"frames": len(lines), "dets": 0, "false": 0, "extra_on_robot": 0}
             for arm in arms}
    all_arms = [arm.name for arm in arms] + [CACHE_ARM]
    for d in lines:
        key = str(d["frame"])
        masks = [m for m in d["masks"] if m.get("poly")]
        mboxes = [bbox(m["poly"]) for m in masks]
        rboxes = [r["box"] for r in d["robots"]]
        per_arm = {arm.name: dets[arm.name][key] for arm in arms}
        # Cached old-model dets were stored at det_conf; filter up if --conf is higher.
        per_arm[CACHE_ARM] = [b for b in d["bumpers"] if b["conf"] >= a.conf]
        matched = {name: match_bumpers(masks, per_arm[name]) for name in all_arms}
        for arm in arms:
            used = {id(b) for b in matched[arm.name].values() if b is not None}
            st = boxes[arm.name]
            for b in per_arm[arm.name]:
                st["dets"] += 1
                if id(b) in used:
                    continue
                cx = (b["box"][0] + b["box"][2]) / 2
                cy = (b["box"][1] + b["box"][3]) / 2
                on_robot = (any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in mboxes)
                            or any(ios(b["box"], r) >= 0.5 for r in rboxes))
                st["extra_on_robot" if on_robot else "false"] += 1
        g = groups.get(key, {})
        for m in masks:
            team, _ = team_at(d["frame"], m["slot"])
            samples.append(Sample(match, d["frame"], m["slot"], m.get("alliance"),
                                  str(team) if team else None, g.get(m["slot"]),
                                  {name: matched[name][m["slot"]] for name in all_arms}))
    return samples, boxes


# --------------------------------------------------------------------------- #
# Stats + reporting
# --------------------------------------------------------------------------- #

def summarise(samples, arm, expected):
    """expected(sample) -> 'blue' | 'red' | None (None: colour not scored)."""
    n = len(samples)
    hits = [(s, s.det[arm]) for s in samples if s.det[arm] is not None]
    confs = [b["conf"] for _, b in hits]
    scored = [(s, b) for s, b in hits if expected(s)]
    right = sum(1 for s, b in scored if b["color"] == expected(s))
    return {
        "n": n,
        "n_det": len(hits),
        "det_rate": len(hits) / n if n else None,
        "conf_med": statistics.median(confs) if confs else None,
        "conf_p25": (statistics.quantiles(confs, n=4)[0] if len(confs) >= 4
                     else (min(confs) if confs else None)),
        "colour_ok": right / len(scored) if scored else None,
    }


def group_expected(s):
    return GROUP_COLOUR[s.group or UNCLASSIFIED]


def alliance_expected(s):
    return s.alliance if s.alliance in ("blue", "red") else None


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def num(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def delta(a, b, scale=100.0, nd=1, unit=" pp"):
    if a is None or b is None:
        return "–"
    return f"{scale * (b - a):+.{nd}f}{unit}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


STAT_HEADERS = ["n", "det %", "conf med", "conf p25", "colour ok %"]


def stat_cells(st):
    return [st["n"], pct(st["det_rate"]), num(st["conf_med"]), num(st["conf_p25"]),
            pct(st["colour_ok"])]


def delta_cells(b, a):
    return ["", delta(b["det_rate"], a["det_rate"]),
            delta(b["conf_med"], a["conf_med"], 1.0, 3, ""),
            delta(b["conf_p25"], a["conf_p25"], 1.0, 3, ""),
            delta(b["colour_ok"], a["colour_ok"])]


def build_report(a, arms, matches, samples, boxes, prod_arm):
    names = [arm.name for arm in arms]
    has_delta = "before" in names and "after" in names
    res = {"config": {"root": str(a.root), "stride": a.stride, "max_frames": a.max_frames,
                      "conf": a.conf, "device": a.device, "focus_teams": a.focus,
                      "arms": {arm.name: str(arm.weights) for arm in arms},
                      "matches": matches, "production_weights": str(a.prod_weights)},
           "frames": {m: boxes[names[0]][m]["frames"] for m in matches},
           "groups": {}, "teams": {}, "boxes": {}, "sanity": None}
    md = []

    by_mg = defaultdict(list)       # (match|"pooled", group) -> samples
    for s in samples:
        g = s.group or UNCLASSIFIED
        by_mg[(s.match, g)].append(s)
        by_mg[("pooled", g)].append(s)
    for name in names:
        res["groups"][name] = {m: {g: summarise(by_mg.get((m, g), []), name, group_expected)
                                   for g in REPORT_GROUPS} for m in list(matches) + ["pooled"]}

    # Pooled tables, one per colour group (rows = arm).
    md.append(f"### Bumper detection by colour group, pooled over {len(matches)} matches")
    md.append(f"conf ≥ {a.conf:g}, stride {a.stride}; colour ok = matched box has the "
              f"group's expected colour (teal/royal/navy → blue, red → red).")
    for g in REPORT_GROUPS:
        rows = [[name] + stat_cells(res["groups"][name]["pooled"][g]) for name in names]
        if has_delta:
            rows.append(["Δ after − before"] + delta_cells(res["groups"]["before"]["pooled"][g],
                                                           res["groups"]["after"]["pooled"][g]))
        md.append(f"\n#### {g} (expected {GROUP_COLOUR[g] or 'colour not scored'})\n")
        md.append(md_table(["arm"] + STAT_HEADERS, rows))

    # Per-match tables (rows = group x arm).
    md.append("\n### Per match")
    for m in matches:
        rows = []
        for g in REPORT_GROUPS:
            for name in names:
                rows.append([g, name] + stat_cells(res["groups"][name][m][g]))
            if has_delta:
                rows.append([g, "Δ"] + delta_cells(res["groups"]["before"][m][g],
                                                   res["groups"]["after"][m][g]))
        md.append(f"\n#### {m} ({res['frames'][m]} frames)\n")
        md.append(md_table(["group", "arm"] + STAT_HEADERS, rows))

    # Focus teams (all mask samples of the team; expected colour = mask alliance).
    by_tm = defaultdict(list)
    for s in samples:
        if s.team in a.focus:
            by_tm[(s.team, s.match)].append(s)
            by_tm[(s.team, "pooled")].append(s)
    rows = []
    for team in a.focus:
        res["teams"][team] = {}
        for m in ["pooled"] + list(matches):
            smp = by_tm.get((team, m))
            if not smp:
                continue
            res["teams"][team][m] = {name: summarise(smp, name, alliance_expected)
                                     for name in names}
            for name in names:
                rows.append([team, m, name] + stat_cells(res["teams"][team][m][name]))
            if has_delta:
                rows.append([team, m, "Δ"] + delta_cells(res["teams"][team][m]["before"],
                                                         res["teams"][team][m]["after"]))
    md.append("\n### Focus teams\n")
    md.append("Team = replay slot label at the nearest meta frame; every mask sample of the "
              "team (no colour-group filter); colour ok vs the mask's alliance.\n")
    md.append(md_table(["team", "match", "arm"] + STAT_HEADERS, rows) if rows
              else "_no samples of the focus teams_")

    # False boxes per frame.
    for name in names:
        tot = {"frames": 0, "dets": 0, "false": 0, "extra_on_robot": 0}
        res["boxes"][name] = {}
        for m in matches:
            st = dict(boxes[name][m])
            for k in tot:
                tot[k] += st[k]
            res["boxes"][name][m] = st
        res["boxes"][name]["pooled"] = tot
        for st in res["boxes"][name].values():
            f = st["frames"] or 1
            st["false_per_frame"] = st["false"] / f
            st["extra_per_frame"] = st["extra_on_robot"] / f
            st["dets_per_frame"] = st["dets"] / f
    cols = list(matches) + ["pooled"]
    rows = []
    for name in names:
        r = res["boxes"][name]
        rows.append([name] + [f"{r[m]['false_per_frame']:.3f}" for m in cols]
                    + [r["pooled"]["false"], f"{r['pooled']['extra_per_frame']:.3f}"])
    if has_delta:
        b, af = res["boxes"]["before"], res["boxes"]["after"]
        rows.append(["Δ after − before"]
                    + [f"{af[m]['false_per_frame'] - b[m]['false_per_frame']:+.3f}" for m in cols]
                    + [f"{af['pooled']['false'] - b['pooled']['false']:+d}",
                       f"{af['pooled']['extra_per_frame'] - b['pooled']['extra_per_frame']:+.3f}"])
    md.append("\n### False boxes per frame\n")
    md.append("False = matched to no robot mask, centre in no mask bbox, IoS < 0.5 with every "
              "cached whole-robot box. Extra on-robot = unmatched but on a robot "
              "(duplicate / second box).\n")
    md.append(md_table(["arm"] + [f"{m} false/f" for m in matches]
                       + ["pooled false/f", "pooled false total", "pooled extra on-robot/f"],
                       rows))

    # Sanity: the production-weights arm vs the x-ray cache's own detections.
    if prod_arm:
        sanity = {}
        rows = []
        for m in list(matches) + ["pooled"]:
            smp = [s for s in samples if m == "pooled" or s.match == m]
            arm_rate = (sum(s.det[prod_arm] is not None for s in smp) / len(smp)) if smp else None
            cache_rate = (sum(s.det[CACHE_ARM] is not None for s in smp) / len(smp)) if smp else None
            agree = (sum((s.det[prod_arm] is None) == (s.det[CACHE_ARM] is None) for s in smp)
                     / len(smp)) if smp else None
            both = [(s.det[prod_arm], s.det[CACHE_ARM]) for s in smp
                    if s.det[prod_arm] is not None and s.det[CACHE_ARM] is not None]
            dconf = statistics.mean(abs(x["conf"] - y["conf"]) for x, y in both) if both else None
            same_col = sum(x["color"] == y["color"] for x, y in both) / len(both) if both else None
            sanity[m] = {"n": len(smp), "arm_det_rate": arm_rate, "cache_det_rate": cache_rate,
                         "sample_agreement": agree, "mean_abs_conf_diff": dconf,
                         "same_colour": same_col}
            rows.append([m, len(smp), pct(arm_rate), pct(cache_rate), pct(agree),
                         num(dconf, 3), pct(same_col)])
        res["sanity"] = {"arm": prod_arm, "per_match": sanity}
        md.append(f"\n### Sanity: arm `{prod_arm}` (production weights) vs cached x-ray "
                  "detections\n")
        md.append("All mask samples (no colour filter). Agreement = both matched or both "
                  "unmatched; conf diff / same colour over samples matched in both.\n")
        md.append(md_table(["match", "n", "arm det %", "cache det %", "agreement",
                            "mean |Δconf|", "same colour"], rows))
    else:
        md.append(f"\n_Sanity check skipped: no arm uses the production weights "
                  f"({a.prod_weights})._")
    return res, "\n".join(md) + "\n"


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--root", default=str(REPO_DATA),
                    help="checkout holding data/, runs/, web/replays/ (default: %(default)s)")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="weights to evaluate (repeatable); relative paths are under --root")
    ap.add_argument("--match", action="append", metavar="KEY=REPLAY_ID",
                    help="match key + replay id (repeatable; default: the 4 held-out matches)")
    ap.add_argument("--stride", type=int, default=3,
                    help="use every Nth x-ray cache line (default 3 = 5 fps of 15 fps layers)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per match (smoke)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0", help="ultralytics device (default 0; or cpu)")
    ap.add_argument("--focus-teams", default="4322,6800,498")
    ap.add_argument("--prod-weights", default=PROD_WEIGHTS_REL,
                    help="production bumper weights, for the cache sanity check")
    ap.add_argument("--out-json", default=None,
                    help=f"default <root>/{DEFAULT_OUT_REL}/results.json")
    ap.add_argument("--out-md", default=None, help=f"default <root>/{DEFAULT_OUT_REL}/results.md")
    ap.add_argument("--det-cache", default=None,
                    help=f"per-arm detection cache dir (default <root>/{DEFAULT_OUT_REL}/det_cache)")
    a = ap.parse_args()

    a.root = Path(a.root).expanduser().resolve()
    a.focus = [t.strip() for t in a.focus_teams.split(",") if t.strip()]
    a.prod_weights = resolve(a.root, a.prod_weights).resolve()
    out_dir = a.root / DEFAULT_OUT_REL
    a.out_json = Path(a.out_json) if a.out_json else out_dir / "results.json"
    a.out_md = Path(a.out_md) if a.out_md else out_dir / "results.md"
    a.det_cache = Path(a.det_cache) if a.det_cache else out_dir / "det_cache"
    if a.stride < 1:
        raise SystemExit("--stride must be >= 1")
    matches = parse_kv(a.match, "match") if a.match else dict(DEFAULT_MATCHES)

    arms = []
    for name, p in parse_kv(a.arm, "arm").items():
        if name == CACHE_ARM:
            raise SystemExit(f"arm name {CACHE_ARM!r} is reserved")
        w = resolve(a.root, p).resolve()
        if not w.is_file():
            raise SystemExit(f"arm {name}: no weights at {w}")
        if w.stat().st_size < 4096:
            raise SystemExit(f"arm {name}: {w} looks like a Git LFS pointer; use the main checkout")
        arms.append(Arm(name, w))

    from ultralytics import YOLO
    for arm in arms:       # each model loaded once, reused across all matches
        arm.model = YOLO(str(arm.weights))
        arm.names = dict(arm.model.names)
        print(f"arm {arm.name}: {arm.weights}  classes {arm.names}", flush=True)
    prod_arm = next((arm.name for arm in arms if arm.weights == a.prod_weights), None)

    samples, boxes = [], defaultdict(dict)
    t0 = time.perf_counter()
    for m, rid in matches.items():
        smp, bx = run_match(a, a.root, m, rid, arms)
        samples += smp
        for name, st in bx.items():
            boxes[name][m] = st
    print(f"\nall matches done in {time.perf_counter() - t0:.0f} s; "
          f"{len(samples)} robot samples, {sum(1 for s in samples if s.group)} with a colour group")

    res, md = build_report(a, arms, list(matches), samples, boxes, prod_arm)
    print("\n" + md)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(res, indent=1))
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.write_text(md)
    print(f"wrote {a.out_json}\nwrote {a.out_md}")


if __name__ == "__main__":
    sys.exit(main())
