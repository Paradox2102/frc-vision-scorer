#!/usr/bin/env python3
"""S2 legibility gate + S3 robot/not-robot heads on frozen DINOv3-S CLS features (design §5).

Labels
  Gold (blind Fable judges, scripts/teacher/gold.py): b1 = 2025cur + 2026cur (TRAIN/VAL),
  b2 = 2026alhu + 2025casd (held-out TEST, touched once by ``final``). An item judged twice
  is used only when both judges give the same task label.
    S2: legible    = kind bumper|robot and (legible full, or partial with a named team at
                     conf >= 0.8, gold.gold_team)
        illegible  = kind bumper|robot and legible none
        not_bumper = any other kind (person, game_piece, field, equipment, graphic, other)
        (partial without a named team = ambiguous -> not used)
    S3: robot = kind bumper|robot; not robot = anything else.
  Weak (Jev, training events only: 2025cur, 2026cur, 2026hop):
    S2: per Jev CALL (the exact crop Jev saw): top roster team p >= WEAK_P -> legible;
        "unreadable" p >= WEAK_P -> illegible; "not_robot" (+ "printed") p >= WEAK_P -> not_bumper.
    S3 positives: every crop of a Jev ``team`` bumper tracklet + robot crops with a
        labels_robot.jsonl team. Negatives: crops of Jev ``not_bumper`` tracklets (12-21%
        are robots per gold) -> filtered by the gold-only S3 model (dropped when it says
        p_robot >= NEG_FILTER_P) and down-weighted (NEG_WEIGHT).

Validation = 5-fold CV on b1 grouped by MATCH (weak rows from the held-out fold's matches are
also held out), never b2. Thresholds and model choices come from the out-of-fold predictions.

Commands
  modes  compare preprocessing modes (letterbox/squash/center) on b1 CV, gold only
  cv     all S2/S3 variants on b1 CV -> data/teacher/models/students_cv.json
  final  train the chosen variants on all of b1 (+weak), save checkpoints, evaluate b2 ONCE

  cv-v2     v2 sweep: gold b1 + b3 (targeted hard cases), + curated equipment negatives
            (students/equipment.py, CV grouped by source video) -> students_cv_v2.json
  final-v2  train the chosen v2 variants, save *_v2.pt, evaluate ONCE on b2 + the held-out
            equipment slice next to the v1 checkpoints -> students_b2_report_v2.json

  .venv/bin/python -m scripts.teacher.students.heads cv
  .venv/bin/python -m scripts.teacher.students.heads final --s2 <variant> --s3 <variant>
  .venv/bin/python -m scripts.teacher.students.heads cv-v2 --repeats 3
  .venv/bin/python -m scripts.teacher.students.heads final-v2 --s2 <variant> --s3 <variant> --s3-rule min

v2 result (Nov 10/11): the default checkpoints stay v1 (s2_legibility.pt, s3_robot.pt);
s2_legibility_v2.pt / s3_robot_v2.pt are saved alongside. S2 v2 drops more illegible on b2
(71->77%, 2026 60->71%) but keeps 2 fewer legible crops (98.9->97.8%); S3 v2's equipment
gain is on fixed-region crops, not on detector false fires (OVERFLOW crate 0/150 for both),
and it rejects ~5 pts less gold junk than v1 under the same threshold rule (CV and b2).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.gold import _answers, gold_team  # noqa: E402
from scripts.teacher.harvest import MAIN, PAD_BUMPER, PAD_ROBOT, pad_box  # noqa: E402
from scripts.teacher.students import features as F  # noqa: E402
from web.students.paths import MODELS  # noqa: E402  (data/teacher/models in the main checkout)

DATA = MAIN / "data/teacher"
ROOT = DATA / "harvest_v2"
GOLD = DATA / "gold"
TRAIN_EVENTS = ("2025cur", "2026cur", "2026hop")
TEST_EVENTS = ("2026alhu", "2025casd")
EXCLUDE_MATCHES = {"2026cur_qm8", "2026cur_qm11"}  # GT matches: never trained on (design §3)
S2_CLASSES = ("legible", "illegible", "not_bumper")
ROBOT_KINDS = {"bumper", "robot"}
WEAK_P = 0.9
NEG_FILTER_P = 0.5
NEG_WEIGHT = 0.3
LEGIBILITY_FLOOR = 0.005  # web/app.py LEGIBILITY_FLOOR (production hand gate)
N_FOLDS = 5
EQUIP_PREFIX = "equipment_negs/"
EQUIP_CAP = 12


# ----------------------------------------------------------------------------- labels

def s2_label(ans: dict) -> Optional[str]:
    kind = ans.get("kind")
    if kind not in ROBOT_KINDS:
        return "not_bumper"
    leg = ans.get("legible")
    if leg == "full" or (leg == "partial" and gold_team(ans) is not None):
        return "legible"
    if leg == "none":
        return "illegible"
    return None


def s3_label(ans: dict) -> int:
    return int(ans.get("kind") in ROBOT_KINDS)


def _agreed(values: List) -> Optional[object]:
    vs = set(values)
    return vs.pop() if len(vs) == 1 else None


def gold_items(batch: str) -> List[dict]:
    g = GOLD / batch
    key = [json.loads(l) for l in (g / "key.jsonl").open() if l.strip()]
    ans = _answers(g)
    out = []
    for k in key:
        js = list(ans.get(k["id"], {}).values())
        if not js:
            continue
        s2 = _agreed([s2_label(a) for a in js])
        s3 = _agreed([s3_label(a) for a in js])
        out.append({"id": k["id"], "path": k["path"], "event": k["event"], "match_key": k["match_key"],
                    "year": int(k["event"][:4]), "detector": k["detector"], "px_w": k["px_w"],
                    "s2": s2, "s3": s3, "n_judges": len(js),
                    "s2_disagree": len(js) > 1 and s2 is None and any(s2_label(a) for a in js),
                    "s3_disagree": len(js) > 1 and s3 is None,
                    "kind": js[0].get("kind"), "pool": k.get("pool"), "batch": batch})
    return out


def training_gold() -> List[dict]:
    """b1 (random) + b3 (targeted hard cases) on the training events, GT matches excluded."""
    return [it for b in ("b1", "b3") for it in gold_items(b) if it["match_key"] not in EXCLUDE_MATCHES]


def equipment_rows(split: Optional[str] = None, cap: Optional[int] = None) -> List[dict]:
    """Reviewed equipment crops (students/equipment.py) as S3 negatives. ``cap``: at most this
    many (evenly spaced over the match) per (source, region label, origin), so the ~170
    near-identical 2026cur field-edge slivers do not swamp the other venues
    (default EQUIP_CAP for the train split, all crops for test)."""
    from scripts.teacher.students.equipment import load_manifest
    cap = cap if cap is not None else (EQUIP_CAP if split == "train" else 0)
    groups = defaultdict(list)
    for r in load_manifest():
        if split and r["split"] != split:
            continue
        groups[(r["source"], r["label_dir"], r.get("origin"))].append(r)
    sel = []
    for g in groups.values():
        g = sorted(g, key=lambda r: (r["frame"], r["path"]))
        if cap and len(g) > cap:
            g = [g[int(i * len(g) / cap)] for i in range(cap)]
        sel.extend(g)
    out = []
    for r in sorted(sel, key=lambda r: r["path"]):
        out.append({"path": EQUIP_PREFIX + r["path"], "event": r["event"], "match_key": r["source"],
                    "source": r["source"], "task": "s3", "y": 0, "src": "equipment",
                    "detector": r["detector"], "year": int(r["event"][:4]), "crop_wh": r["crop_wh"],
                    "label": r["label"], "equip_kind": r["kind"], "split": r["split"],
                    "above_prod_conf": r["above_prod_conf"], "origin": r.get("origin"),
                    "kind": "equipment"})
    return out


_MANIFEST_CACHE: Dict[str, Dict[str, dict]] = {}
_FRAME_WH: Dict[str, Tuple[int, int]] = {}


def match_manifest(event: str, mk: str) -> Dict[str, dict]:
    if mk not in _MANIFEST_CACHE:
        mdir = ROOT / event / mk
        rows = (json.loads(l) for l in (mdir / "manifest.jsonl").open() if l.strip())
        _MANIFEST_CACHE[mk] = {r["path"]: r for r in rows}
        s = json.loads((mdir / "summary.json").read_text())
        _FRAME_WH[mk] = (int(s.get("width", 1920)), int(s.get("height", 1080)))
    return _MANIFEST_CACHE[mk]


def crop_wh(row: dict) -> Tuple[int, int]:
    """Size of the harvested (padded) crop, from the manifest box (no image read)."""
    W, H = _FRAME_WH[row["match_key"]]
    pad = PAD_BUMPER if row["detector"] == "bumper" else PAD_ROBOT
    x0, y0, x1, y1 = pad_box(row["box"], pad, W, H)
    return x1 - x0, y1 - y0


def size_feats_from_wh(wh: np.ndarray) -> np.ndarray:
    wh = np.maximum(np.asarray(wh, np.float32).reshape(-1, 2), 1.0)
    w, h = wh[:, 0], wh[:, 1]
    return np.stack([np.log(w / 100.0), np.log(h / 100.0), np.log(w / h)], 1).astype(np.float32)


def weak_rows(events: Sequence[str]) -> List[dict]:
    """Weak S2 (per Jev call) and S3 (tracklet / per-crop team) rows for ``events``."""
    rows = []
    for ev in events:
        for mdir in sorted((ROOT / ev).iterdir()):
            mk = mdir.name
            if mk in EXCLUDE_MATCHES or not (mdir / "labels_jev.jsonl").exists():
                continue
            man = match_manifest(ev, mk)
            by_tr = defaultdict(list)
            for r in man.values():
                by_tr[(r["detector"], r["tracklet_id"])].append(r["path"])
            for l in (mdir / "labels_jev.jsonl").open():
                lab = json.loads(l)
                if lab["detector"] != "bumper":
                    continue
                # S2: the exact crops Jev called
                for c in lab.get("calls", []):
                    p = c.get("probs") or {}
                    if c.get("error") or c["path"] not in man:
                        continue
                    teams = {k: v for k, v in p.items()
                             if k not in ("other", "unreadable", "not_robot", "printed")}
                    s2 = None
                    if teams and max(teams.values()) >= WEAK_P:
                        s2 = "legible"
                    elif p.get("unreadable", 0) >= WEAK_P:
                        s2 = "illegible"
                    elif p.get("not_robot", 0) + p.get("printed", 0) >= WEAK_P:
                        s2 = "not_bumper"
                    if s2:
                        rows.append({"path": c["path"], "match_key": mk, "event": ev, "task": "s2",
                                     "y": S2_CLASSES.index(s2), "src": f"jev_call_{s2}"})
                    if s2 == "not_bumper":  # S3 call-level negative (cleaner than tracklet-level)
                        rows.append({"path": c["path"], "match_key": mk, "event": ev, "task": "s3",
                                     "y": 0, "src": "jev_call_not_robot"})
                # S3: tracklet-level
                dec = lab["decision"]
                if dec in ("team", "not_bumper"):
                    for pth in by_tr[("bumper", lab["tracklet_id"])]:
                        rows.append({"path": pth, "match_key": mk, "event": ev, "task": "s3",
                                     "y": int(dec == "team"), "src": f"jev_tracklet_{dec}"})
            rl = mdir / "labels_robot.jsonl"
            if rl.exists():
                for l in rl.open():
                    r = json.loads(l)
                    if r.get("team") and r["path"] in man:
                        rows.append({"path": r["path"], "match_key": mk, "event": ev, "task": "s3",
                                     "y": 1, "src": "robot_labels_team"})
    for r in rows:
        r["detector"] = match_manifest(r["event"], r["match_key"])[r["path"]]["detector"]
        r["year"] = int(r["event"][:4])
    return rows


# ----------------------------------------------------------------------------- hand gate

def inner_crop(img: np.ndarray, row: dict) -> np.ndarray:
    """The detector box inside a harvested (padded) crop."""
    W, H = _FRAME_WH[row["match_key"]]
    pad = PAD_BUMPER if row["detector"] == "bumper" else PAD_ROBOT
    x0, y0, _, _ = pad_box(row["box"], pad, W, H)
    bx = [int(round(row["box"][0] - x0)), int(round(row["box"][1] - y0)),
          int(round(row["box"][2] - x0)), int(round(row["box"][3] - y0))]
    h, w = img.shape[:2]
    bx = [max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])]
    return img[bx[1]:bx[3], bx[0]:bx[2]]


def hand_gate_score(row: dict) -> float:
    """Production legibility_score on what production would score: the bumper box, or the
    bumper band (web.field.bumper.bumper_crop) of a whole-robot box."""
    from web.field.bumper import bumper_crop
    from web.legibility import legibility_score
    img = cv2.imread(str(ROOT / row["path"]))
    crop = inner_crop(img, row)
    if row["detector"] == "robot":
        band = bumper_crop(crop, (0, 0, crop.shape[1], crop.shape[0]))
        crop = band if band is not None else crop
    return float(legibility_score(crop))


# ----------------------------------------------------------------------------- models

@dataclass
class Head:
    kind: str = "lr"           # lr | mlp
    n_out: int = 3
    wd: float = 1e-3
    hidden: int = 256
    steps: int = 600
    lr: float = 0.0
    seed: int = 0
    state: dict = field(default_factory=dict)

    def _net(self, d_in: int):
        import torch.nn as nn
        if self.kind == "lr":
            return nn.Linear(d_in, self.n_out)
        return nn.Sequential(nn.Linear(d_in, self.hidden), nn.GELU(), nn.Dropout(0.2),
                             nn.Linear(self.hidden, self.n_out))

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "Head":
        import torch
        torch.manual_seed(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)
        yt = torch.from_numpy(y.astype(np.int64)).to(dev)
        w = np.ones(len(y), np.float32) if w is None else w.astype(np.float32)
        # class balance: each class gets the same total weight
        cw = np.zeros(self.n_out, np.float32)
        for c in range(self.n_out):
            s = w[y == c].sum()
            cw[c] = (w.sum() / self.n_out / s) if s > 0 else 0.0
        wt = torch.from_numpy(w * cw[y]).to(dev)
        net = self._net(X.shape[1]).to(dev)
        lr = self.lr or (1e-2 if self.kind == "lr" else 1e-3)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        bs = min(len(y), 4096)
        net.train()
        for _ in range(self.steps):
            idx = torch.randperm(len(y), generator=g)[:bs].to(dev) if bs < len(y) else slice(None)
            logits = net(Xt[idx])
            loss = (torch.nn.functional.cross_entropy(logits, yt[idx], reduction="none") * wt[idx]).sum() \
                / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        self.state = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                      "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                      "d_in": X.shape[1]}
        self._cached = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if getattr(self, "_cached", None) is None:
            net = self._net(self.state["d_in"])
            net.load_state_dict(self.state["net"])
            net.eval()
            self._cached = net
        Xn = (X - self.state["mu"]) / self.state["sd"]
        with torch.no_grad():
            return torch.softmax(self._cached(torch.from_numpy(Xn.astype(np.float32))), 1).numpy()

    def to_ckpt(self) -> dict:
        import torch
        return {"kind": self.kind, "n_out": self.n_out, "hidden": self.hidden, "wd": self.wd,
                "mu": torch.from_numpy(self.state["mu"]), "sd": torch.from_numpy(self.state["sd"]),
                "d_in": self.state["d_in"], "net": self.state["net"]}

    @classmethod
    def from_ckpt(cls, ck: dict) -> "Head":
        h = cls(kind=ck["kind"], n_out=ck["n_out"], hidden=ck["hidden"], wd=ck["wd"])
        h.state = {"mu": np.asarray(ck["mu"], np.float32), "sd": np.asarray(ck["sd"], np.float32),
                   "net": ck["net"],
                   "d_in": ck["d_in"]}
        h._cached = None
        return h


# ----------------------------------------------------------------------------- metrics

def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thr_for_recall(score: np.ndarray, recall: float) -> float:
    """Largest threshold t with mean(score >= t) >= recall."""
    s = np.sort(score)
    k = int(math.floor((1 - recall) * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


def s2_gate_score(p: np.ndarray) -> np.ndarray:
    """Gate score from S2 probs: P(legible)."""
    return p[:, 0]


def s2_report(score: np.ndarray, y: np.ndarray, t: float) -> dict:
    keep = score >= t
    out = {"t": float(t)}
    for c, name in enumerate(S2_CLASSES):
        m = y == c
        out[f"n_{name}"] = int(m.sum())
        out[f"keep_{name}"] = float(keep[m].mean()) if m.any() else float("nan")
    out["drop_illegible"] = 1 - out["keep_illegible"]
    out["drop_not_bumper"] = 1 - out["keep_not_bumper"]
    return out


def confusion(y: np.ndarray, pred: np.ndarray, n: int = 3) -> List[List[int]]:
    m = np.zeros((n, n), int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m.tolist()


def s3_report(score: np.ndarray, y: np.ndarray, t: float) -> dict:
    keep = score >= t
    return {"t": float(t), "n_robot": int((y == 1).sum()), "n_junk": int((y == 0).sum()),
            "robot_recall": float(keep[y == 1].mean()) if (y == 1).any() else float("nan"),
            "robots_dropped": int((~keep[y == 1]).sum()),
            "junk_rejection": float((~keep[y == 0]).mean()) if (y == 0).any() else float("nan"),
            "auc": auc(score, y)}


# ----------------------------------------------------------------------------- data assembly

class FeatureStore:
    def __init__(self, out: Path = F.DEFAULT_OUT, events: Optional[Sequence[str]] = None,
                 equipment: bool = True):
        self.index, self.X = F.load_cache(out, events)
        if equipment:
            from scripts.teacher.students.equipment import load_features
            ei, EX = load_features()
            if len(EX):
                n = len(self.X)
                self.index.update({p: n + i for p, i in ei.items()})
                self.X = np.concatenate([self.X, EX])

    def get(self, paths: Sequence[str]) -> np.ndarray:
        missing = [p for p in paths if p not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} crops not in feature cache, e.g. {missing[0]}")
        return self.X[[self.index[p] for p in paths]]


def design_matrix(store: FeatureStore, rows: List[dict], use_size: bool) -> np.ndarray:
    X = store.get([r["path"] for r in rows])
    if not use_size:
        return X
    wh = []
    for r in rows:
        if "crop_wh" in r:
            wh.append(tuple(r["crop_wh"]))
            continue
        mr = match_manifest(r["event"], r["match_key"])[r["path"]]
        wh.append(crop_wh(mr))
    return np.concatenate([X, size_feats_from_wh(np.array(wh))], 1)


def folds_by_match(items: List[dict], k: int = N_FOLDS, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    fold = {}
    for ev in sorted({it["event"] for it in items}):
        ms = sorted({it["match_key"] for it in items if it["event"] == ev})
        rng.shuffle(ms)
        for i, m in enumerate(ms):
            fold[m] = i % k
    return fold


@dataclass
class Variant:
    name: str
    task: str                  # s2 | s3
    model: str = "lr"
    wd: float = 1e-3
    use_size: bool = False
    weak: bool = False
    weak_weight: float = 0.3   # per-row weight of weak rows vs 1.0 for gold
    neg_filter: bool = True    # s3: drop weak negatives the gold-only model calls robot
    neg_weight: float = NEG_WEIGHT
    neg_source: str = "tracklet"  # s3 weak negatives: tracklet (Jev not_bumper tracklets) | call
    gold: Tuple[str, ...] = ("b1",)  # gold batches trained on (b3 = targeted hard cases)
    equip: bool = False        # s3: + curated equipment false fires as negatives
    equip_weight: float = 1.0
    equip_s2: bool = False     # s2: + equipment crops as not_bumper


def _gold_xy(store, items, task, use_size):
    lab = "s2" if task == "s2" else "s3"
    use = [it for it in items if it[lab] is not None]
    X = design_matrix(store, use, use_size)
    y = np.array([S2_CLASSES.index(it["s2"]) if task == "s2" else it["s3"] for it in use])
    return use, X, y


def train_variant(v: Variant, store: FeatureStore, gold_train: List[dict], weak_train: List[dict],
                  seed: int = 0, equip_train: Sequence[dict] = ()) -> Head:
    gold_train = [it for it in gold_train if it.get("batch", "b1") in v.gold]
    use, Xg, yg = _gold_xy(store, gold_train, v.task, v.use_size)
    n_out = 3 if v.task == "s2" else 2
    Xs, ys, ws = [Xg], [yg], [np.ones(len(yg), np.float32)]
    if v.weak:
        neg_src = "jev_call_not_robot" if v.neg_source == "call" else "jev_tracklet_not_bumper"
        wr = [r for r in weak_train if r["task"] == v.task
              and (v.task == "s2" or r["y"] == 1 or r["src"] == neg_src)]
        if wr:
            Xw = design_matrix(store, wr, v.use_size)
            yw = np.array([r["y"] for r in wr])
            ww = np.full(len(yw), v.weak_weight, np.float32)
            if v.task == "s3":
                neg = yw == 0
                ww[neg] = v.neg_weight * v.weak_weight
                if v.neg_filter and neg.any():
                    g0 = Head(kind=v.model, n_out=2, wd=v.wd, seed=seed).fit(Xg, yg)
                    pr = g0.predict_proba(Xw[neg])[:, 1]
                    keep = np.ones(len(yw), bool)
                    keep[np.where(neg)[0][pr >= NEG_FILTER_P]] = False
                    Xw, yw, ww = Xw[keep], yw[keep], ww[keep]
            Xs.append(Xw)
            ys.append(yw)
            ws.append(ww)
    if equip_train and ((v.task == "s3" and v.equip) or (v.task == "s2" and v.equip_s2)):
        Xs.append(design_matrix(store, list(equip_train), v.use_size))
        ys.append(np.full(len(equip_train), 0 if v.task == "s3" else S2_CLASSES.index("not_bumper")))
        ws.append(np.full(len(equip_train), v.equip_weight, np.float32))
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    return Head(kind=v.model, n_out=n_out, wd=v.wd, seed=seed).fit(X, y, w)


def folds_by_source(rows: Sequence[dict], k: int = N_FOLDS, seed: int = 0) -> Dict[str, int]:
    """Equipment rows: one fold per source match/video (round robin after a seeded shuffle)."""
    srcs = sorted({r["source"] for r in rows})
    random.Random(seed).shuffle(srcs)
    return {s: i % k for i, s in enumerate(srcs)}


def run_cv(v: Variant, store: FeatureStore, items: List[dict], weak: List[dict],
           fold: Dict[str, int], equip: Sequence[dict] = (), efold: Optional[Dict[str, int]] = None,
           seed: int = 0) -> dict:
    """Out-of-fold probs for gold items (and S3 weak positives in held-out matches, and
    equipment negatives of held-out sources)."""
    lab = "s2" if v.task == "s2" else "s3"
    use = [it for it in items if it[lab] is not None]
    P = np.zeros((len(use), 3 if v.task == "s2" else 2), np.float32)
    weak_val = [r for r in weak if r["task"] == "s3" and r["y"] == 1 and r["match_key"] in fold]
    Pw = np.zeros(len(weak_val), np.float32)
    equip = list(equip)
    Pe = np.zeros((len(equip), 3 if v.task == "s2" else 2), np.float32)
    for k in range(N_FOLDS):
        tr = [it for it in items if fold[it["match_key"]] != k]
        wtr = [r for r in weak if fold.get(r["match_key"], -1) != k]
        etr = [r for r in equip if efold[r["source"]] != k] if equip else []
        head = train_variant(v, store, tr, wtr, seed=seed, equip_train=etr)
        te = [i for i, it in enumerate(use) if fold[it["match_key"]] == k]
        P[te] = head.predict_proba(design_matrix(store, [use[i] for i in te], v.use_size))
        ei = [i for i, r in enumerate(equip) if efold[r["source"]] == k]
        if ei:
            Pe[ei] = head.predict_proba(design_matrix(store, [equip[i] for i in ei], v.use_size))
        if v.task == "s3":
            wi = [i for i, r in enumerate(weak_val) if fold[r["match_key"]] == k]
            if wi:
                Pw[wi] = head.predict_proba(design_matrix(store, [weak_val[i] for i in wi],
                                                          v.use_size))[:, 1]
    return {"use": use, "P": P, "weak_val": weak_val, "Pw": Pw, "equip": equip, "Pe": Pe}


# ----------------------------------------------------------------------------- reporting

def per_slice(rows: List[dict], fn, keys=("year", "detector")) -> dict:
    out = {"all": fn(np.arange(len(rows)))}
    for key in keys:
        for val in sorted({r[key] for r in rows}):
            idx = np.array([i for i, r in enumerate(rows) if r[key] == val])
            out[f"{key}={val}"] = fn(idx)
    return out


def s2_eval(use, P, y, t, hand, hand_t_95) -> dict:
    gate = s2_gate_score(P)

    def fn(idx):
        r = {"student": s2_report(gate[idx], y[idx], t),
             "confusion(rows=gold legible/illegible/not_bumper, cols=pred)":
                 confusion(y[idx], P[idx].argmax(1)),
             "hand_gate_floor": s2_report(hand[idx], y[idx], LEGIBILITY_FLOOR),
             "hand_gate_keep95": s2_report(hand[idx], y[idx], hand_t_95)}
        li = idx[y[idx] != 2]
        r["auc_legible_vs_illegible"] = {"student": auc(gate[li], (y[li] == 0).astype(int)),
                                         "hand_gate": auc(hand[li], (y[li] == 0).astype(int))}
        return r
    return per_slice(use, fn)


def s3_eval(use, P, y, t) -> dict:
    s = P[:, 1]
    return per_slice(use, lambda idx: s3_report(s[idx], y[idx], t))


def hand_scores(items: List[dict]) -> np.ndarray:
    out = []
    for it in items:
        mr = match_manifest(it["event"], it["match_key"])[it["path"]]
        out.append(hand_gate_score(mr))
    return np.array(out, np.float32)


def fmt_s2(name, r):
    return (f"{name:34s} keep legible {r['keep_legible']:.3f} | drop illegible {r['drop_illegible']:.3f} "
            f"| drop not_bumper {r['drop_not_bumper']:.3f}  (n {r['n_legible']}/{r['n_illegible']}/"
            f"{r['n_not_bumper']}, t={r['t']:.4g})")


def fmt_s3(name, r):
    return (f"{name:34s} recall {r['robot_recall']:.4f} ({r['robots_dropped']} of {r['n_robot']} dropped) "
            f"| junk rejection {r['junk_rejection']:.3f} of {r['n_junk']} | AUC {r['auc']:.3f} | t={r['t']:.4g}")


# ----------------------------------------------------------------------------- commands

S2_VARIANTS = [
    Variant("s2_lr", "s2", "lr", 1e-3),
    Variant("s2_lr_wd1e-1", "s2", "lr", 1e-1),
    Variant("s2_lr_size", "s2", "lr", 1e-3, use_size=True),
    Variant("s2_mlp", "s2", "mlp", 1e-2),
    Variant("s2_mlp_size", "s2", "mlp", 1e-2, use_size=True),
    Variant("s2_lr_weak", "s2", "lr", 1e-3, weak=True, weak_weight=0.3),
    Variant("s2_mlp_weak", "s2", "mlp", 1e-2, weak=True, weak_weight=0.3),
    Variant("s2_mlp_size_weak", "s2", "mlp", 1e-2, use_size=True, weak=True, weak_weight=0.3),
    Variant("s2_mlp_weak_w1", "s2", "mlp", 1e-2, weak=True, weak_weight=1.0),
    Variant("s2_mlp_weak_w01", "s2", "mlp", 1e-2, weak=True, weak_weight=0.1),
]
S3_VARIANTS = [
    Variant("s3_lr", "s3", "lr", 1e-3),
    Variant("s3_lr_wd1e-1", "s3", "lr", 1e-1),
    Variant("s3_lr_size", "s3", "lr", 1e-3, use_size=True),
    Variant("s3_mlp", "s3", "mlp", 1e-2),
    Variant("s3_lr_weak", "s3", "lr", 1e-3, weak=True, weak_weight=0.3),
    Variant("s3_mlp_weak", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3),
    Variant("s3_mlp_weak_nofilter", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_filter=False),
    Variant("s3_mlp_size_weak", "s3", "mlp", 1e-2, use_size=True, weak=True, weak_weight=0.3),
    Variant("s3_lr_weak_callneg", "s3", "lr", 1e-3, weak=True, weak_weight=0.3, neg_source="call"),
    Variant("s3_mlp_weak_callneg", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call"),
    Variant("s3_mlp_weak_callneg_w1", "s3", "mlp", 1e-2, weak=True, weak_weight=1.0, neg_source="call"),
]
VARIANTS = {v.name: v for v in S2_VARIANTS + S3_VARIANTS}
S3_RECALL = 0.995
S2_KEEP = 0.95


def choose_s3_threshold(cv: dict) -> Tuple[float, dict]:
    use, P = cv["use"], cv["P"]
    y = np.array([it["s3"] for it in use])
    t_gold = thr_for_recall(P[y == 1, 1], S3_RECALL)
    t_weak = thr_for_recall(cv["Pw"], S3_RECALL) if len(cv["Pw"]) else t_gold
    return min(t_gold, t_weak), {"t_gold_oof": t_gold, "t_weak_oof": t_weak,
                                 "n_weak_val_pos": len(cv["Pw"])}


def cmd_cv(a) -> None:
    store = FeatureStore()
    items = [it for it in gold_items("b1") if it["match_key"] not in EXCLUDE_MATCHES]
    fold = folds_by_match(items)
    weak = weak_rows(TRAIN_EVENTS)
    print(f"b1 items {len(items)}: s2 {Counter(it['s2'] for it in items)}, "
          f"s3 {Counter(it['s3'] for it in items)}, s2 judge-disagree dropped "
          f"{sum(it['s2_disagree'] for it in items)}, s3 judge-disagree dropped "
          f"{sum(it['s3_disagree'] for it in items)}")
    print("weak rows:", dict(Counter((r["task"], r["src"]) for r in weak)))
    weak_label_precision(items, weak)
    s2_items = [it for it in items if it["s2"] is not None]
    hand = hand_scores(s2_items)
    y2 = np.array([S2_CLASSES.index(it["s2"]) for it in s2_items])
    results = {"variants": {}}
    names = a.variants.split(",") if a.variants else list(VARIANTS)
    for name in names:
        v = VARIANTS[name]
        cv = run_cv(v, store, items, weak, fold)
        if v.task == "s2":
            gate = s2_gate_score(cv["P"])
            t = thr_for_recall(gate[y2 == 0], S2_KEEP)
            leg_vs_ill = (y2 != 2)
            r = {"oof_at_keep95": s2_report(gate, y2, t),
                 "auc_legible_vs_illegible": auc(gate[leg_vs_ill], (y2[leg_vs_ill] == 0).astype(int)),
                 "acc": float((cv["P"].argmax(1) == y2).mean()), "t": t}
            print(fmt_s2(name, r["oof_at_keep95"]), f"AUC(leg|ill) {r['auc_legible_vs_illegible']:.3f}",
                  f"acc {r['acc']:.3f}")
        else:
            y3 = np.array([it["s3"] for it in cv["use"]])
            t, info = choose_s3_threshold(cv)
            r = {"oof": s3_report(cv["P"][:, 1], y3, t), **info,
                 "oof_weak_pos_recall": float((cv["Pw"] >= t).mean()) if len(cv["Pw"]) else None,
                 "oof_robot_detector": s3_report(cv["P"][[i for i, it in enumerate(cv["use"])
                                                         if it["detector"] == "robot"], 1],
                                                 y3[[i for i, it in enumerate(cv["use"])
                                                     if it["detector"] == "robot"]], t)}
            print(fmt_s3(name, r["oof"]), f"| weak-pos recall {r['oof_weak_pos_recall']:.4f} "
                  f"(t_gold {info['t_gold_oof']:.3g}, t_weak {info['t_weak_oof']:.3g})")
        results["variants"][name] = r
    t_h = thr_for_recall(hand[y2 == 0], S2_KEEP)
    results["hand_gate"] = {"floor": s2_report(hand, y2, LEGIBILITY_FLOOR),
                            "keep95": s2_report(hand, y2, t_h),
                            "auc_legible_vs_illegible": auc(hand[y2 != 2], (y2[y2 != 2] == 0).astype(int))}
    print(fmt_s2("hand gate @ floor 0.005", results["hand_gate"]["floor"]))
    print(fmt_s2("hand gate @ keep95 (b1-fit)", results["hand_gate"]["keep95"]),
          f"AUC(leg|ill) {results['hand_gate']['auc_legible_vs_illegible']:.3f}")
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / (a.out or "students_cv.json")
    prev = json.loads(out.read_text()) if out.exists() and a.variants else {"variants": {}}
    prev["variants"].update(results["variants"])
    prev["hand_gate"] = results["hand_gate"]
    out.write_text(json.dumps(prev, indent=1, default=float))
    print("->", out)


def weak_label_precision(items: List[dict], weak: List[dict]) -> None:
    """Weak labels vs gold on the same crop (b1)."""
    by_path = {it["path"]: it for it in items}
    c = Counter()
    for r in weak:
        it = by_path.get(r["path"])
        if not it:
            continue
        g = it["s2"] if r["task"] == "s2" else it["s3"]
        if g is None:
            c[(r["src"], "gold_ambiguous")] += 1
            continue
        gy = S2_CLASSES.index(g) if r["task"] == "s2" else g
        c[(r["src"], "agree" if gy == r["y"] else f"gold={g}")] += 1
    print("weak vs gold (same crop, b1):")
    for k in sorted(c):
        print("  ", k, c[k])


def cmd_modes(a) -> None:
    """Preprocessing choice: b1 grouped CV, gold only, LR heads, features computed on the fly."""
    items = [it for it in gold_items("b1") if it["match_key"] not in EXCLUDE_MATCHES]
    fold = folds_by_match(items)
    bb = F.Backbone()
    imgs = [cv2.imread(str(ROOT / it["path"])) for it in items]
    for mode in ("letterbox", "squash", "center"):
        X = bb.embed_crops(imgs, mode=mode)
        store = FeatureStore.__new__(FeatureStore)
        store.index = {it["path"]: i for i, it in enumerate(items)}
        store.X = X
        line = [mode]
        for v in (VARIANTS["s2_lr"], VARIANTS["s3_lr"]):
            cv = run_cv(v, store, items, [], fold)
            if v.task == "s2":
                y2 = np.array([S2_CLASSES.index(it["s2"]) for it in cv["use"]])
                gate = s2_gate_score(cv["P"])
                t = thr_for_recall(gate[y2 == 0], S2_KEEP)
                r = s2_report(gate, y2, t)
                line.append(f"S2 acc {(cv['P'].argmax(1) == y2).mean():.3f} drop-ill@keep95 "
                            f"{r['drop_illegible']:.3f} AUC(leg|ill) "
                            f"{auc(gate[y2 != 2], (y2[y2 != 2] == 0).astype(int)):.3f}")
            else:
                y3 = np.array([it["s3"] for it in cv["use"]])
                line.append(f"S3 AUC {auc(cv['P'][:, 1], y3):.3f}")
        print(" | ".join(line))


def cmd_final(a) -> None:
    store = FeatureStore()
    b1 = [it for it in gold_items("b1") if it["match_key"] not in EXCLUDE_MATCHES]
    b2 = gold_items("b2")
    weak = weak_rows(TRAIN_EVENTS)
    cvres = json.loads((MODELS / "students_cv.json").read_text())
    report = {"b1_items": len(b1), "b2_items": len(b2)}
    MODELS.mkdir(parents=True, exist_ok=True)
    import torch
    # ---- S2
    v2 = VARIANTS[a.s2]
    t2 = cvres["variants"][a.s2]["t"]
    h2 = train_variant(v2, store, b1, weak)
    use2 = [it for it in b2 if it["s2"] is not None]
    y2 = np.array([S2_CLASSES.index(it["s2"]) for it in use2])
    P2 = h2.predict_proba(design_matrix(store, use2, v2.use_size))
    hand2 = hand_scores(use2)
    t_h = cvres["hand_gate"]["keep95"]["t"]
    report["s2"] = {"variant": a.s2, "t_from_b1_cv": t2, "hand_t_keep95_from_b1": t_h,
                    "b2_dropped_judge_disagree": sum(it["s2_disagree"] for it in b2),
                    "b2": s2_eval(use2, P2, y2, t2, hand2, t_h)}
    torch.save({"head": h2.to_ckpt(), "classes": list(S2_CLASSES), "use_size": v2.use_size,
                "threshold_keep95": t2, "variant": a.s2, "feature": {"model": F.MODEL_NAME,
                "mode": F.DEFAULT_MODE, "size": F.DEFAULT_SIZE, "cls": "forward_features[:,0]"}},
               MODELS / "s2_legibility.pt")
    # ---- S3
    v3 = VARIANTS[a.s3]
    cv3 = cvres["variants"][a.s3]
    t3_q995 = min(cv3["t_gold_oof"], cv3["t_weak_oof"])
    # Conservative rule (declared before b2 was opened): the lowest out-of-fold score of any
    # gold-positive over two CV repeats (val recall 100%), since a 99.5% quantile of ~365
    # positives is decided by a single crop.
    lows = []
    for seed in (0, 1):
        r = run_cv(v3, store, b1, weak, folds_by_match(b1, seed=seed))
        yy = np.array([it["s3"] for it in r["use"]])
        lows.append(float(r["P"][yy == 1, 1].min()))
    t3 = min(lows) * 0.999 if a.s3_rule == "min2" else t3_q995
    h3 = train_variant(v3, store, b1, weak)
    use3 = [it for it in b2 if it["s3"] is not None]
    y3 = np.array([it["s3"] for it in use3])
    P3 = h3.predict_proba(design_matrix(store, use3, v3.use_size))
    report["s3"] = {"variant": a.s3, "rule": a.s3_rule, "t_from_b1_cv": t3, "t_q995_b1_cv": t3_q995,
                    "b1_oof_lowest_positive_by_repeat": lows,
                    "b2_dropped_judge_disagree": sum(it["s3_disagree"] for it in b2),
                    "b2": s3_eval(use3, P3, y3, t3),
                    "b2_at_t_q995": s3_eval(use3, P3, y3, t3_q995),
                    "b2_dropped_robots": [{"id": it["id"], "path": it["path"], "kind": it["kind"],
                                           "p_robot": float(P3[i, 1])}
                                          for i, it in enumerate(use3) if y3[i] == 1 and P3[i, 1] < t3]}
    torch.save({"head": h3.to_ckpt(), "use_size": v3.use_size, "threshold_recall995": t3,
                "variant": a.s3, "feature": {"model": F.MODEL_NAME, "mode": F.DEFAULT_MODE,
                "size": F.DEFAULT_SIZE, "cls": "forward_features[:,0]"}}, MODELS / "s3_robot.pt")
    (MODELS / "students_b2_report.json").write_text(json.dumps(report, indent=1, default=float))
    s2b = report["s2"]["b2"]
    print(f"== S2 {a.s2} on b2 (t={t2:.4g} from b1 CV)")
    for sl, r in s2b.items():
        print(f"  [{sl}]")
        print("   ", fmt_s2("student", r["student"]))
        print("   ", fmt_s2("hand gate @ floor 0.005", r["hand_gate_floor"]))
        print("   ", fmt_s2("hand gate @ keep95 t (b1)", r["hand_gate_keep95"]))
        print("    confusion", r["confusion(rows=gold legible/illegible/not_bumper, cols=pred)"],
              "AUC(leg|ill)", {k: round(v, 3) for k, v in r["auc_legible_vs_illegible"].items()})
    print(f"== S3 {a.s3} on b2 (t={t3:.4g} from b1 CV)")
    for sl, r in report["s3"]["b2"].items():
        print("   ", fmt_s3(f"[{sl}]", r))
    print(f"   (alt: t_q995 = {t3_q995:.4g})")
    for sl, r in report["s3"]["b2_at_t_q995"].items():
        print("   ", fmt_s3(f"[{sl}] @q995", r))
    for d in report["s3"]["b2_dropped_robots"]:
        print("    dropped robot:", d)
    print("->", MODELS)


# ----------------------------------------------------------------------------- v2: b1 + b3 + equipment

V2_VARIANTS = [
    # v1 recipes trained on b1 only = the reference under the same folds
    Variant("s2_v1ref", "s2", "mlp", 1e-2, weak=True, weak_weight=0.1),
    Variant("s2_mlp_weak_w01_b13", "s2", "mlp", 1e-2, weak=True, weak_weight=0.1, gold=("b1", "b3")),
    Variant("s2_mlp_weak_b13", "s2", "mlp", 1e-2, weak=True, weak_weight=0.3, gold=("b1", "b3")),
    Variant("s2_mlp_b13", "s2", "mlp", 1e-2, gold=("b1", "b3")),
    Variant("s2_lr_weak_w01_b13", "s2", "lr", 1e-3, weak=True, weak_weight=0.1, gold=("b1", "b3")),
    Variant("s2_mlp_size_weak_w01_b13", "s2", "mlp", 1e-2, use_size=True, weak=True, weak_weight=0.1,
            gold=("b1", "b3")),
    Variant("s2_mlp_weak_w01_b13_eq", "s2", "mlp", 1e-2, weak=True, weak_weight=0.1, gold=("b1", "b3"),
            equip_s2=True),
    Variant("s3_v1ref", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call"),
    Variant("s3_mlp_weak_callneg_b13", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call",
            gold=("b1", "b3")),
    # v1 recipe (b1 gold only) + equipment: isolates the equipment effect from the b3 hard positives
    Variant("s3_mlp_weak_callneg_b1_eq", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call",
            equip=True, equip_weight=1.0),
    Variant("s3_mlp_weak_callneg_b1_eq03", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call",
            equip=True, equip_weight=0.3),
    Variant("s3_mlp_weak_callneg_b13_eq", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3, neg_source="call",
            gold=("b1", "b3"), equip=True, equip_weight=1.0),
    Variant("s3_mlp_weak_callneg_b13_eq03", "s3", "mlp", 1e-2, weak=True, weak_weight=0.3,
            neg_source="call", gold=("b1", "b3"), equip=True, equip_weight=0.3),
    Variant("s3_lr_weak_callneg_b13_eq", "s3", "lr", 1e-3, weak=True, weak_weight=0.3, neg_source="call",
            gold=("b1", "b3"), equip=True, equip_weight=1.0),
    Variant("s3_mlp_weak_callneg_w1_b13_eq", "s3", "mlp", 1e-2, weak=True, weak_weight=1.0,
            neg_source="call", gold=("b1", "b3"), equip=True, equip_weight=1.0),
    Variant("s3_mlp_b13_eq", "s3", "mlp", 1e-2, gold=("b1", "b3"), equip=True, equip_weight=1.0),
    Variant("s3_mlp_size_weak_callneg_b13_eq", "s3", "mlp", 1e-2, use_size=True, weak=True,
            weak_weight=0.3, neg_source="call", gold=("b1", "b3"), equip=True, equip_weight=1.0),
]
VARIANTS.update({v.name: v for v in V2_VARIANTS})


def kind_rejection(rows: Sequence[dict], s: np.ndarray, y: np.ndarray, t: float) -> Dict[str, str]:
    """Junk rejected / junk total per judged kind (S3 negatives only)."""
    c = defaultdict(lambda: [0, 0])
    for r, si, yi in zip(rows, s, y):
        if yi == 0:
            c[r.get("kind") or "?"][1] += 1
            c[r.get("kind") or "?"][0] += int(si < t)
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())}


def equip_rejection(rows: Sequence[dict], s: np.ndarray, t: float,
                    keys=("origin", "above_prod_conf", "equip_kind", "source", "label")) -> dict:
    if not len(rows):
        return {}
    out = {"all": f"{int((s < t).sum())}/{len(rows)}"}
    for key in keys:
        for val in sorted({str(r.get(key)) for r in rows}):
            m = np.array([str(r.get(key)) == val for r in rows])
            out[f"{key}={val}"] = f"{int((s[m] < t).sum())}/{int(m.sum())}"
    return out


def s3_v2_eval(use, s, y, t) -> dict:
    out = s3_eval(use, np.stack([1 - s, s], 1), y, t)
    out["junk_rejection_by_kind"] = kind_rejection(use, s, y, t)
    return out


def _s3_cv_report(cv: dict, t: float) -> dict:
    use, s = cv["use"], cv["P"][:, 1]
    y = np.array([it["s3"] for it in use])
    b1 = np.array([it["batch"] == "b1" for it in use])
    r = {"t": t, "b1": s3_report(s[b1], y[b1], t), "b3": s3_report(s[~b1], y[~b1], t),
         "b3_pools": {p: s3_report(s[~b1 & m], y[~b1 & m], t) for p in sorted({it["pool"] for it in use
                                                                              if it["batch"] == "b3"})
                      for m in [np.array([it["pool"] == p for it in use])]},
         "b1b3_junk_by_kind": kind_rejection(use, s, y, t),
         "weak_pos_recall": float((cv["Pw"] >= t).mean()) if len(cv["Pw"]) else None,
         "equip_train_oof": equip_rejection(cv["equip"], cv["Pe"][:, 1], t) if len(cv["equip"]) else {}}
    return r


def s3_thresholds(cvs: Sequence[dict]) -> dict:
    """min rule = lowest OOF gold-robot score over b1+b3 and all repeats (x0.999); q995 = 99.5%
    recall on b1 gold positives (random sample) and on weak positives, the lower of the two."""
    lows, lows_b1, q = [], [], []
    for cv in cvs:
        s = cv["P"][:, 1]
        y = np.array([it["s3"] for it in cv["use"]])
        b1 = np.array([it["batch"] == "b1" for it in cv["use"]])
        lows.append(float(s[y == 1].min()))
        lows_b1.append(float(s[(y == 1) & b1].min()))
        q.append(min(thr_for_recall(s[(y == 1) & b1], S3_RECALL),
                     thr_for_recall(cv["Pw"], S3_RECALL) if len(cv["Pw"]) else 1.0))
    return {"t_min_b1b3": min(lows) * 0.999, "t_min_b1": min(lows_b1) * 0.999, "t_q995": min(q),
            "lowest_pos_by_repeat_b1b3": lows, "lowest_pos_by_repeat_b1": lows_b1}


def _s2_cv_report(cv: dict, t: float) -> dict:
    use, P = cv["use"], cv["P"]
    y = np.array([S2_CLASSES.index(it["s2"]) for it in use])
    g = s2_gate_score(P)
    b1 = np.array([it["batch"] == "b1" for it in use])
    y26 = np.array([it["year"] == 2026 for it in use])
    li = b1 & (y != 2)
    r = {"t": t, "b1": s2_report(g[b1], y[b1], t), "b1_2026": s2_report(g[b1 & y26], y[b1 & y26], t),
         "b3": s2_report(g[~b1], y[~b1], t),
         "b3_pools": {p: s2_report(g[~b1 & m], y[~b1 & m], t) for p in sorted({it["pool"] for it in use
                                                                              if it["batch"] == "b3"})
                      for m in [np.array([it["pool"] == p for it in use])]},
         "auc_b1_leg_vs_ill": auc(g[li], (y[li] == 0).astype(int)),
         "acc_b1": float((P[b1].argmax(1) == y[b1]).mean())}
    if len(cv["equip"]):
        r["equip_train_oof_dropped"] = f"{int((cv['Pe'][:, 0] < t).sum())}/{len(cv['equip'])}"
    return r


def cmd_cv_v2(a) -> None:
    store = FeatureStore()
    items = training_gold()
    fold = folds_by_match(items)
    weak = weak_rows(TRAIN_EVENTS)
    eq = equipment_rows("train")
    efold = folds_by_source(eq)
    print(f"gold b1+b3 items {len(items)} ({Counter(it['batch'] for it in items)}): "
          f"s2 {Counter((it['batch'], it['s2']) for it in items)}, s3 {Counter((it['batch'], it['s3']) for it in items)}")
    print(f"equipment train rows {len(eq)}: {Counter((r['source'], r['origin'] if 'origin' in r else '') for r in eq)}")
    print("weak rows:", dict(Counter((r["task"], r["src"]) for r in weak)))
    names = a.variants.split(",") if a.variants else [v.name for v in V2_VARIANTS]
    out = MODELS / "students_cv_v2.json"
    res = json.loads(out.read_text()) if out.exists() and a.variants else {"variants": {}}
    for name in names:
        v = VARIANTS[name]
        cvs = [run_cv(v, store, items, weak, folds_by_match(items, seed=sd), eq, folds_by_source(eq, seed=sd),
                      seed=sd) for sd in range(a.repeats if v.task == "s3" else 1)]
        if v.task == "s2":
            y = np.array([S2_CLASSES.index(it["s2"]) for it in cvs[0]["use"]])
            b1 = np.array([it["batch"] == "b1" for it in cvs[0]["use"]])
            t = thr_for_recall(s2_gate_score(cvs[0]["P"])[b1 & (y == 0)], S2_KEEP)
            r = _s2_cv_report(cvs[0], t)
            print(fmt_s2(f"{name} [b1]", r["b1"]), f"AUC {r['auc_b1_leg_vs_ill']:.3f} acc {r['acc_b1']:.3f}")
            print(fmt_s2(f"{'':>{len(name)}} [b1 2026]", r["b1_2026"]))
            print(fmt_s2(f"{'':>{len(name)}} [b3]", r["b3"]), r.get("equip_train_oof_dropped", ""))
        else:
            th = s3_thresholds(cvs)
            r = {"thresholds": th, "at_min_b1b3": _s3_cv_report(cvs[0], th["t_min_b1b3"]),
                 "at_q995": _s3_cv_report(cvs[0], th["t_q995"])}
            for tn in ("at_min_b1b3", "at_q995"):
                rr = r[tn]
                print(fmt_s3(f"{name} {tn} [b1]", rr["b1"]))
                print(fmt_s3(f"{'':>{len(name)}} {tn} [b3]", rr["b3"]))
                print(f"{'':>{len(name)}}   junk by kind {rr['b1b3_junk_by_kind']} | equip OOF "
                      f"{rr['equip_train_oof'].get('all')} | weak-pos recall {rr['weak_pos_recall']:.4f}")
        res["variants"][name] = r
    MODELS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("->", out)


def _load_ck(name: str) -> dict:
    import torch
    return torch.load(MODELS / name, map_location="cpu", weights_only=True)


def _ck_probs(ck: dict, store: FeatureStore, rows: List[dict]) -> np.ndarray:
    return Head.from_ckpt(ck["head"]).predict_proba(design_matrix(store, rows, ck["use_size"]))


def cmd_final_v2(a) -> None:
    """Train the chosen v2 variants on all training gold (+weak, +equipment train), save
    *_v2.pt, evaluate ONCE on b2 + the held-out equipment slice next to the v1 checkpoints."""
    import torch
    store = FeatureStore()
    items = training_gold()
    weak = weak_rows(TRAIN_EVENTS)
    eq_tr, eq_te = equipment_rows("train"), equipment_rows("test")
    b2 = gold_items("b2")
    cvres = json.loads((MODELS / "students_cv_v2.json").read_text())["variants"]
    v1_2 = _load_ck(a.v1_s2)
    v1_3 = _load_ck(a.v1_s3)
    rep = {"train_items": dict(Counter(it["batch"] for it in items)), "b2_items": len(b2),
           "equipment_train": len(eq_tr), "equipment_test": len(eq_te),
           "note": "second look at b2 for the S2/S3 students; no iteration after it"}
    feat = {"model": F.MODEL_NAME, "mode": F.DEFAULT_MODE, "size": F.DEFAULT_SIZE, "cls": "forward_features[:,0]"}
    # ---- S2
    v2 = VARIANTS[a.s2]
    t2 = cvres[a.s2]["t"]
    h2 = train_variant(v2, store, items, weak, equip_train=eq_tr)
    use2 = [it for it in b2 if it["s2"] is not None]
    y2 = np.array([S2_CLASSES.index(it["s2"]) for it in use2])
    P2 = h2.predict_proba(design_matrix(store, use2, v2.use_size))
    P2o = _ck_probs(v1_2, store, use2)
    hand2 = hand_scores(use2)
    t_h = json.loads((MODELS / "students_cv.json").read_text())["hand_gate"]["keep95"]["t"]
    rep["s2"] = {"variant": a.s2, "t_from_cv": t2, "v1_variant": v1_2.get("variant"),
                 "v1_t": float(v1_2["threshold_keep95"]),
                 "new_b2": s2_eval(use2, P2, y2, t2, hand2, t_h),
                 "v1_b2": s2_eval(use2, P2o, y2, float(v1_2["threshold_keep95"]), hand2, t_h)}
    if eq_te:
        Pe2 = h2.predict_proba(design_matrix(store, eq_te, v2.use_size))[:, 0]
        Pe2o = _ck_probs(v1_2, store, eq_te)[:, 0]
        rep["s2"]["equip_test_dropped"] = {"new": equip_rejection(eq_te, Pe2, t2),
                                           "v1": equip_rejection(eq_te, Pe2o, float(v1_2["threshold_keep95"]))}
    torch.save({"head": h2.to_ckpt(), "classes": list(S2_CLASSES), "use_size": v2.use_size,
                "threshold_keep95": t2, "variant": a.s2, "gold": list(v2.gold), "feature": feat},
               MODELS / "s2_legibility_v2.pt")
    # ---- S3
    v3 = VARIANTS[a.s3]
    th = cvres[a.s3]["thresholds"]
    t3 = th["t_min_b1b3"] if a.s3_rule == "min" else th["t_q995"]
    h3 = train_variant(v3, store, items, weak, equip_train=eq_tr)
    use3 = [it for it in b2 if it["s3"] is not None]
    y3 = np.array([it["s3"] for it in use3])
    s3n = h3.predict_proba(design_matrix(store, use3, v3.use_size))[:, 1]
    s3o = _ck_probs(v1_3, store, use3)[:, 1]
    to = float(v1_3["threshold_recall995"])
    v1th = cvres.get("s3_v1ref", {}).get("thresholds")
    rep["s3"] = {"variant": a.s3, "rule": a.s3_rule, "thresholds_from_cv": th, "t": t3,
                 "v1_variant": v1_3.get("variant"), "v1_t": to,
                 "new_b2": s3_v2_eval(use3, s3n, y3, t3),
                 "new_b2_at_alt": s3_v2_eval(use3, s3n, y3, th["t_q995"] if a.s3_rule == "min" else th["t_min_b1b3"]),
                 # informational: v1's rule (lowest OOF b1 positive) on the new head
                 "new_b2_at_t_min_b1": s3_v2_eval(use3, s3n, y3, th["t_min_b1"]),
                 "v1_b2": s3_v2_eval(use3, s3o, y3, to),
                 # v1's checkpoint under the v2 rules (thresholds of the v1 recipe s3_v1ref from the
                 # same b1+b3 CV): v1's 0.013 was set without the b3 hard positives
                 "v1_ref_thresholds": cvres.get("s3_v1ref", {}).get("thresholds"),
                 "v1_b2_at_v2rule_min": s3_v2_eval(use3, s3o, y3, v1th["t_min_b1b3"]) if v1th else None,
                 "v1_b2_at_v2rule_q995": s3_v2_eval(use3, s3o, y3, v1th["t_q995"]) if v1th else None,
                 "new_b2_dropped_robots": [{"id": it["id"], "path": it["path"], "kind": it["kind"],
                                            "p_robot": float(s3n[i])} for i, it in enumerate(use3)
                                           if y3[i] == 1 and s3n[i] < t3]}
    if eq_te:
        se = h3.predict_proba(design_matrix(store, eq_te, v3.use_size))[:, 1]
        seo = _ck_probs(v1_3, store, eq_te)[:, 1]
        t_alt = th["t_q995"] if a.s3_rule == "min" else th["t_min_b1b3"]
        rep["s3"]["equip_test_rejected"] = {"new": equip_rejection(eq_te, se, t3),
                                            "new_at_alt": equip_rejection(eq_te, se, t_alt),
                                            "v1": equip_rejection(eq_te, seo, to)}
        if v1th:
            rep["s3"]["equip_test_rejected"]["v1_at_v2rule_min"] = equip_rejection(eq_te, seo, v1th["t_min_b1b3"])
            rep["s3"]["equip_test_rejected"]["v1_at_v2rule_q995"] = equip_rejection(eq_te, seo, v1th["t_q995"])
    torch.save({"head": h3.to_ckpt(), "use_size": v3.use_size, "threshold_recall995": t3,
                "threshold_rule": a.s3_rule, "thresholds_cv": th, "variant": a.s3, "gold": list(v3.gold),
                "equip": v3.equip, "feature": feat}, MODELS / "s3_robot_v2.pt")
    (MODELS / "students_b2_report_v2.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"== S2 b2: new {a.s2} (t={t2:.4g}) vs v1 {rep['s2']['v1_variant']} (t={rep['s2']['v1_t']:.4g})")
    for sl in rep["s2"]["new_b2"]:
        print(f"  [{sl}]")
        print("   ", fmt_s2("new", rep["s2"]["new_b2"][sl]["student"]))
        print("   ", fmt_s2("v1", rep["s2"]["v1_b2"][sl]["student"]))
    for k, v in (rep["s2"].get("equip_test_dropped") or {}).items():
        print(f"  equipment test dropped by S2 [{k}]:", v)
    print(f"== S3 b2: new {a.s3} (rule {a.s3_rule}, t={t3:.4g}) vs v1 (t={to:.4g})")
    for key in ("new_b2", "new_b2_at_alt", "new_b2_at_t_min_b1", "v1_b2", "v1_b2_at_v2rule_min",
                "v1_b2_at_v2rule_q995"):
        for sl, r in (rep["s3"][key] or {}).items():
            if sl == "junk_rejection_by_kind":
                print(f"    {key} junk by kind {r}")
            else:
                print("   ", fmt_s3(f"{key} [{sl}]", r))
    for k, v in (rep["s3"].get("equip_test_rejected") or {}).items():
        print(f"  equipment test rejected by S3 [{k}]:", v)
    for d in rep["s3"]["new_b2_dropped_robots"]:
        print("    dropped robot:", d)
    print("->", MODELS / "students_b2_report_v2.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("modes").set_defaults(fn=cmd_modes)
    c = sub.add_parser("cv")
    c.add_argument("--variants", default=None)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_cv)
    f = sub.add_parser("final")
    f.add_argument("--s2", required=True)
    f.add_argument("--s3", required=True)
    f.add_argument("--s3-rule", choices=["min2", "q995"], default="min2")
    f.set_defaults(fn=cmd_final)
    c2 = sub.add_parser("cv-v2")
    c2.add_argument("--variants", default=None)
    c2.add_argument("--repeats", type=int, default=2)
    c2.set_defaults(fn=cmd_cv_v2)
    f2 = sub.add_parser("final-v2")
    f2.add_argument("--s2", required=True)
    f2.add_argument("--s3", required=True)
    f2.add_argument("--s3-rule", choices=["min", "q995"], default="min")
    f2.add_argument("--v1-s2", default="s2_legibility.pt")
    f2.add_argument("--v1-s3", default="s3_robot.pt")
    f2.set_defaults(fn=cmd_final_v2)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
