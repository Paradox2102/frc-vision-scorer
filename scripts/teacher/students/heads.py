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

  .venv/bin/python -m scripts.teacher.students.heads cv
  .venv/bin/python -m scripts.teacher.students.heads final --s2 <variant> --s3 <variant>
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

DATA = MAIN / "data/teacher"
ROOT = DATA / "harvest_v2"
GOLD = DATA / "gold"
MODELS = DATA / "models"
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
                    "kind": js[0].get("kind")})
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
    def __init__(self, out: Path = F.DEFAULT_OUT, events: Optional[Sequence[str]] = None):
        self.index, self.X = F.load_cache(out, events)

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


def _gold_xy(store, items, task, use_size):
    lab = "s2" if task == "s2" else "s3"
    use = [it for it in items if it[lab] is not None]
    X = design_matrix(store, use, use_size)
    y = np.array([S2_CLASSES.index(it["s2"]) if task == "s2" else it["s3"] for it in use])
    return use, X, y


def train_variant(v: Variant, store: FeatureStore, gold_train: List[dict], weak_train: List[dict],
                  seed: int = 0) -> Head:
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
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    return Head(kind=v.model, n_out=n_out, wd=v.wd, seed=seed).fit(X, y, w)


def run_cv(v: Variant, store: FeatureStore, items: List[dict], weak: List[dict],
           fold: Dict[str, int]) -> dict:
    """Out-of-fold probs for gold items (and S3 weak positives in held-out matches)."""
    lab = "s2" if v.task == "s2" else "s3"
    use = [it for it in items if it[lab] is not None]
    P = np.zeros((len(use), 3 if v.task == "s2" else 2), np.float32)
    weak_val = [r for r in weak if r["task"] == "s3" and r["y"] == 1 and r["match_key"] in fold]
    Pw = np.zeros(len(weak_val), np.float32)
    for k in range(N_FOLDS):
        tr = [it for it in items if fold[it["match_key"]] != k]
        wtr = [r for r in weak if fold.get(r["match_key"], -1) != k]
        head = train_variant(v, store, tr, wtr)
        te = [i for i, it in enumerate(use) if fold[it["match_key"]] == k]
        P[te] = head.predict_proba(design_matrix(store, [use[i] for i in te], v.use_size))
        if v.task == "s3":
            wi = [i for i, r in enumerate(weak_val) if fold[r["match_key"]] == k]
            if wi:
                Pw[wi] = head.predict_proba(design_matrix(store, [weak_val[i] for i in wi],
                                                          v.use_size))[:, 1]
    return {"use": use, "P": P, "weak_val": weak_val, "Pw": Pw}


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
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
