#!/usr/bin/env python3
"""S4 same-robot embedder (teacher-distillation design §5 S4, results §13).

Fine-tunes DINOv3 ViT-S/16 (``timm/vit_small_patch16_dinov3.lvd1689m``, the production
re-ID backbone) with a supervised-contrastive loss so that crops of the same robot at one
event embed close together across matches, while co-occurring robots stay apart.

* Identity key = (event, team): robots change between events and completely between seasons.
* Labels: bumper tracklets with a Jev ``team`` decision (labels_jev.jsonl, propagated to every
  crop of the tracklet) and per-crop robot labels (labels_robot.jsonl, time-local bumper rule).
  ``off_alliance`` labels are dropped.
* Model: explicit CLS (``forward_features(x)[:, 0]`` -- timm's pooled output is an avg over
  patch tokens, a known regression here) -> MLP -> 128-d L2-normalised embedding. Only the
  last ``train_blocks`` transformer blocks + final norm + projection are trained.
* Input: the harvester crop (robot box padded 10%, ``predict.pad_crop``) letterboxed to 224
  (features.preprocess), the same input the frozen-feature cache uses.
* Sampler: batches are built match-first: pick a match, add its labelled identities (so
  same-match same-alliance robots -- guaranteed different -- are in-batch negatives), repeat
  until P identities; each identity gets K crops spread round-robin over its matches.

Splits: TEST_EVENTS never touched until ``eval --split test``; GT_EXCLUDE never used;
``VAL_EVENTS`` (two whole training events) are held out for model selection -- the standard
10% val matches of an event share its teams with the train matches, so they cannot measure
cross-match generalisation to unseen robots. Training uses split_of(mk) == "train" matches of
the remaining events.

Usage:
  .venv/bin/python -m scripts.teacher.students.embedder train --name s4_robot --detectors robot
  .venv/bin/python -m scripts.teacher.students.embedder eval --ckpt data/teacher/models/s4_robot.pt --split val
  .venv/bin/python -m scripts.teacher.students.embedder baseline --split val

Inference:
  from scripts.teacher.students.embedder import S4Embedder
  emb = S4Embedder()                       # data/teacher/models/s4_embedder.pt
  vecs = emb.embed_crops(crops_bgr)        # [N, 128] float32, L2-normalised
  vecs = emb.embed_boxes(frame_bgr, boxes) # cuts robot boxes like the harvester (10% pad)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.events import GT_EXCLUDE, TEST_EVENTS, split_of  # noqa: E402
from scripts.teacher.students import features as F  # noqa: E402
# runtime inference pieces (moved to web/ so the pipeline never imports scripts.*)
from web.students.s4_embedder import (  # noqa: E402,F401
    S4Embedder, S4RobotEmbedder, build_net, letterbox, load_net, model_tag,
)

MODELS = F.MAIN / "data/teacher/models"
DEFAULT_CKPT = MODELS / "s4_embedder.pt"
VAL_EVENTS = ("2026bcvi", "2024mimil")
EMB_DIM = 128
ROBOT_PAD = 0.10


# ----------------------------------------------------------------------------- data

def record_split(match_key: str) -> str:
    """'test' | 'val' (VAL_EVENTS) | 'train' | 'excluded' (GT or standard val matches)."""
    s = split_of(match_key)
    if s in ("test", "excluded"):
        return s
    if match_key.split("_", 1)[0] in VAL_EVENTS:
        return "val"
    return "train" if s == "train" else "excluded"


def team_alliance(roster: Optional[dict], team: str) -> Optional[str]:
    for col in ("red", "blue"):
        if roster and team in (roster.get(col) or []):
            return col
    return None


def load_match_records(mdir: Path) -> List[dict]:
    """Labelled crops of one harvested match (manifest order)."""
    jev = {}
    jp = mdir / "labels_jev.jsonl"
    if jp.exists():
        for line in jp.open():
            if line.strip():
                d = json.loads(line)
                if d.get("decision") == "team" and d.get("team") and not d.get("off_alliance"):
                    jev[d["tracklet_id"]] = str(d["team"])
    rob = {}
    rp = mdir / "labels_robot.jsonl"
    if rp.exists():
        for line in rp.open():
            if line.strip():
                d = json.loads(line)
                if d.get("team") and not d.get("off_alliance"):
                    rob[d["path"]] = str(d["team"])
    out = []
    for line in (mdir / "manifest.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        team = jev.get(r["tracklet_id"]) if r["detector"] == "bumper" else rob.get(r["path"])
        if not team:
            continue
        out.append({
            "path": r["path"], "event": r["event"], "match_key": r["match_key"],
            "detector": r["detector"], "team": team, "t_sec": float(r["t_sec"]),
            "tracklet_id": r["tracklet_id"], "roster": r.get("roster"), "box": r.get("box"),
            "alliance": team_alliance(r.get("roster"), team),
        })
    return out


def load_records(split: str, root: Path = F.DEFAULT_ROOT,
                 detectors: Sequence[str] = ("robot", "bumper")) -> List[dict]:
    recs = []
    for ev in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for mdir in sorted((root / ev).iterdir()):
            if not (mdir / "manifest.jsonl").exists() or not (mdir / "summary.json").exists():
                continue
            if record_split(mdir.name) != split:
                continue
            recs.extend(r for r in load_match_records(mdir) if r["detector"] in detectors)
    return recs


def ident(r: dict) -> Tuple[str, str]:
    return (r["event"], r["team"])


# ----------------------------------------------------------------------------- sampler

class IdentityBatchSampler:
    """Yields lists of record indices: ``p`` identities x ``k`` crops.

    Match-first: a random match contributes all its labelled identities, so robots that
    co-occur (guaranteed different) land in the same batch as hard negatives. An identity's
    k crops are spread round-robin over its matches (cross-match positives) and, when
    ``bumper_frac`` > 0, about that fraction of them are bumper crops.
    """

    def __init__(self, recs: Sequence[dict], p: int = 48, k: int = 4, steps: int = 1000,
                 bumper_frac: float = 0.0, seed: int = 0, min_crops: int = 2):
        self.p, self.k, self.steps, self.bumper_frac = p, k, steps, bumper_frac
        self.rng = random.Random(seed)
        by = defaultdict(lambda: defaultdict(lambda: {"robot": [], "bumper": []}))
        for i, r in enumerate(recs):
            by[ident(r)][r["match_key"]][r["detector"]].append(i)
        self.ids = {idk: dict(ms) for idk, ms in by.items()
                    if sum(len(v["robot"]) + len(v["bumper"]) for v in ms.values()) >= min_crops}
        self.match_ids = defaultdict(list)
        for idk, ms in self.ids.items():
            for m in ms:
                self.match_ids[m].append(idk)
        self.matches = sorted(self.match_ids)
        if not self.matches:
            raise ValueError("no identities with enough crops")

    def __len__(self) -> int:
        return self.steps

    def _crops(self, idk) -> List[int]:
        ms = list(self.ids[idk].items())
        self.rng.shuffle(ms)
        n_b = sum(1 for _ in range(self.k) if self.rng.random() < self.bumper_frac)
        want = ["bumper"] * n_b + ["robot"] * (self.k - n_b)
        out = []
        for j, det in enumerate(want):
            # round-robin over matches, falling back to the other detector if this one is empty
            for step in range(len(ms)):
                m, v = ms[(j + step) % len(ms)]
                pool = v[det] or v["robot" if det == "bumper" else "bumper"]
                if pool:
                    out.append(self.rng.choice(pool))
                    break
        return out

    def batch(self) -> List[int]:
        chosen, seen = [], set()
        tries = 0
        while len(chosen) < self.p and tries < 50 * self.p:
            tries += 1
            m = self.rng.choice(self.matches)
            cand = [i for i in self.match_ids[m] if i not in seen]
            self.rng.shuffle(cand)
            for idk in cand[: self.p - len(chosen)]:
                seen.add(idk)
                chosen.append(idk)
        out = []
        for idk in chosen:
            out.extend(self._crops(idk))
        return out

    def __iter__(self):
        for _ in range(self.steps):
            yield self.batch()


# ----------------------------------------------------------------------------- augmentation

def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-time augmentation on a BGR crop (before letterbox). No hue change: bumper colour
    is alliance (it flips between matches) and robot body colour is identity."""
    h, w = img.shape[:2]
    if h >= 8 and w >= 8:
        s = rng.uniform(0.6, 1.0)
        ar = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
        ch = int(round(min(h, h * math.sqrt(s / ar))))
        cw = int(round(min(w, w * math.sqrt(s * ar))))
        ch, cw = max(4, ch), max(4, cw)
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        img = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.3:  # simulate far / small robots
        f = rng.uniform(0.35, 0.75)
        hh, ww = img.shape[:2]
        small = cv2.resize(img, (max(4, int(ww * f)), max(4, int(hh * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.15:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.2))
    if rng.random() < 0.8:
        a = rng.uniform(0.75, 1.25)
        b = rng.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


class CropDataset:
    def __init__(self, recs: Sequence[dict], root: Path = F.DEFAULT_ROOT, train: bool = False,
                 seed: int = 0, inset: bool = False):
        self.recs, self.root, self.train, self.seed = recs, root, train, seed
        self.inset = inset  # cut the production region (box inset 8%) out of the padded crop
        self._rng = None

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.root / self.recs[i]["path"]))
        if img is None:
            img = np.zeros((32, 32, 3), np.uint8)
        elif self.inset and self.recs[i].get("box"):
            img = prod_crop(img, self.recs[i]["box"])
        if self.train:
            if self._rng is None:
                import torch
                self._rng = np.random.default_rng(self.seed + int(torch.initial_seed() % 100000))
            img = augment(img, self._rng)
        return F.preprocess(img, "letterbox", F.DEFAULT_SIZE), i


# ----------------------------------------------------------------------------- model + loss

# build_net lives in web.students.s4_embedder (runtime inference); re-exported above.


def supcon_loss(z, labels, tau: float = 0.1):
    """Supervised contrastive loss (Khosla et al., L_out) on L2-normalised ``z`` [N, d]."""
    import torch
    n = z.shape[0]
    sim = (z @ z.T) / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    npos = pos.sum(1)
    keep = npos > 0
    if not keep.any():
        return z.sum() * 0.0
    loss = -(logp * pos).sum(1)[keep] / npos[keep]
    return loss.mean()


# ----------------------------------------------------------------------------- metrics

def auc_fast(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via average ranks (ties handled)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _norm(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)


def pair_aucs(E: np.ndarray, recs: Sequence[dict], max_pairs_per_event: int = 3_000_000,
              seed: int = 0) -> dict:
    """Within-match (all pairs, and same-alliance pairs) and cross-match same-event AUCs."""
    E = _norm(E)
    rng = np.random.default_rng(seed)
    codes = lambda key: np.unique([str(key(r)) for r in recs], return_inverse=True)[1]  # noqa: E731
    ev = codes(lambda r: r["event"])
    mk = codes(lambda r: r["match_key"])
    tm = codes(lambda r: (r["event"], r["team"]))
    al = codes(lambda r: (r["match_key"], r.get("alliance") or ""))
    tk = codes(lambda r: (r["match_key"], r["detector"], r["tracklet_id"]))
    w_s, w_y, wa_s, wa_y, wx_s, wx_y, x_s, x_y = [], [], [], [], [], [], [], []
    for e in np.unique(ev):
        idx = np.where(ev == e)[0]
        S = E[idx] @ E[idx].T
        iu, ju = np.triu_indices(len(idx), 1)
        same_m = mk[idx][iu] == mk[idx][ju]
        same_t = tm[idx][iu] == tm[idx][ju]
        same_a = al[idx][iu] == al[idx][ju]  # alliance code includes the match
        s = S[iu, ju]
        w_s.append(s[same_m]); w_y.append(same_t[same_m])
        sel = same_m & same_a
        wa_s.append(s[sel]); wa_y.append(same_t[sel])
        sel = same_m & (tk[idx][iu] != tk[idx][ju])  # relink-like: different tracklets
        wx_s.append(s[sel]); wx_y.append(same_t[sel])
        xm = np.where(~same_m)[0]
        if len(xm) > max_pairs_per_event:
            xm = rng.choice(xm, max_pairs_per_event, replace=False)
        x_s.append(s[xm]); x_y.append(same_t[xm])
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
    out = {
        "within_match_auc": auc_fast(cat(w_s), cat(w_y)),
        "within_match_same_alliance_auc": auc_fast(cat(wa_s), cat(wa_y)),
        "within_match_cross_tracklet_auc": auc_fast(cat(wx_s), cat(wx_y)),
        "cross_match_auc": auc_fast(cat(x_s), cat(x_y)),
        "n_within_pairs": int(sum(len(a) for a in w_s)),
        "n_cross_pairs": int(sum(len(a) for a in x_s)),
        "n_cross_pos": int(sum(a.sum() for a in x_y)),
    }
    return out


def lineup_top1(E: np.ndarray, recs: Sequence[dict]) -> dict:
    """For every crop: nearest roster identity, gallery = mean embedding of that team's crops
    in OTHER matches of the same event. 6-way (whole roster) and 3-way (alliance known),
    per crop and per tracklet (mean embedding over the tracklet's crops)."""
    E = _norm(E)
    by_ev_team_match = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(recs):
        by_ev_team_match[(r["event"], r["team"])][r["match_key"]].append(i)
    # per (event, team): sum + count per match, so "all but this match" is a subtraction
    sums = {}
    for key, ms in by_ev_team_match.items():
        sums[key] = {m: (E[ix].sum(0), len(ix)) for m, ix in ms.items()}

    def gallery(event, team, excl_match):
        d = sums.get((event, team))
        if not d:
            return None
        tot = sum(v[0] for m, v in d.items() if m != excl_match) if any(m != excl_match for m in d) else None
        if tot is None:
            return None
        n = np.linalg.norm(tot)
        return tot / n if n > 0 else None

    res = {"crop6": [0, 0], "crop3": [0, 0], "trk6": [0, 0], "trk3": [0, 0]}
    ncand = []
    trk = defaultdict(list)
    for i, r in enumerate(recs):
        trk[(r["match_key"], r["detector"], r["tracklet_id"], r["team"])].append(i)
    gcache = {}

    def cands(r, alliance_only):
        roster = r.get("roster") or {}
        cols = [r["alliance"]] if alliance_only and r.get("alliance") else ["red", "blue"]
        teams = [t for c in cols for t in (roster.get(c) or [])]
        out = []
        for t in teams:
            k = (r["event"], t, r["match_key"])
            if k not in gcache:
                gcache[k] = gallery(r["event"], t, r["match_key"])
            if gcache[k] is not None:
                out.append((t, gcache[k]))
        return out

    def score(v, r, alliance_only, key):
        c = cands(r, alliance_only)
        if r["team"] not in [t for t, _ in c]:
            return
        if not alliance_only:
            ncand.append(len(c))
        sims = [float(v @ g) for _, g in c]
        res[key][1] += 1
        res[key][0] += int(c[int(np.argmax(sims))][0] == r["team"])

    for i, r in enumerate(recs):
        score(E[i], r, False, "crop6")
        score(E[i], r, True, "crop3")
    for (_, _, _, _), ix in trk.items():
        v = E[ix].mean(0)
        v = v / max(np.linalg.norm(v), 1e-8)
        r = recs[ix[0]]
        score(v, r, False, "trk6")
        score(v, r, True, "trk3")
    out = {f"lineup_{k}_top1": (a / n if n else float("nan")) for k, (a, n) in res.items()}
    out.update({f"lineup_{k}_n": n for k, (_, n) in res.items()})
    out["lineup_mean_candidates6"] = float(np.mean(ncand)) if ncand else float("nan")
    return out


def evaluate(E: np.ndarray, recs: Sequence[dict], per_event: bool = True) -> dict:
    out = {"all": {**pair_aucs(E, recs), **lineup_top1(E, recs), "n_crops": len(recs)}}
    if per_event:
        evs = sorted({r["event"] for r in recs})
        for e in evs:
            ix = [i for i, r in enumerate(recs) if r["event"] == e]
            sub = [recs[i] for i in ix]
            out[e] = {**pair_aucs(E[ix], sub), **lineup_top1(E[ix], sub), "n_crops": len(sub)}
    return out


# ----------------------------------------------------------------------------- embedding

@dataclass
class TrainCfg:
    name: str = "s4_embedder"
    detectors: Tuple[str, ...] = ("robot",)
    bumper_frac: float = 0.0
    p: int = 48
    k: int = 4
    steps: int = 3000
    train_blocks: int = 4
    lr_backbone: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    tau: float = 0.1
    warmup: int = 100
    eval_every: int = 500
    workers: int = 6
    seed: int = 0
    dim: int = EMB_DIM
    inset: bool = False
    extra: dict = field(default_factory=dict)


def embed_records(net, recs: Sequence[dict], root: Path = F.DEFAULT_ROOT, batch: int = 128,
                  workers: int = 6, device: str = "cuda", which: str = "proj",
                  inset: bool = False) -> np.ndarray:
    import torch
    dl = torch.utils.data.DataLoader(CropDataset(recs, root, inset=inset), batch_size=batch, num_workers=workers,
                                     shuffle=False)
    out = []
    net.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            v = net(x) if which == "proj" else net.cls(x)
            out.append(v.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)


def frozen_cache_feats(recs: Sequence[dict]) -> np.ndarray:
    """Frozen DINOv3-S CLS (letterbox 224) from the features cache, aligned with ``recs``."""
    events = sorted({r["event"] for r in recs})
    pmap, X = F.load_cache(F.DEFAULT_OUT, events)
    missing = [r["path"] for r in recs if r["path"] not in pmap]
    if missing:
        raise KeyError(f"{len(missing)} crops not in the feature cache, e.g. {missing[0]}")
    return X[[pmap[r["path"]] for r in recs]]


def prod_crop(img: np.ndarray, box: Sequence[float], pad: float = ROBOT_PAD) -> np.ndarray:
    """From a harvester crop (box padded by ``pad``) cut the production embedder's input
    region: the box inset by web.embedder.BOX_INSET per side."""
    from web.embedder import BOX_INSET
    bw, bh = box[2] - box[0], box[3] - box[1]
    ox = int(max(0, np.floor(box[0] - pad * bw)))
    oy = int(max(0, np.floor(box[1] - pad * bh)))
    x0, y0 = int(box[0] + BOX_INSET * bw) - ox, int(box[1] + BOX_INSET * bh) - oy
    x1, y1 = int(box[2] - BOX_INSET * bw) - ox, int(box[3] - BOX_INSET * bh) - oy
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return img[y0:y1, x0:x1] if x1 - x0 >= 4 and y1 - y0 >= 4 else img


def frozen_prod_feats(recs: Sequence[dict], batch: int = 128) -> np.ndarray:
    """Frozen DINOv3-S CLS with the production transform (web/embedder.py: inset box, timm
    eval transform = shorter side 256 bicubic + centre crop 256)."""
    import timm
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(F.MODEL_NAME, pretrained=True, num_classes=0).to(dev).eval()
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(m), is_training=False)
    out = []
    for i in range(0, len(recs), batch):
        xs = []
        for r in recs[i:i + batch]:
            img = cv2.imread(str(F.DEFAULT_ROOT / r["path"]))
            xs.append(tf(Image.fromarray(cv2.cvtColor(prod_crop(img, r["box"]), cv2.COLOR_BGR2RGB))))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(m.forward_features(torch.stack(xs).to(dev))[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def _save_ckpt(net, cfg: TrainCfg, path: Path, extra: dict) -> None:
    import torch
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
    torch.save({"state_dict": sd, "cfg": asdict(cfg), "model": F.MODEL_NAME, "dim": cfg.dim,
                "input": {"pad": ROBOT_PAD, "inset": cfg.inset, "mode": "letterbox", "size": F.DEFAULT_SIZE,
                          "cls": "forward_features[:,0]"}, **extra}, path)


# load_net / S4Embedder live in web.students.s4_embedder (runtime inference); re-exported above.


# ----------------------------------------------------------------------------- commands

def _summ(name: str, m: dict) -> str:
    a = m["all"]
    return (f"{name:28s} within {a['within_match_auc']:.4f} (x-trk {a['within_match_cross_tracklet_auc']:.4f}"
            f" same-all {a['within_match_same_alliance_auc']:.4f})"
            f"  cross {a['cross_match_auc']:.4f}  lineup6 crop {a['lineup_crop6_top1']:.3f} trk {a['lineup_trk6_top1']:.3f}"
            f"  lineup3 crop {a['lineup_crop3_top1']:.3f} trk {a['lineup_trk3_top1']:.3f}")


def cmd_train(a) -> None:
    import torch
    cfg = TrainCfg(name=a.name, detectors=tuple(a.detectors.split(",")), bumper_frac=a.bumper_frac,
                   p=a.p, k=a.k, steps=a.steps, train_blocks=a.train_blocks, lr_backbone=a.lr_backbone,
                   lr_head=a.lr_head, tau=a.tau, eval_every=a.eval_every, workers=a.workers, seed=a.seed,
                   inset=a.inset)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = "cuda"
    tr = load_records("train", detectors=cfg.detectors)
    va = [r for r in load_records("val", detectors=("robot",))]
    print(f"train crops {len(tr)} ({sum(r['detector'] == 'robot' for r in tr)} robot), "
          f"identities {len({ident(r) for r in tr})}; val robot crops {len(va)}", flush=True)
    sampler = IdentityBatchSampler(tr, p=cfg.p, k=cfg.k, steps=cfg.steps, bumper_frac=cfg.bumper_frac,
                                   seed=cfg.seed)
    ds = CropDataset(tr, train=True, seed=cfg.seed, inset=cfg.inset)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=cfg.workers,
                                     pin_memory=True, prefetch_factor=4)
    id_index = {k: i for i, k in enumerate(sorted({ident(r) for r in tr}))}
    net = build_net(True, cfg.train_blocks, cfg.dim).to(dev)
    bb = [p for p in net.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg.lr_backbone},
                             {"params": net.proj.parameters(), "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = (step + 1) / cfg.warmup if step < cfg.warmup else \
            0.5 * (1 + math.cos(math.pi * (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * f

    torch.cuda.reset_peak_memory_stats()
    out_path = MODELS / f"{cfg.name}.pt"
    log = {"cfg": asdict(cfg), "evals": []}
    best = -1.0
    t0 = time.time()
    run_loss = 0.0
    for step, (x, idx) in enumerate(dl):
        labels = torch.tensor([id_index[ident(tr[int(i)])] for i in idx], device=dev)
        set_lr(step)
        net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(x.to(dev, non_blocking=True))
        loss = supcon_loss(z.float(), labels, cfg.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()
        run_loss += float(loss.detach())
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{cfg.steps} loss {run_loss / 50:.4f} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            run_loss = 0.0
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            E = embed_records(net, va, workers=cfg.workers, inset=cfg.inset)
            m = evaluate(E, va, per_event=False)["all"]
            m["step"] = step + 1
            log["evals"].append(m)
            print(_summ(f"val@{step + 1}", {"all": m}), flush=True)
            if m["cross_match_auc"] > best:
                best = m["cross_match_auc"]
                _save_ckpt(net, cfg, out_path, {"val": m, "step": step + 1})
    log["train_s"] = time.time() - t0
    log["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    log["best_val_cross_auc"] = best
    (MODELS / f"{cfg.name}.log.json").write_text(json.dumps(log, indent=1))
    print(f"done in {log['train_s'] / 60:.1f} min, peak VRAM {log['peak_vram_gb']:.1f} GB, "
          f"best val cross {best:.4f} -> {out_path}")


def _eval_recs(split: str) -> List[dict]:
    if split == "test":
        recs = load_records("test", detectors=("robot",))
        assert {r["event"] for r in recs} <= set(TEST_EVENTS)
        return recs
    recs = load_records(split, detectors=("robot",))
    assert not any(r["match_key"] in GT_EXCLUDE for r in recs)
    return recs


def cmd_baseline(a) -> None:
    recs = _eval_recs(a.split)
    E = frozen_prod_feats(recs) if a.prod else frozen_cache_feats(recs)
    m = evaluate(E, recs)
    print(_summ(f"frozen-cls{'-prod' if a.prod else ''} {a.split}", m))
    for e in sorted(k for k in m if k != "all"):
        print(_summ(f"  {e}", {"all": m[e]}))
    if a.out:
        Path(a.out).write_text(json.dumps(m, indent=1))


def cmd_eval(a) -> None:
    recs = _eval_recs(a.split)
    net, ck = load_net(Path(a.ckpt))
    res = {}
    for which in a.which.split(","):
        E = embed_records(net, recs, which=which, inset=bool(ck["cfg"].get("inset")))
        m = evaluate(E, recs)
        res[which] = m
        print(_summ(f"{Path(a.ckpt).stem}[{which}] {a.split}", m))
        for e in sorted(k for k in m if k != "all"):
            print(_summ(f"  {e}", {"all": m[e]}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def bench_latency(ckpt: Path, batch: int = 64, n_iter: int = 30) -> dict:
    import torch
    emb = S4Embedder(ckpt)
    x = torch.randn(batch, 3, F.DEFAULT_SIZE, F.DEFAULT_SIZE, device=emb.device, dtype=torch.float16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(5):
            emb.net(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            emb.net(x)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    crops = [np.random.randint(0, 255, (110, 130, 3), np.uint8) for _ in range(batch)]
    t1 = time.time()
    for _ in range(5):
        emb.embed_crops(crops, batch=batch)
    dt_e2e = (time.time() - t1) / 5
    return {"batch": batch, "gpu_ms_per_crop": 1000 * dt / batch,
            "end_to_end_ms_per_crop": 1000 * dt_e2e / batch,
            "peak_vram_gb_inference": torch.cuda.max_memory_allocated() / 2**30}


def _by_year(recs: Sequence[dict], E: np.ndarray) -> dict:
    out = {}
    for y in sorted({r["event"][:4] for r in recs}):
        ix = [i for i, r in enumerate(recs) if r["event"].startswith(y)]
        sub = [recs[i] for i in ix]
        out[y] = {**pair_aucs(E[ix], sub), **lineup_top1(E[ix], sub), "n_crops": len(sub)}
    return out


def cmd_report(a) -> None:
    """Final report: frozen baselines (cache letterbox + production transform) vs S4 on val
    events and, once, on the TEST events; latency + VRAM. Writes s4_report.json."""
    import torch
    ck_path = Path(a.ckpt)
    net, ck = load_net(ck_path)
    rep = {"ckpt": str(ck_path), "cfg": ck["cfg"], "val_events": list(VAL_EVENTS),
           "test_events": sorted(TEST_EVENTS), "selected_step": ck.get("step"),
           "eval_crops": "robot-detector crops with a time-local per-crop team label (labels_robot.jsonl)",
           "splits": {}}
    log = MODELS / f"{ck_path.stem}.log.json"
    if log.exists():
        lj = json.loads(log.read_text())
        rep["train"] = {k: lj.get(k) for k in ("train_s", "peak_vram_gb", "best_val_cross_auc")}
    for split in ("val", "test"):
        recs = _eval_recs(split)
        ent = {"n_crops": len(recs), "n_matches": len({r["match_key"] for r in recs}),
               "n_identities": len({ident(r) for r in recs})}
        for name, E in (("frozen_cls_cache", frozen_cache_feats(recs)),
                        ("frozen_cls_prod", frozen_prod_feats(recs)),
                        ("s4", embed_records(net, recs, inset=bool(ck["cfg"].get("inset"))))):
            m = evaluate(E, recs)
            m["by_year"] = _by_year(recs, E)
            ent[name] = m
            print(_summ(f"{name} {split}", m), flush=True)
        rep["splits"][split] = ent
    del net
    torch.cuda.empty_cache()
    rep["latency"] = bench_latency(ck_path)
    rep["train_peak_vram_gb"] = rep.get("train", {}).get("peak_vram_gb")
    out = Path(a.out)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["latency"]), "->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", default="s4_embedder")
    t.add_argument("--detectors", default="robot")
    t.add_argument("--bumper-frac", type=float, default=0.0)
    t.add_argument("--p", type=int, default=48)
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--train-blocks", type=int, default=4)
    t.add_argument("--lr-backbone", type=float, default=3e-5)
    t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--tau", type=float, default=0.1)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--inset", action="store_true", help="train on the production region (box inset 8%%)")
    b = sub.add_parser("baseline")
    b.add_argument("--split", default="val", choices=["val", "test"])
    b.add_argument("--out", default=None)
    b.add_argument("--prod", action="store_true", help="production transform instead of the cache")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--which", default="proj", help="proj and/or cls (comma list)")
    e.add_argument("--out", default=None)
    rp = sub.add_parser("report")
    rp.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    rp.add_argument("--out", default=str(MODELS / "s4_report.json"))
    lt = sub.add_parser("latency")
    lt.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "baseline":
        cmd_baseline(a)
    elif a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "latency":
        print(json.dumps(bench_latency(Path(a.ckpt)), indent=1))


if __name__ == "__main__":
    main()
