#!/usr/bin/env python3
"""S1 roster-constrained bumper digit reader (design §5, §12).

Model: PARSeq-base (Apache-2.0, vendored in parseq.py) fine-tuned from the official pretrained
scene-text weights with a digits-only vocabulary. Input = the harvester's bumper crop (box + 25%
padding per side, BGR), stretched to 32x128 like upstream PARSeq.

Training data (§9 rule: only crops whose digits are visible):
  * Jev calls on bumper crops whose top answer is a roster team with p >= 0.5 (train split only);
  * gold b1/b3 bumper crops a judge named (conf >= 0.8), train split only;
  * synthetic bumpers (synth.py), ``--synth`` fraction of every epoch.
Never: test events, GT matches, val matches, labels propagated along a tracklet.

Decoding: teacher-forced log P(team digits, [E] | crop) for each roster team; "other" gets the
remaining sequence-probability mass (1 - sum over the roster). Optional bumper-colour prior scales
the off-alliance teams by ``colour_eps``. Abstain when the top option is "other", its probability
is below ``t_p``, or its margin over the runner-up is below ``t_m`` (tuned on val for <=0.5% wrong
among named crops).

  python -m scripts.teacher.students.digits train --out s1_digits.pt          # train + val tune
  python -m scripts.teacher.students.digits train --synth 0 --out s1_nosynth.pt --no-save-best
  python -m scripts.teacher.students.digits test --ckpt s1_digits.pt          # TEST, run once
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.events import split_of  # noqa: E402
from scripts.teacher.gold import _answers, gold_team, size_bucket  # noqa: E402

IMG_H, IMG_W = 32, 128
JEV_MIN_P = 0.5
OTHER = "other"
MAX_WRONG = 0.005  # wrong / named on val


def _paths():
    from scripts.teacher.students.heads import GOLD, MODELS, ROOT
    return ROOT, GOLD, MODELS


# ----------------------------------------------------------------------------- preprocessing
def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> float32 [3, 32, 128], RGB, normalised to [-1, 1] (upstream PARSeq)."""
    h, w = crop_bgr.shape[:2]
    interp = cv2.INTER_AREA if (w > IMG_W and h > IMG_H) else cv2.INTER_CUBIC
    x = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H), interpolation=interp)
    return ((x.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def augment_real(img: np.ndarray, rng: np.random.Generator, progressive: bool) -> np.ndarray:
    from scripts.teacher.students.synth import interlace_comb, jpeg
    h, w = img.shape[:2]
    # box jitter: the bumper sits in the middle 2/3 of the padded crop
    x0, x1 = int(rng.uniform(0, 0.10) * w), w - int(rng.uniform(0, 0.10) * w)
    y0, y1 = int(rng.uniform(0, 0.12) * h), h - int(rng.uniform(0, 0.12) * h)
    if x1 - x0 >= 8 and y1 - y0 >= 6:
        img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    if rng.random() < 0.3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.2 and w > 40:  # lower resolution
        s = rng.uniform(0.45, 0.85)
        small = cv2.resize(img, (max(8, int(w * s)), max(6, int(h * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.9))
    if progressive and rng.random() < 0.15:
        img = interlace_comb(img, rng, shift=int(rng.integers(1, 3)))
    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.random() < 0.3:
        img = jpeg(img, int(rng.integers(40, 90)))
    return img


# ----------------------------------------------------------------------------- data
def jev_rows(splits: Sequence[str] = ("train", "val", "test"), unreadable: bool = False) -> List[dict]:
    """Bumper crops Jev read as a roster team (p >= 0.5). ``unreadable=True`` instead returns the
    crops whose top Jev answer was unreadable / not_robot (p >= 0.5) - used only for evaluation."""
    ROOT, _, _ = _paths()
    out = []
    for lf in sorted(ROOT.glob("*/*/labels_jev.jsonl")):
        mk = lf.parent.name
        sp = split_of(mk)
        if sp not in splits:
            continue
        man = {}
        for l in (lf.parent / "manifest.jsonl").open():
            r = json.loads(l)
            man[r["path"]] = r
        for l in lf.open():
            t = json.loads(l)
            if t.get("detector") != "bumper":
                continue
            for c in t.get("calls", []):
                m = man.get(c["path"])
                top, p = c.get("top"), float(c.get("top_p") or 0.0)
                if m is None or p < JEV_MIN_P:
                    continue
                roster = m["roster"]
                teams = roster["red"] + roster["blue"]
                hit = top in teams if not unreadable else top in ("unreadable", "not_robot")
                if not hit:
                    continue
                out.append({"path": c["path"], "team": top if not unreadable else None, "jev_top": top,
                            "jev_p": p, "roster": roster, "colour": m.get("alliance_colour"),
                            "px_w": m["px_w"], "event": m["event"], "year": int(m["year"]),
                            "match_key": mk, "split": sp, "src": "jev"})
    return out


def gold_rows(batch: str) -> List[dict]:
    """Gold bumper items with the judges' agreed team (None when unsure / unnamed / conflicting)."""
    from scripts.teacher.students.heads import match_manifest
    _, GOLD, _ = _paths()
    g = GOLD / batch
    ans = _answers(g)
    out = []
    for l in (g / "key.jsonl").open():
        if not l.strip():
            continue
        k = json.loads(l)
        js = list(ans.get(k["id"], {}).values())
        if k["detector"] != "bumper" or not js:
            continue
        named = {gold_team(a) for a in js} - {None}
        team = named.pop() if len(named) == 1 else None
        m = match_manifest(k["event"], k["match_key"]).get(k["path"])
        if m is None:
            continue
        roster = m["roster"]
        legs = {a.get("legible") for a in js}
        out.append({"path": k["path"], "team": team, "conflict": len(named) > 1, "roster": roster,
                    "colour": k.get("colour") or m.get("alliance_colour"), "px_w": k["px_w"],
                    "event": k["event"], "year": int(k["event"][:4]), "match_key": k["match_key"],
                    "split": split_of(k["match_key"]), "src": f"gold_{batch}", "id": k["id"],
                    "jev_decision": k.get("jev_decision"), "jev_team": k.get("jev_team"),
                    "legible_none": legs == {"none"}, "kind": js[0].get("kind")})
    return out


def training_rows() -> List[dict]:
    rows = [r for r in jev_rows(("train",))]
    for b in ("b1", "b3"):
        for r in gold_rows(b):
            if r["split"] == "train" and r["team"] and r["team"] in r["roster"]["red"] + r["roster"]["blue"]:
                rows.append(r)
    return rows


def load_bytes(rows: Sequence[dict]) -> List[bytes]:
    ROOT, _, _ = _paths()
    return [(ROOT / r["path"]).read_bytes() for r in rows]


def decode_jpeg(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


class TrainSet:
    """Real crops (augmented) + synthetic bumpers; ``synth`` = synthetic fraction of an epoch."""

    def __init__(self, rows, blobs, backgrounds, synth: float, seed: int = 0):
        self.rows, self.blobs, self.bgs = rows, blobs, backgrounds
        self.n_real = len(rows)
        self.n_syn = int(round(self.n_real * synth / max(1e-6, 1 - synth))) if synth > 0 else 0
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.n_real + self.n_syn

    def __getitem__(self, i):
        import torch
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if i < self.n_real:
            r = self.rows[i]
            img = augment_real(decode_jpeg(self.blobs[i]), rng, progressive=r["year"] < 2026)
            label = r["team"]
        else:
            from scripts.teacher.students.synth import render
            bg = decode_jpeg(self.bgs[rng.integers(len(self.bgs))]) if self.bgs else None
            img, label = render(rng, bg)
        return torch.from_numpy(preprocess(img)), label


def _collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def _worker_init(_):
    cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- decoding
def candidate_probs(model, X, rosters: Sequence[dict], colours: Optional[Sequence[Optional[str]]] = None,
                    colour_eps: Optional[float] = None, with_read: bool = False):
    """Roster-constrained posterior per crop.

    Returns (teams: list of per-crop team lists, P: [N, K+1] numpy (last column = other), reads)."""
    import torch
    teams = [list(r["red"]) + list(r["blue"]) if isinstance(r, dict) else list(r) for r in rosters]
    K = max(len(t) for t in teams)
    cands = [t + [t[0]] * (K - len(t)) for t in teams]
    valid = torch.tensor([[j < len(t) for j in range(K)] for t in teams], device=X.device)
    with torch.no_grad():
        mem = model.encode(X)
        lp = model.score_strings(mem, cands)
        lp = lp.masked_fill(~valid, -1e9)
        reads = model.greedy(mem)[0] if with_read else None
    p = lp.double().exp().cpu().numpy()
    other = np.clip(1.0 - p.sum(1, keepdims=True), 1e-9, None)
    P = np.concatenate([p, other], 1)
    if colour_eps is not None and colours is not None:
        for i, (r, c) in enumerate(zip(rosters, colours)):
            if not isinstance(r, dict) or c not in ("red", "blue"):
                continue
            off = [j for j, t in enumerate(teams[i]) if t not in r[c]]
            P[i, off] *= colour_eps
    P = P / P.sum(1, keepdims=True)
    return teams, P.astype(np.float64), reads


def decide(P: np.ndarray, t_p: float, t_m: float) -> np.ndarray:
    """Index of the chosen team per row, or -1 (abstain). Column -1 is "other"."""
    s = np.sort(P, 1)
    top = P.argmax(1)
    ok = (top != P.shape[1] - 1) & (s[:, -1] >= t_p) & (s[:, -1] - s[:, -2] >= t_m)
    return np.where(ok, top, -1)


def tally(pred_idx: np.ndarray, teams: Sequence[List[str]], truth: Sequence[Optional[str]]) -> dict:
    c = Counter()
    for k, t, y in zip(pred_idx, teams, truth):
        if k < 0:
            c["abstain"] += 1
        elif t[k] == y:
            c["right"] += 1
        else:
            c["wrong"] += 1
    n = sum(c.values())
    named = c["right"] + c["wrong"]
    return {"n": n, "right": c["right"], "wrong": c["wrong"], "abstain": c["abstain"],
            "right_pct": round(100 * c["right"] / max(1, n), 2),
            "wrong_pct_of_named": round(100 * c["wrong"] / max(1, named), 3),
            "wrong_pct_of_all": round(100 * c["wrong"] / max(1, n), 3)}


THRESH_GRID_P = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
THRESH_GRID_M = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]


def tune_thresholds(P, teams, truth, max_wrong: float = MAX_WRONG) -> dict:
    best = None
    for tp in THRESH_GRID_P:
        for tm in THRESH_GRID_M:
            r = tally(decide(P, tp, tm), teams, truth)
            named = r["right"] + r["wrong"]
            if named and r["wrong"] / named <= max_wrong and (best is None or r["right"] > best[2]["right"]):
                best = (tp, tm, r)
    if best is None:
        return {"t_p": 0.999, "t_m": 0.98, "val": tally(decide(P, 0.999, 0.98), teams, truth)}
    return {"t_p": best[0], "t_m": best[1], "val": best[2]}


# ----------------------------------------------------------------------------- model io
def build_model(cfg: dict):
    from scripts.teacher.students.parseq import PARSeq
    return PARSeq(**cfg)


def load_ckpt(path_or_dict, device: str = "cpu"):
    import torch
    ck = path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict, map_location="cpu",
                                                                        weights_only=True)
    model = build_model(ck["config"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return model.to(device).eval(), ck


class DigitReader:
    """S1 inference: ``read(crops_bgr, rosters, colours)`` -> per crop
    {"probs": {team: p, ..., "other": p}, "team": str|None, "p": float, "abstain": bool, "read": str}.

    ``rosters``: per crop {"red": [...], "blue": [...]} (or a flat team list, which disables the colour
    prior for that crop). ``colours``: per crop "red"/"blue"/None (bumper class). ``use_colour``
    defaults to the checkpoint's choice."""

    def __init__(self, ckpt=None, device: Optional[str] = None, batch: int = 256):
        import torch
        if ckpt is None:
            ckpt = _paths()[2] / "s1_digits.pt"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ck = load_ckpt(ckpt, self.device)
        self.half = self.device.startswith("cuda")
        if self.half:
            self.model.half()
        self.batch = batch
        th = self.ck["thresholds"]
        self.colour_eps = float(self.ck.get("colour_eps", 0.05))
        self.use_colour = bool(self.ck.get("use_colour", True))
        self.t = {True: (th["colour"]["t_p"], th["colour"]["t_m"]), False: (th["plain"]["t_p"], th["plain"]["t_m"])}

    def probs(self, crops_bgr, rosters, colours=None, use_colour=None, with_read=False):
        import torch
        use_colour = self.use_colour if use_colour is None else use_colour
        colours = list(colours) if colours is not None else [None] * len(crops_bgr)
        teams_all, P_all, reads_all = [], [], []
        for s in range(0, len(crops_bgr), self.batch):
            X = torch.from_numpy(np.stack([preprocess(c) for c in crops_bgr[s:s + self.batch]])).to(self.device)
            X = X.half() if self.half else X
            t, P, rd = candidate_probs(self.model, X, rosters[s:s + self.batch], colours[s:s + self.batch],
                                       self.colour_eps if use_colour else None, with_read)
            teams_all += t
            P_all += list(P)
            reads_all += rd or [None] * len(t)
        return teams_all, P_all, reads_all

    def read(self, crops_bgr: Sequence[np.ndarray], rosters, colours=None, use_colour=None) -> List[dict]:
        if not len(crops_bgr):
            return []
        use_colour = self.use_colour if use_colour is None else use_colour
        teams, P, reads = self.probs(list(crops_bgr), list(rosters), colours, use_colour, with_read=True)
        t_p, t_m = self.t[bool(use_colour)]
        out = []
        for tm, p, rd in zip(teams, P, reads):
            k = int(decide(p[None], t_p, t_m)[0])
            j = int(p.argmax())
            probs = {t: float(p[i]) for i, t in enumerate(tm)}
            probs[OTHER] = float(p[-1])
            out.append({"probs": probs, "team": tm[k] if k >= 0 else None,
                        "p": float(p[j]), "abstain": k < 0, "read": rd})
        return out


# ----------------------------------------------------------------------------- evaluation
def eval_rows(model, rows, blobs, device, colour_eps=None, batch=256, with_read=False):
    import torch
    teams_all, P_all, reads_all = [], [], []
    for s in range(0, len(rows), batch):
        X = torch.from_numpy(np.stack([preprocess(decode_jpeg(b)) for b in blobs[s:s + batch]])).to(device)
        rs = rows[s:s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            t, P, rd = candidate_probs(model, X, [r["roster"] for r in rs], [r["colour"] for r in rs],
                                       colour_eps, with_read)
        teams_all += t
        P_all.append(P)
        reads_all += rd or [None] * len(t)
    return teams_all, np.concatenate(P_all) if P_all else np.zeros((0, 7)), reads_all


def by_bucket(pred, teams, rows) -> dict:
    out = {}
    for b in ("s", "m", "l"):
        idx = [i for i, r in enumerate(rows) if size_bucket(r["px_w"]) == b]
        out[b] = tally(pred[idx], [teams[i] for i in idx], [rows[i]["team"] for i in idx])
    return out


def by_key(pred, teams, rows, key) -> dict:
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(key(r))].append(i)
    return {g: tally(pred[idx], [teams[i] for i in idx], [rows[i]["team"] for i in idx])
            for g, idx in sorted(groups.items())}


def named_rate(pred) -> dict:
    n = len(pred)
    k = int((pred >= 0).sum())
    return {"n": n, "named": k, "named_pct": round(100 * k / max(1, n), 2)}


# ----------------------------------------------------------------------------- training
def cmd_train(a) -> None:
    import torch
    from scripts.teacher.students.parseq import BASE_CFG, PARSeq, plm_loss
    _, _, MODELS = _paths()
    torch.manual_seed(a.seed)
    dev = a.device
    rows = training_rows()
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)]
    val = jev_rows(("val",))
    if a.val_limit:
        val = val[:: max(1, len(val) // a.val_limit)]
    print(f"train rows {len(rows)} ({Counter(r['src'] for r in rows)}), val rows {len(val)}", flush=True)
    t0 = time.time()
    blobs = load_bytes(rows)
    vblobs = load_bytes(val)
    bgs = []
    if a.synth > 0:  # backgrounds: bumper-detector crops Jev called not_robot (no digits), train split
        neg = [r for r in jev_rows(("train",), unreadable=True) if r["jev_top"] == "not_robot"]
        rng = np.random.default_rng(a.seed)
        neg = [neg[i] for i in rng.permutation(len(neg))[:4000]]
        bgs = load_bytes(neg)
    print(f"loaded {len(blobs)}+{len(vblobs)} crops, {len(bgs)} backgrounds in {time.time() - t0:.0f}s", flush=True)

    sd = torch.load(a.pretrained, map_location="cpu", weights_only=True) if a.pretrained else None
    cfg = dict(BASE_CFG, max_label_length=6)
    model = PARSeq.from_pretrained(sd, **cfg).to(dev)
    ds = TrainSet(rows, blobs, bgs, a.synth, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                     collate_fn=_collate, drop_last=True, persistent_workers=False,
                                     worker_init_fn=_worker_init, pin_memory=True)
    steps = a.epochs * len(dl)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "pos_" in n or "embed" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.06,
                                                cycle_momentum=False)
    rng = np.random.default_rng(a.seed)
    step, hist = 0, []
    for ep in range(a.epochs):
        ds.epoch = ep
        model.train()
        tl, tn, te = 0.0, 0, time.time()
        for X, y in dl:
            X = X.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                loss = plm_loss(model, X, y, rng=rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            sched.step()
            step += 1
            tl += loss.item()
            tn += 1
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {tl / tn:.4f} {tn * a.bs / (time.time() - te):.0f} img/s",
                      flush=True)
        model.eval()
        teams, P, _ = eval_rows(model, val, vblobs, dev)
        top1 = tally(decide(P, 0.0, 0.0), teams, [r["team"] for r in val])
        th = tune_thresholds(P, teams, [r["team"] for r in val])
        hist.append({"epoch": ep, "loss": tl / max(1, tn), "val_top1": top1, "val_tuned": th})
        print(f"epoch {ep} loss {tl / max(1, tn):.4f} val top1 {top1['right']}/{top1['n']} "
              f"tuned t_p={th['t_p']} t_m={th['t_m']} -> {th['val']} ({time.time() - te:.0f}s)", flush=True)

    # final val report + thresholds (plain and colour prior)
    seen = {r["team"] for r in rows}
    truth = [r["team"] for r in val]
    res = {"train_rows": len(rows), "train_sources": dict(Counter(r["src"] for r in rows)),
           "synth": a.synth, "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "history": hist}
    thresholds = {}
    for name, eps in (("plain", None), ("colour", a.colour_eps), ("colour_hard", 0.0)):
        teams, P, _ = eval_rows(model, val, vblobs, dev, colour_eps=eps)
        th = tune_thresholds(P, teams, truth)
        pred = decide(P, th["t_p"], th["t_m"])
        thresholds[name] = {"t_p": th["t_p"], "t_m": th["t_m"]}
        res[f"val_{name}"] = {"thresholds": thresholds[name], "all": tally(pred, teams, truth),
                              "top1_no_abstain": tally(decide(P, 0.0, 0.0), teams, truth),
                              "by_size": by_bucket(pred, teams, val),
                              "by_year": by_key(pred, teams, val, lambda r: r["year"]),
                              "team_seen_in_train": by_key(pred, teams, val, lambda r: r["team"] in seen)}
        print(name, json.dumps(res[f"val_{name}"]["all"]), flush=True)
    # hallucination check: val crops Jev called unreadable / not_robot
    unr = jev_rows(("val",), unreadable=True)
    ublobs = load_bytes(unr)
    for name, eps in (("plain", None), ("colour", a.colour_eps)):
        teams, P, _ = eval_rows(model, unr, ublobs, dev, colour_eps=eps)
        pred = decide(P, thresholds[name]["t_p"], thresholds[name]["t_m"])
        res[f"val_jev_unreadable_{name}"] = {
            k: named_rate(pred[[i for i, r in enumerate(unr) if r["jev_top"] == k]])
            for k in ("unreadable", "not_robot")}
    print("jev-unreadable named:", json.dumps({k: v for k, v in res.items() if k.startswith("val_jev_unr")}))
    ck = {"config": cfg, "state_dict": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
          "thresholds": thresholds, "colour_eps": a.colour_eps, "use_colour": True,
          "img_size": [IMG_H, IMG_W], "pad": 0.25, "arch": "parseq-base digits", "max_wrong": MAX_WRONG,
          "train_teams": sorted(seen), "report": json.loads(json.dumps(res, default=float))}
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / a.out
    torch.save(ck, out)
    (MODELS / (Path(a.out).stem + "_val.json")).write_text(json.dumps(res, indent=1, default=float))
    print(f"saved {out}")


# ----------------------------------------------------------------------------- TEST (once)
def latency(model, device, n: int = 512, batch: int = 64) -> dict:
    import torch
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (int(rng.integers(30, 90)), int(rng.integers(60, 200)), 3), np.uint8)
             for _ in range(n)]
    rosters = [{"red": ["254", "1678", "118"], "blue": ["10254", "971", "6800"]}] * n
    model.half()
    out = {}
    for with_read in (False, True):
        for _ in range(2):
            t0 = time.time()
            for s in range(0, n, batch):
                X = torch.from_numpy(np.stack([preprocess(c) for c in crops[s:s + batch]])).to(device).half()
                candidate_probs(model, X, rosters[s:s + batch], None, None, with_read)
            torch.cuda.synchronize()
            dt = time.time() - t0
        out["ms_per_crop" + ("_with_free_read" if with_read else "")] = round(1000 * dt / n, 3)
    model.float()
    out["batch"] = batch
    return out


def cmd_test(a) -> None:
    import torch
    _, _, MODELS = _paths()
    model, ck = load_ckpt(MODELS / a.ckpt, a.device)
    th = ck["thresholds"]
    eps = ck["colour_eps"]
    rep = {"ckpt": a.ckpt, "val": {k: v for k, v in ck["report"].items() if k.startswith("val_")},
           "synth": ck["report"].get("synth"), "thresholds": th, "colour_eps": eps}
    test = jev_rows(("test",))
    tb = load_bytes(test)
    gold = [r for r in gold_rows("b2")]
    gb = load_bytes(gold)
    for name, e in (("plain", None), ("colour", eps), ("colour_hard", 0.0)):
        t = th[name]
        teams, P, _ = eval_rows(model, test, tb, a.device, colour_eps=e)
        pred = decide(P, t["t_p"], t["t_m"])
        truth = [r["team"] for r in test]
        rep[f"test_jev_{name}"] = {"all": tally(pred, teams, truth), "by_size": by_bucket(pred, teams, test),
                                   "by_event": by_key(pred, teams, test, lambda r: r["event"]),
                                   "top1_no_abstain": tally(decide(P, 0.0, 0.0), teams, truth)}
        teams, P, reads = eval_rows(model, gold, gb, a.device, colour_eps=e, with_read=True)
        pred = decide(P, t["t_p"], t["t_m"])
        named = [i for i, r in enumerate(gold) if r["team"]]
        sub = lambda idx: (pred[idx], [teams[i] for i in idx], [gold[i] for i in idx])  # noqa: E731
        g = {}
        p_, t_, r_ = sub(named)
        g["judge_named"] = {"all": tally(p_, t_, [r["team"] for r in r_]), "by_size": by_bucket(p_, t_, r_),
                            "by_event": by_key(p_, t_, r_, lambda r: r["event"])}
        # Jev's tracklet label on the same crops, for comparison
        jc = Counter()
        for r in r_:
            jc["none" if not r["jev_team"] else ("right" if r["jev_team"] == r["team"] else "wrong")] += 1
        g["judge_named_jev_tracklet_label"] = dict(jc)
        for nm, cond in (("jev_unsure", lambda r: r["jev_decision"] == "unsure"),
                         ("jev_no_team", lambda r: not r["jev_team"])):
            idx = [i for i in named if cond(gold[i])]
            p_, t_, r_ = sub(idx)
            g[f"judge_named_{nm}"] = {"all": tally(p_, t_, [r["team"] for r in r_]),
                                      "by_size": by_bucket(p_, t_, r_)}
        # crops judges could not name: how often does S1 still name a team?
        un = [i for i, r in enumerate(gold) if not r["team"] and not r["conflict"]]
        g["judge_unnamed"] = {
            "all": named_rate(pred[un]),
            "legible_none": named_rate(pred[[i for i in un if gold[i]["legible_none"]]]),
            "kind_bumper_or_robot": named_rate(pred[[i for i in un if gold[i]["kind"] in ("bumper", "robot")]])}
        rep[f"gold_b2_{name}"] = g
        if name == "colour":
            rep["gold_b2_examples_wrong"] = [
                {"id": gold[i]["id"], "path": gold[i]["path"], "judge": gold[i]["team"],
                 "s1": teams[i][pred[i]], "read": reads[i], "px_w": gold[i]["px_w"]}
                for i in named if pred[i] >= 0 and teams[i][pred[i]] != gold[i]["team"]]
        print(name, json.dumps(rep[f"test_jev_{name}"]["all"]), json.dumps(g["judge_named"]["all"]),
              json.dumps(g["judge_named_jev_unsure"]["all"]), flush=True)
    if a.device.startswith("cuda"):
        rep["latency"] = latency(model, a.device)
        print("latency", rep["latency"])
    if a.ablation:
        rep["synth_ablation_val"] = {}
        for f in a.ablation.split(","):
            v = json.loads((MODELS / (Path(f).stem + "_val.json")).read_text())
            rep["synth_ablation_val"][f] = {"synth": v["synth"],
                                           **{k: v[k]["all"] for k in v if k in ("val_plain", "val_colour")}}
    out = MODELS / a.report
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default="s1_digits.pt")
    t.add_argument("--pretrained", default=None, help="local parseq-bb5792a6.pt (default: download)")
    t.add_argument("--synth", type=float, default=0.3)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--bs", type=int, default=192)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--colour-eps", type=float, default=0.05)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--val-limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("test", help="held-out TEST events + gold b2 (run once)")
    e.add_argument("--ckpt", default="s1_digits.pt")
    e.add_argument("--report", default="s1_report.json")
    e.add_argument("--ablation", default="", help="comma list of other ckpt names whose _val.json to include")
    e.add_argument("--device", default="cuda")
    e.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
