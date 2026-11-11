#!/usr/bin/env python3
"""S1 roster-constrained bumper digit reader (design §5, §12).

Model: PARSeq-base (Apache-2.0, vendored in web/students/parseq.py) fine-tuned from the official pretrained
scene-text weights with a digits-only vocabulary. Input = the harvester's bumper crop (box + 25%
padding per side, BGR), stretched to 32x128 like upstream PARSeq.

Training data (§9 rule: only crops whose digits are visible):
  * Jev calls on bumper crops whose top answer is a roster team with p >= 0.5 (train split only);
  * gold b1/b3 bumper crops a judge named (conf >= 0.8), train split only;
  * synthetic bumpers (synth.py), ``--synth`` fraction of every epoch.
Never: test events, GT matches, val matches, labels propagated along a tracklet.

Decoding: teacher-forced log P(team digits, [E] | crop) for each roster team (``score_batch``), then a
decision rule (``posterior`` / ``decide``) tuned on val:
  * "softmax": softmax(LP / T) over the 6 roster numbers (the nearest roster number wins - the reader
    often free-reads one digit off, e.g. "9473" for 2473), abstain below top-1 ``t_p`` / margin ``t_m``
    or when the best raw log-likelihood is under ``floor``;
  * "mass": P(team) = sequence probability, "other" = 1 - sum over the roster (stricter).
Optional bumper-colour prior multiplies off-alliance teams by ``colour_eps``. The rule is chosen
on val for <=0.5% wrong of named Jev-read crops and <=1% wrong vs the tracklet label on val crops
Jev never read (propagated labels are ~99% pure; evaluation only, never training).

  python -m scripts.teacher.students.digits train --out s1_digits.pt          # train + val tune (v1: trim 0.1, synth 0.3)
  python -m scripts.teacher.students.digits train --synth 0 --out s1_nosynth.pt     # ablation
  python -m scripts.teacher.students.digits retune --ckpt s1_digits.pt         # re-pick rule on val
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
def preprocess(crop_bgr: np.ndarray, trim: float = 0.0) -> np.ndarray:
    """BGR crop -> float32 [3, 32, 128], RGB, normalised to [-1, 1] (upstream PARSeq).
    ``trim`` cuts that fraction of the padded crop off every side first (the bumper box itself
    spans 1/6..5/6 of the harvester's 25%-padded crop)."""
    h, w = crop_bgr.shape[:2]
    if trim > 0:
        dy, dx = int(h * trim), int(w * trim)
        if h - 2 * dy >= 4 and w - 2 * dx >= 4:
            crop_bgr = crop_bgr[dy:h - dy, dx:w - dx]
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
                            "match_key": mk, "split": sp, "src": "jev",
                            "tracklet_team": t.get("team") if t.get("decision") == "team" else None})
    return out


def propagated_rows(splits: Sequence[str] = ("val",)) -> List[dict]:
    """Bumper crops Jev did NOT read, in tracklets Jev decided as a team (label = tracklet team).
    Evaluation only (a mix of legible and illegible crops - the population S1 sees live); never
    training data (design §9)."""
    ROOT, _, _ = _paths()
    out = []
    for lf in sorted(ROOT.glob("*/*/labels_jev.jsonl")):
        mk = lf.parent.name
        sp = split_of(mk)
        if sp not in splits:
            continue
        lab = {}
        for l in lf.open():
            t = json.loads(l)
            if t.get("detector") == "bumper" and t.get("decision") == "team" and t.get("team"):
                lab[t["tracklet_id"]] = (t["team"], {c["path"] for c in t.get("calls", [])})
        for l in (lf.parent / "manifest.jsonl").open():
            m = json.loads(l)
            if m["detector"] != "bumper" or m["tracklet_id"] not in lab:
                continue
            team, called = lab[m["tracklet_id"]]
            if m["path"] in called or team not in m["roster"]["red"] + m["roster"]["blue"]:
                continue
            out.append({"path": m["path"], "team": team, "roster": m["roster"], "colour": m.get("alliance_colour"),
                        "px_w": m["px_w"], "event": m["event"], "year": int(m["year"]), "match_key": mk,
                        "split": sp, "src": "propagated"})
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

    def __init__(self, rows, blobs, backgrounds, synth: float, seed: int = 0, trim: float = 0.0):
        self.rows, self.blobs, self.bgs, self.trim = rows, blobs, backgrounds, trim
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
        return torch.from_numpy(preprocess(img, self.trim)), label


def _collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def _worker_init(_):
    cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- decoding
def roster_teams(r) -> List[str]:
    return list(r["red"]) + list(r["blue"]) if isinstance(r, dict) else list(r)


def score_batch(model, X, rosters: Sequence, with_read: bool = False):
    """Teacher-forced log P(team digits [E] | crop) per roster team.

    Returns (teams per crop, LP [N, K] float64 numpy with -1e9 in padding slots, greedy reads)."""
    import torch
    teams = [roster_teams(r) for r in rosters]
    K = max(len(t) for t in teams)
    cands = [t + [t[0]] * (K - len(t)) for t in teams]
    valid = torch.tensor([[j < len(t) for j in range(K)] for t in teams], device=X.device)
    with torch.no_grad():
        mem = model.encode(X)
        lp = model.score_strings(mem, cands).masked_fill(~valid, -1e9)
        reads = model.greedy(mem)[0] if with_read else None
    return teams, lp.double().cpu().numpy(), reads


def colour_logprior(teams, rosters, colours, eps: Optional[float]) -> np.ndarray:
    """log prior per candidate: 0 for the bumper's alliance (or unknown colour), log(eps) off-alliance."""
    K = max(len(t) for t in teams) if teams else 0
    out = np.zeros((len(teams), K))
    if eps is None or colours is None:
        return out
    leps = math.log(eps) if eps > 0 else -1e9
    for i, (t, r, c) in enumerate(zip(teams, rosters, colours)):
        if isinstance(r, dict) and c in ("red", "blue"):
            for j, team in enumerate(t):
                if team not in r[c]:
                    out[i, j] = leps
    return out


def posterior(LP: np.ndarray, logprior: np.ndarray, rule: dict) -> np.ndarray:
    """[N, K+1] probabilities; the last column is "other".

    mode "mass":    P(team) = P(string), P(other) = 1 - sum over the roster (the reader must
                    put most of its sequence mass on one roster number to name it).
    mode "softmax": softmax(LP / T) over the roster only (nearest roster number wins, e.g. a read
                    "9473" still names 2473 when that is the only close candidate); P(other) = 0, the
                    absolute evidence check is ``floor`` on the best raw log-likelihood."""
    if rule["mode"] == "mass":
        p = np.exp(LP + logprior)
        other = np.clip(1.0 - np.exp(LP).sum(1, keepdims=True), 1e-12, None)
        P = np.concatenate([p, other], 1)
    else:
        z = LP / rule["T"] + logprior
        z = z - z.max(1, keepdims=True)
        P = np.concatenate([np.exp(z), np.zeros((len(z), 1))], 1)
    return P / P.sum(1, keepdims=True)


def decide(P: np.ndarray, t_p: float, t_m: float, maxlp: Optional[np.ndarray] = None,
           floor: float = -1e9) -> np.ndarray:
    """Index of the chosen team per row, or -1 (abstain). Column -1 is "other"."""
    s = np.sort(P, 1)
    top = P.argmax(1)
    ok = (top != P.shape[1] - 1) & (s[:, -1] >= t_p) & (s[:, -1] - s[:, -2] >= t_m)
    if maxlp is not None:
        ok &= maxlp >= floor
    return np.where(ok, top, -1)


def apply_rule(LP, teams, rosters, colours, rule: dict) -> np.ndarray:
    P = posterior(LP, colour_logprior(teams, rosters, colours, rule.get("colour_eps")), rule)
    return decide(P, rule["t_p"], rule["t_m"], LP.max(1), rule.get("floor", -1e9)), P


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


def _fast_tally(pred: np.ndarray, truth_idx: np.ndarray) -> dict:
    right = int(((pred >= 0) & (pred == truth_idx)).sum())
    named = int((pred >= 0).sum())
    return {"n": len(pred), "right": right, "wrong": named - right, "abstain": len(pred) - named}


GRID_P = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
GRID_M = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
GRID_RULES = [("mass", 1.0, -1e9)] + [("softmax", T, f) for T in (1.0, 2.0, 4.0) for f in (-1e9, -12.0, -8.0, -5.0, -3.0)]
PROP_MAX_WRONG = 0.01  # propagated tracklet labels are ~99% pure (gold §9/§10)


def tune_rule(ev: dict, colour_eps: Optional[float], max_wrong: float = MAX_WRONG,
              modes: Sequence[str] = ("mass", "softmax")) -> dict:
    """Most val-right rule with wrong <= max_wrong of named on Jev-read val crops and, when
    ``ev['prop']`` is given, <= PROP_MAX_WRONG of named on val crops Jev never read (tracklet label).
    ``ev``: {"val": (LP, teams, rosters, colours, truth), "prop": (...) or None}."""
    best = None
    for mode, T, floor in GRID_RULES:
        if mode not in modes:
            continue
        pre = {}
        for k, v in ev.items():
            if v is None:
                continue
            LP, teams, rosters, colours, truth = v
            base = {"mode": mode, "T": T, "colour_eps": colour_eps}
            ti = np.array([t.index(y) if y in t else -2 for t, y in zip(teams, truth)])
            pre[k] = (posterior(LP, colour_logprior(teams, rosters, colours, colour_eps), base), LP.max(1), ti)
        for tp in GRID_P:
            for tm in GRID_M:
                res = {k: _fast_tally(decide(P, tp, tm, mx, floor), ti) for k, (P, mx, ti) in pre.items()}
                r = res["val"]
                named = r["right"] + r["wrong"]
                if not named or r["wrong"] / named > max_wrong:
                    continue
                if "prop" in res:
                    pn = res["prop"]["right"] + res["prop"]["wrong"]
                    if pn and res["prop"]["wrong"] / pn > PROP_MAX_WRONG:
                        continue
                if best is None or r["right"] > best[1]["val"]["right"]:
                    best = ({"mode": mode, "T": T, "floor": floor, "t_p": tp, "t_m": tm, "colour_eps": colour_eps}, res)
    if best is None:
        rule = {"mode": "mass", "T": 1.0, "floor": -1e9, "t_p": 0.999, "t_m": 0.98, "colour_eps": colour_eps}
        return {"rule": rule, "metrics": {}}
    return {"rule": best[0], "metrics": best[1]}


# ----------------------------------------------------------------------------- model io
def build_model(cfg: dict):
    from web.students.parseq import PARSeq
    return PARSeq(**cfg)


def load_ckpt(path_or_dict, device: str = "cpu"):
    import torch
    ck = path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict, map_location="cpu",
                                                                        weights_only=True)
    model = build_model(ck["config"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model.trim = float(ck.get("trim", 0.0))
    return model.to(device).eval(), ck


class DigitReader:
    """S1 inference: ``read(crops_bgr, rosters, colours)`` -> per crop
    {"probs": {team: p, ..., "other": p}, "team": str|None, "p": float, "abstain": bool,
     "read": free-decoded digit string, "roster_mass": sum of the roster numbers' sequence probability}.

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
        self.rules = self.ck["rules"]
        self.trim = float(self.ck.get("trim", 0.0))
        self.use_colour = bool(self.ck.get("use_colour", True))

    def scores(self, crops_bgr, rosters, with_read=False):
        import torch
        teams_all, LP_all, reads_all = [], [], []
        for s in range(0, len(crops_bgr), self.batch):
            X = torch.from_numpy(np.stack([preprocess(c, self.trim) for c in crops_bgr[s:s + self.batch]])).to(self.device)
            X = X.half() if self.half else X
            t, LP, rd = score_batch(self.model, X, rosters[s:s + self.batch], with_read)
            teams_all += t
            LP_all += list(LP)
            reads_all += rd or [None] * len(t)
        return teams_all, LP_all, reads_all

    def read(self, crops_bgr: Sequence[np.ndarray], rosters, colours=None, use_colour=None) -> List[dict]:
        if not len(crops_bgr):
            return []
        rosters = list(rosters)
        colours = list(colours) if colours is not None else [None] * len(rosters)
        use_colour = self.use_colour if use_colour is None else use_colour
        rule = self.rules["colour" if use_colour else "plain"]
        teams, LPs, reads = self.scores(list(crops_bgr), rosters, with_read=True)
        out = []
        for t, lp, r, c, rd in zip(teams, LPs, rosters, colours, reads):
            k_, P = apply_rule(lp[None, :len(t)], [t], [r], [c], rule)
            k, p = int(k_[0]), P[0]
            probs = {team: float(p[i]) for i, team in enumerate(t)}
            probs[OTHER] = float(p[-1])
            out.append({"probs": probs, "team": t[k] if k >= 0 else None, "p": float(p.max()),
                        "abstain": k < 0, "read": rd, "roster_mass": float(np.exp(lp[:len(t)]).sum())})
        return out


# ----------------------------------------------------------------------------- evaluation
def eval_rows(model, rows, blobs, device, batch=256, with_read=False):
    """-> (LP [N, K], teams, rosters, colours, truth, reads)."""
    import torch
    teams_all, LP_all, reads_all = [], [], []
    for s in range(0, len(rows), batch):
        X = torch.from_numpy(np.stack([preprocess(decode_jpeg(b), getattr(model, 'trim', 0.0))
                                       for b in blobs[s:s + batch]])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            t, LP, rd = score_batch(model, X, [r["roster"] for r in rows[s:s + batch]], with_read)
        teams_all += t
        LP_all.append(LP)
        reads_all += rd or [None] * len(t)
    LP = np.concatenate(LP_all) if LP_all else np.zeros((0, 6))
    return (LP, teams_all, [r["roster"] for r in rows], [r["colour"] for r in rows],
            [r.get("team") for r in rows], reads_all)


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


VARIANTS = (("plain", None), ("colour", "eps"), ("colour_hard", 0.0))


def val_report(model, dev, colour_eps: float, seen: set, cache: dict) -> tuple:
    """Tune one rule per colour variant on val (+ propagated val crops) and report val metrics."""
    for k, fn in (("val", lambda: jev_rows(("val",))), ("prop", lambda: propagated_rows(("val",))),
                  ("unr", lambda: jev_rows(("val",), unreadable=True))):
        if k not in cache:
            rows = fn()
            cache[k] = (rows, load_bytes(rows))
    ev = {k: eval_rows(model, cache[k][0], cache[k][1], dev) for k in ("val", "prop", "unr")}
    rules, res = {}, {}
    for name, eps in VARIANTS:
        eps = colour_eps if eps == "eps" else eps
        tuned = tune_rule({"val": ev["val"][:5], "prop": ev["prop"][:5]}, eps)
        rule = tuned["rule"]
        rules[name] = rule
        out = {"rule": rule}
        for k in ("val", "prop"):
            LP, teams, rosters, colours, truth, _ = ev[k]
            pred, _ = apply_rule(LP, teams, rosters, colours, rule)
            rows = cache[k][0]
            out[k] = {"all": tally(pred, teams, truth), "by_size": by_bucket(pred, teams, rows),
                      "by_year": by_key(pred, teams, rows, lambda r: r["year"])}
            if k == "val":
                out[k]["team_seen_in_train"] = by_key(pred, teams, rows, lambda r: r["team"] in seen)
        # the earlier, stricter "mass" rule on the same val set, for comparison
        mass = tune_rule({"val": ev["val"][:5], "prop": ev["prop"][:5]}, eps, modes=("mass",))
        LP, teams, rosters, colours, truth, _ = ev["val"]
        out["val_mass_rule"] = {"rule": mass["rule"],
                                "all": tally(apply_rule(LP, teams, rosters, colours, mass["rule"])[0], teams, truth)}
        LP, teams, rosters, colours, _, _ = ev["unr"]
        pred, _ = apply_rule(LP, teams, rosters, colours, rule)
        urows = cache["unr"][0]
        out["jev_unreadable_named"] = {j: named_rate(pred[[i for i, r in enumerate(urows) if r["jev_top"] == j]])
                                       for j in ("unreadable", "not_robot")}
        res[name] = out
        print(name, json.dumps(rule), "val", json.dumps(out["val"]["all"]), "prop", json.dumps(out["prop"]["all"]),
              "unr", json.dumps(out["jev_unreadable_named"]), flush=True)
    return rules, res


# ----------------------------------------------------------------------------- training
def cmd_train(a) -> None:
    import torch
    from web.students.parseq import BASE_CFG, PARSeq
    from scripts.teacher.students.parseq_plm import plm_loss
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
    model.trim = a.trim
    ds = TrainSet(rows, blobs, bgs, a.synth, seed=a.seed, trim=a.trim)
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
    soft = {"mode": "softmax", "T": 2.0, "floor": -1e9, "colour_eps": None}
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
        LP, teams, rosters, colours, truth, _ = eval_rows(model, val, vblobs, dev)
        top1 = tally(apply_rule(LP, teams, rosters, colours, dict(soft, t_p=0.0, t_m=0.0))[0], teams, truth)
        hist.append({"epoch": ep, "loss": tl / max(1, tn), "val_top1": top1})
        print(f"epoch {ep} loss {tl / max(1, tn):.4f} val roster top-1 {top1['right']}/{top1['n']} "
              f"({time.time() - te:.0f}s)", flush=True)

    seen = {r["team"] for r in rows}
    rules, vres = val_report(model, dev, a.colour_eps, seen, {})
    res = {"train_rows": len(rows), "train_sources": dict(Counter(r["src"] for r in rows)),
           "synth": a.synth, "trim": a.trim, "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "history": hist,
           **{f"val_{k}": v for k, v in vres.items()}}
    ck = {"config": cfg, "state_dict": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
          "rules": rules, "colour_eps": a.colour_eps, "use_colour": True, "trim": a.trim,
          "img_size": [IMG_H, IMG_W], "pad": 0.25, "arch": "parseq-base digits", "max_wrong": MAX_WRONG,
          "train_teams": sorted(seen), "report": json.loads(json.dumps(res, default=float))}
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / a.out
    torch.save(ck, out)
    (MODELS / (Path(a.out).stem + "_val.json")).write_text(json.dumps(res, indent=1, default=float))
    print(f"saved {out}")


def cmd_retune(a) -> None:
    """Re-tune the decision rules of a trained checkpoint on val (no TEST data touched)."""
    import torch
    _, _, MODELS = _paths()
    model, ck = load_ckpt(MODELS / a.ckpt, a.device)
    seen = set(ck.get("train_teams", []))
    rules, vres = val_report(model, a.device, ck.get("colour_eps", 0.05), seen, {})
    rep = dict(ck.get("report", {}))
    rep = {k: v for k, v in rep.items() if not k.startswith("val_")}
    rep.update({f"val_{k}": v for k, v in vres.items()})
    ck.update(rules=rules, report=json.loads(json.dumps(rep, default=float)))
    ck.pop("thresholds", None)
    torch.save(ck, MODELS / a.ckpt)
    (MODELS / (Path(a.ckpt).stem + "_val.json")).write_text(json.dumps(rep, indent=1, default=float))
    print(f"re-tuned {a.ckpt}")


# ----------------------------------------------------------------------------- TEST (once)
def latency(model, device, n: int = 1024, batch: int = 128) -> dict:
    import torch
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (int(rng.integers(30, 90)), int(rng.integers(60, 200)), 3), np.uint8)
             for _ in range(n)]
    rosters = [{"red": ["254", "1678", "118"], "blue": ["10254", "971", "6800"]}] * n
    model.half()
    out = {"batch": batch, "dtype": "fp16"}
    for with_read in (False, True):
        for _ in range(2):  # second pass timed (warm)
            t_gpu, t0 = 0.0, time.time()
            for s in range(0, n, batch):
                X = torch.from_numpy(np.stack([preprocess(c, getattr(model, 'trim', 0.0)) for c in crops[s:s + batch]])).to(device).half()
                torch.cuda.synchronize()
                g0 = time.time()
                score_batch(model, X, rosters[s:s + batch], with_read)
                torch.cuda.synchronize()
                t_gpu += time.time() - g0
            dt = time.time() - t0
        sfx = "_with_free_read" if with_read else ""
        out["ms_per_crop_total" + sfx] = round(1000 * dt / n, 3)
        out["ms_per_crop_model" + sfx] = round(1000 * t_gpu / n, 3)
    model.float()
    return out


def cmd_test(a) -> None:
    _, _, MODELS = _paths()
    model, ck = load_ckpt(MODELS / a.ckpt, a.device)
    rules = ck["rules"]
    rep = {"ckpt": a.ckpt, "synth": ck["report"].get("synth"), "rules": rules,
           "val": {k: v for k, v in ck["report"].items() if k.startswith("val_")}}
    test = jev_rows(("test",))
    ev_t = eval_rows(model, test, load_bytes(test), a.device)
    prop = propagated_rows(("test",))
    ev_p = eval_rows(model, prop, load_bytes(prop), a.device)
    gold = gold_rows("b2")
    ev_g = eval_rows(model, gold, load_bytes(gold), a.device, with_read=True)
    for name, _ in VARIANTS:
        rule = rules[name]
        LP, teams, rosters, colours, truth, _ = ev_t
        pred, _ = apply_rule(LP, teams, rosters, colours, rule)
        rep[f"test_jev_{name}"] = {"all": tally(pred, teams, truth), "by_size": by_bucket(pred, teams, test),
                                   "by_event": by_key(pred, teams, test, lambda r: r["event"])}
        LP, teams, rosters, colours, truth, _ = ev_p
        pred, _ = apply_rule(LP, teams, rosters, colours, rule)
        rep[f"test_propagated_{name}"] = {"all": tally(pred, teams, truth), "by_size": by_bucket(pred, teams, prop)}
        LP, teams, rosters, colours, _, reads = ev_g
        pred, _ = apply_rule(LP, teams, rosters, colours, rule)
        named = [i for i, r in enumerate(gold) if r["team"]]
        sub = lambda idx: (pred[idx], [teams[i] for i in idx], [gold[i] for i in idx])  # noqa: E731
        g = {}
        p_, t_, r_ = sub(named)
        g["judge_named"] = {"all": tally(p_, t_, [r["team"] for r in r_]), "by_size": by_bucket(p_, t_, r_),
                            "by_event": by_key(p_, t_, r_, lambda r: r["event"])}
        jc = Counter()
        for r in r_:  # Jev's tracklet label on the same crops, for comparison
            jc["none" if not r["jev_team"] else ("right" if r["jev_team"] == r["team"] else "wrong")] += 1
        g["judge_named_jev_tracklet_label"] = dict(jc)
        for nm, cond in (("jev_unsure", lambda r: r["jev_decision"] == "unsure"),
                         ("jev_no_team", lambda r: not r["jev_team"])):
            idx = [i for i in named if cond(gold[i])]
            p_, t_, r_ = sub(idx)
            g[f"judge_named_{nm}"] = {"all": tally(p_, t_, [r["team"] for r in r_]),
                                      "by_size": by_bucket(p_, t_, r_)}
        un = [i for i, r in enumerate(gold) if not r["team"] and not r["conflict"]]
        g["judge_unnamed_s1_named"] = {
            "all": named_rate(pred[un]),
            "legible_none": named_rate(pred[[i for i in un if gold[i]["legible_none"]]]),
            "kind_bumper_or_robot": named_rate(pred[[i for i in un if gold[i]["kind"] in ("bumper", "robot")]]),
            # where Jev's tracklet has a team, compare S1's name with it (a judge could not read the crop)
            "vs_jev_tracklet": tally(np.array([pred[i] for i in un if gold[i]["jev_team"]], int),
                                     [teams[i] for i in un if gold[i]["jev_team"]],
                                     [gold[i]["jev_team"] for i in un if gold[i]["jev_team"]])}
        g["wrong_examples"] = [{"id": gold[i]["id"], "path": gold[i]["path"], "judge": gold[i]["team"],
                                "s1": teams[i][pred[i]], "read": reads[i], "px_w": gold[i]["px_w"]}
                               for i in named if pred[i] >= 0 and teams[i][pred[i]] != gold[i]["team"]]
        rep[f"gold_b2_{name}"] = g
        print(name, "test", json.dumps(rep[f"test_jev_{name}"]["all"]), "gold", json.dumps(g["judge_named"]["all"]),
              "gold-jev-unsure", json.dumps(g["judge_named_jev_unsure"]["all"]), flush=True)
    if a.device.startswith("cuda"):
        rep["latency"] = latency(model, a.device)
        print("latency", rep["latency"])
    if a.ablation:
        rep["synth_ablation_val"] = {}
        for f in [ck_name for ck_name in a.ablation.split(",") if ck_name]:
            v = json.loads((MODELS / (Path(f).stem + "_val.json")).read_text())
            rep["synth_ablation_val"][f] = {"synth": v["synth"], **{
                k: {"val": v[k]["val"]["all"], "prop": v[k]["prop"]["all"], "rule": v[k]["rule"]}
                for k in v if k in ("val_plain", "val_colour", "val_colour_hard")}}
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
    t.add_argument("--trim", type=float, default=0.1, help="fraction cut off each side of the padded crop (val: 0.1 best)")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--val-limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("retune", help="re-tune decision rules on val for a trained checkpoint")
    r.add_argument("--ckpt", default="s1_digits.pt")
    r.add_argument("--device", default="cuda")
    r.set_defaults(fn=cmd_retune)
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
