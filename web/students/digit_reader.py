"""S1 roster-constrained bumper digit reader: runtime inference.

PARSeq-base (Apache-2.0, vendored in web/students/parseq.py) fine-tuned with a digits-only
vocabulary on the harvester's bumper crops (box + 25% padding per side, BGR). Training, rule
tuning and evaluation live in scripts/teacher/students/digits.py, which re-exports this module.

Decoding: teacher-forced log P(team digits, [E] | crop) for each roster team (``score_batch``),
then the checkpoint's decision rule (``posterior`` / ``decide``):
  * "softmax": softmax(LP / T) over the roster numbers, abstain below top-1 ``t_p`` / margin
    ``t_m`` or when the best raw log-likelihood is under ``floor``;
  * "mass": P(team) = sequence probability, "other" = 1 - sum over the roster.
Optional bumper-colour prior multiplies off-alliance teams by ``colour_eps``.

Checkpoints (``torch.save`` dicts of plain tensors / config, loaded with ``weights_only=True``)
pickle no class paths, so moving this code does not affect loading.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import cv2
import numpy as np

IMG_H, IMG_W = 32, 128
OTHER = "other"


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
            from web.students.paths import MODELS
            ckpt = MODELS / "s1_digits.pt"
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
