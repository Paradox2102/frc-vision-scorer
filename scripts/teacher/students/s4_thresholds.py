#!/usr/bin/env python3
"""Per-model appearance thresholds for the S4 embedder (teacher-distillation spec §16).

Every cosine threshold / margin in the pipeline was hand-set on the DINOv3 (production) scale.
S4 cosines live on a different scale, so each one gets its own S4 value. Rule: the FEASIBLE
band is every S4 value whose false-pass rate is <= DINOv3's at today's value AND whose
true-pass rate is >= DINOv3's; inside the band pick the Youden-J maximiser (true - false);
when the band is empty below/above the J optimum it clamps to the band edge (the FPR-matched
end when the unconstrained optimum would let more false passes through). Data: the S4 VAL events (2026bcvi, 2024mimil — whole events never trained on; the
test events 2026alhu / 2025casd are NOT touched), robot-detector crops with their time-local
per-crop team labels.

Embeddings:
  * DINOv3 = the production transform (box inset 8%, timm eval transform, explicit CLS) on the
    harvester crop (``frozen_prod_feats``). Production also blacks out the SAM mask and embeds
    from the 0.5x frame — neither is reproducible on harvest crops (caveat).
  * S4 = ``S4Embedder.embed_crops`` on the harvester crop (its training input).

Proxy tasks (what each threshold decides):
  relink   query crop vs each candidate robot of the SAME match, candidate = max cosine over
           <= 5 exemplar crops from OTHER tracklets (FusedTrackRegistry: max over EMA / anchor /
           5 recent exemplars). pos = own team. -> relink_floor, identity_floor, app_floor;
           margins (top1 - top2) -> update_margin, confident_margin.
  ema      as relink but vs the normalised MEAN of the exemplars (an EMA stand-in)
           -> divergence_sim (pass = "not divergent").
  gallery  query vs same-colour teams of its match, exemplars (<= 12) from OTHER matches of the
           event -> gallery_confirm_sim / supplement_sim (pair false-pass), confirm_gap (margin).
  dedup    within-match same-team cross-tracklet pairs: S4 value at the SAME positive quantile
           (dedup is redundancy, not identity).
  app term selective TBD cost w*(1-cos): scale = DINOv3 / S4 gap between mean cross-tracklet
           negative and positive cosine on same-alliance pairs (equal mean cost differential).

Usage:
  .venv/bin/python -m scripts.teacher.students.s4_thresholds [--out data/teacher/models/s4_thresholds.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.events import TEST_EVENTS  # noqa: E402
from scripts.teacher.students import features as F  # noqa: E402
from scripts.teacher.students.embedder import (  # noqa: E402
    MODELS, VAL_EVENTS, _eval_recs, frozen_prod_feats)
from web.embedder import DINOV3_THRESHOLDS  # noqa: E402
from web.students.s4_embedder import DEFAULT_CKPT, S4Embedder  # noqa: E402

N_RELINK_EX = 5
N_GALLERY_EX = 12


def _norm(E: np.ndarray) -> np.ndarray:
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def s4_feats(recs: Sequence[dict], batch: int = 128) -> np.ndarray:
    emb = S4Embedder(DEFAULT_CKPT)
    out = []
    for i in range(0, len(recs), batch):
        crops = [cv2.imread(str(F.DEFAULT_ROOT / r["path"])) for r in recs[i:i + batch]]
        crops = [c if c is not None else np.zeros((32, 32, 3), np.uint8) for c in crops]
        out.append(emb.embed_crops(crops, batch=batch))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- proxy tasks
def build_queries(recs: Sequence[dict], seed: int = 0) -> dict:
    """Index structure shared by both models (same exemplar draws for both)."""
    rng = np.random.default_rng(seed)
    by_match = defaultdict(lambda: defaultdict(list))   # mk -> team -> [idx]
    by_event_team = defaultdict(lambda: defaultdict(list))  # (event, team) -> mk -> [idx]
    for i, r in enumerate(recs):
        by_match[r["match_key"]][r["team"]].append(i)
        by_event_team[(r["event"], r["team"])][r["match_key"]].append(i)
    relink, gallery = [], []
    for mk, teams in by_match.items():
        if len(teams) < 2:
            continue
        for t, idxs in teams.items():
            for q in idxs:
                qtrk = recs[q]["tracklet_id"]
                cands = {}
                for c, cidx in teams.items():
                    pool = [j for j in cidx if recs[j]["tracklet_id"] != qtrk and j != q]
                    if not pool:
                        continue
                    if len(pool) > N_RELINK_EX:
                        pool = list(rng.choice(pool, N_RELINK_EX, replace=False))
                    cands[c] = pool
                if t in cands and len(cands) >= 2:
                    relink.append((q, t, cands))
                # gallery: same-colour teams of this match, exemplars from other matches
                al = recs[q].get("alliance")
                roster = recs[q].get("roster") or {}
                colour_teams = [str(x) for x in (roster.get(al) or [])] if al else []
                gc = {}
                for c in colour_teams:
                    other = [j for m2, js in by_event_team[(recs[q]["event"], c)].items()
                             if m2 != mk for j in js]
                    if not other:
                        continue
                    if len(other) > N_GALLERY_EX:
                        other = list(rng.choice(other, N_GALLERY_EX, replace=False))
                    gc[c] = other
                if t in gc and len(gc) >= 2:
                    gallery.append((q, t, gc))
    # within-match cross-tracklet pairs (dedup + app-term scale), same alliance only
    pos_pairs, neg_pairs = [], []
    for mk, teams in by_match.items():
        allidx = [j for js in teams.values() for j in js]
        for a_ in range(len(allidx)):
            for b_ in range(a_ + 1, len(allidx)):
                i, j = allidx[a_], allidx[b_]
                if recs[i]["tracklet_id"] == recs[j]["tracklet_id"]:
                    continue
                if (recs[i].get("alliance") or "") != (recs[j].get("alliance") or ""):
                    continue
                (pos_pairs if recs[i]["team"] == recs[j]["team"] else neg_pairs).append((i, j))
    return {"relink": relink, "gallery": gallery, "pos_pairs": pos_pairs, "neg_pairs": neg_pairs}


def score_task(E: np.ndarray, items, how: str = "max"):
    """Per query: (pos score, [neg scores], top1 correct, margin top1-top2)."""
    pos, neg, correct, margin = [], [], [], []
    for q, t, cands in items:
        sc = {}
        for c, ex in cands.items():
            S = E[ex] @ E[q]
            if how == "max":
                sc[c] = float(S.max())
            else:
                m = E[ex].mean(0)
                sc[c] = float(m @ E[q] / max(np.linalg.norm(m), 1e-9))
        pos.append(sc[t])
        neg.extend(v for c, v in sc.items() if c != t)
        order = sorted(sc.items(), key=lambda kv: -kv[1])
        correct.append(order[0][0] == t)
        margin.append(order[0][1] - order[1][1])
    return np.array(pos), np.array(neg), np.array(correct), np.array(margin)


def rate_ge(x: np.ndarray, t: float) -> float:
    return float((x >= t).mean()) if len(x) else float("nan")


def match_fpr(neg_ref: np.ndarray, t_ref: float, neg_new: np.ndarray) -> float:
    """Smallest threshold (0.001 grid) whose false-pass rate on ``neg_new`` <= the reference
    model's false-pass rate at ``t_ref``."""
    target = rate_ge(neg_ref, t_ref)
    for t in np.arange(-1.0, 1.0005, 0.001):
        if rate_ge(neg_new, t) <= target:
            return round(float(t), 3)
    return 1.0


def match_margin(ok_ref, m_ref, t_m_ref, ok_new, m_new) -> float:
    """Smallest margin (0.001 grid) whose WRONG-top1 pass rate (over all queries) <= the
    reference model's at ``t_m_ref``."""
    target = float(((~ok_ref) & (m_ref >= t_m_ref)).mean())
    for t in np.arange(0.0, 1.0005, 0.001):
        if float(((~ok_new) & (m_new >= t)).mean()) <= target:
            return round(float(t), 3)
    return 1.0


def band_pick(pos_new, neg_new, pos_ref, neg_ref, t_ref, lo=-1.0, hi=1.0, step=0.001) -> dict:
    """Threshold rule (see module doc) for a pair-score task."""
    fp_ref, tp_ref = rate_ge(neg_ref, t_ref), rate_ge(pos_ref, t_ref)
    grid = np.round(np.arange(lo, hi + step / 2, step), 3)
    ok = [t for t in grid if rate_ge(neg_new, t) <= fp_ref and rate_ge(pos_new, t) >= tp_ref]
    if not ok:  # no dominating value: fall back to the FPR-matched one
        t = match_fpr(neg_ref, t_ref, neg_new)
        return {"t": t, "band": None}
    best = max(ok, key=lambda t: (rate_ge(pos_new, t) - rate_ge(neg_new, t), -t))
    return {"t": float(best), "band": [float(min(ok)), float(max(ok))]}


def band_pick_margin(ok_new, m_new, ok_ref, m_ref, t_ref, step=0.001) -> dict:
    """Margin rule: wrong-top1 pass <= reference AND correct-top1 pass >= reference, then
    max (correct pass - wrong pass)."""
    w_ref = float(((~ok_ref) & (m_ref >= t_ref)).mean())
    c_ref = float((ok_ref & (m_ref >= t_ref)).mean())
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 3)
    cp = lambda t: float((ok_new & (m_new >= t)).mean())  # noqa: E731
    wp = lambda t: float(((~ok_new) & (m_new >= t)).mean())  # noqa: E731
    ok = [t for t in grid if wp(t) <= w_ref and cp(t) >= c_ref]
    if not ok:
        return {"t": match_margin(ok_ref, m_ref, t_ref, ok_new, m_new), "band": None}
    best = max(ok, key=lambda t: (cp(t) - wp(t), -t))
    return {"t": float(best), "band": [float(min(ok)), float(max(ok))]}


def curve(pos, neg, grid) -> List[dict]:
    return [{"t": round(float(t), 3), "true_pass": rate_ge(pos, t), "false_pass": rate_ge(neg, t)}
            for t in grid]


def auc(pos, neg) -> float:
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(MODELS / "s4_thresholds.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    recs = _eval_recs("val")
    assert {r["event"] for r in recs} <= set(VAL_EVENTS) and not ({r["event"] for r in recs} & TEST_EVENTS)
    print(f"val crops {len(recs)} events {sorted({r['event'] for r in recs})} "
          f"matches {len({r['match_key'] for r in recs})}")
    Q = build_queries(recs, a.seed)
    print(f"relink queries {len(Q['relink'])}  gallery queries {len(Q['gallery'])}  "
          f"same-alliance x-tracklet pairs pos {len(Q['pos_pairs'])} neg {len(Q['neg_pairs'])}")
    E = {"dinov3": _norm(frozen_prod_feats(recs)), "s4": _norm(s4_feats(recs))}

    D = DINOV3_THRESHOLDS
    res: Dict[str, dict] = {}
    for name, X in E.items():
        r = {}
        r["relink"] = score_task(X, Q["relink"], "max")
        r["ema"] = score_task(X, Q["relink"], "mean")
        r["gallery"] = score_task(X, Q["gallery"], "max")
        pi = np.array(Q["pos_pairs"]); ni = np.array(Q["neg_pairs"])
        r["pairs"] = ((X[pi[:, 0]] * X[pi[:, 1]]).sum(1), (X[ni[:, 0]] * X[ni[:, 1]]).sum(1))
        res[name] = r

    d, s = res["dinov3"], res["s4"]
    rl_d, rl_s = d["relink"], s["relink"]
    picks = {
        "relink_floor": band_pick(rl_s[0], rl_s[1], rl_d[0], rl_d[1], D.relink_floor),
        "update_margin": band_pick_margin(rl_s[2], rl_s[3], rl_d[2], rl_d[3], D.update_margin),
        "confident_margin": band_pick_margin(rl_s[2], rl_s[3], rl_d[2], rl_d[3], D.confident_margin),
        "divergence_sim": band_pick(s["ema"][0], s["ema"][1], d["ema"][0], d["ema"][1], D.divergence_sim),
        "gallery_confirm_sim": band_pick(s["gallery"][0], s["gallery"][1], d["gallery"][0],
                                         d["gallery"][1], D.gallery_confirm_sim),
        "gallery_supplement_sim": band_pick(s["gallery"][0], s["gallery"][1], d["gallery"][0],
                                            d["gallery"][1], D.gallery_supplement_sim),
        "gallery_confirm_gap": band_pick_margin(s["gallery"][2], s["gallery"][3], d["gallery"][2],
                                                d["gallery"][3], D.gallery_confirm_gap),
    }
    floor = picks["relink_floor"]["t"]
    upd_m, conf_m = picks["update_margin"]["t"], picks["confident_margin"]["t"]
    div = picks["divergence_sim"]["t"]
    g_conf, g_sup = picks["gallery_confirm_sim"]["t"], picks["gallery_supplement_sim"]["t"]
    g_gap = picks["gallery_confirm_gap"]["t"]
    # FPR-matched values (the permissive end of each band) for reference
    fpr_matched = {
        "relink_floor": match_fpr(rl_d[1], D.relink_floor, rl_s[1]),
        "update_margin": match_margin(rl_d[2], rl_d[3], D.update_margin, rl_s[2], rl_s[3]),
        "confident_margin": match_margin(rl_d[2], rl_d[3], D.confident_margin, rl_s[2], rl_s[3]),
        "divergence_sim": match_fpr(d["ema"][1], D.divergence_sim, s["ema"][1]),
        "gallery_confirm_sim": match_fpr(d["gallery"][1], D.gallery_confirm_sim, s["gallery"][1]),
        "gallery_supplement_sim": match_fpr(d["gallery"][1], D.gallery_supplement_sim, s["gallery"][1]),
        "gallery_confirm_gap": match_margin(d["gallery"][2], d["gallery"][3], D.gallery_confirm_gap,
                                            s["gallery"][2], s["gallery"][3]),
    }
    q = 1.0 - rate_ge(d["pairs"][0], D.gallery_dedup_sim)
    dedup = round(float(np.quantile(s["pairs"][0], q)), 3)
    gap_d = float(d["pairs"][0].mean() - d["pairs"][1].mean())
    gap_s = float(s["pairs"][0].mean() - s["pairs"][1].mean())
    app_scale = round(gap_d / gap_s, 3)
    s4_thr = {"app_floor": floor, "relink_floor": floor, "identity_floor": floor,
              "update_margin": upd_m, "confident_margin": conf_m, "divergence_sim": div,
              "gallery_confirm_sim": g_conf, "gallery_confirm_gap": g_gap,
              "gallery_supplement_sim": g_sup, "gallery_dedup_sim": dedup,
              "selective_app_scale": app_scale}

    def at(task, model, t):
        p, n, ok, m = res[model][task]
        return {"t": t, "true_pass": rate_ge(p, t), "false_pass": rate_ge(n, t)}

    def at_margin(task, model, t):
        p, n, ok, m = res[model][task]
        return {"t": t, "correct_pass": float((ok & (m >= t)).mean()),
                "wrong_pass": float(((~ok) & (m >= t)).mean()), "top1": float(ok.mean())}

    ops = {
        "relink_floor": {"dinov3": at("relink", "dinov3", D.relink_floor), "s4": at("relink", "s4", floor)},
        "update_margin": {"dinov3": at_margin("relink", "dinov3", D.update_margin),
                          "s4": at_margin("relink", "s4", upd_m)},
        "confident_margin": {"dinov3": at_margin("relink", "dinov3", D.confident_margin),
                             "s4": at_margin("relink", "s4", conf_m)},
        "divergence_sim": {"dinov3": at("ema", "dinov3", D.divergence_sim), "s4": at("ema", "s4", div)},
        "gallery_confirm_sim": {"dinov3": at("gallery", "dinov3", D.gallery_confirm_sim),
                                "s4": at("gallery", "s4", g_conf)},
        "gallery_supplement_sim": {"dinov3": at("gallery", "dinov3", D.gallery_supplement_sim),
                                   "s4": at("gallery", "s4", g_sup)},
        "gallery_confirm_gap": {"dinov3": at_margin("gallery", "dinov3", D.gallery_confirm_gap),
                                "s4": at_margin("gallery", "s4", g_gap)},
        "gallery_dedup_sim": {"dinov3": {"t": D.gallery_dedup_sim,
                                         "pos_ge": rate_ge(d["pairs"][0], D.gallery_dedup_sim)},
                              "s4": {"t": dedup, "pos_ge": rate_ge(s["pairs"][0], dedup)}},
        "selective_app_scale": {"gap_dinov3": gap_d, "gap_s4": gap_s, "scale": app_scale},
    }
    grid = np.round(np.arange(-0.2, 1.0001, 0.05), 3)
    out = {
        "data": {"split": "val", "events": sorted(VAL_EVENTS), "n_crops": len(recs),
                 "n_relink_queries": len(Q["relink"]), "n_gallery_queries": len(Q["gallery"]),
                 "n_pos_pairs": len(Q["pos_pairs"]), "n_neg_pairs": len(Q["neg_pairs"])},
        "dinov3": {k: getattr(D, k) for k in s4_thr},
        "s4": s4_thr,
        "picks": picks,
        "fpr_matched": fpr_matched,
        "operating_points": ops,
        "auc": {task: {m: auc(res[m][task][0], res[m][task][1]) for m in res}
                for task in ("relink", "ema", "gallery")},
        "top1": {task: {m: float(res[m][task][2].mean()) for m in res} for task in ("relink", "gallery")},
        "curves": {task: {m: curve(res[m][task][0], res[m][task][1], grid) for m in res}
                   for task in ("relink", "ema", "gallery")},
        "margin_curves": {task: {m: [at_margin(task, m, float(t)) for t in np.round(np.arange(0, 0.31, 0.02), 3)]
                                 for m in res} for task in ("relink", "gallery")},
        "score_quantiles": {task: {m: {"pos": np.quantile(res[m][task][0], [.05, .25, .5, .75, .95]).round(3).tolist(),
                                       "neg": np.quantile(res[m][task][1], [.05, .25, .5, .75, .95]).round(3).tolist()}
                                   for m in res} for task in ("relink", "ema", "gallery")},
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"s4": s4_thr, "picks": picks, "fpr_matched": fpr_matched,
                      "auc": out["auc"], "top1": out["top1"]}, indent=1))
    for k, v in ops.items():
        print(f"{k:24s} {json.dumps(v)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
