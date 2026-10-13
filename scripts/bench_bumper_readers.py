"""Offline bench of bumper-number reader "arms" over the persisted crop bench.

Reads ``data/eval/bumper_crop_bench/manifest.jsonl`` (built by
scripts/build_bumper_crop_bench.py), runs each arm over the crops, and writes
one prediction row per crop to ``results/<arm>-<timestamp>.jsonl`` plus a
``results/run-<timestamp>.json`` summary. Nothing here touches the pipeline,
and nothing here starts vLLM.

Arms
----
freeread     The incumbent: web.app._vllm_read_bumpers (real prompt, guided JSON
             schema _BUMPER_READ_SCHEMA, roster match, logprob-blended
             confidence) against the local vLLM server, one crop per call.
vllm-choice  Closed-set "poor man's decision model" on the SAME vLLM model:
             lettered options, max_tokens=1, top_logprobs=20; the option-letter
             probabilities are renormalised into a distribution.
clef-flash   Cloudflare Workers AI decision models (Jev-API compatible): one
clef         request per crop (per permutation) answering `legible` (noul),
             `surface` (choice) and `team` (choice) jointly.

Option sets (``--options``)
---------------------------
alliance      the 3 teams of the hinted alliance (all 6 when the hint is
              unknown) + "none" (no legible number) + "other" (not listed)
decoy         all 6 roster teams + the same two escapes; no colour hint, so the
              opposite alliance's 3 are known-wrong decoys when the bumper
              colour is known (label_alliance, else the hint)
missing-true  roster-forced-misread stress test: the alliance set with the TRUE
              team removed (label_team when labelled legible, else the
              baseline's matched team; crops with neither are skipped). The
              right answer is an escape, so the headline metric is the rate at
              which an arm still picks a listed team ("assert" column). The
              freeread arm honours this too: its roster has the team removed.
Order is shuffled per crop (seeded by crop id) and recorded per row.

Position bias / calibration (choice arms)
-----------------------------------------
--permutations K   score K cyclic rotations of the seeded order and average the
                   distributions (default 3 for vllm-choice, 1 for clef arms -
                   every permutation is a paid request). Per-permutation
                   distributions are stored; "slotA" reports how often the
                   argmax sits in the first slot vs the 1/n expectation.
--blank-calibrate  contextual calibration for vllm-choice: score a mid-gray image
                   of the same size with the same options -> p_blank; calibrated
                   = p / max(p_blank, eps), renormalised. Raw and calibrated are
                   both stored and scored ("+cal" rows). --blank-calibrate-clef
                   does the same for clef arms (doubles the paid requests). NB:
                   a blank image SHOULD pick "none", so calibration
                   systematically pushes mass away from the escapes.

The alliance hint (``--hint-source``) stands in for the pipeline's CV
bumper-colour hint: ``baseline`` = the incumbent's alliance_read (default),
``label`` = label_alliance (a perfect hint), ``none`` = never hinted.

Scoring (``score`` subcommand, or ``--score`` after a run)
----------------------------------------------------------
legible crops: correct / wrong / abstained (argmax is an escape, or freeread
returned no roster team); illegible/not_bumper: false-positive rate (any team
asserted). Probabilistic arms add ECE (10 bins) on the top choice, precision and
coverage at 0.5/0.7/0.9, and mean probability mass on decoys. Label-free
behaviour: assert rate, agreement with the baseline's matched team, assert rate
on legibility-filtered crops, first-slot argmax rate.

Credentials: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN from the environment,
else from the repo-root .env (never printed, never overriding the environment).

Usage
-----
    .venv/bin/python scripts/bench_bumper_readers.py --dry-run --arms clef-flash,vllm-choice
    .venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice --limit 40 --score
    .venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice --options missing-true --score
    .venv/bin/python scripts/bench_bumper_readers.py --arms clef-flash --options decoy --score
    .venv/bin/python scripts/bench_bumper_readers.py score            # latest file per arm
    .venv/bin/python scripts/bench_bumper_readers.py score data/eval/bumper_crop_bench/results/clef-*.jsonl
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH_DIR = REPO / "data" / "eval" / "bumper_crop_bench"
ENV_FILE = REPO / ".env"

ARMS = ("freeread", "vllm-choice", "clef-flash", "clef", "clef-omni")
OPTION_MODES = ("alliance", "decoy", "missing-true")
HINT_SOURCES = ("baseline", "label", "none")
ESCAPES = ("none", "other")
ESCAPE_TEXT = {
    "none": ("No legible team number on a robot bumper (blurry, occluded, cut off, "
             "not a robot, or the only digits are printed on field equipment)"),
    "other": "A legible team number that is NOT any of the listed teams",
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THRESHOLDS = (0.5, 0.7, 0.9)
BLANK_EPS = 1e-3
BLANK_GRAY = (128, 128, 128)
DEFAULT_PERMS = {"vllm-choice": 3, "clef-flash": 1, "clef": 1, "clef-omni": 1}
CF_ENV_KEYS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")

VLM_SERVER_URL = os.environ.get("VLM_SERVER_URL", "http://localhost:8006/v1")

# Workers AI decision models. Schema verified 2026-10-12 against
#   https://developers.cloudflare.com/workers-ai/models/clef-flash/  (+ /models/clef/)
#   .../clef-flash/schema-input.json and schema-output.json:
# * request: {"model": "clef"|"clef-flash"|"clef-omni" (required, selects the model),
#   "state": str|object (required), "questions": {id: q} (1-64),
#   "images": [...] (optional, max 4, placed before the state)}.
# * images items: a base64 DATA URL string ("data:image/jpeg;base64,...") or
#   {"content_type": "image/jpeg", "base64": "..."}; remote URLs are rejected.
#   PNG/JPEG/WebP, 4 MiB + 16 MP each, 8 MiB total decoded, body <= 13 MiB.
# * questions: {"type": "noul", "instructions", "criteria"?: {"true","false"}},
#   {"type": "choice", "instructions", "criteria": {option_id: description}}
#   (2-255 options), {"type": "score", ...}.
# * answers: noul -> {"type": "noul", "noul": p_yes};
#   choice -> {"type": "choice", "choice", "probabilities": {id: p}, "confidence"}.
# * usage: {"input_tokens", "output_tokens"}; billed on input only.
# * image tokens: sides rounded to 32-px multiples, one token per 32x32 block
#   + 3 markers, <= 1024 tokens/image; min ~65k px (so upscale to >= 256 side).
# * context: clef-flash 24,576 tokens, clef 65,536.
# The REST endpoint wraps the model output in Cloudflare's usual envelope
# ({"result": ..., "success": ..., "errors": [...]}) - the docs show only the
# model-level schema, so both shapes are accepted.
CLEF_MODELS = {
    "clef-flash": {"model": "clef-flash", "route": "@cf/cloudflare/clef-flash",
                   "usd_per_m_input": 0.038},
    "clef": {"model": "clef", "route": "@cf/cloudflare/clef",
             "usd_per_m_input": 0.24},
    # 30B MoE (3B active), released 2026-10-09; same System One request
    # schema (model pattern ^clef-omni$), plus audio/videos inputs we don't use.
    "clef-omni": {"model": "clef-omni", "route": "@cf/cloudflare/clef-omni",
                  "usd_per_m_input": 0.15},
}
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{route}"
CLEF_MIN_SIDE = 256
CLEF_MAX_IMAGE_TOKENS = 1024
CLEF_MAX_IMAGE_BYTES = 4 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Pure helpers: hints, option sets, shuffles, permutations, distributions
# --------------------------------------------------------------------------- #

def _stable_int(*parts) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def hint_alliance(row: dict, source: str = "baseline") -> str:
    """The bumper-colour hint an arm gets for this crop: red|blue|unknown."""
    if source == "none":
        return "unknown"
    val = row.get("label_alliance") if source == "label" else row.get("baseline_alliance_read")
    return val if val in ("red", "blue") else "unknown"


def true_team(row: dict) -> Optional[str]:
    """The team the missing-true set removes: label_team for a crop labelled
    legible, else the baseline's matched team; None -> crop is skipped."""
    if row.get("label_status") == "legible" and row.get("label_team"):
        return str(row["label_team"])
    if row.get("baseline_matched_team"):
        return str(row["baseline_matched_team"])
    return None


def offered_roster(row: dict, mode: str) -> tuple[list[str], list[str]]:
    """(red, blue) roster lists offered to an arm; missing-true drops the true team."""
    red = [str(t) for t in row["red_teams"]]
    blue = [str(t) for t in row["blue_teams"]]
    if mode == "missing-true":
        tt = true_team(row)
        red = [t for t in red if t != tt]
        blue = [t for t in blue if t != tt]
    return red, blue


def build_options(row: dict, mode: str, hint: str) -> list[dict]:
    """Unshuffled option set: roster teams (alliance or all 6) + two escapes.

    Each option: {"id", "kind": "team"|"escape", "team"?, "alliance"?}. Team ids
    are the team-number strings, so they never collide with the escape ids."""
    if mode not in OPTION_MODES:
        raise ValueError(f"unknown option mode {mode!r}")
    if mode == "missing-true" and true_team(row) is None:
        raise ValueError(f"{row.get('id')}: missing-true needs label_team or baseline_matched_team")
    red, blue = offered_roster(row, mode)
    if mode in ("alliance", "missing-true") and hint in ("red", "blue"):
        pairs = [(t, hint) for t in (red if hint == "red" else blue)]
    else:
        pairs = [(t, "red") for t in red] + [(t, "blue") for t in blue]
    opts = [{"id": t, "kind": "team", "team": t, "alliance": a} for t, a in pairs]
    opts += [{"id": e, "kind": "escape"} for e in ESCAPES]
    return opts


def shuffle_options(options: list[dict], crop_id: str, seed: int = 0) -> list[dict]:
    """Deterministic per-crop permutation (avoids position bias, reproducible)."""
    out = list(options)
    random.Random(_stable_int("options", seed, crop_id)).shuffle(out)
    return out


def permutation_orders(options: list, k: int) -> list[list]:
    """K cyclic rotations of ``options`` with offsets spread over the list
    (rotation 0 is the input order). K is capped at len(options)."""
    n = len(options)
    k = max(1, min(int(k), n))
    offsets = sorted({round(i * n / k) % n for i in range(k)})
    return [options[o:] + options[:o] for o in offsets]


def average_distributions(dists: list[dict]) -> Optional[dict]:
    """Mean of per-permutation distributions keyed by option id."""
    dists = [d for d in dists if d]
    if not dists:
        return None
    keys = sorted({k for d in dists for k in d})
    return {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in keys}


def blank_calibrate(p: dict, p_blank: dict, eps: float = BLANK_EPS) -> dict:
    """Contextual calibration: p / max(p_blank, eps), renormalised.

    Falls back to ``p`` unchanged when there is nothing to divide by."""
    q = {k: v / max(float(p_blank.get(k, 0.0)), eps) for k, v in p.items()}
    s = sum(q.values())
    if s <= 0 or not math.isfinite(s):
        return dict(p)
    return {k: v / s for k, v in q.items()}


def option_text(opt: dict) -> str:
    if opt["kind"] == "team":
        return f"Team {opt['team']} ({opt['alliance']} alliance)"
    return ESCAPE_TEXT[opt["id"]]


def decoy_teams(row: dict, hint: str) -> list[str]:
    """Known-wrong teams: the alliance opposite the bumper colour, taken from
    label_alliance when labelled, else from the hint. [] when colour unknown."""
    colour = row.get("label_alliance")
    if colour not in ("red", "blue"):
        colour = hint
    if colour == "red":
        return [str(t) for t in row["blue_teams"]]
    if colour == "blue":
        return [str(t) for t in row["red_teams"]]
    return []


def _token_letter(tok: str) -> Optional[str]:
    s = (tok or "").strip()
    if len(s) == 1 and "A" <= s <= "Z":
        return s
    if len(s) == 2 and "A" <= s[0] <= "Z" and s[1] in ".):":
        return s[0]
    return None


def letter_distribution(top_logprobs: list[dict], letters: str) -> tuple[Optional[dict], float]:
    """First-token top_logprobs -> renormalised distribution over ``letters``.

    Token variants of one letter (" A", "A", "A.") are summed. Returns
    (dist, raw_mass) where raw_mass is the pre-normalisation probability that
    landed on a valid letter (a format-compliance signal); dist is None when
    no valid letter appears in the top-k."""
    mass = {L: 0.0 for L in letters}
    seen = set()
    for ent in top_logprobs or []:
        tok = ent.get("token") or ""
        L = _token_letter(tok)
        if L in mass and tok not in seen and ent.get("logprob") is not None:
            seen.add(tok)
            mass[L] += math.exp(float(ent["logprob"]))
    total = sum(mass.values())
    if total <= 0:
        return None, 0.0
    return {L: m / total for L, m in mass.items()}, total


def top_choice(probs: Optional[dict]) -> tuple[Optional[str], float]:
    if not probs:
        return None, 0.0
    k = max(probs, key=lambda o: (probs[o], o))
    return k, float(probs[k])


def _sum_usage(usages: list) -> Optional[dict]:
    out: dict = {}
    for u in usages:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    return out or None


# --------------------------------------------------------------------------- #
# Pure helpers: scoring
# --------------------------------------------------------------------------- #

def target_option(label_row: dict, option_ids: Optional[list]) -> Optional[str]:
    """The correct option id for a labelled crop, given the options offered
    (None = the full roster was offered)."""
    st = label_row.get("label_status")
    if st == "legible" and label_row.get("label_team") is not None:
        t = str(label_row["label_team"])
        if option_ids is None:
            return t
        return t if t in option_ids else "other"
    if st in ("illegible", "not_bumper"):
        return "none"
    return None


def ece(confs: list[float], corrects: list[bool], n_bins: int = 10) -> Optional[float]:
    """Expected calibration error over equal-width confidence bins."""
    if not confs:
        return None
    n = len(confs)
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confs):
        bins[min(n_bins - 1, max(0, int(c * n_bins)))].append(i)
    total = 0.0
    for b in bins:
        if b:
            acc = sum(1 for i in b if corrects[i]) / len(b)
            conf = sum(confs[i] for i in b) / len(b)
            total += len(b) / n * abs(acc - conf)
    return total


def calibrated_view(pred: dict) -> dict:
    """The same prediction row with the blank-calibrated distribution swapped in."""
    q = dict(pred)
    q["probs"] = pred.get("probs_cal")
    q["top"], q["top_p"] = top_choice(q["probs"])
    q["pred_team"] = q["top"] if q["top"] not in ESCAPES else None
    q["perms"] = [dict(pm, probs=pm.get("probs_cal")) for pm in pred.get("perms") or []]
    return q


def _pred_team(pred: dict) -> Optional[str]:
    if pred.get("probs") is not None:
        top = pred.get("top")
        return top if top is not None and top not in ESCAPES else None
    t = pred.get("pred_team")
    return str(t) if t else None


def score_predictions(preds: list[dict], manifest: dict[str, dict]) -> dict:
    """Metrics for ONE arm's prediction rows joined to the manifest by id."""
    m: dict = {"n": 0, "errors": 0, "n_legible": 0, "correct": 0, "wrong": 0,
               "abstain": 0, "n_neg": 0, "fp": 0}
    probabilistic = any(p.get("probs") is not None for p in preds)
    confs, corrects = [], []
    thr = {t: {"asserted": 0, "correct": 0, "neg_asserted": 0} for t in THRESHOLDS}
    decoy_mass, listed_mass = [], []
    n_ok = n_assert = 0
    agree_n = agree = filt_n = filt_assert = 0
    slot_n = slot_a = 0
    slot_expect = []
    noul_n = noul_ok = 0
    lat = []
    cost = 0.0
    for p in preds:
        row = manifest.get(p.get("id"))
        if row is None:
            continue
        m["n"] += 1
        if p.get("latency_s") is not None:
            lat.append(p["latency_s"])
        cost += p.get("est_cost_usd") or 0.0
        if p.get("error"):
            m["errors"] += 1
            continue
        team = _pred_team(p)
        probs = p.get("probs")
        # ---- label-free behaviour ----
        n_ok += 1
        n_assert += int(team is not None)
        if row.get("stratum") == "matched" and row.get("baseline_matched_team"):
            agree_n += 1
            agree += int(team == str(row["baseline_matched_team"]))
        if row.get("stratum") == "filtered":
            filt_n += 1
            filt_assert += int(team is not None)
        if probs is not None:
            listed_mass.append(sum(v for k, v in probs.items() if k not in ESCAPES))
            dts = [t for t in decoy_teams(row, p.get("hint_alliance", "unknown")) if t in probs]
            if dts:
                decoy_mass.append(sum(probs[t] for t in dts))
            for pm in p.get("perms") or []:
                if pm.get("probs") and pm.get("option_ids"):
                    slot_n += 1
                    slot_a += int(top_choice(pm["probs"])[0] == pm["option_ids"][0])
                    slot_expect.append(1.0 / len(pm["option_ids"]))
        # ---- labelled metrics ----
        if probs is not None:
            offered = p.get("option_ids")
        elif p.get("roster_offered") is not None:
            offered = list(p["roster_offered"]) + list(ESCAPES)
        else:
            offered = None
        tgt = target_option(row, offered)
        if tgt is None:
            continue
        st = row["label_status"]
        if st == "legible":
            m["n_legible"] += 1
            if probs is not None:
                ok = p.get("top") == tgt
            elif tgt == "other":  # freeread: right digits, correctly NOT roster-forced
                ok = team is None and (p.get("freeread") or {}).get("raw_read") == str(row["label_team"])
            else:
                ok = team == tgt
            if ok:
                m["correct"] += 1
            elif team is not None:
                m["wrong"] += 1
            else:
                m["abstain"] += 1
        else:
            m["n_neg"] += 1
            m["fp"] += int(team is not None)
        if probs is not None:
            top_p = float(p.get("top_p") or 0.0)
            confs.append(top_p)
            # A negative crop is answered correctly by EITHER escape: "other"
            # on an unreadable crop asserts no team, same as the fp metric.
            corrects.append(p.get("top") in ESCAPES if st != "legible"
                            else p.get("top") == tgt)
            for t in THRESHOLDS:
                asserted = team is not None and top_p >= t
                if st == "legible":
                    thr[t]["asserted"] += int(asserted)
                    thr[t]["correct"] += int(asserted and p.get("top") == tgt)
                else:
                    thr[t]["neg_asserted"] += int(asserted)
        if p.get("legible_p") is not None:
            noul_n += 1
            noul_ok += int((p["legible_p"] >= 0.5) == (st == "legible"))
    nl, nn = m["n_legible"], m["n_neg"]
    m["acc"] = m["correct"] / nl if nl else None
    m["fp_rate"] = m["fp"] / nn if nn else None
    m["probabilistic"] = probabilistic
    m["ece"] = ece(confs, corrects) if probabilistic else None
    m["at"] = {}
    if probabilistic:
        for t in THRESHOLDS:
            a = thr[t]["asserted"]
            m["at"][str(t)] = {
                "precision": thr[t]["correct"] / a if a else None,
                "coverage": a / nl if nl else None,
                "fp_rate": thr[t]["neg_asserted"] / nn if nn else None,
            }
    m["assert_rate"] = n_assert / n_ok if n_ok else None
    m["listed_mass"] = statistics.fmean(listed_mass) if listed_mass else None
    m["decoy_mass"] = statistics.fmean(decoy_mass) if decoy_mass else None
    m["agree_baseline"] = agree / agree_n if agree_n else None
    m["assert_filtered"] = filt_assert / filt_n if filt_n else None
    m["slot_a_top"] = slot_a / slot_n if slot_n else None
    m["slot_a_expected"] = statistics.fmean(slot_expect) if slot_expect else None
    m["noul_acc"] = noul_ok / noul_n if noul_n else None
    m["latency_med_s"] = statistics.median(lat) if lat else None
    m["est_cost_usd"] = cost
    return m


def score_file_variants(preds: list[dict], manifest: dict, label: str) -> list[tuple[str, dict]]:
    """Score a results file; blank-calibrated rows add a '+cal' variant."""
    out = [(label, score_predictions(preds, manifest))]
    if any(p.get("probs_cal") is not None for p in preds):
        out.append((label + "+cal", score_predictions(
            [calibrated_view(p) if p.get("probs_cal") is not None else p for p in preds],
            manifest)))
    return out


def _f(x, pct=False, nd=2):
    if x is None:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def format_table(scored: list[tuple[str, dict]]) -> str:
    w = max([24] + [len(n) for n, _ in scored])
    h1 = (f"{'arm':<{w}} {'n':>4} {'err':>3} {'leg':>4} {'ok':>4} {'wrong':>5} {'abst':>4} "
          f"{'acc':>5} {'FP':>5} {'ECE':>5} {'p@.5(cov)':>10} {'p@.7(cov)':>10} {'p@.9(cov)':>10} "
          f"{'noul':>5}")
    lines = ["labelled accuracy", h1, "-" * len(h1)]
    for name, m in scored:
        at = []
        for t in THRESHOLDS:
            a = m["at"].get(str(t)) if m.get("at") else None
            at.append(f"{_f(a['precision'], True)}({_f(a['coverage'], True)})" if a else "-")
        lines.append(
            f"{name:<{w}} {m['n']:>4} {m['errors']:>3} {m['n_legible']:>4} {m['correct']:>4} "
            f"{m['wrong']:>5} {m['abstain']:>4} {_f(m['acc'], True):>5} {_f(m['fp_rate'], True):>5} "
            f"{_f(m['ece']):>5} {at[0]:>10} {at[1]:>10} {at[2]:>10} {_f(m['noul_acc'], True):>5}")
    h2 = (f"{'arm':<{w}} {'assert':>6} {'listedP':>7} {'decoy':>5} {'agreeB':>6} {'filtA':>5} "
          f"{'slotA(exp)':>11} {'lat_s':>5} {'est$':>7}")
    lines += ["", "label-free behaviour", h2, "-" * len(h2)]
    for name, m in scored:
        slot = (f"{_f(m['slot_a_top'], True)}({_f(m['slot_a_expected'], True)})"
                if m.get("slot_a_top") is not None else "-")
        lines.append(
            f"{name:<{w}} {_f(m['assert_rate'], True):>6} {_f(m['listed_mass']):>7} "
            f"{_f(m['decoy_mass']):>5} {_f(m['agree_baseline'], True):>6} "
            f"{_f(m['assert_filtered'], True):>5} {slot:>11} {_f(m['latency_med_s'], nd=1):>5} "
            f"{m['est_cost_usd']:>7.4f}")
    lines.append("assert = picks a listed team (HEADLINE for missing-true: lower is better); "
                 "listedP = mean prob on listed teams; decoy = mean prob on opposite-alliance "
                 "teams; agreeB = same team as baseline on 'matched' crops; filtA = asserts on "
                 "legibility-filtered crops; slotA = argmax in first slot (vs 1/n)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #

def load_env_file(path: Path, keys=CF_ENV_KEYS) -> list[str]:
    """Set ``keys`` from a KEY=VALUE .env file unless already in os.environ.

    Tiny parser (comments, blank lines, ``export``, single/double quotes,
    trailing `` # comment`` on unquoted values). Returns the NAMES it set;
    values are never printed."""
    loaded: list[str] = []
    try:
        text = Path(path).read_text()
    except OSError:
        return loaded
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        k, sep, v = s.partition("=")
        if not sep:
            continue
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k in keys and v and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def app_style_data_uri(jpg: bytes) -> str:
    """Byte-for-byte the image prep in web/app.py _vllm_read_bumpers (~L9853):
    512-px min-dim LANCZOS upscale, RGB, JPEG q92, base64 data URI."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpg)).convert("RGB")
    min_dim = 512
    if img.width < min_dim or img.height < min_dim:
        scale = max(min_dim / img.width, min_dim / img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_jpeg_like(jpg: bytes) -> bytes:
    """A mid-gray JPEG with the crop's dimensions (contextual-calibration probe)."""
    from PIL import Image
    with Image.open(io.BytesIO(jpg)) as im:
        size = im.size
    buf = io.BytesIO()
    Image.new("RGB", size, BLANK_GRAY).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def clef_image_tokens(w: int, h: int) -> int:
    """Workers AI image tokens: 32-px blocks + 3 markers, capped at 1024."""
    return min(CLEF_MAX_IMAGE_TOKENS,
               max(1, round(w / 32)) * max(1, round(h / 32)) + 3)


def clef_image(jpg: bytes) -> tuple[bytes, int, int]:
    """Prep a crop for Clef: upscale to >= 256 px min side (LANCZOS); shrink if
    it would exceed 1024 image tokens; otherwise send the original bytes."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpg))
    w, h = img.size
    scale = 1.0
    if min(w, h) < CLEF_MIN_SIDE:
        scale = CLEF_MIN_SIDE / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if round(nw / 32) * round(nh / 32) + 3 > CLEF_MAX_IMAGE_TOKENS:
        shrink = math.sqrt((CLEF_MAX_IMAGE_TOKENS - 3) * 32 * 32 / (nw * nh)) * 0.98
        nw, nh = max(32, int(nw * shrink)), max(32, int(nh * shrink))
    if (nw, nh) == (w, h) and img.format == "JPEG" and len(jpg) <= CLEF_MAX_IMAGE_BYTES:
        return jpg, w, h
    img = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), nw, nh


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #

def _roster_line(red: list, blue: list) -> str:
    return (f"Match roster: red alliance {', '.join(map(str, red))}; "
            f"blue alliance {', '.join(map(str, blue))}.")


def choice_prompt(options: list[dict], mode: str, hint: str) -> str:
    """vllm-choice prompt: the incumbent's bumper description + lettered options."""
    lines = [
        "This image shows ONE FRC robot from a side or elevated camera angle. "
        "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
        "WHITE team number printed on it. FRC team numbers are 1 to 5 digits.",
    ]
    if mode in ("alliance", "missing-true") and hint in ("red", "blue"):
        lines.append(f"Computer vision sees {hint.upper()} bumper pixels.")
    lines.append("Which option matches the WHITE digits on this robot's bumper?")
    for L, opt in zip(LETTERS, options):
        lines.append(f"{L}. {option_text(opt)}")
    lines.append("Answer with ONLY the letter of the correct option.")
    return "\n".join(lines)


def vllm_choice_payload(model: str, extra_body: dict, data_uri: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }
    if extra_body:
        payload.update(extra_body)
    return payload


def clef_state(row: dict, mode: str, hint: str) -> str:
    red, blue = offered_roster(row, mode)
    s = ("One image: a crop from FRC robotics match video around ONE robot's bumper, a "
         "solid RED or BLUE fabric band around the robot's base with the team number in "
         "WHITE digits. Field structures (crates, bins, signs) can also carry printed "
         "numbers. " + _roster_line(red, blue))
    if mode in ("alliance", "missing-true") and hint in ("red", "blue"):
        s += f" Computer vision sees {hint.upper()} bumper pixels."
    return s


def clef_questions(options: list[dict]) -> dict:
    return {
        "legible": {
            "type": "noul",
            "instructions": "Is a team number legible on this robot's bumper band?",
            "criteria": {
                "true": "White digits on the robot's red or blue bumper band can be read.",
                "false": ("The digits are blurry, occluded, cut off or absent, the object is "
                          "not a robot, or the only number is printed on field equipment."),
            },
        },
        "surface": {
            "type": "choice",
            "instructions": "Where is the most prominent number in the image?",
            "criteria": {
                "bumper": "On a robot's red or blue bumper band",
                "printed": "Printed on a field structure, crate, bin, sign or other equipment",
                "none": "No number is visible",
            },
        },
        "team": {
            "type": "choice",
            "instructions": "Which option matches the white team-number digits on this robot's bumper?",
            "criteria": {o["id"]: option_text(o) for o in options},
        },
    }


def clef_payload(model_key: str, row: dict, img_bytes: bytes, options: list[dict],
                 mode: str, hint: str) -> dict:
    return {
        "model": CLEF_MODELS[model_key]["model"],
        "state": clef_state(row, mode, hint),
        "questions": clef_questions(options),
        "images": ["data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()],
    }


def clef_estimate_tokens(payload: dict, w: int, h: int) -> int:
    """Rough input tokens: image blocks + ~4 chars/token for state+questions."""
    text = len(json.dumps(payload["state"])) + len(json.dumps(payload["questions"]))
    return clef_image_tokens(w, h) + text // 4


def truncate_payload(obj, keep: int = 60):
    """Copy of a payload with data-URI / base64 strings cut down for printing."""
    if isinstance(obj, dict):
        return {k: truncate_payload(v, keep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncate_payload(v, keep) for v in obj]
    if isinstance(obj, str) and (obj.startswith("data:") or len(obj) > 2000):
        return obj[:keep] + f"...<{len(obj)} chars>"
    return obj


# --------------------------------------------------------------------------- #
# Transport (retry/backoff); never logs headers
# --------------------------------------------------------------------------- #

class ArmError(RuntimeError):
    pass


def post_json(url: str, payload: dict, headers: Optional[dict] = None,
              timeout: float = 120.0, retries: int = 5) -> tuple[dict, float, int]:
    """POST with exponential backoff on 429/5xx/connection errors.

    Returns (body, latency_s_of_successful_attempt, attempts)."""
    import requests
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    last = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            r = requests.post(url, json=payload, headers=hdrs, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"{type(e).__name__}"
        else:
            lat = time.monotonic() - t0
            if r.status_code == 200:
                return r.json(), lat, attempt + 1
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code != 429 and r.status_code < 500:
                raise ArmError(last)
            ra = r.headers.get("Retry-After")
            if ra and ra.replace(".", "", 1).isdigit() and attempt < retries:
                time.sleep(min(60.0, float(ra)))
                continue
        if attempt < retries:
            time.sleep(min(30.0, 2 ** attempt) + random.random())
    raise ArmError(f"gave up after {retries + 1} attempts: {last}")


def probe_vllm(url: str) -> list[str]:
    import requests
    try:
        r = requests.get(f"{url}/models", timeout=3)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]
    except Exception as e:
        raise ArmError(f"vLLM server unreachable at {url} ({type(e).__name__}); start it with "
                       "web/start_vllm.sh (and wake it if sleeping) - this bench never starts it")


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

def vllm_extra_body(model: str, no_thinking: bool) -> dict:
    """Per-model payload keys: the web/vlm_models.py registry entry's extra_body
    when the served model is registered, plus enable_thinking=false on
    --no-thinking (e.g. Qwen3.5-9B)."""
    extra: dict = {}
    try:
        from web import vlm_models
        for key in vlm_models.model_keys():
            if vlm_models.get(key).hf_repo == model:
                extra = json.loads(json.dumps(vlm_models.extra_body(key)))
    except Exception:
        pass
    if no_thinking:
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        extra["chat_template_kwargs"] = ctk
    return extra


class Ctx:
    def __init__(self, args, bench_dir: Path):
        self.args = args
        self.bench_dir = bench_dir
        self.vlm_model: Optional[str] = None
        self.vlm_extra: dict = {}
        self.vlm_model_note = ""

    def resolve_vllm(self, dry_run: bool) -> None:
        """--vllm-model, else the first id GET /v1/models reports. A dry run
        never probes: it falls back to VLM_MODEL_NAME / the registry default."""
        if self.vlm_model is not None:
            return
        a = self.args
        if a.vllm_model:
            self.vlm_model = a.vllm_model
            if not dry_run:
                served = probe_vllm(a.server_url)
                if a.vllm_model not in served:
                    print(f"WARNING: vLLM serves {served}, payload model is {a.vllm_model}",
                          file=sys.stderr)
        elif dry_run:
            from web import vlm_models
            self.vlm_model = os.environ.get(
                "VLM_MODEL_NAME", vlm_models.get(vlm_models.DEFAULT_KEY).hf_repo)
            self.vlm_model_note = " (dry run: server not probed; a real run uses GET /v1/models)"
        else:
            served = probe_vllm(a.server_url)
            if not served:
                raise ArmError(f"vLLM at {a.server_url} reports no models")
            self.vlm_model = served[0]
        self.vlm_extra = vllm_extra_body(self.vlm_model, a.no_thinking)

    def perms_for(self, arm: str) -> int:
        k = self.args.permutations
        return int(k) if k is not None else DEFAULT_PERMS.get(arm, 1)

    def crop(self, row: dict) -> bytes:
        return (self.bench_dir / row["file"]).read_bytes()


def _base_pred(arm: str, row: dict, ctx: Ctx, hint: str, model: str,
               mode: Optional[str]) -> dict:
    return {"id": row["id"], "arm": arm, "model": model, "options_mode": mode,
            "hint_source": ctx.args.hint_source, "hint_alliance": hint,
            "option_ids": None, "probs": None, "top": None, "top_p": None,
            "pred_team": None, "latency_s": None, "latencies_s": [], "attempts": None,
            "usage": None, "est_cost_usd": 0.0, "error": None}


# ---- freeread: the real web.app reader -------------------------------------

class FreeRead:
    """Drives web.app._vllm_read_bumpers (import is cheap: ~4 s, no model or
    CUDA init - verified 2026-10-12). Its HTTP helper _post_vlm_chat is wrapped
    to capture latency, usage and raw text, keyed by the request's image data
    URI (reproduced with app_style_data_uri)."""

    def __init__(self, ctx: Ctx, dry_run: bool):
        import web.app as appmod
        self.app = appmod
        self.ctx = ctx
        self.dry = dry_run
        appmod.VLM_SERVER_URL = ctx.args.server_url
        appmod._active_vlm_payload_model = ctx.vlm_model
        appmod._active_vlm_extra_body = dict(ctx.vlm_extra)
        self.captured: dict[str, dict] = {}
        self.lock = threading.Lock()
        orig = appmod._post_vlm_chat

        def wrapped(payload: dict) -> dict:
            key = hashlib.sha1(
                payload["messages"][0]["content"][0]["image_url"]["url"].encode()).hexdigest()
            t0 = time.monotonic()
            if self.dry:
                body = {"choices": [{"message": {"content": '{"team": null, "color": "red"}'},
                                     "logprobs": None}]}
            else:
                body = orig(payload)
            rec = {"latency_s": time.monotonic() - t0, "usage": body.get("usage"),
                   "raw_text": ((body.get("choices") or [{}])[0].get("message") or {}).get("content"),
                   "payload": payload}
            with self.lock:
                self.captured[key] = rec
            return body

        appmod._post_vlm_chat = wrapped

    @contextlib.contextmanager
    def _offline_probe(self):
        """Dry-run only: answer the app's GET /models probe without a socket."""
        import urllib.request
        real = urllib.request.urlopen

        def fake(req, *a, **k):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.endswith("/models"):
                return contextlib.nullcontext()
            raise RuntimeError(f"dry-run blocked network call to {url}")
        urllib.request.urlopen = fake
        try:
            yield
        finally:
            urllib.request.urlopen = real

    def predict(self, row: dict) -> tuple[dict, Optional[dict]]:
        a = self.ctx.args
        hint = hint_alliance(row, a.hint_source)
        mode = "missing-true" if a.options == "missing-true" else None
        pred = _base_pred("freeread", row, self.ctx, hint, self.ctx.vlm_model, mode)
        red, blue = offered_roster(row, a.options if mode else "alliance")
        if mode:
            pred["roster_offered"] = red + blue
            pred["removed_team"] = true_team(row)
        jpg = self.ctx.crop(row)
        key = hashlib.sha1(app_style_data_uri(jpg).encode()).hexdigest()
        # stdout (the app's per-read prints) is silenced once per arm by the
        # caller: redirect_stdout per call is not thread-safe.
        guard = self._offline_probe() if self.dry else contextlib.nullcontext()
        with guard:
            res = self.app._vllm_read_bumpers(
                [base64.b64encode(jpg).decode()], red + blue, blue, red,
                expected_alliances=[hint])[0]
        with self.lock:
            cap = self.captured.pop(key, None)
        pred.update({
            "pred_team": res.get("team_number"),
            "freeread": {k: res.get(k) for k in
                         ("team_number", "alliance", "confidence", "raw_read",
                          "digit_prob", "surface")},
            "error": res.get("error"),
        })
        if cap:
            pred.update({"latency_s": round(cap["latency_s"], 3),
                         "latencies_s": [round(cap["latency_s"], 3)],
                         "usage": cap["usage"], "attempts": 1})
            pred["freeread"]["raw_text"] = cap["raw_text"]
        elif not pred["error"]:
            pred["error"] = "payload capture missed (image key mismatch)"
        return pred, (cap or {}).get("payload")


# ---- shared choice-arm plumbing ----------------------------------------------

def _finish_choice_pred(pred: dict, perms: list[dict], blank: bool) -> None:
    """Average per-permutation distributions (raw and calibrated) into the row."""
    pred["perms"] = perms
    probs = average_distributions([pm.get("probs") for pm in perms])
    top, top_p = top_choice(probs)
    pred.update({"probs": probs, "top": top, "top_p": top_p,
                 "pred_team": top if top is not None and top not in ESCAPES else None})
    if blank:
        cal = average_distributions([pm.get("probs_cal") for pm in perms])
        pred["probs_cal"] = cal
        pred["top_cal"], pred["top_p_cal"] = top_choice(cal)
    lats = pred["latencies_s"]
    pred["latency_s"] = round(statistics.fmean(lats), 3) if lats else None


# ---- vllm-choice ------------------------------------------------------------

class VllmChoice:
    def __init__(self, ctx: Ctx, dry_run: bool):
        self.ctx = ctx
        self.dry = dry_run
        self.k = ctx.perms_for("vllm-choice")
        self.blank = bool(ctx.args.blank_calibrate)

    def _score(self, payload: dict, letters: str, pred: dict, usages: list):
        body, lat, attempts = post_json(f"{self.ctx.args.server_url}/chat/completions",
                                        payload, retries=self.ctx.args.retries)
        usages.append(body.get("usage"))
        pred["attempts"] = (pred["attempts"] or 0) + attempts
        choice = (body.get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        raw = (choice.get("message") or {}).get("content")
        if not content:
            raise ArmError("no logprobs in response")
        dist, mass = letter_distribution(content[0].get("top_logprobs") or [], letters)
        if dist is None:
            raise ArmError(f"no option letter in top_logprobs (raw={raw!r})")
        return dist, mass, raw, lat

    def predict(self, row: dict) -> tuple[dict, dict]:
        a = self.ctx.args
        hint = hint_alliance(row, a.hint_source)
        base = shuffle_options(build_options(row, a.options, hint), row["id"], a.option_seed)
        pred = _base_pred("vllm-choice", row, self.ctx, hint, self.ctx.vlm_model, a.options)
        pred["option_ids"] = [o["id"] for o in base]
        pred["permutations"] = self.k
        pred["blank_calibrated"] = self.blank
        jpg = self.ctx.crop(row)
        uri = app_style_data_uri(jpg)
        blank_uri = app_style_data_uri(gray_jpeg_like(jpg)) if self.blank else None
        first_payload = None
        perms, usages = [], []
        try:
            for order in permutation_orders(base, self.k):
                letters = LETTERS[:len(order)]
                prompt = choice_prompt(order, a.options, hint)
                payload = vllm_choice_payload(self.ctx.vlm_model, self.ctx.vlm_extra, uri, prompt)
                first_payload = first_payload or payload
                if self.dry:
                    continue
                pm = {"option_ids": [o["id"] for o in order], "letters": letters}
                dist, mass, raw, lat = self._score(payload, letters, pred, usages)
                pred["latencies_s"].append(round(lat, 3))
                pm.update({"probs": {o["id"]: dist[L] for o, L in zip(order, letters)},
                           "letter_mass": round(mass, 4), "raw_text": raw})
                if self.blank:
                    bp = vllm_choice_payload(self.ctx.vlm_model, self.ctx.vlm_extra,
                                             blank_uri, prompt)
                    bdist, bmass, _r, _l = self._score(bp, letters, pred, usages)
                    pm["blank_probs"] = {o["id"]: bdist[L] for o, L in zip(order, letters)}
                    pm["blank_letter_mass"] = round(bmass, 4)
                    pm["probs_cal"] = blank_calibrate(pm["probs"], pm["blank_probs"])
                perms.append(pm)
        except ArmError as e:
            pred["error"] = str(e)
        pred["usage"] = _sum_usage(usages)
        if not self.dry:
            _finish_choice_pred(pred, perms, self.blank)
        pred["requests_per_crop"] = len(permutation_orders(base, self.k)) * (2 if self.blank else 1)
        return pred, first_payload


# ---- clef / clef-flash (Workers AI) ----------------------------------------

class Clef:
    def __init__(self, ctx: Ctx, model_key: str, dry_run: bool):
        self.ctx = ctx
        self.key = model_key
        self.spec = CLEF_MODELS[model_key]
        self.dry = dry_run
        self.k = ctx.perms_for(model_key)
        self.blank = bool(ctx.args.blank_calibrate_clef)
        self.env_loaded = load_env_file(ENV_FILE)
        self.account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.token = os.environ.get("CLOUDFLARE_API_TOKEN")
        self.missing = [n for n, v in zip(CF_ENV_KEYS, (self.account, self.token)) if not v]
        if self.missing and not dry_run:
            raise ArmError(f"{model_key}: missing {', '.join(self.missing)} "
                           "(set in the environment or the repo-root .env)")

    def url(self, redact: bool = False) -> str:
        acct = "{CLOUDFLARE_ACCOUNT_ID}" if redact or not self.account else self.account
        return CF_URL.format(account=acct, route=self.spec["route"])

    def requests_per_crop(self, n_options: int) -> int:
        return max(1, min(self.k, n_options)) * (2 if self.blank else 1)

    def build(self, row: dict):
        a = self.ctx.args
        hint = hint_alliance(row, a.hint_source)
        base = shuffle_options(build_options(row, a.options, hint), row["id"], a.option_seed)
        jpg = self.ctx.crop(row)
        img, w, h = clef_image(jpg)
        return hint, base, jpg, img, w, h

    def _post(self, payload: dict, pred: dict, usages: list) -> dict:
        body, lat, attempts = post_json(
            self.url(), payload, headers={"Authorization": f"Bearer {self.token}"},
            retries=self.ctx.args.retries)
        pred["attempts"] = (pred["attempts"] or 0) + attempts
        pred["latencies_s"].append(round(lat, 3))
        if isinstance(body, dict) and body.get("success") is False:
            raise ArmError(f"API error: {body.get('errors')}")
        res = body.get("result", body) if isinstance(body, dict) else {}
        usages.append(res.get("usage") or body.get("usage"))
        return res.get("answers") or {}

    @staticmethod
    def _team_probs(ans: dict) -> dict:
        probs = (ans.get("team") or {}).get("probabilities")
        if not probs:
            raise ArmError(f"no team probabilities in answers: {str(ans)[:300]}")
        return {str(k): float(v) for k, v in probs.items()}

    def predict(self, row: dict) -> tuple[dict, dict]:
        a = self.ctx.args
        hint, base, jpg, img, w, h = self.build(row)
        pred = _base_pred(self.key, row, self.ctx, hint, self.spec["route"], a.options)
        pred["option_ids"] = [o["id"] for o in base]
        pred["permutations"] = self.k
        pred["blank_calibrated"] = self.blank
        pred["image_wh"] = [w, h]
        blank_img = clef_image(gray_jpeg_like(jpg))[0] if self.blank else None
        orders = permutation_orders(base, self.k)
        first_payload = None
        est = 0
        perms, usages, legible, surfaces = [], [], [], []
        try:
            for order in orders:
                payload = clef_payload(self.key, row, img, order, a.options, hint)
                first_payload = first_payload or payload
                est += clef_estimate_tokens(payload, w, h) * (2 if self.blank else 1)
                if self.dry:
                    continue
                pm = {"option_ids": [o["id"] for o in order]}
                ans = self._post(payload, pred, usages)
                pm["probs"] = self._team_probs(ans)
                pm["api_choice"] = (ans.get("team") or {}).get("choice")
                pm["api_confidence"] = (ans.get("team") or {}).get("confidence")
                if (ans.get("legible") or {}).get("noul") is not None:
                    legible.append(float(ans["legible"]["noul"]))
                if (ans.get("surface") or {}).get("probabilities"):
                    surfaces.append(ans["surface"]["probabilities"])
                if self.blank:
                    bp = clef_payload(self.key, row, blank_img, order, a.options, hint)
                    pm["blank_probs"] = self._team_probs(self._post(bp, pred, usages))
                    pm["probs_cal"] = blank_calibrate(pm["probs"], pm["blank_probs"])
                perms.append(pm)
        except ArmError as e:
            pred["error"] = str(e)
        pred["est_input_tokens"] = est
        pred["est_cost_usd"] = est * self.spec["usd_per_m_input"] / 1e6
        pred["usage"] = _sum_usage(usages)
        if pred["usage"] and pred["usage"].get("input_tokens") is not None:
            pred["est_cost_usd"] = pred["usage"]["input_tokens"] * self.spec["usd_per_m_input"] / 1e6
        if not self.dry:
            _finish_choice_pred(pred, perms, self.blank)
            pred["legible_p"] = statistics.fmean(legible) if legible else None
            pred["surface_probs"] = average_distributions(surfaces)
        pred["requests_per_crop"] = self.requests_per_crop(len(base))
        return pred, first_payload


def make_arm(name: str, ctx: Ctx, dry_run: bool):
    if name in ("freeread", "vllm-choice"):
        ctx.resolve_vllm(dry_run)
    if name == "freeread":
        return FreeRead(ctx, dry_run)
    if name == "vllm-choice":
        return VllmChoice(ctx, dry_run)
    if name in CLEF_MODELS:
        return Clef(ctx, name, dry_run)
    raise ArmError(f"unknown arm {name!r}; valid: {', '.join(ARMS)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_manifest(bench_dir: Path) -> list[dict]:
    path = bench_dir / "manifest.jsonl"
    if not path.exists():
        sys.exit(f"no manifest at {path}; run scripts/build_bumper_crop_bench.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_rows(rows: list[dict], only_labelled: bool, limit: Optional[int],
                mode: str = "alliance") -> list[dict]:
    if only_labelled:
        rows = [r for r in rows if (r.get("label_status") or "unlabelled") != "unlabelled"]
    if mode == "missing-true":
        rows = [r for r in rows if true_team(r) is not None]
    if limit is not None and limit < len(rows):
        # Stable hash order so a small --limit still spans many matches.
        rows = sorted(rows, key=lambda r: _stable_int("limit", r["id"]))[:limit]
    return rows


def run_arm(name: str, arm, rows: list[dict], out_path: Path, concurrency: int) -> list[dict]:
    preds: list[dict] = []
    t0 = time.time()
    # Progress goes to stderr; stdout is silenced for the arm's duration (the
    # web.app reader prints per crop). Done once here, never per thread.
    with open(out_path, "w") as fh, contextlib.redirect_stdout(io.StringIO()), \
            ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = {pool.submit(arm.predict, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                pred, _ = fut.result()
            except Exception as e:  # an arm bug must not kill the whole run
                pred = {"id": row["id"], "arm": name, "error": f"{type(e).__name__}: {e}"}
            fh.write(json.dumps(pred) + "\n")
            fh.flush()
            preds.append(pred)
            if i % 10 == 0 or i == len(rows):
                errs = sum(1 for p in preds if p.get("error"))
                print(f"  [{name}] {i}/{len(rows)} done, {errs} errors "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return preds


def _label(pred: dict) -> str:
    arm, mode = pred.get("arm"), pred.get("options_mode")
    return f"{arm}:{mode}" if mode else str(arm)


def dry_run(arm_names: list[str], ctx: Ctx, rows: list[dict]) -> None:
    a = ctx.args
    row = rows[0]
    print(f"DRY RUN - no network calls. {len(rows)} crops selected, options={a.options}, "
          f"hint={a.hint_source}. Example crop: {row['id']} ({row['width']}x{row['height']}, "
          f"stratum={row['stratum']}, baseline={row.get('baseline_matched_team')})\n")
    for name in arm_names:
        arm = make_arm(name, ctx, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            pred, payload = arm.predict(row)
        print(f"=== {name} ===")
        if isinstance(arm, Clef):
            print(f"POST {arm.url(redact=True)}")
            print("Authorization: Bearer <CLOUDFLARE_API_TOKEN>   [credentials: "
                  + ("missing " + ", ".join(arm.missing) if arm.missing else "present")
                  + (f"; loaded from .env: {', '.join(arm.env_loaded)}" if arm.env_loaded else "")
                  + "]")
            total = 0.0
            for r in rows:
                rp, _ = arm.predict(r)
                total += rp["est_cost_usd"]
            print(f"permutations={arm.k} blank={arm.blank} -> {pred['requests_per_crop']} "
                  f"request(s)/crop; est. input tokens this crop: {pred['est_input_tokens']} "
                  f"(image {pred['image_wh'][0]}x{pred['image_wh'][1]}); "
                  f"est. cost for {len(rows)} crops: ${total:.4f}")
        else:
            print(f"POST {a.server_url}/chat/completions   model={ctx.vlm_model}{ctx.vlm_model_note}")
            if isinstance(arm, VllmChoice):
                print(f"permutations={arm.k} blank={arm.blank} -> "
                      f"{pred['requests_per_crop']} request(s)/crop")
        if pred.get("option_ids"):
            print(f"option order (perm 0): {pred['option_ids']}")
        if pred.get("roster_offered") is not None:
            print(f"roster offered: {pred['roster_offered']} (removed {pred.get('removed_team')})")
        print(json.dumps(truncate_payload(payload), indent=2))
        print()


def cmd_run(args) -> None:
    bench_dir = Path(args.bench)
    manifest_rows = load_manifest(bench_dir)
    rows = select_rows(manifest_rows, args.only_labelled, args.limit, args.options)
    if not rows:
        sys.exit("no rows selected (labelled-only with no labels yet?)")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; valid: {', '.join(ARMS)}")
    ctx = Ctx(args, bench_dir)
    if args.dry_run:
        dry_run(arm_names, ctx, rows)
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    res_dir = bench_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    manifest = {r["id"]: r for r in manifest_rows}
    summary = {"timestamp": stamp, "argv": sys.argv[1:], "n_rows": len(rows),
               "options": args.options, "hint_source": args.hint_source, "arms": {}}
    scored = []
    for name in arm_names:
        try:
            arm = make_arm(name, ctx, dry_run=False)
        except ArmError as e:
            print(f"[{name}] ERROR: {e}", file=sys.stderr)
            summary["arms"][name] = {"error": str(e)}
            continue
        out_path = res_dir / f"{name}-{stamp}.jsonl"
        print(f"[{name}] {len(rows)} crops -> {out_path}", file=sys.stderr)
        preds = run_arm(name, arm, rows, out_path, args.concurrency)
        variants = score_file_variants(preds, manifest, _label(preds[0]) if preds else name)
        model = ctx.vlm_model if name in ("freeread", "vllm-choice") else CLEF_MODELS[name]["route"]
        summary["arms"][name] = {"file": out_path.name, "model": model,
                                 "scores": dict(variants)}
        scored.extend(variants)
    (res_dir / f"run-{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    if scored:
        print(format_table(scored) if args.score else
              "\n".join(f"{n}: n={m['n']} errors={m['errors']} lat_med={_f(m['latency_med_s'])}s "
                        f"cost=${m['est_cost_usd']:.4f}" for n, m in scored))


_RESULT_RE = re.compile(r"^(?P<arm>.+)-(?P<stamp>\d{8}-\d{6})\.jsonl$")


def latest_results(res_dir: Path) -> list[Path]:
    best: dict[str, Path] = {}
    for p in sorted(res_dir.glob("*.jsonl")):
        m = _RESULT_RE.match(p.name)
        if m and m.group("arm") in ARMS:
            best[m.group("arm")] = p  # sorted -> last stamp wins
    return [best[a] for a in ARMS if a in best]


def cmd_score(args) -> None:
    bench_dir = Path(args.bench)
    manifest = {r["id"]: r for r in load_manifest(bench_dir)}
    files = [Path(f) for f in args.files] or latest_results(bench_dir / "results")
    if not files:
        sys.exit("no results files")
    scored = []
    for f in files:
        preds = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if preds:
            scored.extend(score_file_variants(preds, manifest, f"{_label(preds[0])} {f.stem[-15:]}"))
    n_lab = sum(1 for r in manifest.values() if (r.get("label_status") or "unlabelled") != "unlabelled")
    print(f"manifest: {len(manifest)} crops, {n_lab} labelled")
    print(format_table(scored))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(scored), indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="run arms over the bench (default command)")
    run.add_argument("--arms", default="freeread,vllm-choice",
                     help=f"comma list of {', '.join(ARMS)}")
    run.add_argument("--options", choices=OPTION_MODES, default="alliance")
    run.add_argument("--hint-source", choices=HINT_SOURCES, default="baseline")
    run.add_argument("--option-seed", type=int, default=0, help="per-crop option shuffle seed")
    run.add_argument("--permutations", type=int, default=None,
                     help="cyclic option orderings averaged per crop "
                          "(default: 3 for vllm-choice, 1 for clef arms)")
    run.add_argument("--blank-calibrate", action="store_true",
                     help="vllm-choice: contextual calibration against a mid-gray crop")
    run.add_argument("--blank-calibrate-clef", action="store_true",
                     help="same for the clef arms (doubles paid requests)")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--only-labelled", action="store_true")
    run.add_argument("--dry-run", action="store_true",
                     help="print one truncated payload per arm; NO network calls")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--retries", type=int, default=5)
    run.add_argument("--score", action="store_true", help="print the comparison table after the run")
    run.add_argument("--server-url", default=VLM_SERVER_URL, help="vLLM base URL (VLM_SERVER_URL)")
    run.add_argument("--vllm-model", default=None,
                     help="served model id for the local arms (default: first id from GET /v1/models)")
    run.add_argument("--no-thinking", action="store_true",
                     help="inject chat_template_kwargs.enable_thinking=false (e.g. Qwen3.5-9B)")
    run.add_argument("--bench", default=str(BENCH_DIR))
    sc = sub.add_parser("score", help="score results files against the manifest labels")
    sc.add_argument("files", nargs="*", help="results/*.jsonl (default: latest per arm)")
    sc.add_argument("--json", default=None, help="also dump metrics as JSON here")
    sc.add_argument("--bench", default=str(BENCH_DIR))
    return ap


def main(argv: Optional[list] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "score", "-h", "--help"):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    (cmd_score if args.cmd == "score" else cmd_run)(args)


if __name__ == "__main__":
    main()
