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
jev          TypeSafe's Jev decision model (api.typesafe.ai/v1/systemone, image
             capable): one request per crop (per permutation, default 1),
             --frames single only. Asks `subject` (bumper / no_digits / printed /
             not_robot) BEFORE `team`; the team options are the offered roster
             plus four escapes (not_robot, unreadable, printed, other) instead
             of none/other - rows carry escape_ids, scoring treats all four as
             escapes. Scored as `jev:<opts>` and `jev:<opts>+subj` (team kept
             only when the subject argmax is "bumper"). Crops are rescaled into
             the API's 280-1568 px window (gray letterbox when the aspect ratio
             needs it). Needs JEV_TYPESAFE_API_TOKEN (env, else repo-root .env).

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

Multi-frame inputs (``--frames``, clef arms)
------------------------------------------
Tracklets (``tracklets.jsonl`` + ``tracklets/<id>/<offset>.jpg``, built by
scripts/build_tracklet_bench.py) follow the cropped bumper +-1.5 s through the
source video at ~12 fps.
single   one image: the bench crop (default; byte-identical payloads).
multi4   up to 4 images (the API max): the bench crop + the sharpest frame of
         each third of the remaining tracklet frames, in time order, each
         prepped like the single crop (clef_image); the state says they show
         the SAME robot over ~N seconds.
multi8   freeread only: up to 8 images (the bench crop + 7 frames, one per
         time seventh).
freeread multi4/multi8: ONE chat request carrying all the images (each with the
         incumbent's 512-px LANCZOS prep), the incumbent prompt reworded for N
         time-ordered moments of the same robot, and the incumbent's
         _vlm_chat_payload settings (guided JSON _BUMPER_READ_SCHEMA, logprob
         confidence, roster hint); the reply goes through the incumbent's
         parse -> roster match -> confidence blend. The server must allow
         several images per prompt (--limit-mm-per-prompt '{"image": 8}').
--prompt escape (freeread, --frames single only; default ``incumbent``): replaces
         the incumbent prompt with one that admits detector junk (balls, piles,
         structures) and asks for "subject" (bumper / no_digits / printed /
         not_robot) BEFORE the digits (ESCAPE_READ_SCHEMA, subject first), with a
         neutral roster line ("Teams on the red alliance in this match: ...").
         Any subject but "bumper" forces team=None. Rows carry prompt_variant and
         freeread.subject; scores are labelled ``freeread+escape``.
video    clef-omni only: every tracklet frame, letterboxed to a fixed canvas
         (long side 512, even sides), encoded H.264 yuv420p at 2 fps - the
         rate clef-omni samples video at, so each crop is one sampled frame -
         sent as ``videos: ["data:video/mp4;base64,..."]``; no image unless
         ``--video-with-still``.
registered  every arm: the tracklet registered + fused into ONE image
         (web/bumper_register.py: interlaced frames split into fields, ECC
         affine alignment to the offset-0 anchor with bad alignments dropped,
         per-pixel anti-ghost merge, mild unsharp), then the app's crop prep
         (integer upscale to >= 384 px tall, JPEG q90 - _bumper_crop_b64) and
         read as the arm's single crop. ``--reg-opts '<json>'`` overrides
         register_and_fuse kwargs. Rows carry reg_fused / reg_samples /
         reg_rejected / reg_fallback.
anchor   every arm: the same pipeline on the anchor frame alone (deinterlace +
         upscale + unsharp + app prep, no fusion) - the control that separates
         the prep's effect (single vs anchor) from the fusion's (anchor vs
         registered).
--labels tracklet|truth|<path> scores against tracklet_labels.jsonl /
track_truth.jsonl / a file instead of the manifest's crop labels.
A crop without a usable tracklet (no source video, not localised, < 2 frames)
falls back to the single-frame request. Every row records ``frames_used`` and
``tracklet_frames``; ``score --subset tracklet`` restricts every file to the
crop ids whose tracklet has >= 3 frames, so single and multi-frame runs are
compared on identical crops.

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

Credentials: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN (clef arms) and
JEV_TYPESAFE_API_TOKEN (jev) from the environment, else from the repo-root .env
(never printed, never overriding the environment).

Usage
-----
    .venv/bin/python scripts/bench_bumper_readers.py --dry-run --arms clef-flash,vllm-choice
    .venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice --limit 40 --score
    .venv/bin/python scripts/bench_bumper_readers.py --arms freeread,vllm-choice --options missing-true --score
    .venv/bin/python scripts/bench_bumper_readers.py --arms clef-flash --options decoy --score
    .venv/bin/python scripts/bench_bumper_readers.py --arms clef-flash --frames multi4 --options decoy
    .venv/bin/python scripts/bench_bumper_readers.py --arms clef-omni --frames video --dry-run
    .venv/bin/python scripts/bench_bumper_readers.py --arms jev --options decoy --score
    .venv/bin/python scripts/bench_bumper_readers.py score --subset tracklet results/clef-*.jsonl
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

ARMS = ("freeread", "vllm-choice", "clef-flash", "clef", "clef-omni", "jev", "local",
        "local-clef")
# `local` = the jev payload POSTed to a self-hosted /v1/systemone server
# (scripts/decision_local/systemone_shim.py); --local-label names the results file.
# `local-clef` = the clef-flash arm's Workers AI payload (legible noul + surface + team,
# images list) POSTed to the shim's /v1/clef route: Clef's native request format.
LOCAL_SYSTEMONE_URL = os.environ.get("LOCAL_SYSTEMONE_URL", "http://127.0.0.1:8100/v1/systemone")
OPTION_MODES = ("alliance", "decoy", "missing-true")
HINT_SOURCES = ("baseline", "label", "none")
ESCAPES = ("none", "other")
ESCAPE_TEXT = {
    "none": ("No legible team number on a robot bumper (blurry, occluded, cut off, "
             "not a robot, or the only digits are printed on field equipment)"),
    "other": "A legible team number that is NOT any of the listed teams",
}
# jev arm escapes (replace none/other); the text rides on the option dict.
JEV_ESCAPES = ("not_robot", "unreadable", "printed", "other")
JEV_ESCAPE_TEXT = {
    "not_robot": ("No robot bumper in the image: balls, field structures, signs, people or "
                  "empty floor"),
    "unreadable": ("A robot bumper, but the digits can't be read: blurred, cut off, turned "
                   "away or too small"),
    "printed": "The digits are printed on a crate, bin, sign or field structure, not a robot bumper",
    "other": "Legible bumper digits that are NOT any of the listed teams",
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THRESHOLDS = (0.5, 0.7, 0.9)
BLANK_EPS = 1e-3
BLANK_GRAY = (128, 128, 128)
DEFAULT_PERMS = {"vllm-choice": 3, "clef-flash": 1, "clef": 1, "clef-omni": 1, "jev": 1}
CF_ENV_KEYS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
JEV_ENV_KEYS = ("JEV_TYPESAFE_API_TOKEN",)

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
# TypeSafe Jev (verified live 2026-11-05): POST {"model", "state", "assets":
# {ref: {"type": "image", "format": "jpg", "data_base64"}}, "questions"} with
# `Authorization: Bearer` and (any image request) `TypeSafe-Danger-Zone-Images:
# true`. Every asset ref must appear in the state or a question's instructions.
# Images: JPEG/PNG, each side 280-1568 px, 78,400-2,458,624 px total; out-of-range
# images are rejected, not resized. Answers: choice -> {"choice", "confidence",
# "probabilities"}; response {"model": "jev-1.14.0", "answers", "usage"}.
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_USD_PER_M_INPUT = 0.042
JEV_MIN_SIDE = 280
JEV_MAX_SIDE = 1568
JEV_PAD_GRAY = (128, 128, 128)
JEV_ASSET_REF = "crop_image"
JEV_SUBJECTS = ("bumper", "no_digits", "printed", "not_robot")
CLEF_MIN_SIDE = 256
CLEF_MAX_IMAGE_TOKENS = 1024
CLEF_MAX_IMAGE_BYTES = 4 * 1024 * 1024

# Multi-frame arms (--frames). Tracklets come from scripts/build_tracklet_bench.py
# (<bench>/tracklets.jsonl + tracklets/<id>/<offset>.jpg).
# clef-omni "videos" input, verified 2026-10-12 against
#   https://developers.cloudflare.com/workers-ai/models/clef-omni/schema-input.json:
# max 2 videos, 16 MiB and 60 s each, data URL "data:video/mp4;base64,..." (or
# {"content_type", "base64"}), sampled at 2 frames per second, up to 256 input
# tokens per second of video (~15,400/min at max resolution; 480p ~144/s).
FRAME_MODES = ("single", "multi4", "multi8", "video", "registered", "anchor")
# registered: the tracklet registered + fused into ONE image (web/bumper_register.py)
# that every arm reads as its single crop; anchor: the same upscale + unsharp + app
# prep on the anchor frame alone (the no-fusion control).
REGISTERED_MODES = ("registered", "anchor")
REG_DEFAULTS: dict = {}         # register_and_fuse overrides (its defaults otherwise)
APP_CROP_MIN_H = 384            # web/app.py _bumper_crop_b64 upscale floor
MULTI_IMAGES = {"multi4": 4, "multi8": 8}   # images per request incl. the bench crop
FREEREAD_FRAME_MODES = ("single", "multi4", "multi8")
VIDEO_ARMS = ("clef-omni",)
TRACKLETS_FILE = "tracklets.jsonl"
MULTI_MAX_IMAGES = 4          # API max images per request
MIN_TRACKLET_FRAMES = 2       # fewer -> the crop falls back to single-frame
SUBSET_MIN_FRAMES = 3         # `score --subset tracklet`
VIDEO_FPS = 2                 # = Clef-omni's sampling rate: one crop per sampled frame
VIDEO_LONG_SIDE = 512
VIDEO_MAX_S = 60
VIDEO_MAX_BYTES = 16 * 1024 * 1024
VIDEO_TOKENS_PER_S = 256
FFMPEG = os.environ.get("FFMPEG", "/usr/bin/ffmpeg")


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
    return opt.get("text") or ESCAPE_TEXT[opt["id"]]


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


def _escapes(pred: dict) -> tuple:
    """Option ids that count as "no team asserted" for this row: the row's own
    ``escape_ids`` (jev), else the classic none/other."""
    return tuple(pred.get("escape_ids") or ESCAPES)


def subject_gated_view(pred: dict) -> dict:
    """The row with the team withheld unless the jev ``subject`` argmax is
    "bumper" (the ``+subj`` variant). A blocked team top becomes the escape
    "gated" so it scores as an abstention on legible crops and as a correct
    escape on negatives."""
    q = dict(pred)
    esc = _escapes(pred)
    if pred.get("subject_top") != "bumper" and pred.get("top") not in esc:
        q["top"] = "gated"
        q["escape_ids"] = list(esc) + ["gated"]
    return q


def calibrated_view(pred: dict) -> dict:
    """The same prediction row with the blank-calibrated distribution swapped in."""
    q = dict(pred)
    q["probs"] = pred.get("probs_cal")
    q["top"], q["top_p"] = top_choice(q["probs"])
    q["pred_team"] = q["top"] if q["top"] not in _escapes(pred) else None
    q["perms"] = [dict(pm, probs=pm.get("probs_cal")) for pm in pred.get("perms") or []]
    return q


def _pred_team(pred: dict) -> Optional[str]:
    if pred.get("probs") is not None:
        top = pred.get("top")
        return top if top is not None and top not in _escapes(pred) else None
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
            listed_mass.append(sum(v for k, v in probs.items() if k not in _escapes(p)))
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
            corrects.append(p.get("top") in _escapes(p) if st != "legible"
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
    if any(p.get("subject_probs") is not None for p in preds):
        out.append((label + "+subj", score_predictions(
            [subject_gated_view(p) if p.get("subject_probs") is not None else p
             for p in preds], manifest)))
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


# ---- jev (TypeSafe) request builders ---------------------------------------

def jev_image(jpg: bytes) -> tuple[bytes, int, int, bool]:
    """Prep a crop for Jev: each side must land in 280..1568 px (the API rejects
    rather than resizes). Scale (LANCZOS) so the short side reaches 280 and the
    long side stays <= 1568; when the aspect ratio makes both impossible, shrink
    the long side to 1568 and letterbox-pad the short side to 280 with neutral
    gray. Unchanged JPEGs pass through byte-identical. Returns
    (jpeg bytes, w, h, padded)."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpg))
    w, h = img.size
    scale = 1.0
    if min(w, h) < JEV_MIN_SIDE:
        scale = JEV_MIN_SIDE / min(w, h)
    if max(w, h) * scale > JEV_MAX_SIDE:
        scale = JEV_MAX_SIDE / max(w, h)
    nw = min(JEV_MAX_SIDE, max(1, int(round(w * scale))))
    nh = min(JEV_MAX_SIDE, max(1, int(round(h * scale))))
    cw, ch = max(nw, JEV_MIN_SIDE), max(nh, JEV_MIN_SIDE)
    if (cw, ch) == (w, h) and img.format == "JPEG":
        return jpg, w, h, False
    img = img.convert("RGB")
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    padded = (cw, ch) != (nw, nh)
    if padded:
        canvas = Image.new("RGB", (cw, ch), JEV_PAD_GRAY)
        canvas.paste(img, ((cw - nw) // 2, (ch - nh) // 2))
        img = canvas
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), cw, ch, padded


def jev_options(row: dict, mode: str, hint: str) -> list[dict]:
    """Unshuffled jev option set: the offered roster teams + the four JEV_ESCAPES
    (replacing the classic none/other)."""
    opts = [o for o in build_options(row, mode, hint) if o["kind"] == "team"]
    return opts + [{"id": e, "kind": "escape", "text": JEV_ESCAPE_TEXT[e]} for e in JEV_ESCAPES]


def jev_roster_line(row: dict, mode: str, hint: str) -> str:
    red, blue = offered_roster(row, mode)
    expected = hint if mode in ("alliance", "missing-true") else "unknown"
    return freeread_escape_hint_line(expected, red, blue, red + blue)


def jev_state(row: dict, mode: str, hint: str, ref: str = JEV_ASSET_REF) -> dict:
    """State: the asset ref + the escape prompt's framing (detector crop that may
    not be a robot, what a bumper looks like, neutral roster line)."""
    return {
        "crop": ref,
        "context": (
            "`crop` is an automatic crop from a robot detector running on FRC match video. "
            "It usually shows one FRC robot, but the detector makes mistakes: the crop may "
            "instead show game pieces (balls), a pile of balls, field structures, signs, "
            "people, or only part of a robot. An FRC robot bumper is a solid RED or BLUE "
            "fabric band around the base of a robot with a WHITE team number (1 to 5 digits, "
            "e.g. 294, 2485, 4322) printed on it. "
            f"{jev_roster_line(row, mode, hint)} "
            "The team list is only for matching partially visible digits. Never pick a team "
            "unless you can see its digits on a bumper in `crop`."),
    }


def jev_questions(options: list[dict]) -> dict:
    """``subject`` first, then ``team`` (dict order is the JSON order)."""
    return {
        "subject": {
            "type": "choice",
            "instructions": "What does `crop` show?",
            "criteria": {
                "bumper": "A robot bumper with white digits you can actually see",
                "no_digits": ("A robot, but its bumper digits are not visible or not "
                              "readable (blurred, cut off, turned away, too small)"),
                "printed": ("Digits printed on a crate, bin, sign or field structure, "
                            "not on a robot bumper"),
                "not_robot": ("No robot at all (balls, field elements, people, empty "
                              "floor)"),
            },
        },
        "team": {
            "type": "choice",
            "instructions": ("Which option matches the white team-number digits on the "
                             "robot bumper in `crop`? Choose an escape option unless you "
                             "can actually see a listed team's digits on a bumper."),
            "criteria": {o["id"]: option_text(o) for o in options},
        },
    }


def jev_payload(row: dict, img_bytes: bytes, options: list[dict], mode: str, hint: str,
                ref: str = JEV_ASSET_REF) -> dict:
    return {
        "model": JEV_MODEL,
        "state": jev_state(row, mode, hint, ref),
        "assets": {ref: {"type": "image", "format": "jpg",
                         "data_base64": base64.b64encode(img_bytes).decode()}},
        "questions": jev_questions(options),
    }


def jev_estimate_tokens(payload: dict, w: int, h: int) -> int:
    """Rough input tokens, fitted to 5 live crops (2026-11-05): ~1 per 415 image
    px + ~1.9 chars/token for state+questions (the JSON eats extra tokens)."""
    text = len(json.dumps(payload["state"])) + len(json.dumps(payload["questions"]))
    return int(w * h / 415 + text / 1.9)


# --------------------------------------------------------------------------- #
# Multi-frame inputs (--frames multi4 | video)
# --------------------------------------------------------------------------- #

def load_tracklets(bench_dir: Path) -> dict[str, dict]:
    """crop id -> tracklets.jsonl row, for status == "ok" rows only."""
    path = Path(bench_dir) / TRACKLETS_FILE
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            if t.get("status") == "ok":
                out[t["id"]] = t
    return out


def usable_tracklet(trk: Optional[dict], min_frames: int = MIN_TRACKLET_FRAMES) -> bool:
    return bool(trk) and trk.get("status") == "ok" and len(trk.get("frames") or []) >= min_frames


def tracklet_subset_ids(tracklets: dict[str, dict], min_frames: int = SUBSET_MIN_FRAMES) -> set:
    """Crop ids whose tracklet has >= min_frames frames (`score --subset tracklet`)."""
    return {i for i, t in tracklets.items() if usable_tracklet(t, min_frames)}


def select_multi_frames(frames: list[dict], n_other: int = MULTI_MAX_IMAGES - 1) -> list[dict]:
    """The ``n_other`` tracklet frames shown next to the bench crop: the centre
    frame (offset 0, which IS the bench crop) is excluded, the remaining frames
    are split into ``n_other`` contiguous time chunks of near-equal size, and the
    sharpest (Laplacian variance; ties -> nearer the centre) of each chunk is
    taken. Returned in time order."""
    others = sorted((f for f in frames if f["offset"] != 0), key=lambda f: f["offset"])
    if len(others) <= n_other:
        return others
    n = len(others)
    picks = []
    for i in range(n_other):
        chunk = others[(i * n) // n_other:((i + 1) * n) // n_other]
        picks.append(max(chunk, key=lambda f: (f.get("sharpness") or 0.0, -abs(f["offset"]))))
    return sorted(picks, key=lambda f: f["offset"])


def multi_frame_sequence(crop_jpg: bytes, trk: dict, bench_dir: Path,
                         n_images: int) -> list[dict]:
    """[{"offset", "bytes"}] in time order: the bench crop (offset 0) plus the
    ``n_images - 1`` frames picked by select_multi_frames."""
    picks = select_multi_frames(trk["frames"], n_images - 1)
    return sorted([{"offset": 0, "bytes": crop_jpg}]
                  + [{"offset": f["offset"], "bytes": (Path(bench_dir) / f["file"]).read_bytes()}
                     for f in picks], key=lambda d: d["offset"])


# ---- freeread multi-image read (the incumbent prompt, several moments) ------

def freeread_hint_line(expected_alliance: str, red: list, blue: list, match_teams: list) -> str:
    """The incumbent's roster-hint line (web/app.py _vllm_read_bumpers._read_one)."""
    if expected_alliance in ("blue", "red"):
        cand = ", ".join(red if expected_alliance == "red" else blue)
        return (f"Computer vision sees {expected_alliance.upper()} bumper pixels, so partial or "
                f"uncertain reads should be matched only against these {expected_alliance} "
                f"alliance teams: {cand}.\n")
    return f"Possible current-match teams are: {', '.join(match_teams)}.\n"


def freeread_multi_prompt(n: int, span_s: float, hint_line: str) -> str:
    """The incumbent bumper-read prompt reworded for ``n`` time-ordered images of
    one robot; same reply instructions, so _BUMPER_READ_SCHEMA still applies."""
    return (
        f"These {n} images show the SAME FRC robot at {n} moments over about {span_s:.1f} "
        "seconds, ordered in time, from a side or elevated camera angle. "
        "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
        "WHITE team number printed on it. FRC team numbers are 1 to 5 digits "
        "(for example 294, 2485, 4322). The digits may be readable in some images and "
        "blurred, cut off or hidden in others.\n\n"
        f"{hint_line}"
        "Carefully read the WHITE digits on the bumper, left to right, using the images "
        "where they are clearest, and give ONE reading for this robot. Report the "
        "exact number you SEE. Do not use robot paint or mechanisms for alliance color; "
        "only the bumper band matters. If the digits are clear, report them even if the "
        "color looks surprising.\n\n"
        'Reply with ONLY compact JSON and nothing else: '
        '{"team": "<the digits, or null if you cannot read them>", '
        '"color": "red" or "blue", '
        '"surface": "bumper" if the digits are on a robot\'s bumper band, '
        'or "printed" if they are printed on a crate, bin, sign, or other '
        'field structure}.'
    )


def multi_image_payload(base_payload: dict, data_uris: list[str], prompt: str) -> dict:
    """A web.app._vlm_chat_payload body with its user content swapped for
    ``data_uris`` (one image_url each, in order) followed by ``prompt``. Model,
    sampling, logprobs, response_format and extra body keys are kept."""
    out = json.loads(json.dumps(base_payload))
    out["messages"] = [{"role": "user", "content": (
        [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
        + [{"type": "text", "text": prompt}])}]
    return out


def parse_bumper_reply(text: str) -> tuple[Optional[str], str, Optional[str]]:
    """(team, color, raw surface) from a reader reply - the incumbent's parse."""
    team, color, surface = None, "unknown", None
    try:
        clean = (text or "").strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        start, end = clean.find("{"), clean.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(clean[start:end])
            team = parsed.get("team")
            color = parsed.get("color", "unknown")
            surface = parsed.get("surface")
            if team is not None:
                team = str(team)
    except Exception:
        pass
    return team, color, surface


# ---- freeread --prompt escape (explicit "not a robot" option, decided first) --

PROMPT_VARIANTS = ("incumbent", "escape")

# Property order matters: under guided decoding the model emits "subject" BEFORE
# any digits, so it decides "is there a bumper at all" first.
ESCAPE_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "enum": ["bumper", "no_digits", "printed", "not_robot"]},
        "team": {"type": ["string", "null"]},
        "color": {"type": "string", "enum": ["red", "blue", "none"]},
    },
    "required": ["subject", "team", "color"],
}


def freeread_escape_hint_line(expected_alliance: str, red: list, blue: list,
                              match_teams: list) -> str:
    """Roster line for the escape prompt: no 'computer vision sees ... pixels',
    no 'should be matched only against' (both invite guessing)."""
    if expected_alliance in ("blue", "red"):
        cand = ", ".join(red if expected_alliance == "red" else blue)
        return f"Teams on the {expected_alliance} alliance in this match: {cand}."
    return f"Teams in this match: {', '.join(match_teams)}."


def freeread_escape_prompt(hint_line: str) -> str:
    """Single-image prompt for --prompt escape (reply shape: ESCAPE_READ_SCHEMA)."""
    return (
        "This image is an automatic crop from a robot detector running on FRC match video. "
        "It usually shows one FRC robot, but the detector makes mistakes: the crop may instead "
        "show game pieces (balls), a pile of balls, field structures, signs, people, or only "
        "part of a robot.\n\n"
        "An FRC robot bumper is a solid RED or BLUE fabric band around the base of a robot "
        "with a WHITE team number (1 to 5 digits, e.g. 294, 2485, 4322) printed on it.\n\n"
        'First decide what the image shows ("subject"):\n'
        '- "bumper": a robot bumper with white digits you can actually see\n'
        '- "no_digits": a robot, but its bumper digits are not visible or not readable '
        "(blurred, cut off, turned away, too small)\n"
        '- "printed": digits printed on a crate, bin, sign or field structure, not on a robot '
        "bumper\n"
        '- "not_robot": no robot at all (balls, field elements, people, empty floor)\n\n'
        'Only if subject is "bumper", read the WHITE digits left to right and report exactly '
        "what you see. Use only the bumper band for color, not robot paint.\n"
        f"{hint_line}\n"
        "The team list is only for matching partially visible digits. Never answer with a team "
        "from the list unless you can see its digits on a bumper in this image. If subject is "
        'not "bumper", team must be null and color "none" is allowed.\n\n'
        'Reply with ONLY compact JSON: {"subject": "...", "team": "<digits>" or null, '
        '"color": "red" | "blue" | "none"}'
    )


def parse_escape_reply(text: str) -> tuple[Optional[str], str, Optional[str], Optional[str]]:
    """(team, color, surface, subject) from an escape-schema reply. Unless the
    subject is "bumper" the team is forced to None (no roster matching on junk);
    surface: bumper -> "bumper", printed -> "printed", anything else -> "unknown"."""
    team, color, _surf = parse_bumper_reply(text)
    subject = None
    try:
        clean = (text or "").strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        start, end = clean.find("{"), clean.rfind("}") + 1
        if start >= 0 and end > start:
            s = json.loads(clean[start:end]).get("subject")
            subject = str(s).strip().lower() if s is not None else None
    except Exception:
        pass
    if subject != "bumper":
        team = None
    surface = subject if subject in ("bumper", "printed") else "unknown"
    return team, color, surface, subject


def _even(x: float) -> int:
    return max(2, 2 * int(round(x / 2.0)))


def letterbox_dims(w: int, h: int, long_side: int = VIDEO_LONG_SIDE) -> tuple[int, int]:
    """Video canvas (width, height) for a w x h tracklet box: long side
    ``long_side``, aspect kept, both sides even (H.264 yuv420p)."""
    s = long_side / max(w, h)
    return _even(w * s), _even(h * s)


def letterbox(img, cw: int, ch: int):
    """PIL image scaled (LANCZOS) to fit cw x ch, centred on black."""
    from PIL import Image
    img = img.convert("RGB")
    s = min(cw / img.width, ch / img.height)
    nw, nh = max(1, int(round(img.width * s))), max(1, int(round(img.height * s)))
    canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((cw - nw) // 2, (ch - nh) // 2))
    return canvas


def encode_video(frames_jpg: list[bytes], cw: int, ch: int, fps: int = VIDEO_FPS,
                 ffmpeg: str = FFMPEG) -> bytes:
    """Letterbox each JPEG onto a cw x ch canvas and encode H.264 yuv420p mp4 at
    ``fps`` (one input frame per output frame) with ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix="bench_vid_") as td:
        for i, b in enumerate(frames_jpg):
            with Image.open(io.BytesIO(b)) as im:
                letterbox(im, cw, ch).save(Path(td) / f"{i:04d}.png")
        out = Path(td) / "out.mp4"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-framerate", str(fps), "-i", str(Path(td) / "%04d.png"),
               "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out.exists():
            raise ArmError(f"ffmpeg failed ({res.returncode}): {res.stderr[-300:]}")
        return out.read_bytes()


def video_token_estimate(n_frames: int, cw: int, ch: int, fps: int = VIDEO_FPS) -> int:
    """Rough video input tokens: 32-px blocks per frame x fps, capped at
    VIDEO_TOKENS_PER_S per second of video."""
    per_frame = max(1, round(cw / 32)) * max(1, round(ch / 32))
    return int(math.ceil(n_frames / fps * min(VIDEO_TOKENS_PER_S, per_frame * fps)))


def _span_s(offsets: list[int], fps: Optional[float]) -> float:
    return (max(offsets) - min(offsets)) / (fps or 60.0) if offsets else 0.0


def build_frame_media(frames_mode: str, crop_jpg: bytes, trk: Optional[dict],
                      bench_dir: Path, with_still: bool = False) -> dict:
    """Images / video for one crop under ``frames_mode``.

    Returns {"kind": "single"|"multi"|"video", "images": [(bytes, w, h)],
    "video": bytes|None, "video_wh", "frames_used", "tracklet_frames", "offsets",
    "span_s"}. A crop without a usable tracklet (or ``single`` mode) is the
    single-frame bench crop, prepared exactly as the single-frame arm does."""
    n_trk = len((trk or {}).get("frames") or []) if (trk or {}).get("status") == "ok" else 0
    single = {"kind": "single", "images": [clef_image(crop_jpg)], "video": None,
              "video_wh": None, "frames_used": 1, "tracklet_frames": n_trk,
              "offsets": [0], "span_s": 0.0}
    if frames_mode == "single" or not usable_tracklet(trk):
        return single
    bench_dir = Path(bench_dir)
    fps = trk.get("fps")
    if frames_mode == "multi4":
        seq = multi_frame_sequence(crop_jpg, trk, bench_dir, MULTI_IMAGES["multi4"])
        offsets = [d["offset"] for d in seq]
        return {"kind": "multi", "images": [clef_image(d["bytes"]) for d in seq],
                "video": None, "video_wh": None, "frames_used": len(seq),
                "tracklet_frames": n_trk, "offsets": offsets, "span_s": _span_s(offsets, fps)}
    if frames_mode == "video":
        seq = sorted(trk["frames"], key=lambda f: f["offset"])
        if len(seq) / VIDEO_FPS > VIDEO_MAX_S:  # keep the centre-most frames
            keep = int(VIDEO_MAX_S * VIDEO_FPS)
            seq = sorted(sorted(seq, key=lambda f: abs(f["offset"]))[:keep], key=lambda f: f["offset"])
        cw, ch = letterbox_dims(*trk["box"][2:4])
        video = encode_video([(bench_dir / f["file"]).read_bytes() for f in seq], cw, ch)
        if len(video) > VIDEO_MAX_BYTES:
            raise ArmError(f"video {len(video)} bytes > {VIDEO_MAX_BYTES}")
        offsets = [f["offset"] for f in seq]
        return {"kind": "video", "images": [clef_image(crop_jpg)] if with_still else [],
                "video": video, "video_wh": (cw, ch), "frames_used": len(seq),
                "tracklet_frames": n_trk, "offsets": offsets, "span_s": _span_s(offsets, fps)}
    raise ValueError(f"unknown frames mode {frames_mode!r}")


# ---- registered: multi-frame registration + fusion -> one image ------------

def app_bumper_jpeg(img) -> bytes:
    """Mirror web/app.py _bumper_crop_b64's reader prep: a crop under 384 px tall
    is upscaled by the integer factor max(2, ceil(384 / h)) with INTER_LINEAR,
    then JPEG q90."""
    import cv2
    h = img.shape[0]
    if h < APP_CROP_MIN_H:
        k = max(2, int(math.ceil(APP_CROP_MIN_H / max(1, h))))
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ArmError("imencode failed")
    return buf.tobytes()


def fuse_tracklet(trk: dict, bench_dir: Path, anchor_only: bool = False, **kw):
    """web.bumper_register.register_and_fuse over a tracklet's native frames,
    anchored on offset 0 (the bench crop's source frame; else the nearest).
    ``anchor_only`` runs the same upscale + unsharp path on the anchor alone
    (the no-fusion control). Returns (BGR uint8 image, diagnostics)."""
    import cv2
    from web.bumper_register import wants_fields, register_and_fuse
    frames = sorted(trk["frames"], key=lambda f: f["offset"])
    a = min(range(len(frames)), key=lambda i: abs(frames[i]["offset"]))
    imgs = []
    for f in frames:
        im = cv2.imread(str(Path(bench_dir) / f["file"]), cv2.IMREAD_COLOR)
        if im is None:
            raise ArmError(f"unreadable tracklet frame {f['file']}")
        imgs.append(im)
    opts = dict(REG_DEFAULTS)
    opts.update(kw)
    if anchor_only:
        # the control gets the full tracklet's interlace decision, then only its anchor
        if opts.get("deinterlace", "auto") == "auto":
            opts["deinterlace"] = ("fields" if wants_fields(
                imgs, a, times=[f["offset"] for f in frames]) else "none")
        frames, imgs, a = [frames[a]], [imgs[a]], 0
    # sharpness is left to register_and_fuse: the tracklet's Laplacian variance
    # is inflated by interlace combing; it measures each (field) sample itself.
    out, diag = register_and_fuse(
        imgs, a, opts.pop("upscale", None), times=[f["offset"] for f in frames],
        match_scores=[f.get("match_score") for f in frames], **opts)
    diag["offsets_used"] = [frames[i]["offset"] for i in diag["used"]]
    diag["anchor_offset"] = frames[a]["offset"]
    return out, diag


def registered_media(frames_mode: str, crop_jpg: bytes, trk: Optional[dict],
                     bench_dir: Path) -> tuple[bytes, dict]:
    """(reader-ready JPEG, info) for --frames registered | anchor. A crop with no
    usable tracklet falls back to the bench crop bytes (``reg_fallback``)."""
    n_trk = len((trk or {}).get("frames") or []) if (trk or {}).get("status") == "ok" else 0
    info = {"frames_used": 1, "tracklet_frames": n_trk, "media_kind": "single",
            "reg_fused": False, "reg_fallback": None}
    if not (trk and trk.get("status") == "ok" and trk.get("frames")):
        info["reg_fallback"] = "no_tracklet"
        return crop_jpg, info
    img, diag = fuse_tracklet(trk, bench_dir, anchor_only=(frames_mode == "anchor"))
    info.update({"media_kind": frames_mode, "frames_used": diag["n_used"],
                 "reg_fused": diag["fused"], "reg_samples": diag["n_samples"],
                 "reg_fields": diag["split_fields"],
                 "frame_offsets": diag["offsets_used"],
                 "reg_rejected": [[i, r] for i, r in diag["rejected"]],
                 "reg_ecc": {str(k): v for k, v in diag["ecc"].items()},
                 "reg_ms": diag["ms"], "image_native_wh": list(trk["box"][2:4])})
    return app_bumper_jpeg(img), info


_BUMPER_DESC =("a solid RED or BLUE fabric band around the robot's base with the team number "
                "in WHITE digits. Field structures (crates, bins, signs) can also carry printed "
                "numbers. ")


def clef_media_state(row: dict, mode: str, hint: str, media: dict) -> str:
    """State text for a multi-image / video request (the single-frame state is
    clef_state, unchanged)."""
    red, blue = offered_roster(row, mode)
    n, span = media["frames_used"], media["span_s"]
    if media["kind"] == "multi":
        s = (f"{n} images of the SAME robot: crops from FRC robotics match video taken at "
             f"{n} moments over about {span:.1f} seconds, ordered in time, each around ONE "
             f"robot's bumper, " + _BUMPER_DESC
             + "The digits may be readable in some images and blurred or hidden in others. ")
    else:
        s = (f"A short video of the SAME robot: {n} crops from FRC robotics match video, "
             f"one per video frame, ordered in time, covering about {span:.1f} seconds of the "
             f"match, each around ONE robot's bumper, " + _BUMPER_DESC
             + "The digits may be readable in some frames and blurred or hidden in others. ")
        if media["images"]:
            s += "The still image is the middle moment of the video. "
    s += _roster_line(red, blue)
    if mode in ("alliance", "missing-true") and hint in ("red", "blue"):
        s += f" Computer vision sees {hint.upper()} bumper pixels."
    return s


def clef_media_payload(model_key: str, row: dict, media: dict, options: list[dict],
                       mode: str, hint: str) -> dict:
    """Request body for a multi-image or video crop (``media`` from
    build_frame_media with kind != "single")."""
    payload = {
        "model": CLEF_MODELS[model_key]["model"],
        "state": clef_media_state(row, mode, hint, media),
        "questions": clef_questions(options),
    }
    if media["images"]:
        payload["images"] = ["data:image/jpeg;base64," + base64.b64encode(b).decode()
                             for b, _w, _h in media["images"]]
    if media["video"] is not None:
        payload["videos"] = ["data:video/mp4;base64," + base64.b64encode(media["video"]).decode()]
    return payload


def clef_media_estimate_tokens(payload: dict, media: dict) -> int:
    text = len(json.dumps(payload["state"])) + len(json.dumps(payload["questions"]))
    tok = sum(clef_image_tokens(w, h) for _b, w, h in media["images"]) + text // 4
    if media["video"] is not None:
        tok += video_token_estimate(media["frames_used"], *media["video_wh"])
    return tok


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


class QuotaExhausted(ArmError):
    """Workers AI daily free allocation used up: retrying is pointless."""


# Set once any request hits the daily-quota 429; later crops are skipped
# without a request instead of each burning the full retry/backoff budget.
QUOTA_EXHAUSTED = threading.Event()
_QUOTA_MARKERS = ("daily free allocation", "upgrade to Cloudflare's Workers Paid")


def is_quota_error(status: int, text: str) -> bool:
    return status == 429 and any(m in (text or "") for m in _QUOTA_MARKERS)


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
            if is_quota_error(r.status_code, r.text):
                QUOTA_EXHAUSTED.set()
                raise QuotaExhausted(last)
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
        """The image every single-image arm reads: the bench crop, or under
        --frames registered|anchor the fused / anchor-control image (bench crop
        when the crop has no tracklet; see media_info)."""
        if self.frames_mode() in REGISTERED_MODES:
            return self._registered(row)[0]
        return (self.bench_dir / row["file"]).read_bytes()

    def frames_mode(self) -> str:
        return getattr(self.args, "frames", "single") or "single"

    def _registered(self, row: dict) -> tuple[bytes, dict]:
        if getattr(self, "_reg_cache", None) is None:
            self._reg_cache, self._reg_lock = {}, threading.Lock()
        with self._reg_lock:
            hit = self._reg_cache.get(row["id"])
        if hit is None:
            crop = (self.bench_dir / row["file"]).read_bytes()
            hit = registered_media(self.frames_mode(), crop, self.tracklet(row["id"]),
                                   self.bench_dir)
            with self._reg_lock:
                self._reg_cache[row["id"]] = hit
        return hit

    def media_info(self, row: dict) -> dict:
        """Prediction-row fields for a single-image read: frames_mode plus, under
        --frames registered|anchor, what the registration used."""
        if self.frames_mode() not in REGISTERED_MODES:
            return {"frames_mode": "single", "frames_used": 1, "media_kind": "single"}
        return {"frames_mode": self.frames_mode(), **self._registered(row)[1]}

    def tracklet(self, crop_id: str) -> Optional[dict]:
        if getattr(self, "_tracklets", None) is None:
            self._tracklets = load_tracklets(self.bench_dir)
        return self._tracklets.get(crop_id)


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
        self.orig_post = orig  # multi-frame reads post directly (no capture keying)
        self.frames = getattr(ctx.args, "frames", "single") or "single"
        if self.frames not in FREEREAD_FRAME_MODES + REGISTERED_MODES:
            raise ArmError(f"freeread: --frames {self.frames} not supported "
                           f"({', '.join(FREEREAD_FRAME_MODES)})")
        self.prompt = getattr(ctx.args, "prompt", "incumbent") or "incumbent"
        if self.prompt not in PROMPT_VARIANTS:
            raise ArmError(f"freeread: --prompt {self.prompt} not supported "
                           f"({', '.join(PROMPT_VARIANTS)})")
        if self.prompt == "escape" and self.frames not in ("single",) + REGISTERED_MODES:
            raise ArmError("--prompt escape supports --frames single only "
                           "(or the single fused image of registered / anchor)")
        ctx.tracklet("")  # load tracklets.jsonl before the worker threads start

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
        pred["prompt_variant"] = self.prompt
        red, blue = offered_roster(row, a.options if mode else "alliance")
        if mode:
            pred["roster_offered"] = red + blue
            pred["removed_team"] = true_team(row)
        jpg = self.ctx.crop(row)
        trk = self.ctx.tracklet(row["id"])
        pred.update({"frames_mode": self.frames, "frames_used": 1,
                     "tracklet_frames": len(trk["frames"]) if trk else 0, "media_kind": "single"})
        if self.frames in REGISTERED_MODES:
            pred.update(self.ctx.media_info(row))
        if self.prompt == "escape":
            return self._predict_escape(pred, jpg, red, blue, hint)
        if self.frames in MULTI_IMAGES and usable_tracklet(trk):
            return self._predict_multi(row, pred, jpg, trk, red, blue, hint,
                                       MULTI_IMAGES[self.frames])
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

    def build_multi_payload(self, jpg: bytes, trk: dict, red: list, blue: list, hint: str,
                            n_images: int) -> tuple[dict, list[int], float]:
        """ONE chat request with the bench crop + the picked tracklet frames
        (time order, each prepped exactly like the single crop), the incumbent
        prompt reworded for several moments, and the incumbent's
        _vlm_chat_payload settings (guided JSON _BUMPER_READ_SCHEMA, logprobs)."""
        app = self.app
        seq = multi_frame_sequence(jpg, trk, self.ctx.bench_dir, n_images)
        offsets = [d["offset"] for d in seq]
        span = _span_s(offsets, trk.get("fps"))
        uris = [app_style_data_uri(d["bytes"]) for d in seq]
        expected = hint if hint in ("red", "blue") else "unknown"
        prompt = freeread_multi_prompt(len(seq), span,
                                       freeread_hint_line(expected, red, blue, red + blue))
        base = app._vlm_chat_payload(
            uris[0], prompt, max_tokens=64, logprobs=app.VLM_LOGPROB_CONFIDENCE,
            schema_name="bumper_read", schema=app._BUMPER_READ_SCHEMA)
        return multi_image_payload(base, uris, prompt), offsets, span

    def _predict_escape(self, pred: dict, jpg: bytes, red: list, blue: list,
                        hint: str) -> tuple[dict, dict]:
        """--prompt escape, single image: the escape prompt + ESCAPE_READ_SCHEMA
        (subject decided before the digits), posted directly (no incumbent
        _vllm_read_bumpers). _vlm_chat_payload merges appmod._active_vlm_extra_body
        (set from Ctx in __init__), so --no-thinking still applies."""
        app = self.app
        expected = hint if hint in ("red", "blue") else "unknown"
        prompt = freeread_escape_prompt(
            freeread_escape_hint_line(expected, red, blue, red + blue))
        payload = app._vlm_chat_payload(
            app_style_data_uri(jpg), prompt, max_tokens=64,
            logprobs=app.VLM_LOGPROB_CONFIDENCE, schema_name="bumper_read_escape",
            schema=ESCAPE_READ_SCHEMA)
        pred["prompt_variant"] = "escape"
        return self._post_and_read(pred, payload, red, blue, expected, escape=True), payload

    def _post_and_read(self, pred: dict, payload: dict, red: list, blue: list,
                       expected: str, escape: bool = False) -> dict:
        """Shared by the multi-frame and escape paths: post -> logprob digit
        confidence -> parse -> roster match -> confidence blend -> pred fields."""
        app = self.app
        try:
            t0 = time.monotonic()
            if self.dry:
                content = ('{"subject": "not_robot", "team": null, "color": "none"}'
                           if escape else '{"team": null, "color": "red"}')
                body = {"choices": [{"message": {"content": content}, "logprobs": None}]}
            else:
                body = self.orig_post(payload)
            lat = time.monotonic() - t0
            choice = body["choices"][0]
            text = (choice["message"]["content"] or "").strip()
            digit_prob = None
            lp = ((choice.get("logprobs") or {}).get("content")) or []
            if app.VLM_LOGPROB_CONFIDENCE and lp:
                digit_prob = app.digit_token_conf([math.exp(t["logprob"]) for t in lp],
                                                  [t.get("token", "") for t in lp])
            subject = None
            if escape:
                team, color, surf, subject = parse_escape_reply(text)
            else:
                team, color, surf = parse_bumper_reply(text)
            read = "".join(ch for ch in team if ch.isdigit()) if team else ""
            model_color = color if color in ("blue", "red") else "unknown"
            matched, alliance, conf = app._match_bumper_read_to_roster(
                read, model_color, expected, red + blue, blue, red)
            conf = app.blend_read_confidence(conf, digit_prob)
            fr = {"team_number": matched, "alliance": alliance, "confidence": conf,
                  "raw_read": read, "digit_prob": digit_prob,
                  "surface": app.normalize_surface(surf), "raw_text": text}
            if escape:
                fr["subject"] = subject
            pred.update({
                "pred_team": matched, "freeread": fr,
                "latency_s": round(lat, 3), "latencies_s": [round(lat, 3)],
                "usage": body.get("usage"), "attempts": 1})
        except Exception as e:
            pred["error"] = f"{type(e).__name__}: {e}"
        return pred

    def _predict_multi(self, row: dict, pred: dict, jpg: bytes, trk: dict, red: list,
                       blue: list, hint: str, n_images: int) -> tuple[dict, dict]:
        expected = hint if hint in ("red", "blue") else "unknown"
        payload, offsets, span = self.build_multi_payload(jpg, trk, red, blue, hint, n_images)
        pred.update({"frames_used": len(offsets), "frame_offsets": offsets,
                     "span_s": round(span, 3), "media_kind": "multi"})
        return self._post_and_read(pred, payload, red, blue, expected), payload


# ---- shared choice-arm plumbing ----------------------------------------------

def _finish_choice_pred(pred: dict, perms: list[dict], blank: bool,
                        escapes: tuple = ESCAPES) -> None:
    """Average per-permutation distributions (raw and calibrated) into the row."""
    pred["perms"] = perms
    probs = average_distributions([pm.get("probs") for pm in perms])
    top, top_p = top_choice(probs)
    pred.update({"probs": probs, "top": top, "top_p": top_p,
                 "pred_team": top if top is not None and top not in escapes else None})
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
        if (getattr(self.ctx.args, "frames", "single") or "single") in REGISTERED_MODES:
            pred.update(self.ctx.media_info(row))
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
        self.frames = getattr(ctx.args, "frames", "single") or "single"
        self.with_still = bool(getattr(ctx.args, "video_with_still", False))
        if self.frames == "multi8":
            raise ArmError(f"{model_key}: --frames multi8 exceeds the 4-image API limit")
        if self.frames == "video" and model_key not in VIDEO_ARMS:
            raise ArmError(f"{model_key}: --frames video needs a model with a videos input "
                           f"({', '.join(VIDEO_ARMS)})")
        if self.frames not in ("single",) + REGISTERED_MODES and self.blank:
            raise ArmError("--blank-calibrate-clef is only implemented for single-image --frames")
        ctx.tracklet("")  # load tracklets.jsonl once, before the worker threads start
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
        if QUOTA_EXHAUSTED.is_set():
            raise QuotaExhausted("skipped: Workers AI daily free allocation exhausted earlier "
                                 "in this run (upgrade to Workers Paid or wait for the 00:00 UTC reset)")
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
        trk = self.ctx.tracklet(row["id"])
        pred["frames_mode"] = self.frames
        pred["frames_used"] = 1
        pred["tracklet_frames"] = len(trk["frames"]) if trk else 0
        if self.frames in REGISTERED_MODES:
            pred.update(self.ctx.media_info(row))
        media = None
        try:
            if self.frames in MULTI_IMAGES or self.frames == "video":
                media = build_frame_media(self.frames, jpg, trk, self.ctx.bench_dir,
                                          self.with_still)
                if media["kind"] == "single":
                    media = None  # no usable tracklet: the plain single-frame request
                else:
                    pred.update({"frames_used": media["frames_used"],
                                 "frame_offsets": media["offsets"],
                                 "span_s": round(media["span_s"], 3),
                                 "image_wh": [[iw, ih] for _b, iw, ih in media["images"]]})
                    if media["video"] is not None:
                        pred["video_wh"] = list(media["video_wh"])
                        pred["video_bytes"] = len(media["video"])
            if self.frames not in REGISTERED_MODES:
                pred["media_kind"] = media["kind"] if media else "single"
            for order in orders:
                if media is None:
                    payload = clef_payload(self.key, row, img, order, a.options, hint)
                    est += clef_estimate_tokens(payload, w, h) * (2 if self.blank else 1)
                else:
                    payload = clef_media_payload(self.key, row, media, order, a.options, hint)
                    est += clef_media_estimate_tokens(payload, media)
                first_payload = first_payload or payload
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


# ---- jev (TypeSafe) ---------------------------------------------------------

class Jev:
    """TypeSafe Jev decision model, single frame: `subject` then `team` with
    four escape options (JEV_ESCAPES)."""

    arm_name = "jev"

    def __init__(self, ctx: Ctx, dry_run: bool):
        self.ctx = ctx
        self.dry = dry_run
        self.k = ctx.perms_for("jev")
        frames = self.frames = getattr(ctx.args, "frames", "single") or "single"
        if frames not in ("single",) + REGISTERED_MODES:
            raise ArmError(f"jev: --frames {frames} is not supported (single frame only, "
                           f"or the single fused image of {' / '.join(REGISTERED_MODES)})")
        if getattr(ctx.args, "blank_calibrate_clef", False):
            raise ArmError("jev: blank calibration is not implemented")
        self.post_url = JEV_URL
        self.cost_per_m = JEV_USD_PER_M_INPUT
        self.model_id = JEV_MODEL
        if self.arm_name != "jev":
            return  # LocalSystemOne: no token, no cost
        self.env_loaded = load_env_file(ENV_FILE, keys=JEV_ENV_KEYS)
        self.token = os.environ.get("JEV_TYPESAFE_API_TOKEN")
        self.missing = [] if self.token else list(JEV_ENV_KEYS)
        if self.missing and not dry_run:
            raise ArmError(f"jev: missing {', '.join(self.missing)} "
                           "(set in the environment or the repo-root .env)")

    @staticmethod
    def url() -> str:
        return JEV_URL

    def requests_per_crop(self, n_options: int) -> int:
        return max(1, min(self.k, n_options))

    @staticmethod
    def _probs(ans: dict, name: str) -> dict:
        probs = (ans.get(name) or {}).get("probabilities")
        if not probs:
            raise ArmError(f"no {name} probabilities in answers: {str(ans)[:300]}")
        return {str(k): float(v) for k, v in probs.items()}

    def _post(self, payload: dict, pred: dict, usages: list) -> dict:
        headers = ({"Authorization": f"Bearer {self.token}",
                    "TypeSafe-Danger-Zone-Images": "true"} if self.arm_name == "jev" else {})
        body, lat, attempts = post_json(self.post_url, payload, headers=headers,
                                        retries=self.ctx.args.retries)
        pred["attempts"] = (pred["attempts"] or 0) + attempts
        pred["latencies_s"].append(round(lat, 3))
        if not isinstance(body, dict):
            raise ArmError(f"unexpected response: {str(body)[:300]}")
        if body.get("model"):
            pred["model"] = body["model"]  # versioned id, e.g. jev-1.14.0
        usages.append(body.get("usage"))
        return body.get("answers") or {}

    def predict(self, row: dict) -> tuple[dict, dict]:
        a = self.ctx.args
        hint = hint_alliance(row, a.hint_source)
        base = shuffle_options(jev_options(row, a.options, hint), row["id"], a.option_seed)
        img, w, h, padded = jev_image(self.ctx.crop(row))
        pred = _base_pred(self.arm_name, row, self.ctx, hint, self.model_id, a.options)
        pred["option_ids"] = [o["id"] for o in base]
        pred["escape_ids"] = list(JEV_ESCAPES)
        pred["permutations"] = self.k
        pred["image_wh"] = [w, h]
        pred["image_padded"] = padded
        pred["frames_mode"] = "single"
        pred["frames_used"] = 1
        if self.frames in REGISTERED_MODES:
            pred.update(self.ctx.media_info(row))
        first_payload = None
        est = 0
        perms, usages, subjects = [], [], []
        try:
            for order in permutation_orders(base, self.k):
                payload = jev_payload(row, img, order, a.options, hint)
                est += jev_estimate_tokens(payload, w, h)
                first_payload = first_payload or payload
                if self.dry:
                    continue
                pm = {"option_ids": [o["id"] for o in order]}
                ans = self._post(payload, pred, usages)
                pm["probs"] = self._probs(ans, "team")
                pm["api_choice"] = (ans.get("team") or {}).get("choice")
                pm["api_confidence"] = (ans.get("team") or {}).get("confidence")
                if (ans.get("subject") or {}).get("probabilities"):
                    pm["subject_probs"] = self._probs(ans, "subject")
                    subjects.append(pm["subject_probs"])
                perms.append(pm)
        except ArmError as e:
            pred["error"] = str(e)
        pred["est_input_tokens"] = est
        pred["est_cost_usd"] = est * self.cost_per_m / 1e6
        pred["usage"] = _sum_usage(usages)
        if pred["usage"] and pred["usage"].get("input_tokens") is not None:
            pred["est_cost_usd"] = pred["usage"]["input_tokens"] * self.cost_per_m / 1e6
        if not self.dry:
            _finish_choice_pred(pred, perms, False, escapes=JEV_ESCAPES)
            sp = average_distributions(subjects)
            pred["subject_probs"] = sp
            pred["subject_top"], pred["subject_top_p"] = top_choice(sp)
        pred["requests_per_crop"] = self.requests_per_crop(len(base))
        return pred, first_payload


class LocalClef(Clef):
    """The clef-flash arm's exact Workers AI request (clef_state / clef_questions /
    clef_image) POSTed to a self-hosted shim's /v1/clef route. No auth, no cost."""

    def __init__(self, ctx: Ctx, dry_run: bool):
        super().__init__(ctx, "clef-flash", dry_run=True)  # skip the CF credential check
        self.dry = dry_run
        self.local_url = ctx.args.local_url.rsplit("/v1/", 1)[0] + "/v1/clef"
        self.token = "local"

    def url(self, redact: bool = False) -> str:
        return self.local_url

    def predict(self, row: dict) -> tuple[dict, dict]:
        pred, payload = super().predict(row)
        pred["arm"] = "local-clef"
        pred["model"] = f"local:{self.ctx.args.local_label}"
        pred["est_cost_usd"] = 0.0
        return pred, payload


class LocalSystemOne(Jev):
    """The jev arm's exact payload (subject + team, four escapes, jev_image prep)
    POSTed to a self-hosted Jev-wire server, e.g. an open decision model behind
    scripts/decision_local/systemone_shim.py. No auth header, no cost."""
    arm_name = "local"

    def __init__(self, ctx: Ctx, dry_run: bool):
        super().__init__(ctx, dry_run)
        self.post_url = ctx.args.local_url
        self.cost_per_m = 0.0
        self.model_id = f"local:{ctx.args.local_label}"
        self.token = None
        self.missing = []


def make_arm(name: str, ctx: Ctx, dry_run: bool):
    if name in ("freeread", "vllm-choice"):
        ctx.resolve_vllm(dry_run)
    if name == "freeread":
        return FreeRead(ctx, dry_run)
    if name == "vllm-choice":
        return VllmChoice(ctx, dry_run)
    if name in CLEF_MODELS:
        return Clef(ctx, name, dry_run)
    if name == "jev":
        return Jev(ctx, dry_run)
    if name == "local":
        return LocalSystemOne(ctx, dry_run)
    if name == "local-clef":
        return LocalClef(ctx, dry_run)
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
                mode: str = "alliance", ids: Optional[set] = None) -> list[dict]:
    if ids is not None:
        rows = [r for r in rows if r["id"] in ids]
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
    out = f"{arm}:{mode}" if mode else str(arm)
    if pred.get("frames_mode") not in (None, "single"):
        out += f"+{pred['frames_mode']}"
    if pred.get("prompt_variant") not in (None, "incumbent"):
        out += f"+{pred['prompt_variant']}"
    return out


def _file_label(preds: list[dict]) -> str:
    """Label from the first row that carries the arm's fields (an arm-crash row
    from run_arm carries only id/arm/error)."""
    return _label(next((p for p in preds if "options_mode" in p), preds[0]))


LABEL_FILES = {"tracklet": "tracklet_labels.jsonl", "truth": "track_truth.jsonl"}
LABEL_KEYS = ("label_status", "label_team", "label_alliance")


def apply_label_file(manifest: dict[str, dict], spec: Optional[str], bench_dir: Path) -> int:
    """Score against another label file instead of the manifest's crop labels:
    ``tracklet`` (tracklet_labels.jsonl), ``truth`` (track_truth.jsonl) or a
    path. Each row's label_status / label_team / label_alliance (or status /
    team / alliance) replace the manifest row's IN PLACE; manifest rows the file
    does not cover become unlabelled, so one run never mixes two truths. A
    track_truth row ({"truth_team": ...}, identity not legibility) becomes
    legible + that team, or unlabelled when truth_team is null.
    Returns the number of rows overlaid."""
    if not spec or spec == "manifest":
        return 0
    path = Path(bench_dir) / LABEL_FILES[spec] if spec in LABEL_FILES else Path(spec)
    if not path.exists():
        sys.exit(f"--labels {spec}: {path} does not exist")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows if r.get("id")}
    n = 0
    for cid, mrow in manifest.items():
        r = by_id.get(cid)
        if r is None:
            mrow.update({"label_status": "unlabelled", "label_team": None})
            continue
        if "truth_team" in r and not ("label_status" in r or "status" in r):
            # track_truth.jsonl: robot IDENTITY, not legibility. A crop with a
            # truth team scores as legible-with-that-team (asserts are right or
            # wrong by identity; abstains stay abstains); no truth -> unlabelled.
            mrow["label_status"] = "legible" if r["truth_team"] else "unlabelled"
            mrow["label_team"] = r["truth_team"]
            n += 1
            continue
        for k in LABEL_KEYS:
            alt = k.replace("label_", "")
            if k in r:
                mrow[k] = r[k]
            elif alt in r:
                mrow[k] = r[alt]
        if mrow.get("label_team") is not None:
            mrow["label_team"] = str(mrow["label_team"])
        n += 1
    return n


def filter_subset(preds: list[dict], ids: Optional[set]) -> list[dict]:
    """Rows whose crop id is in ``ids`` (None = no filter)."""
    return list(preds) if ids is None else [p for p in preds if p.get("id") in ids]


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
            kinds: dict = {}
            for r in rows:
                rp, _ = arm.predict(r)
                total += rp["est_cost_usd"]
                kinds[rp.get("media_kind")] = kinds.get(rp.get("media_kind"), 0) + 1
                if rp.get("error"):
                    print(f"  {r['id']}: ERROR {rp['error']}")
            wh = pred["image_wh"]
            img_desc = (f"image {wh[0]}x{wh[1]}" if wh and not isinstance(wh[0], list)
                        else f"images {', '.join(f'{a}x{b}' for a, b in wh) or '-'}")
            if pred.get("video_wh"):
                img_desc += (f"; video {pred['video_wh'][0]}x{pred['video_wh'][1]} "
                             f"{pred['frames_used']} frames @{VIDEO_FPS}fps "
                             f"{pred['video_bytes']} bytes")
            print(f"frames={arm.frames} permutations={arm.k} blank={arm.blank} -> "
                  f"{pred['requests_per_crop']} request(s)/crop; est. input tokens this crop: "
                  f"{pred['est_input_tokens']} ({img_desc}); media per crop: {kinds}; "
                  f"est. cost for {len(rows)} crops: ${total:.4f}")
        elif isinstance(arm, Jev):
            print(f"POST {arm.url()}")
            print("Authorization: Bearer <JEV_TYPESAFE_API_TOKEN>   "
                  "TypeSafe-Danger-Zone-Images: true   [credentials: "
                  + ("missing " + ", ".join(arm.missing) if arm.missing else "present")
                  + (f"; loaded from .env: {', '.join(arm.env_loaded)}" if arm.env_loaded else "")
                  + "]")
            total = 0.0
            for r in rows:
                rp, _ = arm.predict(r)
                total += rp["est_cost_usd"]
                if rp.get("error"):
                    print(f"  {r['id']}: ERROR {rp['error']}")
            wh = pred["image_wh"]
            print(f"permutations={arm.k} -> {pred['requests_per_crop']} request(s)/crop; "
                  f"est. input tokens this crop: {pred['est_input_tokens']} "
                  f"(image {wh[0]}x{wh[1]}{', letterboxed' if pred['image_padded'] else ''}); "
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
    subset = None
    if args.subset == "tracklet":
        subset = tracklet_subset_ids(load_tracklets(bench_dir))
    rows = select_rows(manifest_rows, args.only_labelled, args.limit, args.options, subset)
    if not rows:
        sys.exit("no rows selected (labelled-only with no labels yet?)")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; valid: {', '.join(ARMS)}")
    if args.frames not in ("single",) + REGISTERED_MODES:
        def _ok(arm: str) -> bool:
            if arm == "freeread":
                return args.frames in FREEREAD_FRAME_MODES
            if arm in ("jev", "local"):
                return False
            if arm in CLEF_MODELS:
                return args.frames == "multi4" or (args.frames == "video" and arm in VIDEO_ARMS)
            return False
        bad = [a for a in arm_names if not _ok(a)]
        if bad:
            sys.exit(f"--frames {args.frames} is not supported by {', '.join(bad)} "
                     f"(multi4: clef arms + freeread; multi8: freeread; "
                     f"video: {', '.join(VIDEO_ARMS)}; jev: single only)")
    if (getattr(args, "prompt", "incumbent") == "escape"
            and args.frames not in ("single",) + REGISTERED_MODES):
        sys.exit("--prompt escape supports --frames single only "
                 "(or the single fused image of registered / anchor)")
    if args.reg_opts:
        REG_DEFAULTS.update(json.loads(args.reg_opts))
    ctx = Ctx(args, bench_dir)
    if args.dry_run:
        dry_run(arm_names, ctx, rows)
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    res_dir = bench_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    # scoring copy: --labels may overlay another truth (the run itself never sees it)
    manifest = {r["id"]: dict(r) for r in manifest_rows}
    apply_label_file(manifest, args.labels, bench_dir)
    summary = {"timestamp": stamp, "argv": sys.argv[1:], "n_rows": len(rows),
               "options": args.options, "hint_source": args.hint_source,
               "frames": args.frames, "prompt": getattr(args, "prompt", "incumbent"),
               "reg_opts": dict(REG_DEFAULTS) if args.frames in REGISTERED_MODES else None,
               "labels": args.labels, "subset": args.subset, "arms": {}}
    scored = []
    for name in arm_names:
        try:
            arm = make_arm(name, ctx, dry_run=False)
        except ArmError as e:
            print(f"[{name}] ERROR: {e}", file=sys.stderr)
            summary["arms"][name] = {"error": str(e)}
            continue
        file_arm = (f"local-{args.local_label}" if name == "local"
                    else f"local-clef-{args.local_label}" if name == "local-clef" else name)
        out_path = res_dir / f"{file_arm}-{stamp}.jsonl"
        print(f"[{name}] {len(rows)} crops -> {out_path}", file=sys.stderr)
        preds = run_arm(name, arm, rows, out_path, args.concurrency)
        if QUOTA_EXHAUSTED.is_set():
            print(f"[{name}] ABORTED: Workers AI daily free allocation exhausted; "
                  f"{sum(1 for p in preds if p.get('error'))}/{len(preds)} rows are errors. "
                  "Upgrade to Workers Paid or re-run after 00:00 UTC.", file=sys.stderr)
        variants = score_file_variants(preds, manifest, _file_label(preds) if preds else name)
        model = (ctx.vlm_model if name in ("freeread", "vllm-choice")
                 else JEV_MODEL if name == "jev"
                 else f"local:{args.local_label}" if name in ("local", "local-clef")
                 else CLEF_MODELS[name]["route"])
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
    n_over = apply_label_file(manifest, args.labels, bench_dir)
    files = [Path(f) for f in args.files] or latest_results(bench_dir / "results")
    if not files:
        sys.exit("no results files")
    subset = None
    if args.subset == "tracklet":
        subset = tracklet_subset_ids(load_tracklets(bench_dir))
        if not subset:
            sys.exit(f"--subset tracklet: no tracklets with >= {SUBSET_MIN_FRAMES} frames in "
                     f"{bench_dir / TRACKLETS_FILE} (run scripts/build_tracklet_bench.py)")
    scored, media_notes = [], []
    for f in files:
        preds = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        preds = filter_subset(preds, subset)
        if preds:
            label = f"{_file_label(preds)} {f.stem[-15:]}"
            scored.extend(score_file_variants(preds, manifest, label))
            multi = sum(1 for p in preds if (p.get("frames_used") or 1) > 1)
            reg = [p for p in preds if p.get("frames_mode") in REGISTERED_MODES]
            if reg:
                fb = sum(1 for p in reg if p.get("reg_fallback"))
                fused = sum(1 for p in reg if p.get("reg_fused"))
                ge3 = sum(1 for p in reg if p.get("reg_fused") and (p.get("frames_used") or 1) >= 3)
                media_notes.append(f"  {label}: {fused}/{len(reg)} fused (>=3 frames {ge3}), "
                                   f"{fb} fell back to the bench crop (no tracklet)")
            elif multi:
                fu = [p["frames_used"] for p in preds if (p.get("frames_used") or 1) > 1]
                media_notes.append(f"  {label}: {multi}/{len(preds)} crops multi-frame "
                                   f"(median {statistics.median(fu):g} frames)")
    n_lab = sum(1 for r in manifest.values() if (r.get("label_status") or "unlabelled") != "unlabelled")
    print(f"manifest: {len(manifest)} crops, {n_lab} labelled"
          + (f" (labels: {args.labels}, {n_over} rows)" if n_over else ""))
    if subset is not None:
        print(f"subset: tracklet - {len(subset)} crop ids with >= {SUBSET_MIN_FRAMES} tracklet "
              f"frames (every arm restricted to them)")
    print(format_table(scored))
    if media_notes:
        print("frames used:\n" + "\n".join(media_notes))
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
                          "(default: 3 for vllm-choice, 1 for clef and jev arms)")
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
    run.add_argument("--frames", choices=FRAME_MODES, default="single",
                     help="single = the bench crop (default); multi4 = the bench crop + the 3 "
                          "sharpest other tracklet frames, one per time third (clef arms, "
                          "freeread); multi8 = up to 8 images (freeread); video = all tracklet "
                          "frames as an H.264 clip at 2 fps (clef-omni); registered = the "
                          "tracklet registered + fused into ONE image (web/bumper_register.py, "
                          "every arm); anchor = the same upscale/deinterlace/unsharp on the "
                          "anchor frame alone (the no-fusion control, every arm). "
                          "Crops without a tracklet fall back to single (frames_used=1).")
    run.add_argument("--reg-opts", default=None,
                     help="--frames registered|anchor: JSON overrides for register_and_fuse, "
                          "e.g. '{\"deinterlace\": \"fields\", \"fuse\": \"median\"}'")
    run.add_argument("--labels", default="manifest",
                     help="score against: manifest (crop labels, default), tracklet "
                          "(tracklet_labels.jsonl), truth (track_truth.jsonl) or a path")
    run.add_argument("--prompt", choices=PROMPT_VARIANTS, default="incumbent",
                     help="freeread prompt: incumbent = web.app's prompt + _BUMPER_READ_SCHEMA "
                          "(default, unchanged); escape = detector-junk-aware prompt with an "
                          "explicit not_robot/no_digits/printed subject decided BEFORE the "
                          "digits (--frames single only)")
    run.add_argument("--subset", choices=("all", "tracklet"), default="all",
                     help=f"tracklet = only crops whose tracklet has >= {SUBSET_MIN_FRAMES} frames "
                          "(the `score --subset tracklet` crops; about half the paid requests)")
    run.add_argument("--video-with-still", action="store_true",
                     help="--frames video: also send the bench crop as an image")
    run.add_argument("--local-url", default=LOCAL_SYSTEMONE_URL,
                     help="local arm: Jev-wire /v1/systemone endpoint (LOCAL_SYSTEMONE_URL)")
    run.add_argument("--local-label", default="local",
                     help="local arm: model label for the results file (local-<label>-<stamp>.jsonl)")
    run.add_argument("--bench", default=str(BENCH_DIR))
    sc = sub.add_parser("score", help="score results files against the manifest labels")
    sc.add_argument("--subset", choices=("all", "tracklet"), default="all",
                    help=f"tracklet = only crop ids whose tracklet has >= {SUBSET_MIN_FRAMES} "
                         "frames (identical crops for single vs multi-frame comparisons)")
    sc.add_argument("files", nargs="*", help="results/*.jsonl (default: latest per arm)")
    sc.add_argument("--json", default=None, help="also dump metrics as JSON here")
    sc.add_argument("--labels", default="manifest",
                    help="manifest (crop labels, default), tracklet (tracklet_labels.jsonl), "
                         "truth (track_truth.jsonl) or a path to a jsonl with id + label_*")
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
