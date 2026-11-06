"""TypeSafe "Jev" decision-model bumper reader (team_reader="jev").

Jev speaks the same "System One" decision schema as Cloudflare's Clef
(web/clef_reader.py): a ``state``, image assets and choice questions in, one
probability per option out. ``JevSession`` reuses ClefSession's transport
(retries, auth latch, concurrency, cost/latency accounting) and only swaps
the URL, headers, image prep, payload, option set and answer mapping.

API (verified live 2026-11-05):
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer {JEV_TYPESAFE_API_TOKEN}
  TypeSafe-Danger-Zone-Images: true          (400 without it on image requests)
  body {"model": "jev-1.14.0", "state": {"crop": "crop_image", "context": ...},
        "assets": {"crop_image": {"type": "image", "format": "jpg",
                                  "data_base64": "<no data: prefix>"}},
        "questions": {...}}
  -> {"model", "answers": {q: {"type": "choice", "choice", "confidence",
                               "probabilities": {...}}},
      "usage": {"input_tokens", "output_tokens"}}
  Every asset ref must be referenced from the state / instructions / criteria.
  Images: each side 280..1568 px, 78,400..2,458,624 px total; out-of-range
  images are REJECTED, not resized (``jev_image`` scales/letterboxes).
  ~0.2 s and ~1-2k input tokens per read, $0.042/M input tokens.

Request design is ported from scripts/bench_bumper_readers.py (branch
decision-reader-bench: jev_image, jev_state, jev_questions, jev_payload): the
escape-prompt framing and four escapes (not_robot / unreadable / printed /
other). The separate ``subject`` question is dropped (it added nothing on the
bench). 200-crop bench (alliance options, 2026-11-05): argmax read 59/90
legible bumpers, 0 wrong, 2% false positives.

Tentative reads: on the bench all 31 legible abstains had top = "unreadable",
yet the true team was the best TEAM option in 23/31. When the top option is
``unreadable`` and the best team's probability is >= ``tentative_min_p``
(job knob jev_tentative_min_p, default 0.17, fitted to the bench), the read
names that team with ``tentative: True``. Live qm8 evidence (Clef reads below
0.65 were 3/4 wrong on red, see identity_resolve.apply_evidence_floor) says
low-probability reads are WEAK evidence: app.py votes a tentative read with
weight p x source_weight (always below a normal read's 0.5 floor) and never
lets it drive a lock.

Credentials come from the environment (web/app.py loads the repo-root .env at
import, exactly as for the CLOUDFLARE_* keys Clef uses); the token is never
logged. ``clef_reader.http_post`` stays the only network call.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Iterable, Optional

from web import clef_reader
from web.clef_reader import ClefError, ClefSession, _choice_probs, _shuffled, _top

# --------------------------------------------------------------------------- #
# Models, limits, option text
# --------------------------------------------------------------------------- #

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODELS = {
    # Pinned: the version the bench numbers (and the 0.17 threshold) are from.
    "jev-1.14.0": {"model": "jev-1.14.0", "usd_per_m_input": 0.042},
    "jev-latest": {"model": "jev-latest", "usd_per_m_input": 0.042},
}
DEFAULT_MODEL = "jev-1.14.0"
JEV_ENV_KEYS = ("JEV_TYPESAFE_API_TOKEN",)
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"

JEV_MIN_SIDE = 280
JEV_MAX_SIDE = 1568
JEV_MIN_PIXELS = 78_400
JEV_MAX_PIXELS = 2_458_624
JEV_PAD_GRAY = (128, 128, 128)
JEV_ASSET_REF = "crop_image"

# Default tentative-read floor (job knob jev_tentative_min_p; 0 disables).
DEFAULT_TENTATIVE_MIN_P = 0.17

JEV_ESCAPES = ("not_robot", "unreadable", "printed", "other")
JEV_ESCAPE_TEXT = {
    "not_robot": ("No robot bumper in the image: balls, field structures, signs, people or "
                  "empty floor"),
    "unreadable": ("A robot bumper, but the digits can't be read: blurred, cut off, turned "
                   "away or too small"),
    "printed": "The digits are printed on a crate, bin, sign or field structure, not a robot bumper",
    "other": "Legible bumper digits that are NOT any of the listed teams",
}

# Offer only the expected alliance's teams when the bumper colour is known
# (the bench's "alliance" option mode, which the 59/90 / 0-wrong numbers and
# the 0.17 tentative fit come from); all roster teams when it is unknown. The
# "other" escape covers a wrong colour hint (missing-true bench: 0/116 named).
JEV_ALLIANCE_OPTIONS = True


# --------------------------------------------------------------------------- #
# Image prep
# --------------------------------------------------------------------------- #

def jev_image(jpg: bytes) -> tuple[bytes, int, int, bool]:
    """Prep a crop for Jev: each side must land in 280..1568 px (the API rejects
    rather than resizes). Scale (LANCZOS) so the short side reaches 280 and the
    long side stays <= 1568; when the aspect ratio makes both impossible,
    shrink the long side to 1568 and letterbox-pad the short side to 280 with
    neutral gray. Unchanged in-range JPEGs pass through byte-identical. Both
    side bounds imply the total-pixel bounds (280^2 = 78,400; 1568^2 =
    2,458,624). Returns (jpeg bytes, w, h, padded)."""
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


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #

def _alliance_mode(red, blue, expected_alliance: str) -> bool:
    """Offer one alliance only: knob-free constant on, colour known, and that
    alliance has roster teams (an empty alliance list falls back to all)."""
    if not JEV_ALLIANCE_OPTIONS or expected_alliance not in ("red", "blue"):
        return False
    return bool(list(red if expected_alliance == "red" else blue))


def jev_options(red: Iterable, blue: Iterable, extra: Iterable = (),
                expected_alliance: str = "unknown") -> list[dict]:
    """Unshuffled options: roster teams (the expected alliance's only when
    known and JEV_ALLIANCE_OPTIONS, else every roster team + extras), each
    labelled with its alliance, then the four JEV_ESCAPES. Option ids are the
    team-number strings, so they never collide with the escape ids."""
    opts = []
    if _alliance_mode(red, blue, expected_alliance):
        opts = clef_reader.roster_options(red if expected_alliance == "red" else (),
                                          blue if expected_alliance == "blue" else ())
    else:
        opts = clef_reader.roster_options(red, blue, extra)
    opts = [o for o in opts if o["kind"] == "team"]
    return opts + [{"id": e, "kind": "escape", "text": JEV_ESCAPE_TEXT[e]}
                   for e in JEV_ESCAPES]


def jev_option_text(opt: dict) -> str:
    if opt["kind"] == "team":
        return clef_reader.option_text(opt)
    return opt.get("text") or JEV_ESCAPE_TEXT[opt["id"]]


def jev_roster_line(options: list[dict], expected_alliance: str) -> str:
    """The escape prompt's neutral roster line, naming exactly the offered teams."""
    team_opts = [o for o in options if o["kind"] == "team"]
    teams = [o["team"] for o in team_opts]
    if (expected_alliance in ("red", "blue") and team_opts
            and all(o.get("alliance") == expected_alliance for o in team_opts)):
        return f"Teams on the {expected_alliance} alliance in this match: {', '.join(teams)}."
    return f"Teams in this match: {', '.join(teams)}."


def jev_state(options: list[dict], expected_alliance: str = "unknown",
              ref: str = JEV_ASSET_REF) -> dict:
    """State: the asset ref + the escape prompt's framing (detector crop that
    may not be a robot, what a bumper looks like, neutral roster line)."""
    return {
        "crop": ref,
        "context": (
            "`crop` is an automatic crop from a robot detector running on FRC match video. "
            "It usually shows one FRC robot, but the detector makes mistakes: the crop may "
            "instead show game pieces (balls), a pile of balls, field structures, signs, "
            "people, or only part of a robot. An FRC robot bumper is a solid RED or BLUE "
            "fabric band around the base of a robot with a WHITE team number (1 to 5 digits, "
            "e.g. 294, 2485, 4322) printed on it. "
            f"{jev_roster_line(options, expected_alliance)} "
            "The team list is only for matching partially visible digits. Never pick a team "
            "unless you can see its digits on a bumper in `crop`."),
    }


def jev_questions(options: list[dict]) -> dict:
    """The single ``team`` question (the bench's ``subject`` is dropped)."""
    return {
        "team": {
            "type": "choice",
            "instructions": ("Which option matches the white team-number digits on the "
                             "robot bumper in `crop`? Choose an escape option unless you "
                             "can actually see a listed team's digits on a bumper."),
            "criteria": {o["id"]: jev_option_text(o) for o in options},
        },
    }


def build_jev_payload(model_key: str, img_bytes: bytes, options: list[dict],
                      expected_alliance: str = "unknown", seed=None,
                      ref: str = JEV_ASSET_REF) -> dict:
    """One Jev request. ``options`` is the UNSHUFFLED offered set (jev_state
    names its teams); the team question carries a per-``seed`` permutation."""
    return {
        "model": JEV_MODELS[model_key]["model"],
        "state": jev_state(options, expected_alliance, ref),
        "assets": {ref: {"type": "image", "format": "jpg",
                         "data_base64": base64.b64encode(img_bytes).decode()}},
        "questions": jev_questions(_shuffled(options, seed)),
    }


_CLASSIFY_CONTEXT = ("`crop` shows ONE object cut out of video of an FRC (FIRST Robotics "
                     "Competition) match.")


def build_jev_classify_payload(model_key: str, img_bytes: bytes,
                               ref: str = JEV_ASSET_REF) -> dict:
    """Equipment adjudication (robot / equipment / person / unclear)."""
    return {
        "model": JEV_MODELS[model_key]["model"],
        "state": {"crop": ref, "context": _CLASSIFY_CONTEXT},
        "assets": {ref: {"type": "image", "format": "jpg",
                         "data_base64": base64.b64encode(img_bytes).decode()}},
        "questions": {"object": {
            "type": "choice",
            "instructions": "What is the object in `crop`?",
            "criteria": dict(clef_reader._CLASSIFY_CRITERIA),
        }},
    }


def jev_estimate_tokens(payload: dict, image_wh) -> int:
    """Rough input tokens, fitted to 5 live crops (2026-11-05): ~1 per 415
    image px + ~1.9 chars/token for state+questions."""
    text = len(json.dumps(payload.get("state", ""))) + len(json.dumps(payload.get("questions", {})))
    w, h = image_wh
    return int(w * h / 415 + text / 1.9)


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #

def unwrap_jev_response(body) -> tuple[dict, dict]:
    """(answers, usage) from a Jev response."""
    if not isinstance(body, dict):
        raise ClefError(f"unexpected response type {type(body).__name__}")
    if body.get("error"):
        raise ClefError(f"API error: {str(body.get('error'))[:300]}")
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ClefError("response has no answers")
    usage = body.get("usage") or {}
    return answers, usage if isinstance(usage, dict) else {}


def _surface(top: Optional[str], roster_alliance: dict) -> str:
    """Jev has no surface question: derive it from the team answer's top."""
    if top == "printed":
        return "printed"
    if top in roster_alliance or top in ("unreadable", "other"):
        return "bumper"
    return "unknown"


def map_jev_answers(answers: dict, roster_alliance: dict, expected_alliance: str,
                    min_team_prob: float,
                    tentative_min_p: float = DEFAULT_TENTATIVE_MIN_P) -> dict:
    """Map a Jev answer set to the per-crop dict the other readers return.

    team_number: the best TEAM option when its p >= ``min_team_prob`` (a
                 normal read, confidence = p); else, when the top option is
                 ``unreadable`` and the best team's p >= ``tentative_min_p``
                 (> 0), that team as a TENTATIVE read (``tentative: True``,
                 confidence = p). not_robot / printed / other tops name no
                 team. p is the closed-set probability, used as-is (no
                 blend_read_confidence), like Clef.
    runner_up / runner_up_p: the second-best TEAM option (any team-naming
                 read; omitted when its p is 0).
    surface:     "printed" when the top is ``printed``; "bumper" for a team /
                 unreadable / other top; else "unknown".
    jev_probs:   the full option distribution (roster teams + escapes).
    """
    probs = _choice_probs(answers.get("team"))
    if not probs:
        raise ClefError(f"no team probabilities in answers: {str(answers)[:200]}")
    top = _top(probs)
    ranked = sorted(((p, t) for t, p in probs.items() if t in roster_alliance),
                    key=lambda pt: (-pt[0], pt[1]))
    best_team, p_team = (ranked[0][1], float(ranked[0][0])) if ranked else (None, 0.0)
    team, tentative = None, False
    if best_team is not None and p_team >= min_team_prob:
        team = best_team
    elif (best_team is not None and top == "unreadable" and tentative_min_p > 0.0
          and p_team >= tentative_min_p):
        team, tentative = best_team, True
    expected = expected_alliance if expected_alliance in ("red", "blue") else "unknown"
    alliance = expected
    if team is not None and roster_alliance.get(team) in ("red", "blue"):
        alliance = roster_alliance[team]
    out = {
        "team_number": team,
        "alliance": alliance,
        "confidence": round(p_team, 4) if team else 0.0,
        "raw_read": top,
        "digit_prob": round(float(probs[top]), 4),
        "surface": _surface(top, roster_alliance),
        "error": None,
        "tentative": tentative,
        "jev_probs": {k: round(float(v), 6) for k, v in probs.items()},
    }
    if team is not None and len(ranked) > 1 and ranked[1][0] > 0.0:
        out["runner_up"] = ranked[1][1]
        out["runner_up_p"] = round(float(ranked[1][0]), 4)
    return out


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

class JevSession(ClefSession):
    """Per-job Jev client on ClefSession's transport (same latch, retries,
    concurrency and counters; ``stats()`` / ``summary_line()`` unchanged).

    Single-frame only: a multi-frame list item is read from its newest crop
    (clef_multiframe is ignored for jev). There is no quota code to latch on:
    429s are retried (honouring retry-after); 401/403 latch the session off."""

    name = "jev"
    label = "Jev"
    vendor = "TypeSafe"
    auth_hint = "JEV_TYPESAFE_API_TOKEN"

    def __init__(self, model_key: str = DEFAULT_MODEL, *, token: Optional[str] = None,
                 tentative_min_p: Optional[float] = None, **kw):
        kw.pop("account", None)
        kw.pop("same_robot_min", None)
        self.tentative_min_p = (DEFAULT_TENTATIVE_MIN_P if tentative_min_p is None
                                else max(0.0, float(tentative_min_p)))
        super().__init__(model_key, token=token, **kw)

    # ---- transport hooks ----
    def _configure(self, model_key, account, token) -> None:
        if model_key not in JEV_MODELS:
            raise ValueError(f"unknown jev model {model_key!r}; valid: {sorted(JEV_MODELS)}")
        self.model_key = model_key
        self.spec = JEV_MODELS[model_key]
        self._account = ""
        self._token = token if token is not None else os.environ.get(JEV_ENV_KEYS[0], "")

    def _missing_credentials(self) -> list:
        return [] if self._token else list(JEV_ENV_KEYS)

    def _url(self) -> str:
        return JEV_URL

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
                IMAGES_HEADER: "true"}

    def _prep_image(self, jpg: bytes) -> tuple[bytes, int, int]:
        img, w, h, _padded = jev_image(jpg)
        return img, w, h

    def _options(self, red, blue, extra):
        # Options depend on the per-crop colour hint: built in _read_payload.
        return clef_reader.roster_options(red, blue, extra)

    def _read_payload(self, img, red, blue, expected, options, seed) -> dict:
        extra = [o["team"] for o in options
                 if o["kind"] == "team" and o.get("alliance") is None]
        return build_jev_payload(self.model_key, img,
                                 jev_options(red, blue, extra, expected), expected, seed)

    def _classify_payload(self, img) -> dict:
        return build_jev_classify_payload(self.model_key, img)

    def _unwrap(self, body):
        return unwrap_jev_response(body)

    def _map_read(self, answers, roster_alliance, expected) -> dict:
        return map_jev_answers(answers, roster_alliance, expected, self.min_team_prob,
                               self.tentative_min_p)

    def _estimate_tokens(self, payload, image_wh) -> int:
        return jev_estimate_tokens(payload, image_wh)

    def _is_quota(self, status, text) -> bool:
        return False

    def _read_multi(self, crops, meta, red, blue, expected, options, roster_alliance):
        return self._read_single(crops[0], red, blue, expected, options, roster_alliance)
