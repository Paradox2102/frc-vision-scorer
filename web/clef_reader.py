"""Cloudflare Workers AI "Clef" decision-model bumper reader (team_reader="clef").

Clef models never generate text: each request carries a ``state`` string, up
to 4 images and a set of questions, and returns a probability per option. The
bumper read is a closed-set choice over the FULL 6-team match roster plus two
escapes (``none``: no legible number; ``other``: a legible number not listed).
Offering only the not-yet-confirmed teams makes forced-choice readers pick a
wrong roster team when the true one is missing; the full roster plus escapes
resisted that best on the 200-crop bench (clef-flash 4% vs Qwen-4B 39%).

Payload/option design is ported from scripts/bench_bumper_readers.py on branch
decision-reader-bench (clef_image, clef_state, clef_questions, post_json):
clef-flash read 68/90 legible bumpers there with 0 wrong reads and a 2%
false-positive rate, ~0.5 s and ~$0.00004 per read, and needs no GPU.

API (verified 2026-10-12):
  POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}
  Authorization: Bearer {CLOUDFLARE_API_TOKEN}
  body {"model", "state", "images": [data URLs], "questions": {...}}
  -> {"result": {"model", "answers", "usage": {"input_tokens", ...}},
      "success": true, "errors": []}   (the unwrapped result is accepted too)

Multi-frame reads (job knob clef_multiframe): ``read_bumpers`` also accepts a
LIST of crops of one identity (newest first) per item and sends them as ONE
request with up to 4 images, the state reworded for N moments of the same
robot, plus a ``same_robot`` noul question. On the labelled tracklet bench
(newest + the 3 sharpest earlier crops) clef-flash went from 31/48 to 39-41/48
legible bumpers with 0 wrong reads and 2% false positives. A same_robot
probability below ``same_robot_min`` discards the read as
error="tracklet_inconsistent" (the app raises a hijack alarm). A string item
is today's single-image request byte-for-byte.

Failure policy: HTTP 429 with error 4006 / "daily free allocation" means the
quota is exhausted and 401/403 means bad credentials. Both latch the session
unavailable for the rest of the job: one log line, then every later read
returns an error entry immediately with no HTTP. Other 429s, 5xx and network
errors are retried with backoff. Credentials are never logged.

``http_post`` is the module's ONLY network call; tests replace it.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

from web.field.identity_confidence import normalize_surface

# --------------------------------------------------------------------------- #
# Models, limits, option text
# --------------------------------------------------------------------------- #

CLEF_MODELS = {
    "clef-flash": {"model": "clef-flash", "route": "@cf/cloudflare/clef-flash",
                   "usd_per_m_input": 0.038},
    "clef": {"model": "clef", "route": "@cf/cloudflare/clef",
             "usd_per_m_input": 0.24},
    "clef-omni": {"model": "clef-omni", "route": "@cf/cloudflare/clef-omni",
                  "usd_per_m_input": 0.15},
}
DEFAULT_MODEL = "clef-flash"
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{route}"
CF_ENV_KEYS = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")

# Images under ~65k px are upscaled server-side; upscale small crops ourselves
# (LANCZOS) and shrink anything that would exceed the per-image token cap.
CLEF_MIN_SIDE = 256
CLEF_MAX_IMAGE_TOKENS = 1024
CLEF_MAX_IMAGE_BYTES = 4 * 1024 * 1024
CLEF_MAX_IMAGE_PIXELS = 16_000_000
# Images per request (API max). A multi-frame read sends the newest crop plus
# up to 3 earlier crops of the same identity in ONE request.
CLEF_MAX_IMAGES = 4
# Multi-frame reads ask "do all images show the same robot?" (noul); below this
# probability the read is discarded as tracklet_inconsistent (hijack alarm).
DEFAULT_SAME_ROBOT_MIN = 0.5
INCONSISTENT_ERROR = "tracklet_inconsistent"

ESCAPES = ("none", "other")
ESCAPE_TEXT = {
    "none": ("No legible team number on a robot bumper (blurry, occluded, cut off, "
             "not a robot, or the only digits are printed on field equipment)"),
    "other": "A legible team number that is NOT any of the listed teams",
}
OBJECT_CLASSES = ("robot", "equipment", "person", "unclear")

# Quota exhaustion: Workers AI error 4006 or its message text.
_QUOTA_CODES = (4006,)
_QUOTA_MARKERS = ("daily free allocation", "upgrade to cloudflare's workers paid")


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Image prep
# --------------------------------------------------------------------------- #

def clef_image_tokens(w: int, h: int) -> int:
    """Workers AI image tokens: 32-px blocks + 3 markers, capped at 1024."""
    return min(CLEF_MAX_IMAGE_TOKENS,
               max(1, round(w / 32)) * max(1, round(h / 32)) + 3)


def clef_image(jpg: bytes) -> tuple[bytes, int, int]:
    """Prep a crop: upscale to >= 256 px on the short side (LANCZOS); shrink if
    it would exceed 1024 image tokens; otherwise send the original bytes.

    The token cap (~1M px) keeps every output far below the API's 16 MP and
    4 MiB per-image limits."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpg))
    w, h = img.size
    scale = 1.0
    if min(w, h) < CLEF_MIN_SIDE:
        scale = CLEF_MIN_SIDE / max(1, min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if (round(nw / 32) * round(nh / 32) + 3 > CLEF_MAX_IMAGE_TOKENS
            or nw * nh > CLEF_MAX_IMAGE_PIXELS):
        shrink = math.sqrt((CLEF_MAX_IMAGE_TOKENS - 3) * 32 * 32 / (nw * nh)) * 0.98
        nw, nh = max(32, int(nw * shrink)), max(32, int(nh * shrink))
    if (nw, nh) == (w, h) and img.format == "JPEG" and len(jpg) <= CLEF_MAX_IMAGE_BYTES:
        return jpg, w, h
    img = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), nw, nh


def _data_url(img_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #

def roster_options(red: Iterable, blue: Iterable, extra: Iterable = ()) -> list[dict]:
    """Unshuffled options: every roster team labelled with its alliance, then
    any extra (alliance-unknown) teams, then the two escapes. Option ids are
    the team-number strings, so they never collide with the escape ids."""
    opts, seen = [], set()
    for teams, alliance in ((red, "red"), (blue, "blue"), (extra, None)):
        for t in teams or ():
            t = str(t)
            if t in seen:
                continue
            seen.add(t)
            opts.append({"id": t, "kind": "team", "team": t, "alliance": alliance})
    opts += [{"id": e, "kind": "escape"} for e in ESCAPES]
    return opts


def option_text(opt: dict) -> str:
    if opt["kind"] == "team":
        if opt.get("alliance") in ("red", "blue"):
            return f"Team {opt['team']} ({opt['alliance']} alliance)"
        return f"Team {opt['team']}"
    return ESCAPE_TEXT[opt["id"]]


def _shuffled(options: list[dict], seed) -> list[dict]:
    """Deterministic per-crop permutation: spreads any option-position bias
    across teams instead of always favouring the first-listed one."""
    if seed is None:
        return list(options)
    key = int(hashlib.sha1(str(seed).encode("utf-8")).hexdigest()[:16], 16)
    out = list(options)
    random.Random(key).shuffle(out)
    return out


def read_state(red: list, blue: list, expected_alliance: str = "unknown") -> str:
    s = ("One image: a crop from FRC robotics match video around ONE robot's bumper, a "
         "solid RED or BLUE fabric band around the robot's base with the team number in "
         "WHITE digits. Field structures (crates, bins, signs) can also carry printed "
         "numbers. "
         f"Match roster: red alliance {', '.join(map(str, red))}; "
         f"blue alliance {', '.join(map(str, blue))}.")
    if expected_alliance in ("red", "blue"):
        s += f" Computer vision sees {expected_alliance.upper()} bumper pixels."
    return s


def read_state_multi(n: int, span_s: Optional[float], red: list, blue: list,
                     expected_alliance: str = "unknown") -> str:
    """State for a multi-image read: ``n`` crops of ONE identity, newest first."""
    when = f" over ~{float(span_s):.1f} s" if span_s is not None else ""
    s = (f"{n} images: the same robot's bumper at different moments{when}, newest first. "
         "Each is a crop from FRC robotics match video around ONE robot's bumper, a solid "
         "RED or BLUE fabric band around the robot's base with the team number in WHITE "
         "digits. The digits may be readable in some images and blurred, cut off or hidden "
         "in others. Field structures (crates, bins, signs) can also carry printed numbers. "
         f"Match roster: red alliance {', '.join(map(str, red))}; "
         f"blue alliance {', '.join(map(str, blue))}.")
    if expected_alliance in ("red", "blue"):
        s += f" Computer vision sees {expected_alliance.upper()} bumper pixels."
    return s


SAME_ROBOT_QUESTION = {
    "type": "noul",
    "instructions": "Do all images show the same robot?",
    "criteria": {
        "true": ("Every image shows the same robot: the same bumper colour and number and the "
                 "same robot. Blur, partial occlusion or a different angle does not make it a "
                 "different robot."),
        "false": ("At least one image shows a different robot, or a non-robot object such as "
                  "field equipment or a person."),
    },
}


def read_questions(options: list[dict]) -> dict:
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
            "instructions": ("Which option matches the white team-number digits on this "
                             "robot's bumper?"),
            "criteria": {o["id"]: option_text(o) for o in options},
        },
    }


def read_questions_multi(options: list[dict]) -> dict:
    """The single-image questions plus ``same_robot`` (noul)."""
    q = read_questions(options)
    q["same_robot"] = json.loads(json.dumps(SAME_ROBOT_QUESTION))
    return q


def build_read_payload(model_key: str, img_bytes: bytes, red: list, blue: list,
                       expected_alliance: str = "unknown",
                       options: Optional[list[dict]] = None, seed=None) -> dict:
    red = [str(t) for t in red or ()]
    blue = [str(t) for t in blue or ()]
    if options is None:
        options = roster_options(red, blue)
    return {
        "model": CLEF_MODELS[model_key]["model"],
        "state": read_state(red, blue, expected_alliance),
        "questions": read_questions(_shuffled(options, seed)),
        "images": [_data_url(img_bytes)],
    }


def build_read_payload_multi(model_key: str, imgs: list, red: list, blue: list,
                             expected_alliance: str = "unknown",
                             options: Optional[list[dict]] = None, seed=None,
                             span_s: Optional[float] = None) -> dict:
    """One request carrying ``imgs`` (prepped JPEG bytes, NEWEST FIRST, <= 4).
    ``seed`` should be the newest crop, so the option order matches what a
    single-crop read of that crop would show."""
    if not imgs or len(imgs) > CLEF_MAX_IMAGES:
        raise ValueError(f"multi-image read needs 1..{CLEF_MAX_IMAGES} images, got {len(imgs)}")
    red = [str(t) for t in red or ()]
    blue = [str(t) for t in blue or ()]
    if options is None:
        options = roster_options(red, blue)
    return {
        "model": CLEF_MODELS[model_key]["model"],
        "state": read_state_multi(len(imgs), span_s, red, blue, expected_alliance),
        "questions": read_questions_multi(_shuffled(options, seed)),
        "images": [_data_url(b) for b in imgs],
    }


_CLASSIFY_STATE = ("One image: ONE object cut out of video of an FRC (FIRST Robotics "
                   "Competition) match.")
_CLASSIFY_CRITERIA = {
    "robot": ("A competition robot: a driving machine with wheels/mechanisms and a solid "
              "RED or BLUE bumper band around its base"),
    "equipment": ("Static field equipment: a crate, bin, tote, feeder rig, machine, or "
                  "other structure that does not drive"),
    "person": "A human (referees wear striped or orange vests)",
    "unclear": "You cannot tell",
}


def build_classify_payload(model_key: str, img_bytes: bytes) -> dict:
    return {
        "model": CLEF_MODELS[model_key]["model"],
        "state": _CLASSIFY_STATE,
        "questions": {"object": {
            "type": "choice",
            "instructions": "What is this object?",
            "criteria": dict(_CLASSIFY_CRITERIA),
        }},
        "images": [_data_url(img_bytes)],
    }


def estimate_tokens(payload: dict, image_wh) -> int:
    """Rough input tokens (image blocks + ~4 chars/token of text) for responses
    that carry no usage block. ``image_wh`` is one (w, h) or, for a
    multi-image request, a list of them (summed)."""
    text = len(json.dumps(payload.get("state", ""))) + len(json.dumps(payload.get("questions", {})))
    if image_wh and isinstance(image_wh[0], (tuple, list)):
        img_tok = sum(clef_image_tokens(*wh) for wh in image_wh)
    else:
        img_tok = clef_image_tokens(*image_wh)
    return img_tok + text // 4


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #

class ClefError(RuntimeError):
    """A read failed (not retryable, or retries exhausted)."""


class ClefTransientError(ClefError):
    """Retryable: 429 rate limit, 5xx, network error."""


class ClefQuotaExhausted(ClefError):
    """Workers AI daily allocation used up — not transient."""


class ClefAuthError(ClefError):
    """401/403: bad token or missing Workers AI permission."""


class ClefUnavailable(ClefError):
    """The session latched off earlier in this job; no HTTP was attempted."""


def _error_codes(text: str) -> list[int]:
    try:
        body = json.loads(text)
    except (TypeError, ValueError):
        return []
    codes = []
    for e in (body.get("errors") or []) if isinstance(body, dict) else []:
        try:
            codes.append(int(e.get("code")))
        except (AttributeError, TypeError, ValueError):
            pass
    return codes


def is_quota_error(status: int, text: str) -> bool:
    if status != 429:
        return False
    low = (text or "").lower()
    return (any(c in _QUOTA_CODES for c in _error_codes(text))
            or any(m in low for m in _QUOTA_MARKERS))


def unwrap_response(body) -> tuple[dict, dict]:
    """(answers, usage) from the Cloudflare envelope or the bare model result."""
    if not isinstance(body, dict):
        raise ClefError(f"unexpected response type {type(body).__name__}")
    if body.get("success") is False:
        raise ClefError(f"API error: {str(body.get('errors'))[:300]}")
    res = body.get("result", body)
    if not isinstance(res, dict):
        raise ClefError("response has no result object")
    answers = res.get("answers")
    if not isinstance(answers, dict):
        raise ClefError("response has no answers")
    usage = res.get("usage") or body.get("usage") or {}
    return answers, usage if isinstance(usage, dict) else {}


def _choice_probs(ans) -> dict:
    """{option_id: p} for a choice answer; falls back to {choice: confidence}."""
    if not isinstance(ans, dict):
        return {}
    probs = ans.get("probabilities")
    if isinstance(probs, dict) and probs:
        return {str(k): float(v) for k, v in probs.items()}
    choice = ans.get("choice")
    if choice is not None:
        return {str(choice): float(ans.get("confidence") or 0.0)}
    return {}


def _top(probs: dict) -> Optional[str]:
    return max(probs, key=lambda k: probs[k]) if probs else None


def map_read_answers(answers: dict, roster_alliance: dict, expected_alliance: str,
                     min_team_prob: float) -> dict:
    """Map a Clef answer set to the per-crop dict the vLLM reader returns.

    team_number: the top TEAM option when its probability >= min_team_prob.
    confidence:  that probability (0.0 with no team). It is the closed-set
                 probability of an exact roster match, so app.py's vote rule
                 (full weight at >= 0.85, else half) applies to it directly;
                 blend_read_confidence is NOT applied on top (it folds a
                 token-level certainty into a fuzzy roster-match base, and
                 would double-count here).
    digit_prob:  the probability of the chosen option (raw_read): the model's
                 certainty in what it read, like digit_token_conf.
    raw_read:    the chosen option id ("none"/"other" for the escapes).
    """
    probs = _choice_probs(answers.get("team"))
    if not probs:
        raise ClefError(f"no team probabilities in answers: {str(answers)[:200]}")
    top = _top(probs)
    team_probs = {k: v for k, v in probs.items() if k in roster_alliance}
    best_team = _top(team_probs)
    p_team = team_probs.get(best_team, 0.0) if best_team else 0.0
    team = best_team if best_team is not None and p_team >= min_team_prob else None
    expected = expected_alliance if expected_alliance in ("red", "blue") else "unknown"
    alliance = expected
    if team is not None and roster_alliance.get(team) in ("red", "blue"):
        alliance = roster_alliance[team]
    surface = normalize_surface(_top(_choice_probs(answers.get("surface"))))
    legible = (answers.get("legible") or {}).get("noul") if isinstance(
        answers.get("legible"), dict) else None
    return {
        "team_number": team,
        "alliance": alliance,
        "confidence": round(float(p_team), 4) if team else 0.0,
        "raw_read": top,
        "digit_prob": round(float(probs[top]), 4),
        "surface": surface,
        "error": None,
        "clef_probs": {k: round(float(v), 6) for k, v in probs.items()},
        "clef_legible": None if legible is None else round(float(legible), 4),
    }


def _noul(ans) -> Optional[float]:
    """The probability of a noul answer ({"noul": p}), or None if absent."""
    if not isinstance(ans, dict):
        return None
    v = ans.get("noul")
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def map_classify_answers(answers: dict) -> dict:
    probs = _choice_probs(answers.get("object"))
    top = _top(probs)
    return {"object": top if top in OBJECT_CLASSES else "unclear", "error": None,
            "clef_probs": {k: round(float(v), 6) for k, v in probs.items()}}


def error_entry(msg: str) -> dict:
    return {"team_number": None, "alliance": "unknown", "confidence": 0,
            "raw_read": None, "error": msg, "surface": "unknown"}


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

def http_post(url: str, data: bytes, headers: dict, timeout: float):
    """POST ``data``; return (status, body_text, retry_after_or_None).

    The ONLY network call in this module. HTTP error statuses are returned,
    not raised; connection errors and timeouts raise (the caller retries)."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), None
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        return e.code, text, (e.headers.get("Retry-After") if e.headers else None)


class ClefSession:
    """Per-job Clef client: credentials, quota/auth latch, cost counters.

    One session per job (built at job start for team_reader="clef"); every
    read and classify of that job goes through it, so a latch set by the first
    quota-exhausted response stops all later HTTP for the job."""

    def __init__(self, model_key: str = DEFAULT_MODEL, *,
                 account: Optional[str] = None, token: Optional[str] = None,
                 concurrency: Optional[int] = None, timeout_s: Optional[float] = None,
                 retries: Optional[int] = None, backoff_s: Optional[float] = None,
                 min_team_prob: Optional[float] = None,
                 same_robot_min: Optional[float] = None,
                 sleep_fn: Optional[Callable[[float], None]] = None,
                 log_fn: Optional[Callable[[str], None]] = None):
        if model_key not in CLEF_MODELS:
            raise ValueError(f"unknown clef model {model_key!r}; valid: {sorted(CLEF_MODELS)}")
        self.model_key = model_key
        self.spec = CLEF_MODELS[model_key]
        self._account = account if account is not None else os.environ.get(CF_ENV_KEYS[0], "")
        self._token = token if token is not None else os.environ.get(CF_ENV_KEYS[1], "")
        self.concurrency = max(1, concurrency if concurrency is not None
                               else _env_int("CLEF_CONCURRENCY", 8))
        self.timeout_s = timeout_s if timeout_s is not None else _env_float("CLEF_TIMEOUT_S", 15.0)
        self.retries = max(0, retries if retries is not None else _env_int("CLEF_RETRIES", 3))
        self.backoff_s = backoff_s if backoff_s is not None else _env_float("CLEF_BACKOFF_S", 0.5)
        self.min_team_prob = (min_team_prob if min_team_prob is not None
                              else _env_float("CLEF_MIN_TEAM_PROB", 0.5))
        self.same_robot_min = (DEFAULT_SAME_ROBOT_MIN if same_robot_min is None
                               else float(same_robot_min))
        # The job sets this when clef_multiframe is on, so the summary line
        # reports the multi-frame counters even when none fired.
        self.multiframe = False
        self._sleep = sleep_fn
        self._log = log_fn or print
        self._lock = threading.Lock()
        self._disabled = threading.Event()
        self.disabled_kind: Optional[str] = None     # quota | auth | credentials
        self.disabled_reason: Optional[str] = None
        self._c = {"requests": 0, "ok": 0, "input_tokens": 0, "output_tokens": 0,
                   "estimated_token_responses": 0, "errors": 0, "quota_errors": 0,
                   "auth_errors": 0, "retries": 0, "skipped": 0,
                   "mf_reads": 0, "mf_images": 0, "inconsistent": 0}
        self._latencies: list[float] = []
        missing = [n for n, v in zip(CF_ENV_KEYS, (self._account, self._token)) if not v]
        if missing:
            self._latch("credentials", f"{' / '.join(missing)} not set",
                        f"[clef] {' / '.join(missing)} not set — reads disabled for this job")

    # ---- state ----
    @property
    def available(self) -> bool:
        return not self._disabled.is_set()

    def _latch(self, kind: str, reason: str, line: str) -> None:
        with self._lock:
            if self._disabled.is_set():
                return
            self.disabled_kind = kind
            self.disabled_reason = reason
            self._disabled.set()
        self._log(line)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] += n

    def _redact(self, text: str) -> str:
        out = str(text)
        for secret in (self._token, self._account):
            if secret:
                out = out.replace(secret, "***")
        return out

    def _url(self) -> str:
        return CF_URL.format(account=self._account, route=self.spec["route"])

    # ---- transport ----
    def _request(self, payload: dict, image_wh) -> dict:
        """POST one payload with retries; returns the answers dict."""
        if self._disabled.is_set():
            self._bump("skipped")
            raise ClefUnavailable(f"clef unavailable: {self.disabled_reason}")
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self._token}"}
        sleep = self._sleep or time.sleep
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if attempt and self._disabled.is_set():   # another thread latched mid-retry
                self._bump("skipped")
                raise ClefUnavailable(f"clef unavailable: {self.disabled_reason}")
            retry_after = None
            self._bump("requests")
            t0 = time.monotonic()
            try:
                result = http_post(self._url(), data, headers, self.timeout_s)
            except OSError as e:
                # URLError, timeouts and connection resets are all OSErrors.
                last = f"{type(e).__name__}: {self._redact(e)}"
            else:
                status, text, *rest = result
                retry_after = rest[0] if rest else None
                if status == 200:
                    latency = time.monotonic() - t0
                    try:
                        answers, usage = unwrap_response(json.loads(text))
                    except ValueError as e:
                        raise ClefError(f"bad response JSON: {self._redact(e)}") from None
                    tokens = usage.get("input_tokens")
                    with self._lock:
                        self._c["ok"] += 1
                        self._latencies.append(latency)
                        if isinstance(tokens, (int, float)):
                            self._c["input_tokens"] += int(tokens)
                        else:
                            self._c["input_tokens"] += estimate_tokens(payload, image_wh)
                            self._c["estimated_token_responses"] += 1
                        out_tok = usage.get("output_tokens")
                        if isinstance(out_tok, (int, float)):
                            self._c["output_tokens"] += int(out_tok)
                    return answers
                msg = f"HTTP {status}: {self._redact((text or '')[:300])}"
                if is_quota_error(status, text):
                    self._bump("quota_errors")
                    self._latch("quota", "Workers AI quota exhausted",
                                "[clef] Workers AI quota exhausted — reads disabled for this job")
                    raise ClefQuotaExhausted(f"Workers AI quota exhausted ({msg})")
                if status in (401, 403):
                    self._bump("auth_errors")
                    self._latch("auth", f"Workers AI auth failed (HTTP {status})",
                                f"[clef] Workers AI auth failed (HTTP {status}) — check "
                                "CLOUDFLARE_API_TOKEN / Workers AI permission; reads "
                                "disabled for this job")
                    raise ClefAuthError(msg)
                if status != 429 and status < 500:
                    raise ClefError(msg)          # e.g. 400: payload problem, don't retry
                last = msg
            if attempt < self.retries:
                self._bump("retries")
                delay = min(8.0, self.backoff_s * (2 ** attempt)) + random.uniform(0, self.backoff_s)
                try:
                    if retry_after is not None:
                        delay = min(30.0, max(delay, float(retry_after)))
                except (TypeError, ValueError):
                    pass
                sleep(delay)
        raise ClefTransientError(f"gave up after {self.retries + 1} attempts: {last}")

    def _run(self, items: list, fn) -> list:
        if not items:
            return []
        if len(items) == 1:
            return [fn(0, items[0])]
        with ThreadPoolExecutor(max_workers=min(len(items), self.concurrency)) as pool:
            return list(pool.map(lambda iv: fn(*iv), enumerate(items)))

    # ---- public API ----
    def read_bumpers(self, crops_b64: list, red: list, blue: list,
                     expected_alliances: Optional[list] = None,
                     extra_teams: Iterable = (),
                     item_meta: Optional[list] = None) -> list[dict]:
        """One Clef request per item; results in the vLLM reader's dict shape.

        An item is a crop string (one image, today's request byte-for-byte) or
        a LIST of crop strings of ONE identity, newest first: a multi-frame
        read sent as ONE request with up to CLEF_MAX_IMAGES images (a 1-element
        list is read exactly like its string). ``item_meta[i]`` (optional dict)
        describes a list item: ``span_s`` (newest-to-oldest source seconds, for
        the state text) and ``frames`` (echoed back as ``clef_frames``). A
        multi-frame result also carries ``clef_same_robot`` (the noul answer);
        below ``same_robot_min`` it becomes team None with
        error="tracklet_inconsistent"."""
        red = [str(t) for t in red or ()]
        blue = [str(t) for t in blue or ()]
        options = roster_options(red, blue, extra_teams)
        roster_alliance = {o["id"]: o.get("alliance") or "unknown"
                           for o in options if o["kind"] == "team"}
        expected_alliances = list(expected_alliances or [])
        item_meta = list(item_meta or [])

        def _one(i: int, item) -> dict:
            expected = (expected_alliances[i] if i < len(expected_alliances)
                        and expected_alliances[i] in ("red", "blue") else "unknown")
            meta = item_meta[i] if i < len(item_meta) and isinstance(item_meta[i], dict) else None
            if isinstance(item, (list, tuple)):
                crops = [c for c in item if c][:CLEF_MAX_IMAGES]
                if len(crops) >= 2:
                    res = self._read_multi(crops, meta, red, blue, expected, options,
                                           roster_alliance)
                    if meta and meta.get("frames") is not None:
                        res["clef_frames"] = list(meta["frames"])
                    return res
                if not crops:
                    self._bump("errors")
                    return error_entry("clef: empty crop list")
                item = crops[0]          # a 1-element list reads like its string
            crop_b64 = item
            try:
                if self._disabled.is_set():
                    self._bump("skipped")
                    return error_entry(f"clef unavailable: {self.disabled_reason}")
                img, w, h = clef_image(base64.b64decode(crop_b64))
                payload = build_read_payload(self.model_key, img, red, blue, expected,
                                             options=options, seed=crop_b64)
                answers = self._request(payload, (w, h))
                return map_read_answers(answers, roster_alliance, expected, self.min_team_prob)
            except ClefUnavailable as e:
                return error_entry(str(e))
            except Exception as e:
                self._bump("errors")
                return error_entry(self._redact(f"clef: {e}"))

        results = self._run(list(crops_b64), _one)
        identified = sum(1 for r in results if r.get("team_number"))
        print(f"  Clef [{self.model_key}]: identified {identified}/{len(results)} robots: "
              f"{[r.get('team_number') for r in results]}")
        return results

    def _read_multi(self, crops: list, meta: Optional[dict], red: list, blue: list,
                    expected: str, options: list, roster_alliance: dict) -> dict:
        """One request with 2..4 images of one identity (newest first)."""
        try:
            if self._disabled.is_set():
                self._bump("skipped")
                return error_entry(f"clef unavailable: {self.disabled_reason}")
            imgs = [clef_image(base64.b64decode(c)) for c in crops]
            payload = build_read_payload_multi(
                self.model_key, [b for b, _w, _h in imgs], red, blue, expected,
                options=options, seed=crops[0],
                span_s=(meta or {}).get("span_s"))
            answers = self._request(payload, [(w, h) for _b, w, h in imgs])
            with self._lock:
                self._c["mf_reads"] += 1
                self._c["mf_images"] += len(imgs)
            res = map_read_answers(answers, roster_alliance, expected, self.min_team_prob)
            p_same = _noul(answers.get("same_robot"))
            res["clef_same_robot"] = None if p_same is None else round(p_same, 4)
            if p_same is not None and p_same < self.same_robot_min:
                self._bump("inconsistent")
                res.update(team_number=None, confidence=0.0, error=INCONSISTENT_ERROR,
                           alliance=expected if expected in ("red", "blue") else "unknown")
            return res
        except ClefUnavailable as e:
            return error_entry(str(e))
        except Exception as e:
            self._bump("errors")
            return error_entry(self._redact(f"clef: {e}"))

    def classify_crops(self, crops_b64: list) -> list[dict]:
        """robot / equipment / person / unclear, same shape as _vllm_classify_crops."""
        def _one(_i: int, crop_b64: str) -> dict:
            try:
                if self._disabled.is_set():
                    self._bump("skipped")
                    return {"object": "unclear",
                            "error": f"clef unavailable: {self.disabled_reason}"}
                img, w, h = clef_image(base64.b64decode(crop_b64))
                answers = self._request(build_classify_payload(self.model_key, img), (w, h))
                return map_classify_answers(answers)
            except ClefUnavailable as e:
                return {"object": "unclear", "error": str(e)}
            except Exception as e:
                self._bump("errors")
                return {"object": "unclear", "error": self._redact(f"clef: {e}")}

        return self._run(list(crops_b64), _one)

    # ---- accounting ----
    def stats(self) -> dict:
        with self._lock:
            c = dict(self._c)
            lat = list(self._latencies)
        return {
            "model": self.model_key,
            **c,
            "cost_usd": c["input_tokens"] * self.spec["usd_per_m_input"] / 1e6,
            "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
            "disabled": self.disabled_kind,
        }

    def summary_line(self) -> str:
        st = self.stats()
        tok = st["input_tokens"]
        tok_s = (f"{tok / 1e6:.2f}M" if tok >= 1e6 else
                 f"{tok / 1e3:.1f}k" if tok >= 1e3 else str(tok))
        line = (f"[clef] {st['model']}: {st['ok']} reads, {tok_s} tokens, "
                f"~${st['cost_usd']:.4f}, {st['errors']} errors")
        if st["latency_p50_s"] is not None:
            line += f", p50 {st['latency_p50_s']:.2f}s"
        if self.multiframe or st["mf_reads"] or st["inconsistent"]:
            line += (f", {st['mf_reads']} multi-frame ({st['mf_images']} images), "
                     f"{st['inconsistent']} inconsistent")
        if st["quota_errors"]:
            line += f", {st['quota_errors']} quota-exhausted"
        if self.disabled_kind:
            line += f"; DISABLED ({self.disabled_reason}), {st['skipped']} reads skipped"
        return line


# Fallback session for reads outside a job (no per-job session passed in).
_default_session: Optional[ClefSession] = None
_default_lock = threading.Lock()


def default_session() -> ClefSession:
    global _default_session
    with _default_lock:
        if _default_session is None:
            _default_session = ClefSession(os.environ.get("CLEF_MODEL") or DEFAULT_MODEL)
        return _default_session
