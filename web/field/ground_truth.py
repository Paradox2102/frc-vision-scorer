"""Hand-annotated ground-truth robot tracks — storage, validation, projection.

One file per (match, robot): ``data/eval/ground_truth/<match_key>/<team>.json``.
A human annotates ONE robot per pass through the match video (tools/
gt_annotator.html via scripts/gt_annotator_server.py), clicking the centre of
the robot's bumper footprint on the carpet at keyframes. Each click is stored in
composite-frame pixels (``uv``) AND projected to WPILib field metres
(``field_xy``) with the event's camera calibration — the same projection the
pipeline uses — so the scorer (web/field/gt_scoring.py) compares like with like.

Keyframe states
---------------
* ``visible``   — robot clearly seen; ``uv``/``field_xy`` are a confident click.
* ``estimated`` — robot occluded/blurred; ``uv``/``field_xy`` are a best guess.
* ``unknown``   — robot on the field but can't be located; no position.

Video alignment
---------------
``frame`` is a 0-based frame index into ``video`` (identified by
``fingerprint``). ``match_t0_frame`` is the video frame where autonomous starts;
it lets the scorer align against a replay made from a *different* download of
the same match (different trim) by match time instead of raw frame index.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
GT_DIR = REPO / "data" / "eval" / "ground_truth"
SCHEMA_VERSION = 1
STATES = ("visible", "estimated", "unknown")
POSITIONED_STATES = ("visible", "estimated")
ON_FIELD_MARGIN_M = 0.5

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_FP_BYTES = 4 * 1024 * 1024
_fp_cache: dict[tuple[str, int, float], str] = {}


# --------------------------------------------------------------------------- #
# Video identity
# --------------------------------------------------------------------------- #
def video_fingerprint(path: str | os.PathLike) -> str:
    """Content identity for a video file: ``<size>:<sha1 of first 4 MB>``.

    The uploads dir holds many byte-identical copies of the same match under
    different names, so names can't identify content; hashing whole 120 MB files
    is too slow for a listing. Size + head hash is unique in practice."""
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime)
    if key not in _fp_cache:
        h = hashlib.sha1()
        with open(p, "rb") as fh:
            h.update(fh.read(_FP_BYTES))
        _fp_cache[key] = f"{st.st_size}:{h.hexdigest()[:16]}"
    return _fp_cache[key]


# --------------------------------------------------------------------------- #
# Paths / IO
# --------------------------------------------------------------------------- #
def _check_key(value: str, what: str) -> str:
    value = str(value)
    if not _SAFE_KEY.match(value):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def gt_path(match_key: str, team: str, gt_dir: Path = GT_DIR) -> Path:
    return Path(gt_dir) / _check_key(match_key, "match_key") / f"{_check_key(team, 'team')}.json"


def load_gt(match_key: str, team: str, gt_dir: Path = GT_DIR) -> Optional[dict]:
    p = gt_path(match_key, team, gt_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_gt(match_key: str, gt_dir: Path = GT_DIR) -> list[dict]:
    """All GT docs for a match, sorted by team."""
    d = Path(gt_dir) / _check_key(match_key, "match_key")
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("."):  # save_gt's in-flight .tmp-*.json
            continue
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def list_gt_matches(gt_dir: Path = GT_DIR) -> list[str]:
    d = Path(gt_dir)
    if not d.is_dir():
        return []
    return sorted(c.name for c in d.iterdir() if c.is_dir() and any(c.glob("*.json")))


def save_gt(doc: dict, gt_dir: Path = GT_DIR) -> Path:
    """Validate, normalise (sort keyframes, one per frame) and write atomically.
    The previous version is kept as ``<team>.json.bak``."""
    doc = normalize_gt(doc)
    validate_gt(doc)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc.setdefault("created_at", doc["updated_at"])
    p = gt_path(doc["match_key"], doc["team"], gt_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        bak = p.with_suffix(".json.bak")
        bak.write_bytes(p.read_bytes())
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    os.chmod(tmp, 0o644)  # mkstemp creates 0600
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def new_gt(match_key: str, event_key: str, team: str, alliance: Optional[str],
           station: Optional[int], video: dict, match_t0_frame: Optional[int] = None,
           annotator: str = "") -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "match_key": match_key,
        "event_key": event_key,
        "team": str(team),
        "alliance": alliance,
        "station": station,
        "video": video,                 # {name, fingerprint, fps, width, height, frame_count}
        "match_t0_frame": match_t0_frame,
        "annotator": annotator,
        "calibration": {},              # {camera: version_id} used for field_xy
        "keyframes": [],
        "notes": "",
    }


def normalize_gt(doc: dict) -> dict:
    """Sort keyframes by frame; the LAST entry for a duplicated frame wins."""
    doc = dict(doc)
    by_frame: dict[int, dict] = {}
    for kf in doc.get("keyframes") or []:
        kf = dict(kf)
        kf["frame"] = int(kf["frame"])
        by_frame[kf["frame"]] = kf
    doc["keyframes"] = [by_frame[f] for f in sorted(by_frame)]
    doc["team"] = str(doc.get("team", ""))
    return doc


def validate_gt(doc: dict) -> None:
    for k in ("match_key", "event_key", "team", "video", "keyframes"):
        if k not in doc:
            raise ValueError(f"ground truth missing {k!r}")
    _check_key(doc["match_key"], "match_key")
    _check_key(doc["team"], "team")
    v = doc["video"]
    for k in ("name", "fingerprint", "fps"):
        if not v.get(k):
            raise ValueError(f"ground truth video missing {k!r}")
    if doc.get("alliance") not in (None, "blue", "red"):
        raise ValueError(f"bad alliance {doc.get('alliance')!r}")
    last = -1
    for kf in doc["keyframes"]:
        f = kf.get("frame")
        if not isinstance(f, int) or f < 0 or f <= last:
            raise ValueError(f"keyframes must have strictly increasing int frames (at {f!r})")
        last = f
        st = kf.get("state")
        if st not in STATES:
            raise ValueError(f"bad keyframe state {st!r} at frame {f}")
        if st in POSITIONED_STATES:
            uv = kf.get("uv")
            if not (isinstance(uv, (list, tuple)) and len(uv) == 2):
                raise ValueError(f"{st} keyframe at frame {f} needs uv [x, y]")


def keyframe_seconds(doc: dict, kf: dict) -> float:
    return kf["frame"] / float(doc["video"]["fps"])


def match_seconds(doc: dict, kf: dict) -> Optional[float]:
    """Seconds since autonomous start, or None without ``match_t0_frame``."""
    t0 = doc.get("match_t0_frame")
    if t0 is None:
        return None
    return (kf["frame"] - t0) / float(doc["video"]["fps"])


# --------------------------------------------------------------------------- #
# Projection (composite pixel -> field metres), identical to the pipeline's
# --------------------------------------------------------------------------- #
def choose_view(views: Iterable, uv) -> Optional[tuple]:
    """Pick the camera view whose composite crop contains ``uv``.

    ``views`` is ``web.field.camera_view.load_views()`` output (main first).
    The main and corner crops overlap by ~20 px at the seam; main wins, matching
    the pipeline's main-first reference ordering. Returns (view, local_uv)."""
    u, v = float(uv[0]), float(uv[1])
    for view in views:
        x, y, w, h = view.crop
        if x <= u < x + w and y <= v < y + h:
            return view, (u - x, v - y)
    return None


def project_uv(views: Iterable, uv, field_dims: Optional[tuple] = None) -> dict:
    """Composite-frame pixel -> {view, field_xy, on_field}.

    Projects the clicked point on the carpet (z=0) with the view's calibration.
    ``field_xy`` is None when no calibrated view contains the point or the ray
    misses the floor; ``on_field`` checks ``field_dims`` (length, width) with a
    0.5 m margin when given."""
    hit = choose_view(views, uv)
    if hit is None:
        return {"view": None, "field_xy": None, "on_field": False}
    view, local = hit
    cal = getattr(view, "calibration", None)
    xy = cal.image_to_field(local) if cal is not None else None
    on_field = xy is not None
    if xy is not None and field_dims is not None:
        L, W = field_dims
        m = ON_FIELD_MARGIN_M
        on_field = (-m <= xy[0] <= L + m) and (-m <= xy[1] <= W + m)
    return {"view": view.name,
            "field_xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)] if xy else None,
            "on_field": bool(on_field)}


def reproject_doc(doc: dict, views, field_dims: Optional[tuple] = None) -> dict:
    """Recompute every positioned keyframe's ``view``/``field_xy`` from ``uv``.
    The server does this on every save so field_xy always reflects the
    calibration in force (recorded under ``calibration``)."""
    doc = dict(doc)
    out = []
    for kf in doc.get("keyframes") or []:
        kf = dict(kf)
        if kf.get("state") in POSITIONED_STATES and kf.get("uv") is not None:
            r = project_uv(views, kf["uv"], field_dims)
            kf["view"], kf["field_xy"] = r["view"], r["field_xy"]
        else:
            kf.pop("field_xy", None)
            kf.pop("view", None)
        out.append(kf)
    doc["keyframes"] = out
    return doc
