#!/usr/bin/env python3
"""Local server for the ground-truth robot-track annotator (tools/gt_annotator.html).

Lists matches that have pipeline replays with a playable source video, serves
those videos (HTTP Range, so the browser can seek), and loads/saves hand
annotated per-robot tracks via web/field/ground_truth.py. ``field_xy`` is always
recomputed SERVER-SIDE from the clicked composite pixels on save, with the
calibration epoch the pipeline itself would use for the match (select_epoch).

Run:  .venv/bin/python scripts/gt_annotator_server.py [--port 8766] [--gt-dir DIR]
Then open http://localhost:8766
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

from web.field import ground_truth as gt  # noqa: E402
from web.field import store as field_store  # noqa: E402
from web.field.camera_view import load_views  # noqa: E402
from web.field.composite_layout import load_layout  # noqa: E402
from web.field.field_model import FieldModel  # noqa: E402
from web.field.match_order import select_epoch  # noqa: E402
from web.field.season import (DEFAULT_LAYOUT, layout_for_event,  # noqa: E402
                               season_for_event, season_for_layout)

HTML_PATH = ROOT / "tools" / "gt_annotator.html"
DB_PATH = Path(os.environ.get("GT_ANNOTATOR_DB", ROOT / "frc_scorer.db"))
GT_DIR = Path(os.environ.get("GT_ANNOTATOR_DIR", gt.GT_DIR))
DEFAULT_SEASON = DEFAULT_LAYOUT     # only when the event has no tag layout
GRID_STEP_PX = 8           # client-side projection lookup grid spacing
KF_FIELDS = ("frame", "state", "uv", "src", "rate", "note")

app = FastAPI(title="GT annotator")

_lock = threading.Lock()
_probe_cache: dict[tuple[str, float], dict] = {}     # (path, mtime) -> probe
_videos: dict[str, dict] = {}                        # video_id -> {path, fingerprint, ...}
_grid_cache: dict[tuple, dict] = {}


# --------------------------------------------------------------------------- #
# DB / video discovery
# --------------------------------------------------------------------------- #
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _probe(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime)
    with _lock:
        hit = _probe_cache.get(key)
    if hit is not None:
        return hit
    cap = cv2.VideoCapture(str(path))
    try:
        info = {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}
    finally:
        cap.release()
    with _lock:
        _probe_cache[key] = info
    return info


def _video_id(fingerprint: str) -> str:
    return fingerprint.replace(":", "-")


def _resolve(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    try:
        rp = Path(p).resolve()
    except OSError:
        return None
    return rp if rp.is_file() else None


def _auto_period(meta_path: Optional[str]) -> Optional[dict]:
    mp = _resolve(meta_path)
    if mp is None:
        return None
    try:
        meta = json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for per in (meta.get("summary") or {}).get("periods") or []:
        if per.get("label") == "auto" and per.get("start_frame") is not None:
            return per
    return None


def _roster(con, match_key: str) -> dict:
    rows = con.execute(
        "SELECT r.alliance, r.position, r.team_number FROM matches m "
        "JOIN match_robots r ON r.match_id = m.id WHERE m.tba_match_key = ? "
        "ORDER BY r.alliance, r.position", (match_key,)).fetchall()
    out = {"blue": [], "red": []}
    for r in rows:
        out[r["alliance"]].append({"team": str(r["team_number"]), "station": r["position"]})
    return out


def _roster_from_meta(meta_path: Optional[str]) -> dict:
    """Fallback roster when TBA has none: teams the pipeline labelled."""
    out = {"blue": [], "red": []}
    mp = _resolve(meta_path)
    if mp is None:
        return out
    try:
        meta = json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    seen = set()
    for fr in meta.get("frames") or []:
        for rb in fr.get("robots") or []:
            t, a = rb.get("team_number"), rb.get("alliance")
            if t and a in out and t not in seen and gt._SAFE_KEY.match(str(t)):
                seen.add(t)
                out[a].append({"team": str(t), "station": None})
    return out


def build_matches() -> list[dict]:
    """Every match with >=1 replay whose source video exists. Rebuilds the
    video allow-list as a side effect."""
    con = _db()
    try:
        reps = con.execute(
            "SELECT id, created_at, event_key, match_key, source_video_path, meta_path, status "
            "FROM replays WHERE match_key IS NOT NULL AND source_video_path IS NOT NULL "
            "ORDER BY created_at DESC").fetchall()
        by_match: dict[str, dict] = {}
        videos: dict[str, dict] = {}
        for r in reps:
            mk = r["match_key"]
            if not gt._SAFE_KEY.match(mk or ""):
                continue
            path = _resolve(r["source_video_path"])
            if path is None:
                continue
            try:
                fp = gt.video_fingerprint(path)
            except OSError:
                continue
            vid = _video_id(fp)
            m = by_match.setdefault(mk, {"match_key": mk, "event_key": r["event_key"],
                                          "videos": {}, "latest_meta": None})
            v = m["videos"].get(vid)
            if v is None:
                # replays are newest-first: the first hit is the most recent use
                v = m["videos"][vid] = {"video_id": vid, "fingerprint": fp, "name": path.name,
                                         "path": str(path), "replays": 0,
                                         "latest_replay_at": r["created_at"],
                                         "suggested_t0_frame": None, "_t0_done": False}
            v["replays"] += 1
            if r["status"] == "complete" and not v["_t0_done"]:
                per = _auto_period(r["meta_path"])
                if per is not None:
                    v["suggested_t0_frame"] = int(per["start_frame"])
                    v["_t0_done"] = True
            if m["latest_meta"] is None and r["status"] == "complete":
                m["latest_meta"] = r["meta_path"]
            videos[vid] = {"path": str(path), "fingerprint": fp, "name": path.name}

        out = []
        for mk, m in by_match.items():
            roster = _roster(con, mk)
            roster_src = "tba"
            if not roster["blue"] and not roster["red"]:
                roster, roster_src = _roster_from_meta(m["latest_meta"]), "pipeline"
            vlist = []
            for v in m["videos"].values():
                v.pop("_t0_done", None)
                v.pop("path", None)
                v.update(_probe(Path(videos[v["video_id"]]["path"])))
                videos[v["video_id"]].update(v)
                vlist.append(v)
            vlist.sort(key=lambda v: v["latest_replay_at"] or "", reverse=True)
            status = {}
            for doc in gt.list_gt(mk, GT_DIR):
                status[doc.get("team")] = {"keyframes": len(doc.get("keyframes") or []),
                                           "updated_at": doc.get("updated_at"),
                                           "video_fingerprint": (doc.get("video") or {}).get("fingerprint")}
            out.append({"match_key": mk, "event_key": m["event_key"], "roster": roster,
                        "roster_source": roster_src, "videos": vlist,
                        "default_video_id": vlist[0]["video_id"] if vlist else None,
                        "gt_status": status})
    finally:
        con.close()
    with _lock:
        _videos.clear()
        _videos.update(videos)
    from web.field.match_order import match_sort_key
    out.sort(key=lambda m: (m["event_key"] or "", match_sort_key(m["match_key"])))
    return out


def _video(video_id: str) -> dict:
    with _lock:
        empty = not _videos
    if empty:
        build_matches()
    with _lock:
        v = _videos.get(video_id)
    if v is None:
        raise HTTPException(404, "unknown video")
    return v


def _match_row(match_key: str) -> dict:
    try:
        gt._check_key(match_key, "match_key")
    except ValueError as e:
        raise HTTPException(400, str(e))
    con = _db()
    try:
        r = con.execute("SELECT event_key FROM replays WHERE match_key = ? AND event_key IS NOT NULL "
                        "ORDER BY created_at DESC LIMIT 1", (match_key,)).fetchone()
        if r is None:
            raise HTTPException(404, "unknown match")
        roster = _roster(con, match_key)
    finally:
        con.close()
    return {"event_key": r["event_key"], "roster": roster}


# --------------------------------------------------------------------------- #
# Calibration / projection
# --------------------------------------------------------------------------- #
def calibration_selection(event_key: str, match_key: Optional[str]) -> dict:
    """{camera: version_id} — the epoch the pipeline uses for this match
    (web/app.py uses select_epoch(load_versions(...), match_key))."""
    sel = {}
    for name in load_layout(event_key=event_key):
        chosen = select_epoch(field_store.load_versions(event_key, name), match_key)
        if chosen is not None:
            sel[name] = chosen["version_id"]
    return sel


def views_for(event_key: str, match_key: Optional[str]):
    """(views, selection, field_dims)."""
    sel = calibration_selection(event_key, match_key)
    views = load_views(event_key, sel or None)
    # the event's field (its key's year / override); 2026 only with no layout
    season = layout_for_event(event_key) or DEFAULT_SEASON
    for v in views:
        if v.calibration is not None and getattr(v.calibration, "field_season", ""):
            season = v.calibration.field_season
            break
    try:
        fm = FieldModel.load(season)
        dims = (fm.length_m, fm.width_m)
    except (OSError, KeyError, json.JSONDecodeError):
        fm = FieldModel.load(DEFAULT_SEASON)
        dims = (fm.length_m, fm.width_m)
    return views, sel, dims, season


def projection_grid(views, step: int = GRID_STEP_PX) -> list[dict]:
    """Per-view lookup grid of carpet projections at ``step`` px spacing in
    crop-local pixels (row-major; null where the ray misses the floor). The
    client bilinearly interpolates it for live feedback / the teleport guard;
    the server still recomputes the exact field_xy on save."""
    out = []
    for v in views:
        x, y, w, h = v.crop
        nx, ny = w // step + 2, h // step + 2
        pts = []
        cal = v.calibration
        for j in range(ny):
            for i in range(nx):
                xy = cal.image_to_field((i * step, j * step)) if cal is not None else None
                if xy is None or not all(map(math.isfinite, xy)):
                    pts.append(None)
                else:
                    pts.append([round(xy[0], 3), round(xy[1], 3)])
        out.append({"name": v.name, "crop": list(v.crop), "step": step, "nx": nx, "ny": ny,
                    "xy": pts})
    return out


ROBOT_FOOTPRINT_M = 0.95   # frame + bumpers, square
ROBOT_HEIGHT_M = 0.6
BOX_GRID_STEP_PX = 32


def robot_box_grid(views, field_dims, step: int = BOX_GRID_STEP_PX,
                   footprint: float = ROBOT_FOOTPRINT_M, height: float = ROBOT_HEIGHT_M) -> list[dict]:
    """Per-view grid of perspective-correct robot boxes for the annotator cursor.

    At each grid node (crop-local px, ``step`` spacing) the ground point (z=0) is
    projected to the field; if on-field, a ``footprint`` x ``footprint`` x
    ``height`` cuboid, axis-aligned to the field and centred on that point, is
    projected back and its image bounding box returned as offsets
    ``[dx0, dy0, dx1, dy1]`` (px) relative to the ground point's own image
    position. Off-field / unprojectable nodes are null."""
    L, W = field_dims
    m = gt.ON_FIELD_MARGIN_M
    h2 = footprint / 2.0
    corners = [(sx * h2, sy * h2, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, height)]
    out = []
    for v in views:
        x0, y0, w, h = v.crop
        cal = v.calibration
        nx, ny = w // step + 2, h // step + 2
        boxes = []
        for j in range(ny):
            for i in range(nx):
                b = None
                xy = cal.image_to_field((i * step, j * step)) if cal is not None else None
                if (xy is not None and all(map(math.isfinite, xy))
                        and -m <= xy[0] <= L + m and -m <= xy[1] <= W + m):
                    try:
                        g = cal.field_to_image((xy[0], xy[1], 0.0))
                        pts = [cal.field_to_image((xy[0] + cx, xy[1] + cy, cz)) for cx, cy, cz in corners]
                        us, vs = [p[0] - g[0] for p in pts], [p[1] - g[1] for p in pts]
                        b = [round(min(us), 1), round(min(vs), 1), round(max(us), 1), round(max(vs), 1)]
                        if not all(map(math.isfinite, b)) or (b[2] - b[0]) > w * 0.6 or (b[3] - b[1]) > h * 0.9:
                            b = None
                    except Exception:  # noqa: BLE001 — degenerate projection → no box
                        b = None
                boxes.append(b)
        out.append({"name": v.name, "crop": list(v.crop), "step": step, "nx": nx, "ny": ny, "box": boxes})
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text()


FIELD2D_DIR = ROOT / "data" / "field" / "advantagescope"


def _field2d_paths(season: str):
    """Drawing files for ``season``: a year ("2025"), a tag-layout stem
    ("2025-reefscape-andymark") or a drawing name ("2025_andymark"). The
    AndyMark variant gets its own drawing when one exists, else the welded one."""
    s = str(season)
    year = s[:4]
    if not year.isdigit():
        raise HTTPException(400, "bad season")
    name = year
    if "andymark" in s.lower() and (FIELD2D_DIR / f"{year}_andymark_field2d.json").is_file():
        name = f"{year}_andymark"
    return FIELD2D_DIR / f"{name}_field2d.json", FIELD2D_DIR / f"{name}_field2d.jpg", name


@app.get("/api/field2d")
def api_field2d(season: str = "2026"):
    """Top-down field drawing (AdvantageScope Field2d, pre-rotated to the
    WPILib blue-left frame) + the pixel rect of the field carpet in it."""
    cfg_p, img_p, name = _field2d_paths(season)
    if not cfg_p.is_file() or not img_p.is_file():
        raise HTTPException(404, f"no field drawing for {name}")
    cfg = json.loads(cfg_p.read_text())
    cfg["image_url"] = f"/api/field2d/{name}.jpg"
    return cfg


@app.get("/api/field2d/{year}.jpg")
def api_field2d_image(year: str):
    _cfg_p, img_p, _ = _field2d_paths(year)
    if not img_p.is_file():
        raise HTTPException(404, "missing")
    return FileResponse(img_p, media_type="image/jpeg")


@app.get("/api/matches")
def api_matches():
    return {"matches": build_matches(), "gt_dir": str(GT_DIR)}


@app.get("/api/video/{video_id}")
def api_video(video_id: str):
    v = _video(video_id)
    p = Path(v["path"])
    if not p.is_file():
        raise HTTPException(404, "video missing")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/event/{event_key}/views")
def api_views(event_key: str, match_key: Optional[str] = None, grid: bool = False):
    try:
        gt._check_key(event_key, "event_key")
        if match_key:
            gt._check_key(match_key, "match_key")
    except ValueError as e:
        raise HTTPException(400, str(e))
    views, sel, dims, season = views_for(event_key, match_key)
    info = season_for_layout(season) or season_for_event(event_key)
    out = {"event_key": event_key, "match_key": match_key, "season": season,
           "season_info": info.public(),
           "field": {"length_m": dims[0], "width_m": dims[1]},
           "on_field_margin_m": gt.ON_FIELD_MARGIN_M,
           "calibration": sel,
           "views": [{"name": v.name, "crop": list(v.crop),
                      "mode": getattr(v.calibration, "mode", None),
                      "version_id": sel.get(v.name)} for v in views]}
    if grid:
        key = (event_key, tuple(sorted(sel.items())))
        with _lock:
            g = _grid_cache.get(key)
        if g is None:
            g = projection_grid(views)
            with _lock:
                _grid_cache[key] = g
        out["grid"] = g
    return out


@app.get("/api/event/{event_key}/robot_box_grid")
def api_robot_box_grid(event_key: str, match_key: Optional[str] = None):
    try:
        gt._check_key(event_key, "event_key")
        if match_key:
            gt._check_key(match_key, "match_key")
    except ValueError as e:
        raise HTTPException(400, str(e))
    views, sel, dims, _ = views_for(event_key, match_key)
    key = ("box", event_key, tuple(sorted(sel.items())))
    with _lock:
        g = _grid_cache.get(key)
    if g is None:
        g = robot_box_grid(views, dims)
        with _lock:
            _grid_cache[key] = g
    return {"event_key": event_key, "calibration": sel, "footprint_m": ROBOT_FOOTPRINT_M,
            "height_m": ROBOT_HEIGHT_M, "grid": g}


@app.post("/api/project")
def api_project(body: dict = Body(...)):
    event_key = str(body.get("event_key") or "")
    match_key = body.get("match_key")
    try:
        gt._check_key(event_key, "event_key")
        if match_key:
            gt._check_key(match_key, "match_key")
        u, v = (float(c) for c in body["uv"])
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, f"bad request: {e}")
    views, _sel, dims, _ = views_for(event_key, match_key)
    return gt.project_uv(views, (u, v), dims, z=gt.DEFAULT_CLICK_HEIGHT_M)


@app.get("/api/gt/{match_key}")
def api_gt_match(match_key: str):
    try:
        return {"docs": gt.list_gt(match_key, GT_DIR)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/gt/{match_key}/{team}")
def api_gt_get(match_key: str, team: str):
    try:
        doc = gt.load_gt(match_key, team, GT_DIR)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"doc": doc}


def _clean_keyframes(raw) -> list[dict]:
    if not isinstance(raw, list):
        raise HTTPException(400, "keyframes must be a list")
    out = []
    for kf in raw:
        if not isinstance(kf, dict):
            raise HTTPException(400, "keyframe must be an object")
        k = {f: kf[f] for f in KF_FIELDS if f in kf and kf[f] is not None}
        try:
            k["frame"] = int(k["frame"])
            if k.get("uv") is not None:
                k["uv"] = [round(float(k["uv"][0]), 2), round(float(k["uv"][1]), 2)]
                if not all(map(math.isfinite, k["uv"])):
                    raise ValueError("non-finite uv")
            if "rate" in k:
                k["rate"] = float(k["rate"])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise HTTPException(400, f"bad keyframe {kf!r}: {e}")
        if k.get("state") == "unknown":
            k.pop("uv", None)
        out.append(k)
    return out


@app.put("/api/gt/{match_key}/{team}")
def api_gt_put(match_key: str, team: str, body: dict = Body(...)):
    try:
        gt._check_key(match_key, "match_key")
        gt._check_key(team, "team")
    except ValueError as e:
        raise HTTPException(400, str(e))
    m = _match_row(match_key)
    v = _video(str(body.get("video_id") or ""))
    existing = gt.load_gt(match_key, team, GT_DIR)
    if (existing and existing.get("keyframes") and not body.get("force_video")
            and (existing.get("video") or {}).get("fingerprint") != v["fingerprint"]):
        raise HTTPException(409, "existing ground truth was annotated on a different video "
                                 f"({existing['video'].get('name')}); frames would not line up")
    alliance, station = None, None
    for a in ("blue", "red"):
        for slot in m["roster"][a]:
            if slot["team"] == team:
                alliance, station = a, slot["station"]
    if alliance is None and body.get("alliance") in ("blue", "red"):
        alliance = body["alliance"]
    t0 = body.get("match_t0_frame")
    doc = gt.new_gt(match_key, m["event_key"], team, alliance, station,
                    video={"name": v["name"], "fingerprint": v["fingerprint"],
                           "fps": v.get("fps"), "width": v.get("width"),
                           "height": v.get("height"), "frame_count": v.get("frame_count")},
                    match_t0_frame=int(t0) if t0 is not None else None,
                    annotator=str(body.get("annotator") or "")[:80])
    if existing and existing.get("created_at"):
        doc["created_at"] = existing["created_at"]
    if existing and existing.get("click_height_m") is not None:
        doc["click_height_m"] = existing["click_height_m"]   # keep a re-tuned value
    doc["notes"] = str(body.get("notes") or "")[:4000]
    doc["keyframes"] = _clean_keyframes(body.get("keyframes") or [])
    views, sel, dims, _ = views_for(m["event_key"], match_key)
    doc = gt.reproject_doc(doc, views, dims)
    doc["calibration"] = sel
    try:
        gt.save_gt(doc, GT_DIR)
    except ValueError as e:
        raise HTTPException(400, str(e))
    saved = gt.load_gt(match_key, team, GT_DIR)
    off = [kf["frame"] for kf in saved["keyframes"]
           if kf.get("state") in gt.POSITIONED_STATES and kf.get("field_xy") is None]
    return {"doc": saved, "unprojectable_frames": off}


@app.get("/api/replays/{match_key}")
def api_replays(match_key: str):
    try:
        gt._check_key(match_key, "match_key")
    except ValueError as e:
        raise HTTPException(400, str(e))
    con = _db()
    try:
        rows = con.execute("SELECT id, created_at, label, source_video_path, meta_path FROM replays "
                           "WHERE match_key = ? AND status = 'complete' ORDER BY created_at DESC "
                           "LIMIT 40", (match_key,)).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        if _resolve(r["meta_path"]) is None:
            continue
        path = _resolve(r["source_video_path"])
        fp = gt.video_fingerprint(path) if path is not None else None
        out.append({"id": r["id"], "created_at": r["created_at"], "label": r["label"],
                    "video_fingerprint": fp, "video_id": _video_id(fp) if fp else None})
    return {"replays": out}


@app.get("/api/replay/{replay_id}/tracks")
def api_replay_tracks(replay_id: str):
    con = _db()
    try:
        r = con.execute("SELECT meta_path FROM replays WHERE id = ?", (replay_id,)).fetchone()
    finally:
        con.close()
    if r is None:
        raise HTTPException(404, "unknown replay")
    mp = _resolve(r["meta_path"])
    if mp is None:
        raise HTTPException(404, "replay meta missing")
    meta = json.loads(mp.read_text())
    slots: dict[str, dict] = {}
    for fr in meta.get("frames") or []:
        fn = fr.get("frame_num")
        if fn is None:
            continue
        for rb in fr.get("robots") or []:
            pos = rb.get("field_pos")
            if not pos or pos[0] is None or pos[1] is None:
                continue
            s = slots.setdefault(rb.get("slot") or "?", {"slot": rb.get("slot"),
                                                         "alliance": rb.get("alliance"),
                                                         "team_number": None, "pts": []})
            if rb.get("team_number"):
                s["team_number"] = str(rb["team_number"])
            s["pts"].append([int(fn), round(float(pos[0]), 3), round(float(pos[1]), 3)])
    periods = (meta.get("summary") or {}).get("periods") or []
    return {"replay_id": replay_id, "periods": periods,
            "slots": sorted(slots.values(), key=lambda s: s["slot"] or "")}


def main():
    global GT_DIR, DB_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gt-dir", default=None, help=f"ground-truth dir (default {GT_DIR})")
    ap.add_argument("--db", default=None, help=f"sqlite DB (default {DB_PATH})")
    args = ap.parse_args()
    if args.gt_dir:
        GT_DIR = Path(args.gt_dir).resolve()
    if args.db:
        DB_PATH = Path(args.db).resolve()
    import uvicorn
    print(f"GT annotator on http://{args.host}:{args.port}  (gt dir: {GT_DIR})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
