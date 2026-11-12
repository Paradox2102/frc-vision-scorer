"""HTTP API for the scouting metrics (``/api/scouting``) — feeds web/static/scouting.html.

  GET /api/scouting/events                      events with complete replays + their metrics
  GET /api/scouting/{event}/metrics             applicable metrics + the replay per match used
  GET /api/scouting/{event}/teams               teams seen (lineup or label) + match counts
  GET /api/scouting/{event}/metric/{key}        one metric envelope (?team=N, ?<param>=v)
  GET /api/scouting/{event}/export              every applicable metric (the RoboRecon payload)
  GET /api/scouting/field-image/{year}[/meta]   AdvantageScope field2d jpg + its corner pixels

Everything blocking (sqlite, meta.json parsing, metric compute) runs in a worker
thread. Event loading is cached by registry on the replay-set fingerprint, so the
page's several calls for one event parse the replays once.
"""
from __future__ import annotations

import asyncio
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from web.scouting import export, geometry, loader, registry

router = APIRouter(prefix="/api/scouting", tags=["scouting"])

# Tests point this at a scratch sqlite file; None = the app's DB.
DB_PATH = None


def _db():
    return DB_PATH or loader.default_db_path()


def _year(event_key: str) -> int:
    return loader.event_season_year(event_key)


def _load(event_key: str) -> tuple:
    """(matches, skipped) through registry's fingerprint-keyed load cache."""
    db = DB_PATH
    fp = loader.source_fingerprint(event_key, db)
    return registry._load_cached(event_key, fp, db, loader.load_event)


# ----------------------------------------------------------------- field image
# Declared before the /{event_key}/... routes so "field-image" is never read as
# an event key.
@router.get("/field-image/{year}")
async def field_image(year: int):
    fi = geometry.field_image(year)
    if not fi:
        raise HTTPException(404, f"no field image for {year}")
    return FileResponse(fi["path"], media_type="image/jpeg")


@router.get("/field-image/{year}/meta")
async def field_image_meta(year: int):
    fi = geometry.field_image(year)
    if not fi:
        raise HTTPException(404, f"no field image for {year}")
    out = {k: v for k, v in fi.items() if k != "path"}
    out["year"] = int(year)
    out["url"] = f"/api/scouting/field-image/{int(year)}"
    return out


# ----------------------------------------------------------------- events
def _events_sync() -> list:
    conn = sqlite3.connect(str(_db()))
    try:
        rows = conn.execute(
            "SELECT event_key, COUNT(DISTINCT match_key) FROM replays "
            "WHERE status = 'complete' AND event_key IS NOT NULL AND event_key != '' "
            "AND match_key IS NOT NULL AND match_key != '' "
            "GROUP BY event_key").fetchall()
    finally:
        conn.close()
    out = []
    for event_key, n in rows:
        year = _year(event_key)
        out.append({"event_key": event_key, "n_matches": int(n), "year": year,
                    "metrics": [m.info() for m in registry.available_metrics(year)]})
    out.sort(key=lambda e: (-e["n_matches"], e["event_key"]))
    return out


@router.get("/events")
async def events():
    return await asyncio.to_thread(_events_sync)


# ----------------------------------------------------------------- per event
def _metrics_sync(event_key: str) -> dict:
    year = _year(event_key)
    matches, skipped = _load(event_key)
    return {"event_key": event_key, "year": year,
            "metrics": [m.info() for m in registry.available_metrics(year)],
            "sources": [m.source_ref() for m in matches], "skipped": list(skipped)}


@router.get("/{event_key}/metrics")
async def event_metrics(event_key: str):
    return await asyncio.to_thread(_metrics_sync, event_key)


def _teams_sync(event_key: str) -> dict:
    matches, _ = _load(event_key)
    seen: dict = {}
    for md in matches:
        lineup = {int(t) for a in ("red", "blue") for t in (md.alliances.get(a) or [])
                  if t is not None}
        labelled = {int(t) for t, tm in md.teams.items() if tm.samples or tm.auto_path}
        for t in lineup | labelled:
            row = seen.setdefault(t, {"team_number": t, "n_matches": 0, "n_labelled": 0})
            row["n_matches"] += 1
            row["n_labelled"] += t in labelled
    return {"event_key": event_key, "n_matches": len(matches),
            "teams": [seen[t] for t in sorted(seen)]}


@router.get("/{event_key}/teams")
async def event_teams(event_key: str):
    return await asyncio.to_thread(_teams_sync, event_key)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _params_from_query(metric, query) -> dict:
    """Query-string overrides for keys the metric declares in default_params."""
    out = {}
    for k, default in metric.default_params.items():
        if k not in query:
            continue
        raw = query.get(k)
        try:
            if isinstance(default, bool):
                low = str(raw).strip().lower()
                if low not in _TRUE | _FALSE:
                    raise ValueError(raw)
                out[k] = low in _TRUE
            elif isinstance(default, int):
                out[k] = int(raw)
            elif isinstance(default, float):
                out[k] = float(raw)
            else:
                out[k] = raw
        except (TypeError, ValueError):
            raise HTTPException(400, f"bad value for {k!r}: {raw!r}")
    return out


def _check_metric(event_key: str, key: str):
    metric = registry.get_metric(key)
    if metric is None:
        raise HTTPException(404, f"unknown metric {key!r}")
    year = _year(event_key)
    if not metric.applies_to(year):
        raise HTTPException(400, f"metric {key!r} does not apply to {year}")
    return metric


@router.get("/{event_key}/metric/{key}")
async def event_metric(event_key: str, key: str, request: Request,
                       team: Optional[int] = None):
    metric = _check_metric(event_key, key)
    params = _params_from_query(metric, request.query_params)
    env = await asyncio.to_thread(registry.run_metric, event_key, key, params, DB_PATH)
    return export.filter_team(env, team) if team is not None else env


def _export_sync(event_key: str) -> dict:
    year = _year(event_key)
    envs = [registry.run_metric(event_key, m.key, None, DB_PATH)
            for m in registry.available_metrics(year)]
    return {"event_key": event_key, "year": year, "envelopes": envs}


@router.get("/{event_key}/export")
async def event_export(event_key: str):
    return await asyncio.to_thread(_export_sync, event_key)
