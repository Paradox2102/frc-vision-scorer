"""HTTP API for the scouting metrics (``/api/scouting``) — feeds web/static/scouting.html.

  GET /api/scouting/events                      events with complete replays + their metrics
  GET /api/scouting/{event}/metrics             applicable metrics + the replay per match used
  GET /api/scouting/{event}/teams               teams seen (lineup or label) + match counts
  GET /api/scouting/{event}/metric/{key}        one metric envelope (?team=N, ?<param>=v)
  GET /api/scouting/{event}/export              every applicable metric + per-match video
                                                timeline + every robot's whole-match path
                                                (?paths=false drops the paths; format:
                                                docs/scouting-export/README.md)
  GET /api/scouting/{event}/match/{match_key}   one match: lineup, timeline, robot paths
                                                (?team=N for one robot)
  GET /api/scouting/field-image/{year}[/meta]   AdvantageScope field2d jpg + its corner pixels
  GET  /api/scouting/roborecon/status           RoboRecon connection/auth check (no token echoed)
  POST /api/scouting/{event}/push-roborecon     push metric envelopes to RoboRecon
                                                (?metrics=auto_paths,... ; default all)

Everything blocking (sqlite, meta.json parsing, metric compute) runs in a worker
thread. Event loading is cached by registry on the replay-set fingerprint, so the
page's several calls for one event parse the replays once.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from web.scouting import export, geometry, loader, registry, timeline

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
                out[k] = registry.check_param(metric, k, int(raw))
            elif isinstance(default, float):
                out[k] = registry.check_param(metric, k, float(raw))
            else:
                out[k] = raw
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"bad value for {k!r}: {raw!r} ({e})")
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


def _export_sync(event_key: str, paths: bool = True) -> dict:
    year = _year(event_key)
    envs = [registry.run_metric(event_key, m.key, None, DB_PATH)
            for m in registry.available_metrics(year)]
    matches, skipped = _load(event_key)
    return export.event_export(event_key, year, matches, skipped, envs,
                               {m.match_key: timeline.for_match(m) for m in matches},
                               paths=paths)


@router.get("/{event_key}/export")
async def event_export(event_key: str, paths: bool = True):
    return await asyncio.to_thread(_export_sync, event_key, paths)


def _match_sync(event_key: str, match_key: str, team: Optional[int]) -> dict:
    matches, skipped = _load(event_key)
    md = next((m for m in matches if m.match_key == match_key), None)
    if md is None:
        why = next((s.get("reason") for s in skipped if s.get("match_key") == match_key), None)
        raise HTTPException(404, f"{match_key}: {why}" if why
                            else f"no analysed match {match_key!r} in {event_key}")
    row = export.match_row(md, timeline.for_match(md), paths=True)
    if team is not None:
        row["robots"] = [r for r in row["robots"] if r["team_number"] == int(team)]
        if not row["robots"]:
            raise HTTPException(404, f"team {team} is not in {match_key}")
    return {"schema": export.MATCH_SCHEMA, "schema_version": export.EXPORT_SCHEMA_VERSION,
            "event_key": event_key, "year": md.year,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "path_frame": export.path_frame(md.year), "match": row}


@router.get("/{event_key}/match/{match_key}")
async def match_export(event_key: str, match_key: str, team: Optional[int] = None):
    return await asyncio.to_thread(_match_sync, event_key, match_key, team)


# ----------------------------------------------------------------- RoboRecon push
def push_event(event_key: str, metrics: Optional[list] = None, sink=None,
               team: Optional[int] = None) -> dict:
    """Run the event's metrics (all applicable, or just ``metrics``) and push the
    envelopes to RoboRecon. Each team's payload covers every match of the event
    (RoboRecon replaces per event/metric/team), so re-running after a match is
    (re)analysed sends that match's newest result exactly once.
    Raises RoboReconError if the push as a whole fails."""
    year = _year(event_key)
    available = {m.key for m in registry.available_metrics(year)}
    keys = [k for k in (metrics or sorted(available))]
    unknown = [k for k in keys if k not in available]
    if unknown:
        raise ValueError(f"metric(s) not available for {year}: {', '.join(unknown)}")
    envs = [registry.run_metric(event_key, k, None, DB_PATH) for k in keys]
    if team is not None:
        envs = [export.filter_team(e, team) for e in envs]
    out = (sink or export.RoboReconSink()).push(envs)
    return {"event_key": event_key, "metrics": keys, **out}


# No /{event_key}/status route exists, so "roborecon" is never read as an event key.
@router.get("/roborecon/status")
async def roborecon_status():
    """Connection check plus the auto-push config and last result per event."""
    from web.scouting import autopush
    try:
        conn = await asyncio.to_thread(lambda: export.RoboReconSink().check())
    except export.RoboReconError as e:
        conn = {"ok": False, "error": str(e)}
    return {"connection": conn, "auto_push": autopush.status()}


@router.post("/{event_key}/push-roborecon")
async def push_roborecon(event_key: str, metrics: Optional[str] = None):
    keys = [k.strip() for k in metrics.split(",") if k.strip()] if metrics else None
    try:
        return await asyncio.to_thread(push_event, event_key, keys)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except export.RoboReconError as e:
        raise HTTPException(502, str(e))
