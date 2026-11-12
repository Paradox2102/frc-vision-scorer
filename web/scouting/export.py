"""Versioned export envelope + sinks (where results get sent).

Every metric result leaves this app wrapped in the same envelope so another
system — first RoboRecon (roborecon.net, Paradox2102/RoboRecon3) — can ingest
any metric without knowing our internals:

  {"schema": "frc-vision-scorer.scouting", "schema_version": 1,
   "metric": key, "metric_version": n, "metric_title": ...,
   "event_key": "2026cur", "year": 2026, "generated_at": ISO-8601 UTC,
   "frame": {units, field_m, perspective, symmetry, note},
   "params": {...}, "sources": [match refs], "skipped": [...],
   "teams": {"3128": {"team_number": 3128, ...}}, ...metric extras}

Identifier conventions chosen to match RoboRecon's schema: teams are ints
(``ScoutingReport.teamNumber``), matches are full TBA keys plus comp_level /
set_number / match_number (RoboRecon keeps only the number), driver station is
``alliance_position`` 1..3 in TBA order (``ScoutingReport.alliancePosition``).
Paths carry ``points_norm`` = 0..1 full-field points, the shape of RoboRecon's
``PathValue`` stroke points.

Bump SCHEMA_VERSION on any breaking envelope change; bump a metric's
``version`` on any breaking change to its own payload.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from web.scouting.geometry import frame_info

SCHEMA = "frc-vision-scorer.scouting"
SCHEMA_VERSION = 1


def envelope(metric, event_key: str, year: int, matches: list, skipped: list,
             result: dict, params: dict) -> dict:
    teams = result.get("teams") or {}
    for k, v in teams.items():
        if isinstance(v, dict):
            v.setdefault("team_number", int(k))
    env = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
           "metric": metric.key, "metric_version": metric.version,
           "metric_title": metric.title,
           "event_key": event_key, "year": int(year),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "frame": frame_info(year), "params": dict(params),
           "sources": [m.source_ref() for m in matches], "skipped": list(skipped)}
    for k, v in result.items():
        if k not in env:
            env[k] = v
    env["teams"] = teams
    return env


def filter_team(env: dict, team) -> dict:
    """The envelope reduced to one team (other keys untouched)."""
    out = dict(env)
    key = str(int(team))
    out["teams"] = {key: env["teams"][key]} if key in env.get("teams", {}) else {}
    return out


class Sink(Protocol):
    """Somewhere metric envelopes go. ``push`` returns a small status dict."""
    name: str

    def push(self, envelopes: list) -> dict: ...


class JsonFileSink:
    """Writes one JSON file per envelope: <dir>/<event>/<metric>.json."""
    name = "json_file"

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def push(self, envelopes: list) -> dict:
        written = []
        for env in envelopes:
            d = self.out_dir / env["event_key"]
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{env['metric']}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(env, separators=(",", ":")))
            tmp.replace(p)
            written.append(str(p))
        return {"sink": self.name, "written": written}


class RoboReconSink:
    """FUTURE: push envelopes to RoboRecon (roborecon.net).

    Neither side has the API yet. Planned contract (see
    docs/scouting-metrics-design-2026-11-12.md):
      * RoboRecon adds ``POST /api/integrations/vision-scouting`` authenticated
        like its cron routes (``Authorization: Bearer <secret>``), accepting a
        list of these envelopes, upserting by (event_key, metric, team).
      * This sink POSTs ``{"envelopes": [...]}`` with that bearer token
        (configured via env ROBORECON_URL / ROBORECON_TOKEN, never committed).
    """
    name = "roborecon"

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        self.base_url, self.token = base_url, token

    def push(self, envelopes: list) -> dict:
        raise NotImplementedError("RoboRecon ingestion API does not exist yet")
