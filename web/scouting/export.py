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
import os
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


class RoboReconError(RuntimeError):
    """The push as a whole failed (not configured, bad token, network, server
    error). Per-envelope validation failures are NOT raised: they come back in
    ``push()``'s result. Messages never contain the token."""


class RoboReconSink:
    """Pushes envelopes to RoboRecon (roborecon.net, Paradox2102/RoboRecon3).

    Contract (RoboRecon ``src/app/api/integrations/vision-scouting/route.ts``):
      * ``GET  {url}/api/integrations/vision-scouting`` -> 200
        ``{success, data: {schema, supportedSchemaVersions}}``: connection check.
      * ``POST`` same path, body ``{"envelopes": [...]}`` (<= 100 envelopes,
        <= 25 MB) -> 200 all stored / 207 some rejected / 400 all rejected, body
        ``{success, data: {stored, rejected, results: [{index, metric, eventKey,
        status, teamsStored, teamsSkippedStale, eventKnown, errors?}]}}``.
      * Auth ``Authorization: Bearer <token>``; 401 bad token, 503 when
        RoboRecon has no key configured.
      * RoboRecon REPLACES each (event_key, metric, team) row on every push
        (never appends; an older ``generated_at`` than stored is skipped), so
        a team's payload must carry ALL its matches. Envelopes too big for one
        request are therefore split by team, never by match.

    Configured by env ``ROBORECON_URL`` / ``ROBORECON_TOKEN`` (repo ``.env``,
    gitignored) unless passed in. ``session`` is injectable for tests.
    """
    name = "roborecon"
    PATH = "/api/integrations/vision-scouting"
    MAX_ENVELOPES = 100                 # RoboRecon rejects more per request
    MAX_BYTES = 20 * 1024 * 1024        # RoboRecon's cap is 25 MB; keep headroom

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None,
                 timeout_s: float = 120.0, session=None):
        if base_url is None or token is None:
            _load_dotenv()
        self.base_url = (base_url or os.environ.get("ROBORECON_URL") or "").rstrip("/")
        self.token = token or os.environ.get("ROBORECON_TOKEN") or ""
        self.timeout_s = timeout_s
        self._session = session

    # ---------------------------------------------------------------- helpers
    @property
    def url(self) -> str:
        return self.base_url + self.PATH

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _require_config(self) -> None:
        missing = [n for n, v in (("ROBORECON_URL", self.base_url),
                                  ("ROBORECON_TOKEN", self.token)) if not v]
        if missing:
            raise RoboReconError(f"{' and '.join(missing)} not set (add to .env)")

    def _raise_for_status(self, resp) -> None:
        if resp.status_code in (200, 207, 400):
            return
        detail = _error_text(resp)
        if resp.status_code == 401:
            raise RoboReconError(f"RoboRecon rejected the token (401) at {self.url}; "
                                 "check ROBORECON_TOKEN matches VISION_SCOUTING_API_KEY")
        if resp.status_code == 503:
            raise RoboReconError(f"RoboRecon at {self.base_url} has no VISION_SCOUTING_API_KEY "
                                 f"set, or is restarting (503): {detail}")
        if resp.status_code == 404:
            raise RoboReconError(f"{self.url} not found (404): is that RoboRecon "
                                 "deployment older than the vision-scouting API?")
        raise RoboReconError(f"RoboRecon returned {resp.status_code} at {self.url}: {detail}")

    # ---------------------------------------------------------------- public
    def check(self) -> dict:
        """Connection + auth check. Returns {ok, url, supported_schema_versions};
        raises RoboReconError when unreachable or unauthorised."""
        self._require_config()
        try:
            resp = self._http().get(self.url, headers=self._headers(), timeout=self.timeout_s)
        except Exception as e:                      # requests' ConnectionError, Timeout, ...
            raise RoboReconError(f"could not reach {self.url}: {type(e).__name__}") from None
        self._raise_for_status(resp)
        data = (_json(resp) or {}).get("data") or {}
        versions = data.get("supportedSchemaVersions") or []
        return {"ok": SCHEMA_VERSION in versions, "url": self.url,
                "supported_schema_versions": versions}

    def push(self, envelopes: list) -> dict:
        """POST every envelope (batched / split by team as needed). Returns
        {sink, url, ok, requests, stored, rejected, results}; ``results`` has one
        entry per envelope actually sent (a split envelope appears once per part)."""
        self._require_config()
        results: list = []
        batches = list(_batches(envelopes, self.MAX_ENVELOPES, self.MAX_BYTES))
        for batch, body in batches:
            try:
                resp = self._http().post(self.url, data=body, headers=self._headers(),
                                         timeout=self.timeout_s)
            except Exception as e:
                raise RoboReconError(f"could not reach {self.url}: {type(e).__name__} "
                                     f"(after {len(results)} of {_count(batches)} envelopes)") from None
            self._raise_for_status(resp)
            payload = _json(resp) or {}
            sent = (payload.get("data") or {}).get("results")
            if not isinstance(sent, list):              # 400 for a malformed body
                sent = [{"index": i, "status": "rejected",
                         "errors": [payload.get("error") or f"HTTP {resp.status_code}"]}
                        for i in range(len(batch))]
            for r, env in zip(sent, batch):
                results.append({**r, "metric": r.get("metric", env.get("metric")),
                                "teams": len(env.get("teams") or {})})
        stored = sum(1 for r in results if r.get("status") == "stored")
        return {"sink": self.name, "url": self.url, "ok": stored == len(results),
                "requests": len(batches), "stored": stored,
                "rejected": len(results) - stored, "results": results}


def _load_dotenv() -> None:
    """Fill ROBORECON_* from the repo's .env without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _json(resp) -> Optional[dict]:
    try:
        out = resp.json()
        return out if isinstance(out, dict) else None
    except ValueError:
        return None


def _error_text(resp) -> str:
    body = _json(resp)
    if body and body.get("error"):
        return str(body["error"])[:300]
    return (getattr(resp, "text", "") or "")[:300]


def _dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _split_by_team(env: dict, max_bytes: int) -> list:
    """An envelope too large for one request, as several envelopes with
    disjoint team subsets (each team whole). A single team over the limit is
    still sent alone; RoboRecon will answer 413 for it."""
    teams = list((env.get("teams") or {}).items())
    shell = {k: v for k, v in env.items() if k != "teams"}
    base = len(_dumps({"envelopes": [{**shell, "teams": {}}]}))
    parts, cur, size = [], {}, base
    for key, payload in teams:
        n = len(_dumps({key: payload}))
        if cur and size + n > max_bytes:
            parts.append({**shell, "teams": cur})
            cur, size = {}, base
        cur[key] = payload
        size += n
    if cur or not parts:
        parts.append({**shell, "teams": cur})
    return parts


def _batches(envelopes: list, max_envelopes: int, max_bytes: int):
    """Yield (envelopes, serialized body) request batches within both limits."""
    pieces = []
    for env in envelopes:
        if len(_dumps({"envelopes": [env]})) > max_bytes:
            pieces.extend(_split_by_team(env, max_bytes))
        else:
            pieces.append(env)
    batch: list = []
    for env in pieces:
        candidate = batch + [env]
        if batch and (len(candidate) > max_envelopes
                      or len(_dumps({"envelopes": candidate})) > max_bytes):
            yield batch, _dumps({"envelopes": batch})
            batch = [env]
        else:
            batch = candidate
    if batch:
        yield batch, _dumps({"envelopes": batch})


def _count(batches: list) -> int:
    return sum(len(b) for b, _ in batches)
