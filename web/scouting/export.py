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
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from web.scouting import provenance
from web.scouting.geometry import frame_info

SCHEMA = "frc-vision-scorer.scouting"
SCHEMA_VERSION = 1

# The whole-event file (GET /api/scouting/{event}/export). Documented for
# outside readers in docs/scouting-export/ (README.md + JSON Schema): update
# both when this shape changes. v2 added the top-level header and ``matches``;
# v3 (additive) added camera provenance (``camera_legend``, ``cameras``,
# ``matches[].cameras``, per-point ``cams``), robot ``identity`` and
# ``matches[].unidentified``.
EXPORT_SCHEMA = "frc-vision-scorer.scouting-export"
EXPORT_SCHEMA_VERSION = 3
# One match on its own (GET /api/scouting/{event}/match/{match_key}): the same
# ``matches[]`` row, wrapped in a header.
MATCH_SCHEMA = "frc-vision-scorer.scouting-match"


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


def event_export(event_key: str, year: int, matches: list, skipped: list,
                 envelopes: list, timelines: dict, paths: bool = True) -> dict:
    """The whole-event export: header, the calibrations the matches used, one
    row per match (lineup + video timeline + each robot's whole-match path and
    provenance + unidentified tracks), then every metric envelope.
    ``paths=False`` leaves out ``robots`` / ``unidentified`` (a much smaller
    file) and never reads the draw logs."""
    ordered = sorted(matches, key=match_order)
    legend = camera_legend(ordered) if paths else None
    rows = [match_row(m, timelines.get(m.match_key), paths, legend) for m in ordered]
    out = {"schema": EXPORT_SCHEMA, "schema_version": EXPORT_SCHEMA_VERSION,
           "event_key": event_key, "year": int(year),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if paths:
        out["path_frame"] = path_frame(year)
        out["camera_legend"] = legend
    out["cameras"] = cameras_block(event_key, ordered)
    out.update({"matches": rows, "skipped": list(skipped), "envelopes": envelopes})
    return out


def camera_legend(matches: list) -> list:
    """Bit order of every ``cams`` mask in the file: CAMERA_BITS, then any other
    camera the matches' draw logs name, sorted. Reads (cached) draw logs."""
    scs = (provenance.slot_cameras(m.replay_dir) for m in matches)
    return provenance.event_legend(sc.legend for sc in scs if sc is not None)


def cameras_block(event_key: str, matches: list) -> dict:
    """{camera: {"versions": {version_id: CAMERA_DETAIL}}} for exactly the
    calibration versions ``matches[].cameras`` reference."""
    used: dict = {}
    for m in matches:
        for cam, c in (m.cameras or {}).items():
            if c.get("version_id") is not None:
                used.setdefault(cam, set()).add(str(c["version_id"]))
    order = {n: i for i, n in enumerate(provenance.CAMERA_BITS)}
    return {cam: {"versions": {v: provenance.camera_detail(event_key, cam, v)
                               for v in sorted(used[cam])}}
            for cam in sorted(used, key=lambda c: (order.get(c, len(order)), c))}


_LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}


def match_order(m) -> tuple:
    return (_LEVEL_ORDER.get(m.comp_level, 9), m.set_number, m.match_number)


def match_row(m, timeline: Optional[dict], paths: bool = True,
              legend: Optional[list] = None) -> dict:
    """One ``matches[]`` entry: TBA identity + lineup, video timeline, the
    calibrations used and (``paths``) every robot's whole-match path +
    identity provenance and the tracks we never named. ``legend`` = the
    file's camera bit order (default: this match's own)."""
    row = m.source_ref()
    row["alliances"] = {a: list(m.alliances.get(a) or [None] * 3) for a in ("red", "blue")}
    row["timeline"] = timeline
    row["cameras"] = {k: dict(v) for k, v in (m.cameras or {}).items()}
    row["unidentified_s"] = {k: round(float(v), 1) for k, v in
                             (m.unidentified_s or {"auto": 0.0, "teleop": 0.0}).items()}
    if paths:
        sc = provenance.slot_cameras(m.replay_dir)
        if legend is None:
            legend = provenance.event_legend([sc.legend] if sc else [])
        cams = _Cams(sc, legend) if sc is not None else None
        row["robots_version"] = ROBOTS_VERSION
        row["auto_s"] = _auto_s(m)
        row["robots"] = match_robots(m, cams)
        row["unidentified"] = match_unidentified(m, cams)
    return row


# ----------------------------------------------------------------- robot paths
# Each robot's position over the WHOLE match, for apps that want the raw path
# rather than our aggregates. Documented in docs/scouting-export/README.md
# ("Robot paths"); bump ROBOTS_VERSION on a breaking change to ``robots``.
# v2 (additive): path ``cams`` + ``camera_s``, robot ``identity`` +
# ``identity_s``, and the match's ``unidentified`` tracks.
ROBOTS_VERSION = 2
PATH_GAP_S = 1.0     # consecutive points further apart than this start a new segment
# ...and so does a step faster than any FRC robot drives (~5.5 m/s top speed):
# that is the label hopping to another robot or a bad position, not motion.
PATH_MAX_SPEED_MPS = 7.0
# An unidentified track's candidate team is ruled out when it is labelled on
# another slot in at least this share of the track's frames.
CANDIDATE_ELSEWHERE_SHARE = 0.5


def path_frame(year: int) -> dict:
    """The coordinate frame of ``robots[].path`` (absolute, NOT the metrics'
    alliance perspective)."""
    from web.scouting.geometry import field_dims
    return {"units": "m", "field_m": list(field_dims(year)), "perspective": "absolute",
            "note": ("WPILib field coordinates: origin at the blue alliance wall's right "
                     "corner (as seen from the blue driver station), +x toward the red "
                     "wall, +y to the left; the same for red and blue robots")}


def _auto_s(m) -> float:
    from web.scouting.clock import MatchClock
    from web.scouting.timeline import PERIOD_S
    auto = MatchClock(m.periods).auto_len if m.periods else 0.0
    return float(auto or PERIOD_S.get(int(m.year), {}).get("auto", 15))


class _Cams:
    """(frame, slot) -> camera bitmask on the file's legend, from one replay's
    draw log (provenance.SlotCameras, built on its own legend)."""

    def __init__(self, sc, legend: list):
        self.sc, self.legend = sc, list(legend)

    def __call__(self, frame: int, slot) -> Optional[int]:
        if slot is None:
            return None
        return provenance.remap(self.sc.mask(frame, slot), self.sc.legend, self.legend)

    def seconds(self, masks: list, weights: list) -> dict:
        """Seconds each camera saw the points (a point counts toward every
        camera in its mask; "none" = coasting). Unknown (None) masks skipped."""
        out = {n: 0.0 for n in self.legend}
        out["none"] = 0.0
        for mk, w in zip(masks, weights):
            if mk is None:
                continue
            if mk == 0:
                out["none"] += w
            for i, n in enumerate(self.legend):
                if mk >> i & 1:
                    out[n] += w
        return {k: round(v, 1) for k, v in out.items()}


def _point_weights(ts: list, period_s: float) -> list:
    """Seconds each point stands for: the time to the next point, capped at the
    frame period (auto is processed at twice teleop's rate, so a flat frame
    period would double-count it); the last point gets one frame period."""
    return [min(b - a, period_s) for a, b in zip(ts, ts[1:])] + ([period_s] if ts else [])


def _split_path(pts: list) -> tuple:
    """([first, last] inclusive index ranges, n_jumps) for time-sorted points
    whose first three items are (t, x, y): a new piece starts where points are
    more than PATH_GAP_S apart or a step is faster than PATH_MAX_SPEED_MPS."""
    segments, start, jumps = [], 0, 0
    for i in range(1, len(pts) + 1):
        if i < len(pts):
            dt = pts[i][0] - pts[i - 1][0]
            jump = math.dist(pts[i][1:3], pts[i - 1][1:3]) > PATH_MAX_SPEED_MPS * max(dt, 1e-6)
            if dt <= PATH_GAP_S and not jump:
                continue
            jumps += jump and dt <= PATH_GAP_S
        segments.append([start, i - 1])
        start = i
    return segments, int(jumps)


def _columns(pts: list, cams, period_s: float) -> dict:
    """The columnar t/x/y/frame(/cams) arrays of (t, x, y, frame, slot) points
    + ``camera_s``. ``cams`` None (no draw log) -> both null."""
    out = {"t": [round(p[0], 2) for p in pts], "x": [round(p[1], 2) for p in pts],
           "y": [round(p[2], 2) for p in pts], "frame": [p[3] for p in pts]}
    if cams is None:
        out["cams"], out["camera_s"] = None, None
    else:
        out["cams"] = [cams(p[3], p[4]) for p in pts]
        out["camera_s"] = cams.seconds(out["cams"], _point_weights([p[0] for p in pts], period_s))
    return out


def _robot_points(tm) -> list:
    """(t_match, x, y, frame, slot) sorted by time, one per frame: the
    pipeline's slot-continuous auto path for auto when there is one, labelled
    frames otherwise."""
    use_auto_path = tm.auto_source == "autonomous_paths" and tm.auto_path
    pts = list(tm.auto_path) if use_auto_path else []
    pts += [s for s in tm.samples if not (use_auto_path and s.period == "auto")]
    pts.sort(key=lambda s: (s.t_match, s.frame_num))
    out, seen = [], set()
    for s in pts:
        if s.frame_num in seen:            # two auto pieces overlapping by < 0.4 s
            continue
        seen.add(s.frame_num)
        out.append((float(s.t_match), float(s.x), float(s.y), int(s.frame_num), s.slot))
    return out


def robot_path(tm, match_period_s: float, cams=None) -> dict:
    """One robot's whole-match path as columnar arrays + quality summary.
    ``cams`` = (frame, slot) -> camera bitmask (None: no draw log)."""
    pts = _robot_points(tm) if tm is not None else []
    segments, jumps = _split_path(pts)
    samples = tm.samples if tm is not None else []
    seen = {"auto": 0.0, "teleop": 0.0}
    for s in samples:
        seen[s.period] = seen.get(s.period, 0.0) + s.dt
    if tm is not None and tm.auto_source == "autonomous_paths" and tm.auto_path:
        ts = sorted({round(s.t_match, 3) for s in tm.auto_path})
        seen["auto"] = sum(min(b - a, match_period_s) for a, b in zip(ts, ts[1:])) + (
            match_period_s if ts else 0.0)
    n_conf = sum(1 for s in samples if s.confirmed)
    cols = _columns(pts, cams, match_period_s)
    return {
        "n_points": len(pts),
        "seen_s": {k: round(v, 1) for k, v in seen.items()},
        "first_t": round(pts[0][0], 2) if pts else None,
        "last_t": round(pts[-1][0], 2) if pts else None,
        "auto_source": tm.auto_source if tm is not None else None,
        "auto_label_share": tm.auto_label_share if tm is not None else None,
        "confirmed_share": round(n_conf / len(samples), 3) if samples else None,
        "segments": segments,
        "n_jumps": jumps,
        "camera_s": cols.pop("camera_s"),
        **cols,
    }


# ----------------------------------------------------------------- identity
def _clock_spans(clock) -> list:
    """[(first_frame, end_frame, fps)] of each clock period, matching
    MatchClock.at (true start, else start_frame - lead)."""
    out = []
    for p in clock.periods.values():
        fps = p.frames_per_s
        start = (p.true_start_frame if p.true_start_frame is not None
                 else p.start_frame - clock.lead_s * fps)
        out.append((start, start + p.length_s * fps, fps))
    return out


def _in_clock_s(spans: list, f0: int, f1: int, step: float) -> float:
    """Match-clock seconds covered by frames [f0, f1 + step)."""
    return sum(max(0.0, min(f1 + step, hi) - max(f0, lo)) / fps for lo, hi, fps in spans)


def robot_identity(m, team: int, segs: Optional[list] = None, clock=None) -> tuple:
    """(identity rows, identity_s) of one team: the replay's identity segments
    labelled with it (neighbouring rows of one slot with the same source
    collapsed), and the match-clock seconds those segments cover by source."""
    from web.scouting.clock import MatchClock
    if segs is None:
        segs = provenance.collapse_segments(m.identity_segments or [])
    clock = clock or MatchClock(m.periods)
    spans = _clock_spans(clock)
    step = float(m.frame_period_s or 0.2) * (clock.frames_per_s() or 0.0)

    def t_at(fn):
        at = clock.at(fn)
        return round(at[1], 2) if at else None

    rows, by_source = [], {}
    for s in segs:
        if s["team"] != int(team):
            continue
        rows.append({"slot": s["slot"], "start_frame": s["start_frame"],
                     "end_frame": s["end_frame"], "start_t": t_at(s["start_frame"]),
                     "end_t": t_at(s["end_frame"]), "source": s["source"],
                     "read_camera": provenance.READ_CAMERA.get(s["source"]),
                     "score": None if s["score"] is None else round(s["score"], 4),
                     "confirmed": bool(s["confirmed"])})
        key = s["source"] or "unknown"
        by_source[key] = by_source.get(key, 0.0) + _in_clock_s(
            spans, s["start_frame"], s["end_frame"], step)
    return rows, {k: round(v, 1) for k, v in by_source.items()}


def match_robots(m, cams=None) -> list:
    """Every robot of the match, lineup order (red 1-3, blue 1-3), then any
    labelled team the lineup does not list (lineup unknown). A lineup robot we
    never identified still gets an entry with an empty path."""
    from web.scouting.clock import MatchClock
    out, done = [], set()
    period = float(m.frame_period_s or 0.2)
    segs = provenance.collapse_segments(getattr(m, "identity_segments", None) or [])
    clock = MatchClock(m.periods)

    def entry(team, alliance, position, tm):
        ident, ident_s = robot_identity(m, team, segs, clock)
        return {"team_number": team, "alliance": alliance, "alliance_position": position,
                "path": robot_path(tm, period, cams), "identity": ident,
                "identity_s": ident_s}

    for alliance in ("red", "blue"):
        for i, team in enumerate(m.alliances.get(alliance) or []):
            if team is None or int(team) in done:
                continue
            team = int(team)
            done.add(team)
            out.append(entry(team, alliance, i + 1, m.teams.get(team)))
    for team in sorted(t for t in m.teams if int(t) not in done):
        tm = m.teams[team]
        out.append(entry(int(team), tm.alliance, tm.station, tm))
    return out


# ----------------------------------------------------------------- unidentified
def match_unidentified(m, cams=None) -> list:
    """Tracked robots we never named: each slot's unlabelled samples, split
    into pieces with the robot-path rule (_split_path), with hints about who
    it may be. ``team_before`` / ``team_after`` = the team labelled on the SAME
    slot in the nearest labelled frame before / after the piece (a slot is a
    tracker seat and can be re-seeded onto another robot, so this is a hint).
    ``candidates`` = the alliance's lineup teams NOT labelled on another slot
    in >= CANDIDATE_ELSEWHERE_SHARE of the piece's frames, in station order."""
    import bisect
    from web.scouting.clock import MatchClock
    clock = MatchClock(m.periods)
    period = float(m.frame_period_s or 0.2)
    by_slot: dict = {}                        # slot -> ([frames], [teams]) sorted
    for fn in sorted(m.frame_labels or {}):
        for slot, team in m.frame_labels[fn]:
            fs, ts = by_slot.setdefault(slot, ([], []))
            fs.append(fn)
            ts.append(team)

    def t_of(fn):
        at = clock.at(fn)
        return at[1] if at else None

    def gap(a, b):
        return round(b - a, 2) if a is not None and b is not None else None

    out = []
    for slot in sorted(m.unidentified or {}):
        samples = m.unidentified[slot]
        pts = [(float(s.t_match), float(s.x), float(s.y), int(s.frame_num), slot)
               for s in samples]
        alliance = next((s.alliance for s in samples if s.alliance), None) or (
            "red" if str(slot)[:1].upper() == "R" else "blue")
        lineup = [int(t) for t in (m.alliances.get(alliance) or []) if t is not None]
        fs, ts = by_slot.get(slot, ([], []))
        segments, _ = _split_path(pts)
        for n, (a, b) in enumerate(segments, 1):
            piece = pts[a:b + 1]
            f0, f1 = piece[0][3], piece[-1][3]
            i0 = bisect.bisect_left(fs, f0) - 1
            i1 = bisect.bisect_right(fs, f1)
            before = (ts[i0], t_of(fs[i0])) if i0 >= 0 else (None, None)
            after = (ts[i1], t_of(fs[i1])) if i1 < len(fs) else (None, None)
            elsewhere: dict = {}
            for p in piece:
                for s2, team in (m.frame_labels or {}).get(p[3], ()):
                    if s2 != slot:
                        elsewhere[team] = elsewhere.get(team, 0) + 1
            cands = [t for t in lineup
                     if elsewhere.get(t, 0) < CANDIDATE_ELSEWHERE_SHARE * len(piece)]
            cols = _columns(piece, cams, period)
            weights = _point_weights([p[0] for p in piece], period)
            out.append({
                "track_id": f"{slot}#{n}", "slot": slot, "alliance": alliance,
                "first_t": round(piece[0][0], 2), "last_t": round(piece[-1][0], 2),
                "n_points": len(piece), "duration_s": round(sum(weights), 2),
                "team_before": before[0], "gap_before_s": gap(before[1], piece[0][0]),
                "team_after": after[0], "gap_after_s": gap(piece[-1][0], after[1]),
                "candidates": cands,
                "camera_s": cols.pop("camera_s"),
                **cols,
            })
    return out


def filter_team(env: dict, team) -> dict:
    """The envelope reduced to one team (other keys untouched)."""
    out = dict(env)
    key = str(int(team))
    out["teams"] = {key: env["teams"][key]} if key in env.get("teams", {}) else {}
    return out


_NUM_LIST = re.compile(r"\[\s+([-\d.eE+,\s]+?)\s+\]")
_PAIR_LIST = re.compile(r"\[\s+(\[-?\d+, -?\d+\](?:,\s+\[-?\d+, -?\d+\])*)\s+\]")


def pretty(obj) -> str:
    """Indented JSON with lists of numbers (and of index pairs) on one line
    (the single-match file format: scripts/export_scouting.py --per-match)."""
    txt = json.dumps(obj, indent=2)
    txt = _NUM_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", txt)
    txt = _PAIR_LIST.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", txt)
    return txt + "\n"

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
