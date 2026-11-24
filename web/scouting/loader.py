"""Load an event's finished replays into ``MatchData`` for the scouting metrics.

One replay per match: several exist when a match was A/B'd. The newest
``complete`` replay whose summary carries ``window_labels`` (the current
identity method) wins; with none, the newest complete one is used and flagged
``legacy_labels``.

Identity: a robot sample belongs to the frame's ``team_number``. Unlabelled
samples are counted in ``tracked_s`` but not attributed to a team; the export
lists them per slot (``MatchData.unidentified``). A label naming a team
that is not in the TBA lineup is dropped (``off_roster``). Two slots carrying
the same team in one frame keep one sample (the confirmed one, else the first).

Synchronous sqlite3: called from a thread by the API.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

from web.field.season import season_for_event
from web.scouting.clock import MatchClock
from web.scouting.geometry import to_alliance
from web.scouting.model import MatchData, Sample, TeamMatch
from web.scouting.provenance import identity_segments, match_cameras

REPO = Path(__file__).resolve().parents[2]
REPLAY_DIR = REPO / "web" / "replays"
_KEY_RE = re.compile(r"^(?P<event>[^_]+)_(?P<level>qm|ef|qf|sf|f)(?:(?P<set>\d+)m)?(?P<num>\d+)$")


def event_season_year(event_key: str) -> int:
    """The season year an event is played on (honours the per-event field-season
    override, POST /api/event/field-season), not just the key's prefix."""
    return int(season_for_event(event_key).year)


def default_db_path() -> Path:
    from web import database
    return Path(database.DB_PATH)


def parse_match_key(match_key: str) -> tuple:
    """'2026cur_qm8' -> ('qm', 1, 8); '2026cur_sf3m1' -> ('sf', 3, 1)."""
    m = _KEY_RE.match(match_key or "")
    if not m:
        return "unknown", 1, 0
    level = m.group("level")
    if m.group("set"):
        return level, int(m.group("set")), int(m.group("num"))
    return level, 1, int(m.group("num"))


def _meta_path(row: dict) -> Optional[Path]:
    for cand in (row.get("meta_path"), REPLAY_DIR / str(row["id"]) / "meta.json"):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _read_meta(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _columns(conn, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def candidate_replays(conn, event_key: str) -> dict:
    """{match_key: [replay rows newest first]} for complete replays."""
    conn.row_factory = sqlite3.Row
    extra = ", source_video_path" if "source_video_path" in _columns(conn, "replays") else ""
    rows = conn.execute(
        f"SELECT id, match_key, created_at, meta_path{extra} FROM replays "
        "WHERE event_key = ? AND status = 'complete' AND match_key IS NOT NULL "
        "AND match_key != '' ORDER BY created_at DESC", (event_key,)).fetchall()
    out: dict = defaultdict(list)
    for r in rows:
        out[r["match_key"]].append(dict(r))
    return out


def _is_full_match(meta: dict) -> bool:
    """A replay that processed teleop (not an auto-only ground-truth / A/B run)."""
    summary = meta.get("summary") or {}
    if (summary.get("config") or {}).get("auto_period_only"):
        return False
    return any(p.get("label") == "teleop" for p in summary.get("periods") or [])


def pick_replay(rows: list) -> tuple:
    """(row, meta, legacy) for the best replay of one match, or (None, None, None).

    Rank, newest first within a rank: full match with window labels > full
    match (legacy labels) > auto-only with window labels > auto-only. Stops
    reading meta.json at the first top-rank hit (rows are newest first)."""
    best, best_rank = None, -1
    for row in rows:
        path = _meta_path(row)
        meta = _read_meta(path) if path else None
        if not meta:
            continue
        wl = bool((meta.get("summary") or {}).get("window_labels"))
        rank = 2 * _is_full_match(meta) + wl
        if rank > best_rank:
            best, best_rank = (row, meta, not wl), rank
            if rank == 3:
                break
    return best or (None, None, None)


def match_lineup(conn, match_key: str) -> dict:
    """TBA lineup + hub auto totals from the matches tables (empty if unknown)."""
    conn.row_factory = sqlite3.Row
    extra = ", tba_video_key" if "tba_video_key" in _columns(conn, "matches") else ""
    m = conn.execute("SELECT id, comp_level, match_number, set_number, tba_red_hub_auto, "
                     f"tba_blue_hub_auto{extra} FROM matches WHERE tba_match_key = ?",
                     (match_key,)).fetchone()
    if not m:
        return {}
    alliances = {"red": [None, None, None], "blue": [None, None, None]}
    for r in conn.execute("SELECT alliance, position, team_number FROM match_robots "
                          "WHERE match_id = ?", (m["id"],)):
        try:
            alliances[r["alliance"]][int(r["position"]) - 1] = int(r["team_number"])
        except (KeyError, ValueError, TypeError, IndexError):
            pass
    return {"comp_level": m["comp_level"], "match_number": m["match_number"],
            "set_number": m["set_number"] or 1, "alliances": alliances,
            "hub_auto": {"red": m["tba_red_hub_auto"], "blue": m["tba_blue_hub_auto"]},
            "video_key": m["tba_video_key"] if extra else None}


def _team_int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _slot_alliance(slot) -> str:
    """'R2' -> 'red', 'B1' -> 'blue' (the tracker's slot naming)."""
    return "red" if str(slot)[:1].upper() == "R" else "blue"


def build_match(event_key: str, match_key: str, replay: dict, meta: dict, legacy: bool,
                lineup: dict, year: int) -> tuple:
    """(MatchData, None) or (None, skip_reason)."""
    summary = meta.get("summary") or {}
    clock = MatchClock(summary.get("periods") or [])
    if not clock.ok:
        return None, "no match clock (summary.periods)"
    frames = meta.get("frames") or []
    if len(frames) < 2:
        return None, "no frames"
    fps = clock.frames_per_s()
    steps = [b["frame_num"] - a["frame_num"] for a, b in zip(frames, frames[1:])
             if b["frame_num"] > a["frame_num"]]
    dt = (median(steps) / fps) if steps and fps > 0 else 0.2

    level, set_no, num = parse_match_key(match_key)
    alliances = lineup.get("alliances") or {"red": [None] * 3, "blue": [None] * 3}
    roster = {t: a for a in ("red", "blue") for t in alliances.get(a, []) if t is not None}
    md = MatchData(
        event_key=event_key, match_key=match_key,
        comp_level=lineup.get("comp_level") or level,
        match_number=int(lineup.get("match_number") or num),
        set_number=int(lineup.get("set_number") or set_no), year=year,
        replay_id=str(replay["id"]), replay_created_at=replay.get("created_at"),
        legacy_labels=bool(legacy), alliances=alliances,
        hub_auto=lineup.get("hub_auto") or {"red": None, "blue": None},
        frame_period_s=float(dt),
        labelled_s={"auto": 0.0, "teleop": 0.0}, tracked_s={"auto": 0.0, "teleop": 0.0},
        periods=list(summary.get("periods") or []),
        video_path=replay.get("source_video_path"), youtube_key=lineup.get("video_key"),
        unidentified_s={"auto": 0.0, "teleop": 0.0}, cameras=match_cameras(summary))

    def team_match(team: int, alliance: str) -> TeamMatch:
        tm = md.teams.get(team)
        if tm is None:
            _, station = md.station_of(team)
            tm = md.teams[team] = TeamMatch(team=team, alliance=alliance, station=station)
        return tm

    def sample(fn, at, xy, alliance, confirmed=False, slot=None) -> Sample:
        period, t, remaining = at
        ax, ay = to_alliance(xy[0], xy[1], alliance, year)
        return Sample(frame_num=int(fn), t_match=t, period=period, remaining=remaining,
                      x=float(xy[0]), y=float(xy[1]), ax=ax, ay=ay, dt=dt,
                      confirmed=bool(confirmed), slot=slot, alliance=alliance)

    slot_auto_xy: dict = defaultdict(list)    # slot -> [(frame_num, xy)] in auto
    slot_auto_labels: dict = defaultdict(dict)  # slot -> {frame_num: team} in auto
    off_roster = dups = 0
    for fr in frames:
        at = clock.at(int(fr.get("frame_num", -1)))
        if at is None:
            continue
        period = at[0]
        seen: dict = {}
        labels: list = []
        for rb in fr.get("robots") or []:
            xy = rb.get("field_pos")
            if not xy or len(xy) < 2:
                continue
            md.tracked_s[period] += dt
            slot = rb.get("slot")
            if period == "auto" and slot:
                slot_auto_xy[slot].append((int(fr["frame_num"]), xy))
            team = _team_int(rb.get("team_number"))
            if team is None:
                if slot:
                    alliance = rb.get("alliance") or _slot_alliance(slot)
                    md.unidentified.setdefault(slot, []).append(
                        sample(fr["frame_num"], at, xy, alliance, False, slot))
                    md.unidentified_s[period] += dt
                continue
            if roster and team not in roster:
                off_roster += 1
                continue
            if slot:
                labels.append((slot, team))
            alliance = roster.get(team) or rb.get("alliance") or "blue"
            if period == "auto" and slot:
                slot_auto_labels[slot][int(fr["frame_num"])] = team
            prev = seen.get(team)
            if prev is not None:
                dups += 1
                if prev.get("confirmed") or not rb.get("confirmed"):
                    continue
            seen[team] = rb
        for team, rb in seen.items():
            alliance = roster.get(team) or rb.get("alliance") or "blue"
            team_match(team, alliance).samples.append(
                sample(fr["frame_num"], at, rb["field_pos"], alliance,
                       rb.get("confirmed"), rb.get("slot")))
            md.labelled_s[period] += dt
        if labels:
            md.frame_labels[int(fr["frame_num"])] = tuple(labels)

    _attach_auto_paths(md, summary.get("autonomous_paths") or {}, slot_auto_labels,
                       slot_auto_xy, clock, roster, team_match, sample)
    # An auto slot span is attributed whole to its team (_slot_spans hands the
    # unlabelled frames between label runs to the nearer run), so those frames
    # are already in that team's auto path: not unidentified.
    claimed = {(s.frame_num, s.slot) for tm in md.teams.values()
               if tm.auto_source == "autonomous_paths" for s in tm.auto_path}
    for slot in list(md.unidentified):
        pts = [s for s in md.unidentified[slot] if (s.frame_num, slot) not in claimed]
        for s in md.unidentified[slot]:
            if (s.frame_num, slot) in claimed:
                md.unidentified_s[s.period] -= s.dt
        if pts:
            md.unidentified[slot] = sorted(pts, key=lambda s: (s.t_match, s.frame_num))
        else:
            del md.unidentified[slot]
    md.off_roster_samples, md.duplicate_label_samples = off_roster, dups
    return md, None


AUTO_GAP_S = 1.0       # a slot unseen this long was re-seeded, maybe on another robot
MIN_LABEL_RUN = 3       # shorter label runs inside one track are treated as misreads
OVERLAP_TOL_S = 0.4     # two pieces of one team may overlap this much in time


def _slot_spans(presence: list) -> list:
    """Split one slot's auto track into single-robot time spans.

    A slot is a tracker *seat*, not a robot: after a track is lost the slot can
    be re-seeded on a different robot (2026casnd qm14: R3 = 3128 for 2 s, then
    2839 for the rest of auto). ``presence`` = time-sorted [(t_match, team or
    None)], one entry per frame the slot was on the field. The track is cut
    where the slot vanished for > AUTO_GAP_S and again between label runs of
    different teams (unlabelled frames go to the nearer run; runs shorter than
    MIN_LABEL_RUN are misreads). Returns [(team, t_lo, t_hi, Counter of labels
    in the span)]; spans with no label are dropped (nobody to attribute them to).
    """
    segs: list = []
    for p in presence:
        if segs and p[0] - segs[-1][-1][0] <= AUTO_GAP_S:
            segs[-1].append(p)
        else:
            segs.append([p])
    spans = []
    for seg in segs:
        runs: list = []                      # [team, first_i, last_i, n]
        for i, (_, team) in enumerate(seg):
            if team is None:
                continue
            if runs and runs[-1][0] == team:
                runs[-1][2], runs[-1][3] = i, runs[-1][3] + 1
            else:
                runs.append([team, i, i, 1])
        strong: list = []
        for r in ([r for r in runs if r[3] >= MIN_LABEL_RUN] or runs):
            if strong and strong[-1][0] == r[0]:
                strong[-1][2], strong[-1][3] = r[2], strong[-1][3] + r[3]
            else:
                strong.append(list(r))
        cuts = [0] + [(a[2] + 1 + b[1]) // 2 for a, b in zip(strong, strong[1:])] + [len(seg)]
        for r, lo, hi in zip(strong, cuts, cuts[1:]):
            votes = Counter(team for _, team in seg[lo:hi] if team is not None)
            spans.append((r[0], seg[lo][0], seg[hi - 1][0], votes))
    return spans


def _attach_auto_paths(md, paths, slot_labels, slot_auto_xy, clock, roster, team_match,
                       sample) -> None:
    # The pipeline's per-slot path starts at the clock's auto start, which can
    # trail the real start (clock.py); the slot's earlier frames (the
    # processing lead) are prepended. Each slot track is split into
    # single-robot time spans (_slot_spans, from the slot's per-frame presence
    # and labels, not from the stored path, which skips coasting samples) and
    # every span's path points go to that span's team. So a robot whose track
    # broke and re-seeded on another slot keeps both halves, and a slot re-used
    # for another robot does not leak into it.
    by_team: dict = defaultdict(list)       # team -> [(votes, slot, points, Counter)]
    for slot, path in paths.items():
        first = min((int(p.get("frame_num", 1 << 30)) for p in path), default=1 << 30)
        pts = [(fn, xy, clock.at(fn)) for fn, xy in slot_auto_xy.get(slot, []) if fn < first]
        pts += [(int(p["frame_num"]), p.get("field_pos"), clock.at(int(p.get("frame_num", -1))))
                for p in path if p.get("field_pos") and "frame_num" in p]
        pts = sorted((p for p in pts if p[1] and p[2] and p[2][0] == "auto"),
                     key=lambda p: p[0])
        labels = slot_labels.get(slot, {})
        presence = sorted((at[1], labels.get(fn)) for fn, at in
                          ((fn, clock.at(fn)) for fn, _ in slot_auto_xy.get(slot, []))
                          if at and at[0] == "auto")
        for team, lo, hi, votes in _slot_spans(presence):
            piece = [p for p in pts if lo <= p[2][1] <= hi]
            if piece:
                by_team[team].append((votes[team], slot, piece, votes))
    for team, cands in by_team.items():
        kept: list = []
        for c in sorted(cands, key=lambda c: (-c[0], -len(c[2]))):
            t0, t1 = c[2][0][2][1], c[2][-1][2][1]
            if all(t1 <= k[2][0][2][1] + OVERLAP_TOL_S or t0 >= k[2][-1][2][1] - OVERLAP_TOL_S
                   for k in kept):
                kept.append(c)        # a simultaneous duplicate claim keeps the stronger
        kept.sort(key=lambda c: c[2][0][2][1])
        slot0 = kept[0][1]
        alliance = roster.get(team) or ("red" if str(slot0).startswith("R") else "blue")
        tm = team_match(team, alliance)
        tm.auto_path = [sample(fn, at, xy, alliance, slot=slot)
                        for _, slot, piece, _ in kept for fn, xy, at in piece]
        tm.auto_slot = "+".join(dict.fromkeys(c[1] for c in kept))
        votes = sum((c[3] for c in kept), Counter())
        total = sum(votes.values()) or 1
        runner = max((n for t, n in votes.items() if t != team), default=0)
        tm.auto_label_share = round(votes[team] / total, 3)
        tm.auto_runner_up_share = round(runner / total, 3)
        tm.auto_source = "autonomous_paths"
    for tm in md.teams.values():
        if not tm.auto_path:
            auto = [s for s in tm.samples if s.period == "auto"]
            if auto:
                tm.auto_path, tm.auto_source = auto, "frames"


def load_event(event_key: str, db_path=None) -> tuple:
    """(matches: list[MatchData], skipped: list[{match_key, reason}])."""
    year = event_season_year(event_key)
    conn = sqlite3.connect(str(db_path or default_db_path()))
    try:
        matches, skipped = [], []
        for match_key, rows in sorted(candidate_replays(conn, event_key).items()):
            replay, meta, legacy = pick_replay(rows)
            if replay is None:
                skipped.append({"match_key": match_key, "reason": "no readable meta.json"})
                continue
            md, why = build_match(event_key, match_key, replay, meta, legacy,
                                  match_lineup(conn, match_key), year)
            if md is None:
                skipped.append({"match_key": match_key, "replay_id": replay["id"], "reason": why})
            else:
                # Provenance for the export: the replay dir (draw log, read
                # lazily there) and the identity segments (one indexed query).
                mp = _meta_path(replay)
                md.replay_dir = str(mp.parent) if mp else None
                md.identity_segments = identity_segments(conn, md.replay_id)
                matches.append(md)
        return matches, skipped
    finally:
        conn.close()


def source_fingerprint(event_key: str, db_path=None) -> tuple:
    """Cheap cache key: the set of complete replays (+ their meta mtimes)."""
    conn = sqlite3.connect(str(db_path or default_db_path()))
    try:
        out = []
        for match_key, rows in sorted(candidate_replays(conn, event_key).items()):
            for r in rows:
                p = _meta_path(r)
                out.append((r["id"], p.stat().st_mtime if p else 0.0))
        return tuple(out)
    finally:
        conn.close()
