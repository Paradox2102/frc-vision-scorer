#!/usr/bin/env python3
"""Per-crop team labels for harvested ROBOT crops (design §4, §10).

``label_jev.robot_decision`` gives a robot tracklet a team when every decided
bumper tracklet ever linked to it agrees. Long robot tracklets switch robots
(ByteTrack hands the box to a neighbour after a crossing), so one decided
bumper seen at second 50 can name a crop taken at second 61 of a different
robot (gold b1/b2: 4 wrong of 35).

This module labels each robot CROP from time-local bumper evidence instead
(``crop_label``). For a robot crop at time t with box B:

* a bumper tracklet is a support WITNESS when it is alive at t (its
  first..last detection, widened by ``alive_tol_s`` = 0.25 s) and its box at t
  (interpolated between its saved crops; known only when a saved crop lies
  within ``pos_tol_s`` = 1 s of t) has its centre inside B's lower
  ``lower_frac`` (B padded by ``pad``), lies mostly inside B (``min_ioa``) and
  is >= ``min_w_frac`` of B's width;
* a SUPPORT is a witness decided as team T (not ``off_alliance``), or
  (``weak_support_p``) an undecided witness whose Jev calls name exactly one
  roster team, at p >= 0.85;
* CONFLICT witnesses are looked for loosely (alive within ``conflict_tol_s`` =
  1 s, positions from crops within ``conflict_pos_tol_s`` = 2 s, no width
  floor, plus any bumper LINKED to this robot tracklet whose position is
  unknown). A conflict is a witness decided as another team or off-alliance;
  an undecided witness with a call naming another team at p >=
  ``soft_conflict_p``; or (``colour_conflict``) a witness whose bumper colour
  is the other alliance's.

The crop gets T when there is >= 1 support, every support names T and there is
no conflict; otherwise null with the reason. Bumper decisions themselves come
unchanged from labels_jev.jsonl. Tuned on gold b1 (2025cur + 2026cur) only.

Output per match next to manifest.jsonl: ``labels_robot.jsonl`` (one row per
robot crop: path, tracklet_id, t_sec, team|null, reason, evidence) and
``labels_robot_summary.json``. Post-processes existing outputs only (no GPU,
no network), so it can be re-run any time.

Usage:
  .venv/bin/python -m scripts.teacher.robot_labels --events 2025cur,2026cur,2026alhu,2025casd
  .venv/bin/python -m scripts.teacher.robot_labels --matches 2025cur_qm4 --root data/teacher/harvest_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.label_jev import (  # noqa: E402
    DEFAULT_ROOT, LABELS, match_dirs, read_jsonl, robot_decision, roster_alliance_of,
)

OUT = "labels_robot.jsonl"
OUT_SUMMARY = "labels_robot_summary.json"
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Params:
    alive_tol_s: float = 0.25      # bumper tracklet alive at t: first - tol <= t <= last + tol
    pos_tol_s: float = 1.0         # bumper box at t known only if a saved crop is this close
    lower_frac: float = 0.6        # bumper centre inside the robot box's lower 60% (harvest link rule)
    pad: float = 0.05              # robot box widened by this x its width/height for the centre test
    min_ioa: float = 0.5           # >= this share of the bumper box inside the (padded) robot box
    use_unpositioned: bool = False  # linked + alive but no saved bumper crop near t -> witness
    soft_conflict_p: Optional[float] = 0.5  # unsure witness whose call names another team at p >= this
    conflict_tol_s: float = 1.0    # other-team witnesses looked for this far around t (>= alive_tol_s)
    conflict_pos_tol_s: float = 2.0  # ... with positions known from saved crops this close
    min_w_frac: float = 0.25       # support bumper width >= this x robot box width
    colour_conflict: bool = True   # a witness whose bumper colour is the other alliance's vetoes
    weak_support_p: Optional[float] = 0.85 # unsure witness whose calls name ONLY team T, at p >= this -> support


@dataclass
class Bumper:
    tid: str
    start: float
    end: float
    crops: List[Tuple[float, Box]]          # (t, box) sorted by t
    decision: Optional[str] = None
    team: Optional[str] = None
    off_alliance: bool = False
    robot: Optional[str] = None             # harvest's best robot link (linked_tracklet_id)
    call_teams: Dict[str, float] = field(default_factory=dict)  # roster team -> max top_p over calls
    colour: Optional[str] = None            # bumper class colour (red|blue|None)


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested)
# --------------------------------------------------------------------------- #

def box_at(crops: Sequence[Tuple[float, Box]], t: float, tol: float) -> Optional[Box]:
    """Bumper box at time ``t`` from its saved crops: linear interpolation
    between the bracketing crops when both are within ``tol`` of t, else the
    nearer crop within ``tol``, else None (position unknown)."""
    if not crops:
        return None
    ts = [c[0] for c in crops]
    i = bisect_left(ts, t)
    prev = crops[i - 1] if i > 0 else None
    nxt = crops[i] if i < len(crops) else None
    if nxt is not None and abs(nxt[0] - t) < 1e-9:
        return tuple(nxt[1])
    ok_p = prev is not None and t - prev[0] <= tol
    ok_n = nxt is not None and nxt[0] - t <= tol
    if ok_p and ok_n:
        a = (t - prev[0]) / (nxt[0] - prev[0])
        return tuple(p + a * (n - p) for p, n in zip(prev[1], nxt[1]))
    if ok_p:
        return tuple(prev[1])
    if ok_n:
        return tuple(nxt[1])
    return None


def ioa(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box ``a``'s area inside box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def bumper_inside(bbox: Sequence[float], rbox: Sequence[float], p: Params) -> bool:
    """Bumper centre in the padded robot box's lower ``lower_frac`` and
    >= ``min_ioa`` of the bumper box inside the padded robot box."""
    w, h = rbox[2] - rbox[0], rbox[3] - rbox[1]
    pb = (rbox[0] - p.pad * w, rbox[1] - p.pad * h, rbox[2] + p.pad * w, rbox[3] + p.pad * h)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    top = pb[3] - p.lower_frac * (pb[3] - pb[1])
    if not (pb[0] <= cx <= pb[2] and top <= cy <= pb[3]):
        return False
    return ioa(bbox, pb) >= p.min_ioa


def witnesses(t: float, rbox: Box, robot_tid: str, bumpers: Sequence[Bumper], p: Params,
              conflict: bool = False) -> List[Tuple[Bumper, str]]:
    """Bumper tracklets present in this robot crop at time t -> [(bumper, how)],
    how = 'pos' (box at t inside the robot box) or 'link' (linked to this robot
    tracklet, position at t unknown). Support witnesses use ``alive_tol_s`` /
    ``pos_tol_s`` / ``min_w_frac`` / ``use_unpositioned``; ``conflict=True`` uses
    the wider ``conflict_tol_s`` / ``conflict_pos_tol_s``, no width floor, and
    always admits linked unpositioned bumpers (a veto may be loose, support may not)."""
    tol = p.conflict_tol_s if conflict else p.alive_tol_s
    ptol = max(p.conflict_pos_tol_s, tol) if conflict else p.pos_tol_s
    rw = rbox[2] - rbox[0]
    out = []
    for b in bumpers:
        if not (b.start - tol <= t <= b.end + tol):
            continue
        bb = box_at(b.crops, t, ptol)
        if bb is not None:
            if bumper_inside(bb, rbox, p) and (conflict or bb[2] - bb[0] >= p.min_w_frac * rw):
                out.append((b, "pos"))
        elif (conflict or p.use_unpositioned) and b.robot == robot_tid:
            out.append((b, "link"))
    return out


def _weak_team(b: Bumper, p: Params) -> Optional[str]:
    """Team of an undecided bumper whose Jev calls name exactly one roster team at p >= weak_support_p."""
    if p.weak_support_p is None or b.decision == "team" or len(b.call_teams) != 1:
        return None
    tm, pr = next(iter(b.call_teams.items()))
    return tm if pr >= p.weak_support_p else None


def crop_label(t: float, rbox: Box, robot_tid: str, bumpers: Sequence[Bumper], p: Params,
               roster_alliance: Optional[Dict[str, str]] = None) -> dict:
    """Team for one robot crop from time-local bumper witnesses.
    Support comes from witnesses within ``alive_tol_s``; conflicts are looked
    for in the wider ``conflict_tol_s`` window.
    -> {team, reason, support{team: [tids]}, conflict[], witnesses[]}."""
    ws = witnesses(t, rbox, robot_tid, bumpers, p)
    support: Dict[str, List[str]] = defaultdict(list)
    for b, how in ws:
        if b.decision == "team" and b.team and not b.off_alliance:
            support[b.team].append(b.tid)
        else:
            wt = _weak_team(b, p)
            if wt:
                support[wt].append(b.tid + ":weak")
    teams = set(support)
    cw = witnesses(t, rbox, robot_tid, bumpers, p, conflict=True)
    conflict: List[str] = []
    if p.colour_conflict and roster_alliance and len(teams) == 1:
        al = roster_alliance.get(next(iter(teams)))
        for b, how in cw:
            if al in ("red", "blue") and b.colour in ("red", "blue") and b.colour != al:
                conflict.append(f"{b.tid}:colour:{b.colour}")
    for b, how in cw:
        if b.decision == "team" and b.team:
            if b.off_alliance:
                conflict.append(f"{b.tid}:off_alliance:{b.team}")
            elif b.team not in teams:
                conflict.append(f"{b.tid}:team:{b.team}")
        elif p.soft_conflict_p is not None:
            for tm, pr in sorted(b.call_teams.items()):
                if pr >= p.soft_conflict_p and tm not in teams:
                    conflict.append(f"{b.tid}:call:{tm}")
                    break
    ev = {"support": dict(support), "conflict": conflict,
          "witnesses": [f"{b.tid}:{how}:{b.decision}" + (f":{b.team}" if b.team else "")
                        for b, how in ws]}
    if not support:
        return {"team": None, "reason": "no_witness" if not ws else "no_decided_witness", **ev}
    if len(teams) > 1:
        return {"team": None, "reason": "witnesses_disagree", **ev}
    if conflict:
        return {"team": None, "reason": "conflict", **ev}
    team = next(iter(teams))
    return {"team": team, "reason": f"{len(support[team])}_witness", **ev}


def build_bumpers(rows: Sequence[dict], labels: Dict[str, dict], roster_alliance: Dict[str, str]
                  ) -> List[Bumper]:
    """Bumper tracklets from manifest bumper rows + their labels_jev rows."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["detector"] == "bumper":
            by[r["tracklet_id"]].append(r)
    out = []
    for tid, rs in by.items():
        rs.sort(key=lambda r: r["t_sec"])
        r0 = rs[0]
        lab = labels.get(tid) or {}
        ct: Dict[str, float] = {}
        for c in lab.get("calls") or []:
            top = c.get("top")
            if top in roster_alliance and not c.get("error"):
                ct[top] = max(ct.get(top, 0.0), float(c.get("top_p") or 0.0))
        out.append(Bumper(
            tid=tid, start=float(r0.get("tracklet_start_t", rs[0]["t_sec"])),
            end=float(r0.get("tracklet_end_t", rs[-1]["t_sec"])),
            crops=[(float(r["t_sec"]), tuple(r["box"])) for r in rs],
            decision=lab.get("decision"), team=lab.get("team"),
            off_alliance=bool(lab.get("off_alliance")), robot=r0.get("linked_tracklet_id"),
            call_teams=ct, colour=r0.get("alliance_colour")))
    out.sort(key=lambda b: b.start)
    return out


class _Index:
    """Bumpers bucketed by whole second of their (tolerance-widened) life."""

    def __init__(self, bumpers: Sequence[Bumper], tol: float):
        self.by_s: Dict[int, List[Bumper]] = defaultdict(list)
        for b in bumpers:
            for s in range(int(b.start - tol) - 1, int(b.end + tol) + 2):
                self.by_s[s].append(b)

    def near(self, t: float) -> List[Bumper]:
        return self.by_s.get(int(t), [])


def label_match_rows(rows: Sequence[dict], jev_rows: Sequence[dict], p: Params = Params()
                     ) -> List[dict]:
    """labels_robot rows (one per robot crop) for one match."""
    roster = next((r.get("roster") for r in rows if r.get("roster")), None)
    ra = roster_alliance_of(roster)
    blabels = {r["tracklet_id"]: r for r in jev_rows if r.get("detector") == "bumper"}
    rlabels = {r["tracklet_id"]: r for r in jev_rows if r.get("detector") == "robot"}
    bumpers = build_bumpers(rows, blabels, ra)
    idx = _Index(bumpers, p.alive_tol_s)
    out = []
    for r in rows:
        if r["detector"] != "robot":
            continue
        d = crop_label(float(r["t_sec"]), tuple(r["box"]), r["tracklet_id"],
                       idx.near(float(r["t_sec"])), p, ra)
        old = (rlabels.get(r["tracklet_id"]) or {})
        if not old:
            old = robot_decision(r.get("linked_bumper_tracklets") or [], blabels)
        out.append({"path": r["path"], "match_key": r.get("match_key"), "tracklet_id": r["tracklet_id"],
                    "frame": r.get("frame"), "t_sec": r["t_sec"], "px_w": r.get("px_w"),
                    "team": d["team"],
                    "off_alliance": bool(d["team"] and r.get("alliance_colour") in ("red", "blue")
                                         and ra.get(d["team"]) not in (None, r.get("alliance_colour"))),
                    "reason": d["reason"], "support": d["support"], "conflict": d["conflict"],
                    "witnesses": d["witnesses"], "tracklet_team_old": old.get("team")})
    return out


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    new = sum(1 for r in rows if r["team"])
    old = sum(1 for r in rows if r["tracklet_team_old"])
    both = [r for r in rows if r["team"] and r["tracklet_team_old"]]
    return {"robot_crops": n, "team": new, "coverage": round(new / n, 4) if n else None,
            "old_team": old, "old_coverage": round(old / n, 4) if n else None,
            "new_vs_old_disagree": sum(r["team"] != r["tracklet_team_old"] for r in both),
            "reasons": dict(Counter(r["reason"] for r in rows if not r["team"]))}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run_match(mdir: Path, p: Params, write: bool = True) -> Optional[Tuple[List[dict], dict]]:
    if not (mdir / LABELS).exists():
        return None
    rows = read_jsonl(mdir / "manifest.jsonl")
    out = label_match_rows(rows, read_jsonl(mdir / LABELS), p)
    summ = {"match_key": mdir.name, "params": asdict(p), **summarize(out)}
    if write:
        tmp = mdir / (OUT + ".tmp")
        with open(tmp, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        with open(mdir / OUT_SUMMARY, "w") as f:
            json.dump(summ, f, indent=1)
        os.replace(tmp, mdir / OUT)
    return out, summ


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--events", help="comma list (default: every harvested event)")
    ap.add_argument("--matches", help="comma list of match keys")
    ap.add_argument("--dry-run", action="store_true", help="print stats, write nothing")
    d = Params()
    for f, v in asdict(d).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=lambda s: None if s.lower() == "none" else float(s), default=v)
    a = ap.parse_args(argv)
    p = Params(**{f: getattr(a, f) for f in asdict(d)})
    evs = a.events.split(",") if a.events else None
    ms = a.matches.split(",") if a.matches else None
    tot: Dict[str, Counter] = defaultdict(Counter)
    for mdir in match_dirs(a.root, evs, ms):
        res = run_match(mdir, p, write=not a.dry_run)
        if res is None:
            print(f"[robot_labels] {mdir.name}: no {LABELS}, skipped")
            continue
        _, s = res
        c = tot[mdir.parent.name]
        for k in ("robot_crops", "team", "old_team", "new_vs_old_disagree"):
            c[k] += s[k]
        print(f"[robot_labels] {mdir.name}: {s['team']}/{s['robot_crops']} crops labelled "
              f"(old rule {s['old_team']}), {s['new_vs_old_disagree']} disagree with old")
    for ev, c in sorted(tot.items()):
        n = c["robot_crops"] or 1
        print(f"[robot_labels] {ev}: coverage {c['team'] / n:.3f} (old {c['old_team'] / n:.3f}) "
              f"of {c['robot_crops']} robot crops; {c['new_vs_old_disagree']} disagree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
