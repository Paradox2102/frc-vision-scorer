#!/usr/bin/env python3
"""Replay-level identity metrics for an A/B between finished replays (no GT needed).

    .venv/bin/python scripts/ab_replay_team_metrics.py \
        --arm before=525ed57e,2dcf7631 --arm after=<id>,<id> [--focus 4322,6800,498] [--md out.md]

Replay ids may be prefixes. For each replay (web/replays/<id>/meta.json + the
replays table roster) it reports, per roster team:
  cov      fraction of meta frames in which some slot carries the team's label
  wrong_al fraction of those labelled frames where that slot's alliance is not
           the team's real alliance (roster)
  share    final slot_labels label_share of the slot named for the team
           (0 if no slot ends up named for it)
and per replay: unlabelled slot-frame fraction, mean tracked slots per
alliance per frame (/3), off-roster label fraction, frames where one team sits
on two slots, and the window-label reader's bumper_found / crop counts (the
bumper detector's direct input to the number reader).
Arms are pooled by averaging per-replay numbers for each (match, team).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/nick/AI/frc-vision-scorer")


_ROSTERS: dict = {}


def tba_roster(match_key):
    """True alliances from TBA (the replays table's blue/red_teams are the
    pipeline's OWN identified teams, with nulls — not usable as truth)."""
    if match_key not in _ROSTERS:
        import asyncio
        import sys
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "web"))
        import tba
        m = asyncio.run(tba.get_match(match_key))
        _ROSTERS[match_key] = {t[3:]: al for al in ("red", "blue")
                               for t in m["alliances"][al]["team_keys"]}
    return _ROSTERS[match_key]


def resolve(db, prefix):
    rows = db.execute("SELECT id, match_key FROM replays WHERE id LIKE ?",
                      (prefix + "%",)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"replay prefix {prefix!r}: {len(rows)} matches")
    rid, mk = rows[0]
    return rid, mk, tba_roster(mk)


def replay_metrics(rid, roster):
    meta = json.loads((ROOT / "web/replays" / rid / "meta.json").read_text())
    frames = meta["frames"]
    n = len(frames)
    lab = defaultdict(int)
    wrong = defaultdict(int)
    slot_frames = unl = off = dup = 0
    per_al = {"blue": 0, "red": 0}
    for f in frames:
        seen = defaultdict(int)
        for r in f["robots"]:
            slot_frames += 1
            al = r.get("alliance")
            if al in per_al:
                per_al[al] += 1
            t = r.get("team_number")
            if not t:
                unl += 1
                continue
            t = str(t)
            seen[t] += 1
            if t not in roster:
                off += 1
                continue
            lab[t] += 1
            if al != roster[t]:
                wrong[t] += 1
        if any(v > 1 for v in seen.values()):
            dup += 1
    shares = {}
    for s in (meta["summary"].get("slot_labels") or []):
        t = s.get("team_number")
        if t:
            shares[str(t)] = max(shares.get(str(t), 0.0), float(s.get("label_share") or 0.0))
    wl = meta["summary"].get("window_labels") or {}
    st = wl.get("student") or {}
    teams = {t: {"cov": lab[t] / n if n else 0.0,
                 "wrong_al": wrong[t] / lab[t] if lab[t] else None,
                 "share": shares.get(t, 0.0)} for t in roster}
    return {
        "frames": n,
        "teams": teams,
        "unlabelled": unl / slot_frames if slot_frames else 0.0,
        "blue_slots": per_al["blue"] / n / 3 if n else 0.0,
        "red_slots": per_al["red"] / n / 3 if n else 0.0,
        "off_roster": off / slot_frames if slot_frames else 0.0,
        "dup_frames": dup / n if n else 0.0,
        "bumper_found": st.get("bumper_found"),
        "reader_samples": st.get("samples"),
        "digit_named": wl.get("digit_named"),
        "wl_unlabelled": wl.get("unlabelled"),
        "windows": wl.get("windows"),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f(x, pct=True, nd=1):
    if x is None:
        return "–"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=id1,id2,...")
    ap.add_argument("--focus", default="4322,6800,498")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    focus = a.focus.split(",")
    db = sqlite3.connect(ROOT / "frc_scorer.db")

    data = {}   # arm -> match -> [metrics]
    rosters = {}
    for spec in a.arm:
        name, ids = spec.split("=", 1)
        for p in ids.split(","):
            rid, mk, roster = resolve(db, p.strip())
            rosters[mk] = roster
            m = replay_metrics(rid, roster)
            m["id"] = rid[:8]
            data.setdefault(name, {}).setdefault(mk, []).append(m)
    arms = list(data)
    matches = sorted({mk for d in data.values() for mk in d})
    md = []

    md.append("### Per replay\n")
    md.append("| arm | match | replay | unlabelled slot-frames | blue slots | red slots | off-roster | dup-team frames | reader bumper_found / samples | windows digit-named / unlabelled |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        for mk in matches:
            for m in data[arm].get(mk, []):
                bf = (f"{m['bumper_found']} / {m['reader_samples']} ({100 * m['bumper_found'] / m['reader_samples']:.0f}%)"
                      if m["bumper_found"] is not None and m["reader_samples"] else "–")
                md.append(f"| {arm} | {mk} | {m['id']} | {f(m['unlabelled'])} | {f(m['blue_slots'])} | {f(m['red_slots'])} | "
                          f"{f(m['off_roster'])} | {f(m['dup_frames'])} | {bf} | {m['digit_named']} / {m['wl_unlabelled']} of {m['windows']} |")

    md.append("\n### Per team (mean over an arm's replays of that match)\n")
    md.append("cov = labelled share of frames; wrong-al = labelled frames on the wrong alliance's slot; share = final slot label share.\n")
    hdr = "| match | team | al | " + " | ".join(f"{arm} cov / wrong-al / share" for arm in arms) + " |"
    md.append(hdr)
    md.append("|---|---|---|" + "---:|" * len(arms))
    pooled = defaultdict(lambda: defaultdict(list))   # arm -> key -> [cov]
    for mk in matches:
        roster = rosters[mk]
        for t in sorted(roster, key=lambda t: (t not in focus, roster[t], t)):
            cells = []
            for arm in arms:
                ms = data[arm].get(mk, [])
                cov = mean([m["teams"][t]["cov"] for m in ms])
                wa = mean([m["teams"][t]["wrong_al"] for m in ms])
                sh = mean([m["teams"][t]["share"] for m in ms])
                cells.append(f"{f(cov, nd=0)} / {f(wa, nd=0)} / {f(sh, pct=False, nd=2)}" if ms else "–")
                if ms:
                    key = "focus" if t in focus else "other"
                    pooled[arm][key].append(cov)
                    pooled[arm][key + "_share"].append(sh)
                    pooled[arm][key + "_wa"].append(wa)
            mark = " **(focus)**" if t in focus else ""
            md.append(f"| {mk} | {t}{mark} | {roster[t]} | " + " | ".join(cells) + " |")

    md.append("\n### Pooled team means\n")
    md.append("| arm | focus teams cov | focus share | focus wrong-al | other teams cov | other share | other wrong-al |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        p = pooled[arm]
        md.append(f"| {arm} | {f(mean(p['focus']))} | {f(mean(p['focus_share']), pct=False, nd=2)} | {f(mean(p['focus_wa']))} | "
                  f"{f(mean(p['other']))} | {f(mean(p['other_share']), pct=False, nd=2)} | {f(mean(p['other_wa']))} |")
    text = "\n".join(md)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
