"""Offline check of whole-match per-slot identity labels for a finished job.

Reconstructs the in-memory ``_identity_log`` from the job's
``match_robot_binding`` rows over the replay's processed-frame grid, loads
the job's ``vlm_read`` diagnostic rows, and runs the SAME resolve the
``global_identity_resolve`` finalize step uses
(``web.field.identity_resolve.resolve_slot_labels``) against slot truth.

Usage:
  .venv/bin/python scripts/identity_offline_check.py 80a1cd4c \\
      --truth B1=4786,B2=3181,B3=2992,R1=6800,R2=1477,R3=4256 \\
      [--data-root /home/nick/AI/frc-vision-scorer] [--identity-log dump.json] [--json]

Reconstruction rule. A binding row's ``start_frame`` is the loop's
``_current_frame`` at bind time, which the main loop assigns at the END of
an iteration — i.e. the PREVIOUS processed frame. So an identity holds its
slot over processed frames ``f`` with ``start_frame < f <= next binding's
start_frame on that slot`` (or ``<= end_frame`` when a slot_expired detach
closed it). On 80a1cd4c this joins every team-naming read exactly (the only
unjoined rows are 2 early-OCR reads at frame 420 with no team). The
reconstruction also covers frames where the identity was occluded (the live
log only lists active identities); that doesn't change read attribution —
reads only come from active identities — only ``frames_seen``, which the
solver does not use. ``--identity-log`` takes an exact dump instead
(scripts/rerun_reresolve.py format).

Baseline ("today"): the job's ``match_robot_identity`` labels, i.e. what the
run itself wrote (last-identity labels when the knob was OFF). They are also
the ``slot_meta`` the resolve merges into, as in the live path.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.field.identity_resolve import (  # noqa: E402
    resolve_slot_labels,
    vlm_read_rows_from_jsonl,
)

LABELS = ["B1", "B2", "B3", "R1", "R2", "R3"]
LABEL_TO_SLOT = {lab: i for i, lab in enumerate(LABELS)}


def slot_alliance(slot_id: int) -> str:
    return "blue" if slot_id < 3 else "red"


def reconstruct_identity_log(bindings, frame_nums) -> list[dict]:
    """``bindings``: iterable of (identity_id, slot, start_frame, end_frame)
    in insertion order; ``frame_nums``: the processed source-frame grid.
    Returns identity_log rows ``{frame_num, identity_id, slot_id, alliance}``
    with at most one slot per (frame, identity) — the latest binding wins
    when an identity moved slots without a new holder on the old one."""
    per_slot: dict[int, list[tuple[int, int, int, int | None]]] = defaultdict(list)
    for order, (iid, slot, start, end) in enumerate(bindings):
        per_slot[int(slot)].append((int(start), order, int(iid), end))
    held: dict[tuple[int, int], tuple[int, int, int]] = {}   # (f, iid) -> (start, order, slot)
    for slot, rows in per_slot.items():
        rows.sort()
        starts = [r[0] for r in rows]
        for f in frame_nums:
            k = bisect.bisect_left(starts, f) - 1     # last binding with start < f
            if k < 0:
                continue
            start, order, iid, end = rows[k]
            if end is not None and f > end:
                continue
            key = (f, iid)
            if key not in held or (start, order) > held[key][:2]:
                held[key] = (start, order, slot)
    return [{"frame_num": f, "identity_id": iid, "slot_id": slot,
             "alliance": slot_alliance(slot)}
            for (f, iid), (_s, _o, slot) in sorted(held.items())]


def run_check(bindings, frame_nums, reads, truth: dict, slot_meta=None,
              votes_by_slot=None, blue=None, red=None, identity_log=None) -> dict:
    """Pure core: resolve per-slot labels and score them against ``truth``
    (slot label -> team). Rosters default to the truth's teams."""
    ilog = identity_log if identity_log is not None else reconstruct_identity_log(
        bindings, frame_nums)
    blue = blue or [truth[lab] for lab in LABELS[:3] if lab in truth]
    red = red or [truth[lab] for lab in LABELS[3:] if lab in truth]
    meta = slot_meta if slot_meta is not None else {
        sid: {"team_number": None, "confirmed": False, "alliance": slot_alliance(sid)}
        for sid in range(6)}
    res = resolve_slot_labels(reads or None, ilog, votes_by_slot or {}, {},
                              slot_alliance, blue, red, meta)
    slots, correct, wrong, blank, today_ok = {}, 0, 0, 0, 0
    for sid, lab in enumerate(LABELS):
        got = res["slot_meta"].get(sid, {}).get("team_number")
        want = truth.get(lab)
        today = (meta.get(sid) or {}).get("team_number")
        status = "BLANK" if got is None else ("OK" if got == want else "WRONG")
        correct += status == "OK"
        wrong += status == "WRONG"
        blank += status == "BLANK"
        today_ok += today is not None and today == want
        slots[lab] = {
            "truth": want, "resolved": got, "today": today, "status": status,
            "confirmed": bool(res["slot_meta"].get(sid, {}).get("confirmed")),
            "scores": {t: round(v, 3) for t, v in sorted(
                res["scores"].get(sid, {}).items(), key=lambda kv: -kv[1])},
        }
    per_slot_ids = defaultdict(set)
    for row in ilog:
        per_slot_ids[row["slot_id"]].add(row["identity_id"])
    return {
        "slots": slots, "correct": correct, "wrong": wrong, "blank": blank,
        "today_correct": today_ok,
        "stats": res["stats"],
        "changes": res["changes"],
        "identities": len({r["identity_id"] for r in ilog}),
        "identities_per_slot": {LABELS[s]: len(v) for s, v in sorted(per_slot_ids.items())},
        "team_reads": sum(1 for r in (reads or []) if r.get("matched_team")),
    }


def format_report(res: dict) -> str:
    st = res["stats"]
    lines = [
        f"identities={res['identities']} per slot {res['identities_per_slot']}; "
        f"team-naming reads={res['team_reads']} used={st['reads_used']} "
        f"dropped_misattr={st['reads_dropped_misattributed']} "
        f"fallback={st['fallback_used']} contested={[LABELS[s] for s in st['contested']]}",
        f"{'slot':<5}{'truth':<7}{'resolved':<10}{'today':<8}{'status':<7}scores",
    ]
    for lab, s in res["slots"].items():
        conf = "*" if s["confirmed"] else ""
        lines.append(f"{lab:<5}{str(s['truth']):<7}{str(s['resolved']) + conf:<10}"
                     f"{str(s['today']):<8}{s['status']:<7}{s['scores']}")
    lines.append(f"per-slot resolve: {res['correct']}/6 correct, {res['wrong']} wrong, "
                 f"{res['blank']} blank (today {res['today_correct']}/6)  "
                 f"[* = confirmed, margin >= 2.0]")
    return "\n".join(lines)


def _resolve_job_id(con, prefix: str) -> str:
    ids = [r[0] for r in con.execute(
        "SELECT DISTINCT job_id FROM match_robot_binding WHERE job_id LIKE ?",
        (prefix + "%",))]
    if len(ids) != 1:
        raise SystemExit(f"job prefix {prefix!r} matches {len(ids)} jobs: {ids[:5]}")
    return ids[0]


def load_job(data_root: Path, prefix: str) -> dict:
    con = sqlite3.connect(f"file:{data_root / 'frc_scorer.db'}?mode=ro", uri=True)
    job = _resolve_job_id(con, prefix)
    bindings = con.execute(
        "SELECT identity_id, slot, start_frame, end_frame FROM match_robot_binding "
        "WHERE job_id=? ORDER BY id", (job,)).fetchall()
    ident_rows = con.execute(
        "SELECT slot, identity_id, team_number, confirmed FROM match_robot_identity "
        "WHERE job_id=?", (job,)).fetchall()
    con.close()
    meta = json.loads((data_root / "web" / "replays" / job / "meta.json").read_text())
    frame_nums = [f["frame_num"] for f in meta["frames"]]
    diag = data_root / "web" / "uploads" / f"diagnostic_{job}.jsonl"
    lines = diag.read_text().splitlines() if diag.exists() else []
    reads = vlm_read_rows_from_jsonl(lines)
    # Canonical identities' live votes (last snapshot) = the live fallback source.
    last_snap = None
    for line in lines:
        if '"snapshot"' in line:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "snapshot":
                last_snap = r
    votes_by_id = {r["robot_id"]: r.get("votes") or {}
                   for r in (last_snap or {}).get("robots", [])}
    slot_meta, votes_by_slot = {}, {}
    for slot, iid, team, confirmed in ident_rows:
        if slot is None:
            continue
        slot_meta[int(slot)] = {"team_number": team, "confirmed": bool(confirmed),
                                "alliance": slot_alliance(int(slot))}
        votes_by_slot[int(slot)] = {str(k): float(v)
                                    for k, v in votes_by_id.get(iid, {}).items()}
    return {"job": job, "bindings": bindings, "frame_nums": frame_nums,
            "reads": reads, "slot_meta": slot_meta, "votes_by_slot": votes_by_slot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job", help="job id or unique prefix")
    ap.add_argument("--truth", required=True,
                    help="slot truth, e.g. B1=4786,B2=3181,B3=2992,R1=6800,R2=1477,R3=4256")
    ap.add_argument("--data-root", default=str(ROOT),
                    help="dir holding frc_scorer.db and web/{uploads,replays} "
                         "(default: this checkout)")
    ap.add_argument("--identity-log", help="exact identity_log JSON dump to use "
                                           "instead of the binding reconstruction")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)

    truth = {}
    for part in args.truth.split(","):
        lab, team = part.split("=")
        if lab.strip().upper() not in LABEL_TO_SLOT:
            raise SystemExit(f"bad slot label {lab!r}")
        truth[lab.strip().upper()] = team.strip()

    data = load_job(Path(args.data_root), args.job)
    ilog = json.loads(Path(args.identity_log).read_text()) if args.identity_log else None
    res = run_check(data["bindings"], data["frame_nums"], data["reads"], truth,
                    slot_meta=data["slot_meta"], votes_by_slot=data["votes_by_slot"],
                    identity_log=ilog)
    if args.json:
        print(json.dumps({"job": data["job"], **res}, default=str, indent=2))
    else:
        print(f"== {data['job']}")
        print(format_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
