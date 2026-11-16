"""Offline check of whole-match per-slot identity labels for a finished job.

Reconstructs the in-memory ``_identity_log`` from the job's
``match_robot_binding`` rows over the replay's processed-frame grid, loads
the job's ``vlm_read`` diagnostic rows, and runs the SAME resolve the
``global_identity_resolve`` finalize step uses
(``web.field.identity_resolve.resolve_slot_labels``) against slot truth.

Usage:
  .venv/bin/python scripts/identity_offline_check.py 80a1cd4c \\
      --truth B1=4786,B2=3181,B3=2992,R1=6800,R2=1477,R3=4256 \\
      [--data-root <repo root>] [--identity-log dump.json] [--json]

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

Evicted rows (read_keep_evicted): a read whose identity was evicted while
in flight writes a row flagged ``"evicted": true`` naming the identity that
held the crop's slot at the crop frame; it attributes exactly like any other
row (the reconstruction covers the evicted identity's binding).
``--exclude-evicted`` drops those rows first, so one knob-ON job compares
the resolve with and without the recovered evidence.

Baseline ("today"): the job's ``match_robot_identity`` labels, i.e. what the
run itself wrote (last-identity labels when the knob was OFF). They are also
the ``slot_meta`` the resolve merges into, as in the live path.

``--swap`` (identity_swap_resolve) additionally runs the swap-aware,
time-varying resolve (web.field.identity_swap.resolve_slot_segments) and
prints its per-slot segments, then compares it with whole-match gir on:

- auto-window correctness: labels over frames < ``--auto-end`` (default
  1860, teleop start) vs ``--truth`` (Manta auto paths; teleop truth is not
  available), as slots whose majority auto label is right and as the
  frame-weighted fraction of (slot, auto frame) pairs;
- held-out read agreement: ``--holdout-runs`` seeded splits hold out
  ``--holdout-frac`` of the team-naming reads, resolve on the rest, and
  score the fraction of held-out reads (attributed to a slot, naming a team
  of that slot's alliance) whose team equals the resolved label of their
  slot at their frame. Both arms merge into a BLANK slot_meta here: the
  job's stored labels were computed from all reads and would leak the
  held-out ones.

Per-slot positions for the swap gate come from the replay's per-frame
``field_pos`` rows (the live path reads the identity log's ``pos``). The
replay drops rows of non-robot slots and per-segment-dropped ranges, which
the gate sees as "slot lost" (open) -- slightly more permissive than live.

``--segments`` (identity_segment_resolve) runs the segment-level resolve
(web.field.identity_segments): slot presence + positions from the replay,
cut at presence gaps / > 2.5 m steps plus the job's exact ``slot_boundary``
diagnostic rows when it logged them (slot_boundary_log), else its
``identity_reject`` births (disrupted or > 2 m) -- the reconstruction. It
reports, per arm (gir, swap, seg, seg|gir = gir's label where seg has none
-- a reference arm for the prototype's protocol, never used live -- and with
``--min-conf`` / ``--min-margin``
the Clef evidence floor applied to the TRAINING reads: gir+floor,
seg+floor): auto-window slots/frames right and precision (labelled frames
only), held-out read agreement on the same seeded splits as ``--swap``
(an unlabelled slot counts as a miss) plus the labelled fraction and
precision, slot-time coverage, and exclusivity (instants where two slots
carry one team > 1 m apart / co-located duplicates). Caveat: the auto
truth is one team per slot for the whole auto window, but slots DO change
robots in auto (qm11: 16 of 57 in-alliance auto reads, mostly p >= 0.9,
disagree with it), so segment labels can be right where the auto score
calls them wrong; held-out agreement is the primary metric.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.field.identity_resolve import (  # noqa: E402
    apply_evidence_floor,
    attribute_reads,
    resolve_slot_labels,
    vlm_read_rows_from_jsonl,
)
from web.field.identity_segments import (  # noqa: E402
    SegmentParams,
    boundary_rows_from_rejects,
    exclusivity_violations,
    format_segment_summary,
    labels_at,
    resolve_segment_labels,
)
from web.field.identity_swap import (  # noqa: E402
    SwapParams,
    format_swap_summary,
    resolve_slot_segments,
    segment_at,
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
            # The label's share of the slot's team score + the runner-up
            # (what the replay review list shows; None without evidence).
            **res["label_conf"].get(sid, {"label_share": None, "runner_up_team": None,
                                          "runner_up_share": None}),
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
        "evicted_team_reads": sum(1 for r in (reads or [])
                                  if r.get("matched_team") and r.get("evicted")),
    }


def format_report(res: dict) -> str:
    st = res["stats"]
    ev = res.get("evicted_team_reads")
    lines = [
        f"identities={res['identities']} per slot {res['identities_per_slot']}; "
        f"team-naming reads={res['team_reads']}"
        + (f" ({ev} evicted-in-flight)" if ev else "")
        + f" used={st['reads_used']} "
        f"dropped_misattr={st['reads_dropped_misattributed']} "
        f"fallback={st['fallback_used']} contested={[LABELS[s] for s in st['contested']]}",
        f"{'slot':<5}{'truth':<7}{'resolved':<10}{'today':<8}{'status':<7}"
        f"{'share':<7}{'runner-up':<15}scores",
    ]
    for lab, s in res["slots"].items():
        conf = "*" if s["confirmed"] else ""
        share = s.get("label_share")
        share_s = "-" if share is None else f"{share:.0%}"
        ru = s.get("runner_up_team")
        ru_s = "-" if ru is None else f"{ru} {s.get('runner_up_share') or 0:.0%}"
        lines.append(f"{lab:<5}{str(s['truth']):<7}{str(s['resolved']) + conf:<10}"
                     f"{str(s['today']):<8}{s['status']:<7}{share_s:<7}{ru_s:<15}"
                     f"{s['scores']}")
    lines.append(f"per-slot resolve: {res['correct']}/6 correct, {res['wrong']} wrong, "
                 f"{res['blank']} blank (today {res['today_correct']}/6)  "
                 f"[* = confirmed, margin >= 2.0]")
    return "\n".join(lines)


def attach_positions(ilog: list[dict], positions: dict | None) -> list[dict]:
    """identity_log rows with ``pos`` filled from ``{(frame, slot): [x, y]}``."""
    if not positions:
        return ilog
    return [{**r, "pos": positions.get((r["frame_num"], r["slot_id"]))} for r in ilog]


def _blank_meta() -> dict:
    return {sid: {"team_number": None, "confirmed": False, "alliance": slot_alliance(sid)}
            for sid in range(6)}


def _gir_labels(reads, ilog, votes_by_slot, blue, red, meta) -> dict[int, str | None]:
    res = resolve_slot_labels(reads or None, ilog, votes_by_slot or {}, {},
                              slot_alliance, blue, red, meta)
    return {sid: res["slot_meta"].get(sid, {}).get("team_number") for sid in range(6)}


def _swap_res(reads, ilog, votes_by_slot, blue, red, meta, params):
    return resolve_slot_segments(reads or None, ilog, votes_by_slot or {}, {},
                                 slot_alliance, blue, red, meta, params=params)


def auto_window_score(label_at, truth: dict, auto_frames: list[int]) -> dict:
    """``label_at(slot_id, frame) -> team``; per-slot majority auto label and
    the frame-weighted fraction of (slot, auto frame) pairs labelled right."""
    slots, ok_pairs, n_pairs, ok_slots, lab_pairs = {}, 0, 0, 0, 0
    for sid, lab in enumerate(LABELS):
        want = truth.get(lab)
        counts: dict = defaultdict(int)
        for f in auto_frames:
            counts[label_at(sid, f)] += 1
        n = sum(counts.values())
        ok = counts.get(want, 0) if want is not None else 0
        ok_pairs += ok
        n_pairs += n
        lab_pairs += n - counts.get(None, 0)
        major = max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))[0] if counts else None
        ok_slots += major is not None and major == want
        slots[lab] = {"truth": want, "majority": major, "frac": (ok / n) if n else 0.0,
                      "labels": dict(counts)}
    return {"slots": slots, "slots_ok": ok_slots,
            "frame_frac": (ok_pairs / n_pairs) if n_pairs else 0.0,
            "labelled_frac": (lab_pairs / n_pairs) if n_pairs else 0.0,
            "precision": (ok_pairs / lab_pairs) if lab_pairs else 0.0}


def holdout_agreement(reads, ilog, votes_by_slot, blue, red, params,
                      runs: int = 50, frac: float = 0.3, seed: int = 0) -> dict:
    """Seeded hold-out of team-naming reads: resolve gir and swap-aware on the
    rest, score held-out reads against each arm's label at (slot, frame)."""
    team_reads = [r for r in reads if r.get("matched_team") is not None]
    roster = {**{t: "blue" for t in blue}, **{t: "red" for t in red}}
    rng = random.Random(seed)
    tot = {"gir": 0, "swap": 0, "n": 0}
    for _ in range(runs):
        held = set(rng.sample(range(len(team_reads)), round(frac * len(team_reads))))
        train = [r for i, r in enumerate(team_reads) if i not in held]
        test = [r for i, r in enumerate(team_reads) if i in held]
        gl = _gir_labels(train, ilog, votes_by_slot, blue, red, _blank_meta())
        sres = _swap_res(train, ilog, votes_by_slot, blue, red, _blank_meta(), params)
        for frame, sid, team, _w in attribute_reads(test, ilog)[0]:
            if roster.get(str(team)) != slot_alliance(sid):
                continue        # cross-alliance / off-roster read: no arm can agree
            seg = segment_at(sres["segments"].get(sid), frame)
            tot["n"] += 1
            tot["gir"] += gl.get(sid) == str(team)
            tot["swap"] += seg is not None and seg["team_number"] == str(team)
    n = tot["n"] or 1
    return {"n": tot["n"], "runs": runs, "frac": frac,
            "gir": tot["gir"] / n, "swap": tot["swap"] / n}


def run_swap_check(bindings, frame_nums, reads, truth: dict, positions=None,
                   votes_by_slot=None, params: SwapParams | None = None,
                   auto_end: int = 1860, holdout_runs: int = 50,
                   holdout_frac: float = 0.3, identity_log=None) -> dict:
    """Swap-aware vs whole-match gir on one job (blank slot_meta for both)."""
    params = params or SwapParams()
    ilog = identity_log if identity_log is not None else reconstruct_identity_log(
        bindings, frame_nums)
    ilog = attach_positions(ilog, positions)
    blue = [truth[lab] for lab in LABELS[:3] if lab in truth]
    red = [truth[lab] for lab in LABELS[3:] if lab in truth]
    gl = _gir_labels(reads, ilog, votes_by_slot, blue, red, _blank_meta())
    sres = _swap_res(reads, ilog, votes_by_slot, blue, red, _blank_meta(), params)
    auto_frames = sorted({r["frame_num"] for r in ilog if r["frame_num"] < auto_end})

    def swap_at(sid, f):
        seg = segment_at(sres["segments"].get(sid), f)
        return seg["team_number"] if seg else None

    out = {
        "gir_labels": {LABELS[s]: t for s, t in gl.items()},
        "segments": {LABELS[s]: [(g["start_frame"], g["end_frame"], g["team_number"],
                                  g["score"]) for g in segs]
                     for s, segs in sres["segments"].items()},
        "swaps": [{**sw, "slots": [LABELS[a] for a in sw["slots"]]} for sw in sres["swaps"]],
        "summary": format_swap_summary(sres),
        "gate_open_frac": sres["stats"]["gate_open_frac"],
        "auto_gir": auto_window_score(lambda s, f: gl.get(s), truth, auto_frames),
        "auto_swap": auto_window_score(swap_at, truth, auto_frames),
    }
    if holdout_runs > 0:
        out["holdout"] = holdout_agreement(reads, ilog, votes_by_slot, blue, red, params,
                                           runs=holdout_runs, frac=holdout_frac)
    return out


def format_swap_report(res: dict) -> str:
    lines = [res["summary"], f"gate open fraction: {res['gate_open_frac']}",
             f"{'slot':<5}{'truth':<7}{'gir':<7}segments (start-end team score)"]
    for lab in LABELS:
        segs = " | ".join(f"{a}-{b} {t} {sc:g}" for a, b, t, sc in res["segments"].get(lab, []))
        lines.append(f"{lab:<5}{str(res['auto_swap']['slots'][lab]['truth']):<7}"
                     f"{str(res['gir_labels'][lab]):<7}{segs}")
    ag, asw = res["auto_gir"], res["auto_swap"]
    lines.append(f"auto window: gir {ag['slots_ok']}/6 slots ({ag['frame_frac']:.0%} slot-frames)"
                 f" | swap {asw['slots_ok']}/6 slots ({asw['frame_frac']:.0%} slot-frames)")
    ho = res.get("holdout")
    if ho:
        lines.append(f"held-out read agreement ({ho['runs']} x {ho['frac']:.0%}, "
                     f"n={ho['n']}): gir {ho['gir']:.1%} | swap {ho['swap']:.1%}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# --segments: segment-level labels (identity_segment_resolve) + Clef floor
# --------------------------------------------------------------------------- #
def job_boundary_rows(data: dict) -> tuple[list[dict], str]:
    """The job's exact ``slot_boundary`` rows when it logged them
    (slot_boundary_log / identity_segment_resolve), else its
    ``identity_reject`` births as boundary rows (the reconstruction)."""
    if data.get("boundary_rows"):
        return data["boundary_rows"], "exact"
    return boundary_rows_from_rejects(data.get("reject_rows") or []), "reconstructed"


def _floored(reads, min_conf: float, min_margin: float):
    return apply_evidence_floor(reads, min_conf, min_margin)[0]


def holdout_multi(reads, ilog, blue, red, arms: dict, runs: int = 50,
                  frac: float = 0.3, seed: int = 0) -> dict:
    """The ``holdout_agreement`` protocol (same seeded splits) over any arms:
    ``arms`` maps name -> ``make(train_reads) -> label_at(slot, frame)``.
    Per arm: ``agree`` (held-out reads whose team equals the arm's label at
    their slot+frame; an unlabelled slot counts as a miss), ``labelled``
    (fraction of held-out reads the arm labels at all) and ``precision``
    (agreement among the labelled ones)."""
    team_reads = [r for r in reads if r.get("matched_team") is not None]
    roster = {**{t: "blue" for t in blue}, **{t: "red" for t in red}}
    rng = random.Random(seed)
    hit = {k: 0 for k in arms}
    lab = {k: 0 for k in arms}
    n = 0
    for _ in range(runs):
        held = set(rng.sample(range(len(team_reads)), round(frac * len(team_reads))))
        train = [r for i, r in enumerate(team_reads) if i not in held]
        test = [r for i, r in enumerate(team_reads) if i in held]
        fns = {k: make(train) for k, make in arms.items()}
        for frame, sid, team, _w in attribute_reads(test, ilog)[0]:
            if roster.get(str(team)) != slot_alliance(sid):
                continue
            n += 1
            for k, fn in fns.items():
                got = fn(sid, frame)
                lab[k] += got is not None
                hit[k] += got == str(team)
    out = {"n": n, "runs": runs, "frac": frac}
    for k in arms:
        out[k] = {"agree": hit[k] / n if n else 0.0,
                  "labelled": lab[k] / n if n else 0.0,
                  "precision": hit[k] / lab[k] if lab[k] else 0.0}
    return out


def run_segment_check(data: dict, truth: dict, seg_params: SegmentParams | None = None,
                      swap_params: SwapParams | None = None, auto_end: int = 1860,
                      holdout_runs: int = 50, holdout_frac: float = 0.3,
                      min_conf: float = 0.0, min_margin: float = 0.0,
                      identity_log=None) -> dict:
    """gir / swap / segments / segments+floor on one loaded job
    (``load_job`` dict). Blank slot_meta for gir and swap, as in --swap."""
    sp = seg_params or SegmentParams()
    swp = swap_params or SwapParams(fps=sp.fps)
    ilog = identity_log if identity_log is not None else reconstruct_identity_log(
        data["bindings"], data["frame_nums"])
    ilog = attach_positions(ilog, data.get("positions"))
    reads, votes = data["reads"], data.get("votes_by_slot") or {}
    blue = [truth[lab] for lab in LABELS[:3] if lab in truth]
    red = [truth[lab] for lab in LABELS[3:] if lab in truth]
    brows, bsrc = job_boundary_rows(data)
    grid = list(data.get("frame_nums") or [])
    floor_on = min_conf > 0.0 or min_margin > 0.0

    def seg(train, floor=False):
        rows = _floored(train, min_conf, min_margin) if floor else train
        return resolve_segment_labels(rows or None, ilog, blue, red, boundary_rows=brows,
                                      params=sp, grid=grid)

    def gir_fn(train, floor=False):
        rows = _floored(train, min_conf, min_margin) if floor else train
        gl = _gir_labels(rows, ilog, votes, blue, red, _blank_meta())
        return lambda s, f: gl.get(s)

    def swap_fn(train):
        sres = _swap_res(train, ilog, votes, blue, red, _blank_meta(), swp)
        return lambda s, f: (segment_at(sres["segments"].get(s), f) or {}).get("team_number")

    def seg_fn(train, floor=False):
        segs = seg(train, floor)["segments"]
        return lambda s, f: labels_at(segs, s, f)

    def seg_gir_fn(train):
        # Reference only (the prototype's protocol): gir's whole-match label
        # wherever the segment resolve leaves a segment unlabelled. The live
        # resolve never does this -- it does not invent labels.
        segs, gl = seg_fn(train), gir_fn(train)
        return lambda s, f: segs(s, f) if segs(s, f) is not None else gl(s, f)

    arms = {"gir": gir_fn, "swap": swap_fn, "seg": seg_fn, "seg|gir": seg_gir_fn}
    if floor_on:
        arms["gir+floor"] = lambda tr: gir_fn(tr, True)
        arms["seg+floor"] = lambda tr: seg_fn(tr, True)
    auto_frames = sorted({r["frame_num"] for r in ilog if r["frame_num"] < auto_end})
    full = {k: make(reads) for k, make in arms.items()}
    sres_full = seg(reads)
    out = {
        "boundary_source": bsrc,
        "boundary_rows": len(brows),
        "seg": sres_full,
        "seg_summary": format_segment_summary(sres_full),
        "auto": {k: auto_window_score(fn, truth, auto_frames) for k, fn in full.items()},
        "coverage": {"seg": sres_full["stats"]["coverage"]},
        "exclusivity": {"seg": exclusivity_violations(sres_full["segments"], ilog,
                                                      sp.dup_dist_m)},
        "floor": {"min_conf": min_conf, "min_margin": min_margin,
                  "dropped": apply_evidence_floor(reads, min_conf, min_margin)[1]},
    }
    if floor_on:
        sres_floor = seg(reads, True)
        out["seg_floor"] = sres_floor
        out["coverage"]["seg+floor"] = sres_floor["stats"]["coverage"]
        out["exclusivity"]["seg+floor"] = exclusivity_violations(
            sres_floor["segments"], ilog, sp.dup_dist_m)
    if holdout_runs > 0:
        out["holdout"] = holdout_multi(reads, ilog, blue, red, arms,
                                       runs=holdout_runs, frac=holdout_frac)
    return out


def format_segment_report(res: dict) -> str:
    st = res["seg"]["stats"]
    lines = [res["seg_summary"],
             f"boundaries: {res['boundary_source']} ({res['boundary_rows']} rows) "
             f"cuts={st['cuts']} segments={st['segments']} read={st['read']} "
             f"stitched={st['stitched']} none={st['unlabelled']} contested={st['contested']} "
             f"demoted={st['demoted']} links={st['links']} "
             f"refused(reads={st['refused_reads']}, exclusive={st['refused_exclusive']})"]
    fl = res["floor"]
    if fl["min_conf"] > 0 or fl["min_margin"] > 0:
        lines.append(f"clef floor: conf>={fl['min_conf']:g} margin>={fl['min_margin']:g} "
                     f"drops {fl['dropped']} team read(s)")
    ho = res.get("holdout")
    lines.append(f"{'arm':<10}{'auto slots':>11}{'auto frames':>12}{'auto prec':>10}"
                 + (f"{'held-out':>10}{'labelled':>9}{'prec':>7}" if ho else "")
                 + f"{'coverage':>9}{'excl/dup':>10}")
    for k, a in res["auto"].items():
        cov = res["coverage"].get(k)
        ex = res["exclusivity"].get(k)
        row = f"{k:<10}{a['slots_ok']:>9}/6{a['frame_frac']:>12.0%}{a['precision']:>10.0%}"
        if ho:
            h = ho[k]
            row += f"{h['agree']:>10.1%}{h['labelled']:>9.0%}{h['precision']:>7.1%}"
        cov_txt = f"{cov:.0%}" if cov is not None else "-"
        ex_txt = f"{ex['violations']}/{ex['duplicates']}" if ex else "-"
        lines.append(f"{row}{cov_txt:>9}{ex_txt:>10}")
    if ho:
        lines.append(f"held-out: {ho['runs']} x {ho['frac']:.0%}, n={ho['n']} "
                     f"(unlabelled = miss; floors apply to the training reads only)")
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
    positions = {}          # (frame, slot_id) -> [x, y] from the replay rows
    for fr in meta["frames"]:
        for r in fr.get("robots", []):
            if r.get("field_pos") is not None and r.get("slot") in LABEL_TO_SLOT:
                positions.setdefault((fr["frame_num"], LABEL_TO_SLOT[r["slot"]]),
                                     list(r["field_pos"]))
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
    reject_rows, boundary_rows = [], []
    for line in lines:
        if '"identity_reject"' not in line and '"slot_boundary"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "identity_reject":
            reject_rows.append(r)
        elif r.get("type") == "slot_boundary":
            boundary_rows.append(r)
    slot_meta, votes_by_slot = {}, {}
    for slot, iid, team, confirmed in ident_rows:
        if slot is None:
            continue
        slot_meta[int(slot)] = {"team_number": team, "confirmed": bool(confirmed),
                                "alliance": slot_alliance(int(slot))}
        votes_by_slot[int(slot)] = {str(k): float(v)
                                    for k, v in votes_by_id.get(iid, {}).items()}
    return {"job": job, "bindings": bindings, "frame_nums": frame_nums,
            "reads": reads, "slot_meta": slot_meta, "votes_by_slot": votes_by_slot,
            "positions": positions, "reject_rows": reject_rows,
            "boundary_rows": boundary_rows}


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
    ap.add_argument("--exclude-evicted", action="store_true",
                    help="drop read_keep_evicted rows (\"evicted\": true) before "
                         "resolving: the knob-OFF evidence of a knob-ON job")
    ap.add_argument("--swap", action="store_true",
                    help="also run the swap-aware time-varying resolve "
                         "(identity_swap_resolve) and compare it with gir")
    ap.add_argument("--swap-lambda", type=float, default=SwapParams.swap_lambda,
                    help="swap cost in vote-weight units (default %(default)s)")
    ap.add_argument("--prox-m", type=float, default=SwapParams.prox_m,
                    help="swap-gate proximity threshold, metres (default %(default)s)")
    ap.add_argument("--fps", type=float, default=SwapParams.fps,
                    help="source video fps (default %(default)s)")
    ap.add_argument("--auto-end", type=int, default=1860,
                    help="first teleop source frame (default %(default)s)")
    ap.add_argument("--holdout-runs", type=int, default=50)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--segments", action="store_true",
                    help="also run the segment-level resolve (identity_segment_resolve) "
                         "and compare gir / swap / segments (+floor) on held-out reads, "
                         "the auto window, coverage and exclusivity")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Clef evidence floor (clef_min_evidence_conf) for the +floor "
                         "arms: ignore team reads with p(team) below this (0 = off)")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="alternative floor for the +floor arms: ignore team reads with "
                         "p(team) - p(none) below this (0 = off)")
    ap.add_argument("--seg-slack", type=float, default=SegmentParams.link_slack_m,
                    help="segment link slack, metres (default %(default)s)")
    ap.add_argument("--seg-max-gap", type=float, default=SegmentParams.link_max_gap_s,
                    help="segment link max gap, seconds (default %(default)s)")
    ap.add_argument("--seg-speed", type=float, default=SegmentParams.max_speed_mps,
                    help="segment link max robot speed, m/s (default %(default)s)")
    ap.add_argument("--seg-dup-m", type=float, default=SegmentParams.dup_dist_m,
                    help="duplicate / exclusivity distance, metres (default %(default)s)")
    ap.add_argument("--seg-min-support", type=float, default=SegmentParams.min_support,
                    help="read weight a segment needs to label itself (default %(default)s)")
    args = ap.parse_args(argv)

    truth = {}
    for part in args.truth.split(","):
        lab, team = part.split("=")
        if lab.strip().upper() not in LABEL_TO_SLOT:
            raise SystemExit(f"bad slot label {lab!r}")
        truth[lab.strip().upper()] = team.strip()

    data = load_job(Path(args.data_root), args.job)
    if args.exclude_evicted:
        data["reads"] = [r for r in data["reads"] if not r.get("evicted")]
    ilog = json.loads(Path(args.identity_log).read_text()) if args.identity_log else None
    res = run_check(data["bindings"], data["frame_nums"], data["reads"], truth,
                    slot_meta=data["slot_meta"], votes_by_slot=data["votes_by_slot"],
                    identity_log=ilog)
    sres = None
    if args.swap:
        params = SwapParams(swap_lambda=args.swap_lambda, prox_m=args.prox_m, fps=args.fps)
        sres = run_swap_check(data["bindings"], data["frame_nums"], data["reads"], truth,
                              positions=data.get("positions"),
                              votes_by_slot=data["votes_by_slot"], params=params,
                              auto_end=args.auto_end, holdout_runs=args.holdout_runs,
                              holdout_frac=args.holdout_frac, identity_log=ilog)
    gres = None
    if args.segments:
        sp = SegmentParams(fps=args.fps, link_slack_m=args.seg_slack,
                           link_max_gap_s=args.seg_max_gap, max_speed_mps=args.seg_speed,
                           dup_dist_m=args.seg_dup_m, min_support=args.seg_min_support)
        gres = run_segment_check(
            data, truth, seg_params=sp,
            swap_params=SwapParams(swap_lambda=args.swap_lambda, prox_m=args.prox_m,
                                   fps=args.fps),
            auto_end=args.auto_end, holdout_runs=args.holdout_runs,
            holdout_frac=args.holdout_frac, min_conf=args.min_conf,
            min_margin=args.min_margin, identity_log=ilog)
    if args.json:
        print(json.dumps({"job": data["job"], **res,
                          **({"swap": sres} if sres else {}),
                          **({"segments": {k: v for k, v in gres.items()
                                           if k not in ("seg", "seg_floor")}}
                             if gres else {})}, default=str, indent=2))
    else:
        print(f"== {data['job']}")
        print(format_report(res))
        if sres:
            print(format_swap_report(sres))
        if gres:
            print(format_segment_report(gres))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
