#!/usr/bin/env python
"""Event lineup evaluation report: the spec section 8 pass criteria for one
arm (default event-full) against the baseline arm (default own).

Spec: docs/superpowers/specs/2026-11-10-event-lineup-design.md section 8.
An arm is one or more scripts/event_relabel.py --out-dir run dirs: the first
is the full 20-match run, later ones are qm8 repeats. Read from them:
results.jsonl (unlabelled share, GT score on qm8, row sources) and, from the
first dir, audit/*/score.json (scripts/audit_replay.py score --json on that
arm's relabelled metas). --fable AUDIT_DIR=ANSWERS adds a Fable judge
answers file (Claude Code subagents, offline) for an audit dir of the
target arm; its borrowed-row labels give the per-team check.

    .venv/bin/python scripts/event_lineup_report.py \
        --arm own=RUNS/own,RUNS/own-r2 --arm event-full=RUNS/event-full,RUNS/event-full-r2 \
        --fable RUNS/event-full/audit/2026cur_qm11_xxxxxxxx=answers_fable.jsonl

No API calls, nothing written.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QM8_MISID_MAX_S = 5.0          # arm 3 vs arm 1 on qm8: mis-ID at most +5 s
QM8_WRONG_MAX = 0.01           # ... and wrong share at most +1 pt
UNLABELLED_MIN_DROP = 0.05     # mean unlabelled slot-time down by at least 5 pts
FABLE_WRONG_MAX = 0.10         # no team's borrowed-row labels more than 10% wrong
MIN_ANCHORS = 2                # a team with fewer own anchors had no own row


def _read_jsonl(p) -> list:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def load_arm(dirs) -> dict:
    dirs = [Path(d) for d in dirs]
    rows = _read_jsonl(dirs[0] / "results.jsonl")
    qm8 = [r["score"]["full"] for d in dirs for r in _read_jsonl(d / "results.jsonl")
           if str(r.get("match_key", "")).endswith("_qm8") and r.get("score")]
    audit = {}
    for p in sorted((dirs[0] / "audit").glob("*/score.json")):
        s = json.loads(p.read_text())
        audit[s["match_key"]] = s.get("misid")
    return {"dirs": [str(d) for d in dirs], "rows": rows,
            "unlabelled": {r["match_key"]: r.get("unlabelled_share")
                           for r in rows if not r.get("error")},
            "qm8": qm8, "audit": audit}


def paired_ci(diffs: list, reps: int = 10000, seed: int = 0, alpha: float = 0.05):
    """(mean, lower, upper) of the mean paired difference: percentile
    bootstrap over matches. None for no pairs."""
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return (sum(diffs) / n, means[int(alpha / 2 * reps)],
            means[min(reps - 1, int((1 - alpha / 2) * reps))])


def criteria(base: dict, target: dict, fable_worst: Optional[float]) -> list:
    """The four spec section 8 criteria plus the Fable per-team check:
    [{"name", "value", "pass": True / False / None (not measurable)}]."""
    out = []
    bm, tm = (_mean(q.get("misid_s") for q in a["qm8"]) for a in (base, target))
    d = None if bm is None or tm is None else tm - bm
    out.append({"name": "qm8 misID", "pass": None if d is None else d <= QM8_MISID_MAX_S + 1e-9,
                "value": "-" if d is None else f"{d:+.1f} s (limit +{QM8_MISID_MAX_S:.0f} s)"})
    bw, tw = (_mean(q.get("wrong") for q in a["qm8"]) for a in (base, target))
    d = None if bw is None or tw is None else tw - bw
    out.append({"name": "qm8 wrong", "pass": None if d is None else d <= QM8_WRONG_MAX + 1e-9,
                "value": "-" if d is None else f"{d * 100:+.1f} pt (limit +1 pt)"})
    common = sorted(m for m, v in base["unlabelled"].items()
                    if v is not None and target["unlabelled"].get(m) is not None)
    drop = (None if not common else _mean(base["unlabelled"][m] for m in common)
            - _mean(target["unlabelled"][m] for m in common))
    out.append({"name": "unlabelled drop",
                "pass": None if drop is None else drop >= UNLABELLED_MIN_DROP - 1e-9,
                "value": "-" if drop is None else
                f"{drop * 100:.1f} pt over {len(common)} matches (need >= 5 pt)"})
    diffs = [target["audit"][m] - base["audit"][m] for m in sorted(base["audit"])
             if base["audit"][m] is not None and target["audit"].get(m) is not None]
    ci = paired_ci(diffs)
    out.append({"name": "audit misID paired", "pass": None if ci is None else ci[1] <= 1e-12,
                "value": "-" if ci is None else
                f"{ci[0] * 100:+.1f} pt [{ci[1] * 100:+.1f}, {ci[2] * 100:+.1f}] over "
                f"{len(diffs)} matches (lower bound <= 0)"})
    out.append({"name": "Fable borrowed-row wrong",
                "pass": None if fable_worst is None else fable_worst <= FABLE_WRONG_MAX + 1e-9,
                "value": "not run" if fable_worst is None else
                f"worst team {fable_worst:.0%} (limit 10%)"})
    return out


def borrowed_label_counts(audit_dir, answers_path, meta: dict, row_sources: dict,
                          min_conf: float = 0.7) -> dict:
    """team -> (judge-confident items, wrong) over the audit items whose
    label is a LINEUP label (confirmed False) of a team whose row in this
    match held a borrowed or elimination tile."""
    from scripts.audit_replay import parse_answer
    audit_dir = Path(audit_dir)
    audit = json.loads((audit_dir / "audit.json").read_text())
    roster = set(audit["roster"]["red"]) | set(audit["roster"]["blue"])
    answers = {a["sample_id"]: parse_answer(a, roster) for a in _read_jsonl(answers_path)}
    borrowed = {t for t, s in row_sources.items() if s.get("borrowed") or s.get("elim")}
    out: dict = {}
    for s in _read_jsonl(audit_dir / "samples.jsonl"):
        team = s.get("pipeline_label")
        if team not in borrowed:
            continue
        row = next((r for r in meta["frames"][s["frame_idx"]].get("robots") or []
                    if r.get("slot") == s["slot"]), None)
        if row is None or row.get("confirmed"):
            continue
        a = answers.get(s["sample_id"])
        if not a or a["kind"] != "team" or a["conf"] < min_conf:
            continue
        n, w = out.get(team, (0, 0))
        out[team] = (n + 1, w + (a["team"] != team))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR[,DIR...]")
    ap.add_argument("--baseline", default="own")
    ap.add_argument("--target", default="event-full")
    ap.add_argument("--fable", action="append", default=[], metavar="AUDIT_DIR=ANSWERS")
    ap.add_argument("--min-conf", type=float, default=0.7)
    args = ap.parse_args(argv)
    arms = {}
    for spec in args.arm:
        name, _, dirs = spec.partition("=")
        arms[name] = load_arm([d for d in dirs.split(",") if d])
    for need in (args.baseline, args.target):
        if need not in arms:
            ap.error(f"--arm {need}=DIR is required")

    print(f"{'arm':<16}{'matches':>8}{'unlabelled':>12}{'qm8 misID':>11}{'qm8 wrong':>11}"
          f"{'audit misID':>13}")
    for name, a in arms.items():
        m = _mean(q.get("misid_s") for q in a["qm8"])
        print(f"{name:<16}{len(a['rows']):>8}{_pct(_mean(a['unlabelled'].values())):>12}"
              f"{'-' if m is None else f'{m:.0f} s':>11}"
              f"{_pct(_mean(q.get('wrong') for q in a['qm8'])):>11}"
              f"{_pct(_mean(a['audit'].values())):>13}")

    target = arms[args.target]
    by_mk = {r["match_key"]: r for r in target["rows"]}
    counts: dict = {}
    for spec in args.fable:
        adir, _, ans = spec.partition("=")
        adir = Path(adir)
        mk = json.loads((adir / "audit.json").read_text())["match_key"]
        r = by_mk[mk]
        meta = json.loads(Path(r["meta_out"]).read_text())
        ans_p = Path(ans) if Path(ans).exists() else adir / ans
        for t, (n, w) in borrowed_label_counts(adir, ans_p, meta, r.get("row_sources") or {},
                                               args.min_conf).items():
            n0, w0 = counts.get(t, (0, 0))
            counts[t] = (n0 + n, w0 + w)
    no_own: dict = {}
    for r in target["rows"]:
        for t, s in (r.get("row_sources") or {}).items():
            if s.get("own", 0) < MIN_ANCHORS:
                no_own.setdefault(t, []).append(r["match_key"].split("_")[-1])
    print(f"\nper team ({args.target}): matches with no own row but a borrowed one; "
          "Fable verdicts on its borrowed-row labels")
    for t in sorted(set(no_own) | set(counts), key=lambda x: (len(x), x)):
        n, w = counts.get(t, (0, 0))
        print(f"  {t:>6}  {len(no_own.get(t, [])):>2} matches {','.join(no_own.get(t, [])):<40}"
              f"  Fable {f'{w}/{n} wrong' if n else '-'}")
    worst = (max((w / n for n, w in counts.values() if n), default=None)
             if args.fable else None)

    print(f"\npass criteria: {args.target} vs {args.baseline}")
    rows = criteria(arms[args.baseline], target, worst)
    for c in rows:
        flag = "PASS" if c["pass"] else "n/a" if c["pass"] is None else "FAIL"
        print(f"  {c['name']:<26} {c['value']:<62} {flag}")
    verdict = ("PASS" if all(c["pass"] for c in rows) else
               "FAIL" if any(c["pass"] is False for c in rows) else "INCOMPLETE")
    print(f"verdict: {verdict}")
    for name, a in arms.items():
        if name in (args.baseline, args.target):
            continue
        print(f"\n{name} vs {args.baseline} (informational)")
        for c in criteria(arms[args.baseline], a, None)[:4]:
            flag = "PASS" if c["pass"] else "n/a" if c["pass"] is None else "FAIL"
            print(f"  {c['name']:<26} {c['value']:<62} {flag}")


if __name__ == "__main__":
    main()
