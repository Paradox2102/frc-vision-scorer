#!/usr/bin/env python3
"""Jev bulk labeller for harvested bumper tracklets (design §4, route B).

Per match directory written by scripts/teacher/harvest.py (manifest.jsonl):

1. Per bumper tracklet choose <= 2 crops (largest px_w, then sharpness =
   variance of Laplacian, spread in time; see ``choose_crops``).
2. Ask Jev about each crop exactly like production's single-crop read
   (web/jev_reader: jev_options over the match roster, the bumper colour as the
   expected alliance, build_jev_payload, per-crop option shuffle). Every call's
   full option distribution, top option, tokens, latency and error is kept.
3. Tracklet decision (``decide``; raw evidence is stored so the thresholds can
   be re-chosen against the gold set later):
     team        both calls' top = the same roster team, p >= 0.5 each; or
                 one call only and its top is a roster team with p >= 0.85
     not_bumper  both calls' top is not_robot / printed (one call: p >= 0.85)
     unsure      anything else (incl. a failed call)
   ``off_alliance`` flags a team whose alliance != the bumper colour. A team
   outside the roster is never output (options are roster-only anyway).
4. Robot tracklets inherit from their linked bumper tracklets (``robot_decision``):
   every decided (team) linked bumper names the same team -> team; they
   disagree -> conflict (votes kept); none decided -> none.

Outputs next to manifest.jsonl: ``labels_jev.jsonl`` (one row per tracklet),
``labels_jev_summary.json``, and ``jev_calls.jsonl`` (per-call cache: a
re-run reuses finished calls, so an interrupted / budget-stopped match resumes
without paying twice). Matches with labels_jev.jsonl are skipped unless --force.

Usage:
  .venv/bin/python -m scripts.teacher.label_jev --dry-run                  # all of harvest_v2
  .venv/bin/python -m scripts.teacher.label_jev --events 2025cur --max-usd 3
  .venv/bin/python -m scripts.teacher.label_jev --matches 2025cur_qm1,2026cur_qm22
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.teacher.harvest import MAIN  # noqa: E402

DEFAULT_ROOT = MAIN / "data/teacher/harvest_v2"
LABELS = "labels_jev.jsonl"
SUMMARY = "labels_jev_summary.json"
CALLS = "jev_calls.jsonl"

CROPS_PER_TRACKLET = 2
LARGE_FRAC = 0.8          # "largest": px_w within 80% of the tracklet's widest crop
MIN_SPREAD_S = 1.0        # second crop >= this far in time from the first when possible
TWO_CALL_MIN_P = 0.5
ONE_CALL_MIN_P = 0.85
NOT_BUMPER_TOPS = frozenset({"not_robot", "printed"})
CHUNK = 32                # requests per budget check


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested)
# --------------------------------------------------------------------------- #

def choose_crops(rows: Sequence[dict], k: int = CROPS_PER_TRACKLET,
                 large_frac: float = LARGE_FRAC, min_spread_s: float = MIN_SPREAD_S) -> List[dict]:
    """<= ``k`` crop rows of one tracklet. Rows need ``px_w``, ``t_sec`` and
    ``sharpness``. Ranking: crops with px_w >= ``large_frac`` x the widest come
    first, sharpest first within that set, then the rest by (px_w, sharpness).
    The first pick is the top of that ranking; each later pick is the best
    remaining crop >= ``min_spread_s`` from every pick so far, or, when none
    is, the remaining crop farthest in time from the picks."""
    if not rows or k <= 0:
        return []
    wmax = max(r["px_w"] for r in rows)
    ranked = sorted(rows, key=lambda r: (r["px_w"] < large_frac * wmax,
                                         -(r["sharpness"] if r["px_w"] >= large_frac * wmax
                                           else r["px_w"]),
                                         -r["sharpness"], r["t_sec"]))
    picks = [ranked[0]]
    while len(picks) < min(k, len(rows)):
        rest = [r for r in ranked if all(r is not p for p in picks)]
        far = [r for r in rest if all(abs(r["t_sec"] - p["t_sec"]) >= min_spread_s for p in picks)]
        if far:
            picks.append(far[0])
        else:
            picks.append(max(rest, key=lambda r: (min(abs(r["t_sec"] - p["t_sec"]) for p in picks),
                                                  -ranked.index(r))))
    return sorted(picks, key=lambda r: r["t_sec"])


def call_summary(probs: Dict[str, float], roster_alliance: Dict[str, str]) -> dict:
    """top option, its p, and the best roster team + p from one call's probs."""
    if not probs:
        return {"top": None, "top_p": 0.0, "best_team": None, "best_team_p": 0.0}
    top = max(probs, key=lambda o: (probs[o], o))
    teams = [(p, o) for o, p in probs.items() if o in roster_alliance]
    bp, bt = max(teams, key=lambda x: (x[0], x[1])) if teams else (0.0, None)
    return {"top": top, "top_p": round(float(probs[top]), 4),
            "best_team": bt, "best_team_p": round(float(bp), 4)}


def decide(calls: Sequence[dict], roster_alliance: Dict[str, str], colour: Optional[str],
           two_min_p: float = TWO_CALL_MIN_P, one_min_p: float = ONE_CALL_MIN_P) -> dict:
    """Tracklet decision from its calls (each: ``top``, ``top_p``, ``error``).
    -> {decision: team|not_bumper|unsure, team, off_alliance, reasons[]}."""
    reasons: List[str] = []
    out = {"decision": "unsure", "team": None, "off_alliance": False, "reasons": reasons}
    if not roster_alliance:
        reasons.append("no_roster")
        return out
    if not calls:
        reasons.append("no_calls")
        return out
    if any(c.get("error") for c in calls):
        reasons.append("call_error")
        return out
    tops = [c.get("top") for c in calls]
    ps = [float(c.get("top_p") or 0.0) for c in calls]
    team = None
    if len(calls) >= 2:
        if len(set(tops)) == 1 and tops[0] in roster_alliance and min(ps) >= two_min_p:
            team = tops[0]
            reasons.append("two_calls_agree")
        elif all(t in NOT_BUMPER_TOPS for t in tops):
            out["decision"] = "not_bumper"
            reasons.append("two_calls_not_bumper")
            return out
        else:
            if len(set(tops)) > 1 and sum(t in roster_alliance for t in tops) >= 2:
                reasons.append("teams_disagree")
            elif any(t in roster_alliance for t in tops):
                reasons.append("weak_or_split")
            else:
                reasons.append("escape_tops:" + "+".join(sorted(set(map(str, tops)))))
    else:
        if tops[0] in roster_alliance and ps[0] >= one_min_p:
            team = tops[0]
            reasons.append("one_call_confident")
        elif tops[0] in NOT_BUMPER_TOPS and ps[0] >= one_min_p:
            out["decision"] = "not_bumper"
            reasons.append("one_call_not_bumper")
            return out
        else:
            reasons.append("one_call_weak" if tops[0] in roster_alliance
                           else f"escape_top:{tops[0]}")
    if team is None:
        return out
    if team not in roster_alliance:      # defensive: options are roster-only
        reasons.append("off_roster")
        return out
    out["decision"], out["team"] = "team", team
    al = roster_alliance.get(team)
    if colour in ("red", "blue") and al in ("red", "blue") and al != colour:
        out["off_alliance"] = True
        reasons.append("off_alliance")
    return out


def robot_decision(linked: Iterable[str], bumper_labels: Dict[str, dict]) -> dict:
    """Robot tracklet label from its linked bumper tracklets' decisions."""
    votes = Counter(bumper_labels[b]["team"] for b in linked
                    if b in bumper_labels and bumper_labels[b].get("decision") == "team")
    if not votes:
        return {"decision": "none", "team": None, "votes": {}, "reasons": ["no_decided_bumper"]}
    if len(votes) == 1:
        team = next(iter(votes))
        return {"decision": "team", "team": team, "votes": dict(votes),
                "reasons": [f"{votes[team]}_bumper_tracklets_agree"]}
    (t1, n1), (_t2, n2) = votes.most_common(2)
    return {"decision": "conflict", "team": None, "votes": dict(votes),
            "majority_team": t1 if n1 > n2 else None, "reasons": ["bumper_teams_disagree"]}


def roster_alliance_of(roster: Optional[dict]) -> Dict[str, str]:
    if not roster:
        return {}
    return {str(t): c for c in ("red", "blue") for t in roster.get(c) or []}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                    # torn last line of an interrupted run
    return out


def sharpness(path: Path) -> float:
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g is not None else 0.0


def match_dirs(root: Path, events: Optional[Sequence[str]], matches: Optional[Sequence[str]]
               ) -> List[Path]:
    out = []
    for m in sorted(root.glob("*/*/manifest.jsonl")):
        d = m.parent
        if events and d.parent.name not in events:
            continue
        if matches and d.name not in matches:
            continue
        out.append(d)
    return out


def plan_match(mdir: Path, root: Path) -> Tuple[dict, Dict[str, List[dict]], Dict[str, List[dict]],
                                                Dict[str, List[dict]]]:
    """(match info, bumper rows by tracklet, robot rows by tracklet, chosen crops by tracklet)."""
    rows = read_jsonl(mdir / "manifest.jsonl")
    b: Dict[str, List[dict]] = defaultdict(list)
    r: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        (b if row["detector"] == "bumper" else r)[row["tracklet_id"]].append(row)
    roster = next((row.get("roster") for row in rows if row.get("roster")), None)
    chosen = {}
    for tid, trs in b.items():
        for row in trs:
            row["sharpness"] = sharpness(root / row["path"])
        chosen[tid] = choose_crops(trs)
    info = {"match_key": mdir.name, "event": mdir.parent.name, "roster": roster}
    return info, b, r, chosen


class _UsageJev:
    """JevSession subclass factory (import-time free) recording each request's
    usage in a thread-local, so per-call tokens are exact."""

    @staticmethod
    def make(**kw):
        from web import jev_reader as jr

        class TrackedJev(jr.JevSession):
            _tl = threading.local()

            def _unwrap(self, body):
                answers, usage = super()._unwrap(body)
                TrackedJev._tl.usage = usage
                return answers, usage

        return TrackedJev(jr.DEFAULT_MODEL, **kw)


def build_request(path: Path, roster: dict, colour: Optional[str], seed: str):
    """(payload, (w, h)) for one crop, exactly as JevSession._read_payload builds it."""
    from web import jev_reader as jr
    img, w, h, _pad = jr.jev_image(path.read_bytes())
    expected = colour if colour in ("red", "blue") else "unknown"
    opts = jr.jev_options(roster["red"], roster["blue"], (), expected)
    return jr.build_jev_payload(jr.DEFAULT_MODEL, img, opts, expected, seed), (w, h)


def estimate_call(path: Path, roster: dict, colour: Optional[str]) -> int:
    from web import jev_reader as jr
    payload, wh = build_request(path, roster, colour, str(path))
    return jr.jev_estimate_tokens(payload, wh)


def ask_one(session, root: Path, crop: dict, roster: dict, roster_alliance: dict,
            colour: Optional[str]) -> dict:
    from web import jev_reader as jr
    from web.clef_reader import ClefUnavailable, _choice_probs
    rec = {"path": crop["path"], "t_sec": crop["t_sec"], "px_w": crop["px_w"],
           "px_h": crop["px_h"], "sharpness": round(crop["sharpness"], 1),
           "expected_alliance": colour, "probs": None, "error": None,
           "input_tokens": None, "tokens_estimated": False, "latency_s": None}
    try:
        payload, wh = build_request(root / crop["path"], roster, colour, crop["path"])
        type(session)._tl.usage = None
        t0 = time.monotonic()
        answers = session._request(payload, wh)
        rec["latency_s"] = round(time.monotonic() - t0, 3)
        usage = getattr(type(session)._tl, "usage", None) or {}
        tok = usage.get("input_tokens")
        if isinstance(tok, (int, float)):
            rec["input_tokens"] = int(tok)
        else:
            rec["input_tokens"] = jr.jev_estimate_tokens(payload, wh)
            rec["tokens_estimated"] = True
        probs = _choice_probs(answers.get("team"))
        if not probs:
            raise ValueError("no team probabilities in the answer")
        rec["probs"] = {k: round(float(v), 6) for k, v in probs.items()}
    except ClefUnavailable as e:
        rec["error"] = str(e)
    except Exception as e:
        session._bump("errors")
        rec["error"] = session._redact(f"jev: {type(e).__name__}: {e}")[:300]
    rec.update(call_summary(rec["probs"] or {}, roster_alliance))
    return rec


def finish_match(mdir: Path, info: dict, b: dict, r: dict, chosen: dict, calls: Dict[str, dict]
                 ) -> dict:
    """Decisions + labels_jev.jsonl + labels_jev_summary.json from cached calls."""
    roster_alliance = roster_alliance_of(info["roster"])
    labels: Dict[str, dict] = {}
    out_rows = []
    for tid in sorted(b, key=lambda t: int(t[1:]) if t[1:].isdigit() else t):
        rows = b[tid]
        colour = rows[0].get("alliance_colour")
        cs = [calls[c["path"]] for c in chosen[tid] if c["path"] in calls]
        d = decide(cs, roster_alliance, colour)
        labels[tid] = d
        out_rows.append({"detector": "bumper", "tracklet_id": tid, "colour": colour,
                         "tracklet_len": rows[0].get("tracklet_len"), "n_crops": len(rows),
                         "calls": cs, **d})
    for tid in sorted(r, key=lambda t: int(t[1:]) if t[1:].isdigit() else t):
        rows = r[tid]
        linked = rows[0].get("linked_bumper_tracklets") or []
        d = robot_decision(linked, labels)
        if d["team"] is not None:
            col, al = rows[0].get("alliance_colour"), roster_alliance.get(d["team"])
            d["off_alliance"] = col in ("red", "blue") and al in ("red", "blue") and al != col
        out_rows.append({"detector": "robot", "tracklet_id": tid, "colour": rows[0].get("alliance_colour"),
                         "tracklet_len": rows[0].get("tracklet_len"), "n_crops": len(rows),
                         "linked_bumper_tracklets": linked, "calls": [], **d})
    tmp = mdir / (LABELS + ".tmp")
    with open(tmp, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    summ = summarize(info, out_rows, list(calls.values()))
    with open(mdir / SUMMARY, "w") as f:
        json.dump(summ, f, indent=1)
    os.replace(tmp, mdir / LABELS)
    return summ


def summarize(info: dict, rows: List[dict], calls: List[dict]) -> dict:
    br = [x for x in rows if x["detector"] == "bumper"]
    rr = [x for x in rows if x["detector"] == "robot"]
    ok = [c for c in calls if not c.get("error")]
    lat = [c["latency_s"] for c in ok if c.get("latency_s") is not None]
    tok = sum(c.get("input_tokens") or 0 for c in ok)
    from web import jev_reader as jr
    usd = tok * jr.JEV_MODELS[jr.DEFAULT_MODEL]["usd_per_m_input"] / 1e6
    crops = sum(x["n_crops"] for x in br)
    crops_team = sum(x["n_crops"] for x in br if x["decision"] == "team")
    return {
        "match_key": info["match_key"], "event": info["event"], "model": jr.DEFAULT_MODEL,
        "roster": info["roster"],
        "bumper_tracklets": len(br),
        "bumper_decisions": dict(Counter(x["decision"] for x in br)),
        "bumper_off_alliance": sum(bool(x.get("off_alliance")) for x in br),
        "bumper_crops": crops, "bumper_crops_team": crops_team,
        "bumper_crop_team_coverage": round(crops_team / crops, 4) if crops else None,
        "teams_named": dict(Counter(x["team"] for x in br if x["decision"] == "team")),
        "robot_tracklets": len(rr),
        "robot_decisions": dict(Counter(x["decision"] for x in rr)),
        "robot_crops": sum(x["n_crops"] for x in rr),
        "robot_crops_team": sum(x["n_crops"] for x in rr if x["decision"] == "team"),
        "calls": len(calls), "call_errors": len(calls) - len(ok),
        "input_tokens": tok, "tokens_estimated_calls": sum(bool(c.get("tokens_estimated")) for c in ok),
        "cost_usd": round(usd, 5),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_env() -> None:
    from dotenv import load_dotenv
    for p in (MAIN / ".env", REPO / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="harvest out root")
    ap.add_argument("--events", help="comma list of event keys (default: all under --root)")
    ap.add_argument("--matches", help="comma list of match keys")
    ap.add_argument("--dry-run", action="store_true", help="count requests + estimate cost, no API calls")
    ap.add_argument("--max-usd", type=float, default=3.0, help="stop cleanly before exceeding this spend")
    ap.add_argument("--force", action="store_true", help="relabel matches that have labels (reuses cached calls)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    events = [e for e in (args.events or "").split(",") if e] or None
    matches = [m for m in (args.matches or "").split(",") if m] or None
    dirs = match_dirs(args.root, events, matches)
    if not dirs:
        print(f"[label_jev] no harvested matches under {args.root}", flush=True)
        return 1

    from web import jev_reader as jr
    usd_per_tok = jr.JEV_MODELS[jr.DEFAULT_MODEL]["usd_per_m_input"] / 1e6
    session = None
    spent = 0.0
    totals = Counter()
    for mdir in dirs:
        if (mdir / LABELS).exists() and not args.force and not args.dry_run:
            print(f"[label_jev] {mdir.name}: labelled, skipping (--force to redo)", flush=True)
            continue
        info, b, r, chosen = plan_match(mdir, args.root)
        roster = info["roster"]
        if not roster:
            print(f"[label_jev] {mdir.name}: no roster in manifest, skipping", flush=True)
            continue
        ra = roster_alliance_of(roster)
        cached = {c["path"]: c for c in read_jsonl(mdir / CALLS) if not c.get("error")}
        todo = [(tid, c) for tid in chosen for c in chosen[tid] if c["path"] not in cached]
        n_req = sum(len(v) for v in chosen.values())
        if args.dry_run:
            est = sum(estimate_call(args.root / c["path"], roster, b[tid][0].get("alliance_colour"))
                      for tid, c in todo)
            totals["requests"] += n_req
            totals["todo"] += len(todo)
            totals["tokens"] += est
            totals["tracklets"] += len(chosen)
            print(f"[label_jev] {mdir.name}: {len(chosen)} bumper tracklets, {n_req} requests "
                  f"({len(todo)} not cached), ~{est / 1e3:.0f}k tokens, ~${est * usd_per_tok:.4f}",
                  flush=True)
            continue
        if session is None:
            load_env()
            session = _UsageJev.make(timeout_s=30.0, retries=2, concurrency=args.concurrency)
            if not session.available:
                print(f"[label_jev] Jev unavailable: {session.disabled_reason}", flush=True)
                return 1
        t0 = time.time()
        stopped = False
        with open(mdir / CALLS, "a") as cf:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                est = sum(estimate_call(args.root / c["path"], roster, b[tid][0].get("alliance_colour"))
                          for tid, c in chunk) * usd_per_tok
                if spent + est > args.max_usd:
                    print(f"[label_jev] budget: ${spent:.4f} spent + ~${est:.4f} next chunk > "
                          f"--max-usd {args.max_usd}; stopping (calls so far cached)", flush=True)
                    stopped = True
                    break
                res = session._run(chunk, lambda _i, tc: ask_one(
                    session, args.root, tc[1], roster, ra, b[tc[0]][0].get("alliance_colour")))
                for rec in res:
                    spent += (rec.get("input_tokens") or 0) * usd_per_tok if not rec.get("error") else 0.0
                    cf.write(json.dumps(rec) + "\n")
                    if not rec.get("error"):
                        cached[rec["path"]] = rec
                cf.flush()
                if not session.available:
                    print(f"[label_jev] Jev latched off: {session.disabled_reason}", flush=True)
                    stopped = True
                    break
        # failed calls stay out of ``cached`` -> decision unsure (call_error) needs the record
        errs = {c["path"]: c for c in read_jsonl(mdir / CALLS) if c.get("error")}
        if stopped and any(c["path"] not in cached for v in chosen.values() for c in v):
            print(f"[label_jev] {mdir.name}: incomplete, not writing labels; total ${spent:.4f}", flush=True)
            break
        calls = {**{p: e for p, e in errs.items() if p not in cached}, **cached}
        s = finish_match(mdir, info, b, r, chosen, calls)
        print(f"[label_jev] {mdir.name}: {s['bumper_decisions']} off_alliance {s['bumper_off_alliance']}, "
              f"robots {s['robot_decisions']}, crop coverage {s['bumper_crop_team_coverage']}, "
              f"{s['calls']} calls ({s['call_errors']} err), ${s['cost_usd']:.4f}, "
              f"{time.time() - t0:.0f}s; run total ${spent:.4f}", flush=True)
        if stopped:
            break
    if args.dry_run:
        print(f"[label_jev] DRY RUN: {len(dirs)} matches, {totals['tracklets']} bumper tracklets, "
              f"{totals['requests']} requests ({totals['todo']} not cached), "
              f"~{totals['tokens'] / 1e6:.2f}M tokens, ~${totals['tokens'] * usd_per_tok:.3f}", flush=True)
    elif session is not None:
        print(f"[label_jev] {session.summary_line()}; this run ~${spent:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
