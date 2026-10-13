"""Harvest a stratified, deduped bumper-crop bench from past job diagnostics.

Every analysis job writes ``web/uploads/diagnostic_<job_id>.jsonl``; each
``{"type": "vlm_read"}`` row carries the exact JPEG crop the bumper reader saw
(``crop_b64``) plus the incumbent reader's verdict, and ``vlm_filtered`` rows
carry crops rejected below the legibility floor. ``job_id == replays.id`` in
``frc_scorer.db``; ``replays.match_key`` joins ``matches.tba_match_key`` ->
``match_robots`` for the 6-team roster (``replays.blue_teams/red_teams`` are
per-frame label arrays, NOT the roster, and are ignored here).

The bench lands in ``data/eval/bumper_crop_bench/`` (tracked in git, unlike the
lost July temp-dir bench):

  * ``crops/<id>.jpg``  — the decoded ``crop_b64`` bytes, written as-is
  * ``manifest.jsonl``  — one row per crop: provenance, roster, the incumbent
    reader's output as ``baseline_*``, and empty ``label_*`` fields

Strata (default target 200): ~50% ``matched`` (baseline matched a roster
team), ~35% ``null`` (baseline read nothing / no roster match), ~15%
``filtered`` (``vlm_filtered`` crops). Rows dedupe on
``(match_key, frame, robot_id)`` (many jobs re-run qm8-qm16), each stratum's
quota is spread evenly over matches, and within a match picks are spread over
the video (quantile bins on ``frame`` — ``t_ms`` is processing wall time, not
match time). Deterministic for a given ``--seed`` and set of diagnostics.

Re-running preserves labels already entered for re-selected ids, and refuses to
drop labelled rows unless ``--force``.

Usage
-----
    .venv/bin/python scripts/build_bumper_crop_bench.py
    .venv/bin/python scripts/build_bumper_crop_bench.py --target 300 --seed 7
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "frc_scorer.db"
UPLOADS_DIR = REPO / "web" / "uploads"
DEFAULT_OUT = REPO / "data" / "eval" / "bumper_crop_bench"

STRATA = ("matched", "null", "filtered")
STRATUM_SHARES = {"matched": 0.50, "null": 0.35, "filtered": 0.15}
LABEL_STATUSES = ("unlabelled", "legible", "illegible", "not_bumper")
LABEL_FIELDS = ("label_team", "label_alliance", "label_status", "label_note")

_READ_TAG = b'"vlm_read"'
_FILT_TAG = b'"vlm_filtered"'
_CROP_KEY = re.compile(rb'"crop_b64"\s*:\s*"')


# --------------------------------------------------------------------------- #
# DB: job -> match, match -> roster
# --------------------------------------------------------------------------- #

def load_rosters(db_path: Path) -> dict[str, dict[str, list[str]]]:
    """match_key -> {"red": [3 teams], "blue": [3 teams]} (position order).

    Only complete 3v3 rosters are returned; anything else is treated as no
    roster (the job is skipped)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT m.tba_match_key, mr.alliance, mr.position, mr.team_number "
            "FROM matches m JOIN match_robots mr ON mr.match_id = m.id "
            "ORDER BY m.tba_match_key, mr.alliance, mr.position").fetchall()
    finally:
        con.close()
    rosters: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"red": [], "blue": []})
    for key, alliance, _pos, team in rows:
        rosters[key][alliance].append(str(team).strip())
    return {k: v for k, v in rosters.items()
            if len(v["red"]) == 3 and len(v["blue"]) == 3}


def load_job_matches(db_path: Path) -> dict[str, str]:
    """replays.id (== job_id) -> match_key, for rows with a non-empty key."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, match_key FROM replays").fetchall()
    finally:
        con.close()
    return {jid: mk for jid, mk in rows if jid and mk}


# --------------------------------------------------------------------------- #
# Streaming scan of the diagnostics
# --------------------------------------------------------------------------- #

def _strip_crop(line: bytes) -> tuple[bytes, int]:
    """Cut the (large) crop_b64 value out of a raw JSON line before parsing.

    Returns (line_without_crop, crop_len). base64 never contains a quote, so
    the value ends at the next '"'."""
    m = _CROP_KEY.search(line)
    if m is None:
        return line, 0
    j = m.end()
    k = line.find(b'"', j)
    if k < 0:
        return line, 0
    return line[:j] + line[k:], k - j


def classify_row(row: dict) -> Optional[str]:
    """Stratum for one diagnostic row (crop presence checked by the caller)."""
    t = row.get("type")
    if t == "vlm_read":
        return "matched" if row.get("matched_team") else "null"
    if t == "vlm_filtered":
        return "filtered"
    return None


def scan_diagnostic(path: Path, job_id: str, match_key: str) -> Iterator[dict]:
    """Yield lightweight candidate dicts (no crop bytes) for one diagnostic.

    Streams line by line in binary mode; only lines carrying a vlm_read /
    vlm_filtered tag are JSON-parsed, with the crop payload cut out first. The
    byte offset is kept so the crop can be re-read for the few rows selected."""
    offset = 0
    with open(path, "rb") as fh:
        for line in fh:
            start = offset
            offset += len(line)
            if _READ_TAG not in line and _FILT_TAG not in line:
                continue
            slim, crop_len = _strip_crop(line)
            if crop_len == 0:
                continue  # no crop (e.g. imencode failed on a filtered row)
            try:
                row = json.loads(slim)
            except ValueError:
                continue
            stratum = classify_row(row)
            if stratum is None or row.get("frame") is None or row.get("robot_id") is None:
                continue
            yield {
                "job_id": job_id,
                "match_key": match_key,
                "path": str(path),
                "offset": start,
                "stratum": stratum,
                "frame": int(row["frame"]),
                "t_ms": row.get("t_ms"),
                "robot_id": int(row["robot_id"]),
                "raw_read": row.get("raw_read"),
                "matched_team": row.get("matched_team"),
                "alliance_read": row.get("alliance_read"),
                "confidence": row.get("confidence"),
                "legibility": row.get("legibility"),
                "surface": row.get("surface"),
                "filter_reason": row.get("reason"),
            }


def read_crop_b64(path: str, offset: int) -> str:
    """Re-read one selected row's crop_b64 by byte offset."""
    with open(path, "rb") as fh:
        fh.seek(offset)
        line = fh.readline()
    return json.loads(line)["crop_b64"]


# --------------------------------------------------------------------------- #
# Selection (pure; unit-tested on synthetic candidates)
# --------------------------------------------------------------------------- #

def _stable_int(*parts) -> int:
    """Process-independent hash (Python's str hash is salted per run)."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def dedupe_key(c: dict) -> tuple:
    return (c["match_key"], int(c["frame"]), int(c["robot_id"]))


def dedupe_candidates(cands: Iterable[dict], seed: int) -> list[dict]:
    """Keep ONE candidate per (match_key, frame, robot_id).

    Re-runs of the same match produce the same key many times; the survivor is
    chosen by a seeded stable hash of (job_id, offset), so the result does not
    depend on file scan order. Output is sorted by key."""
    best: dict[tuple, tuple[int, dict]] = {}
    for c in cands:
        k = dedupe_key(c)
        h = _stable_int(seed, c["job_id"], c.get("offset", 0), c["stratum"])
        cur = best.get(k)
        if cur is None or h < cur[0]:
            best[k] = (h, c)
    return [best[k][1] for k in sorted(best)]


def stratum_targets(total: int, shares: dict = STRATUM_SHARES) -> dict[str, int]:
    """Split ``total`` over strata by share (largest-remainder rounding)."""
    raw = {s: total * shares[s] for s in STRATA}
    out = {s: int(raw[s]) for s in STRATA}
    rem = total - sum(out.values())
    for s in sorted(STRATA, key=lambda s: (-(raw[s] - out[s]), STRATA.index(s)))[:rem]:
        out[s] += 1
    return out


def allocate_quota(capacity: dict[str, int], total: int, rng: random.Random) -> dict[str, int]:
    """Spread ``total`` picks over keys as evenly as capacity allows.

    Round-robin (key order shuffled once by ``rng``) giving one pick per pass
    to every key with spare capacity — water-filling, so a match that runs out
    hands its share to the others."""
    keys = sorted(k for k, v in capacity.items() if v > 0)
    rng.shuffle(keys)
    alloc = {k: 0 for k in keys}
    left = min(total, sum(capacity[k] for k in keys))
    while left > 0:
        progressed = False
        for k in keys:
            if left == 0:
                break
            if alloc[k] < capacity[k]:
                alloc[k] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return {k: v for k, v in alloc.items() if v > 0}


def spread_pick(cands: list[dict], q: int, rng: random.Random,
                diversity_key: Optional[str] = None) -> list[dict]:
    """Pick ``q`` candidates spread over the match video.

    Candidates are sorted by frame and split into ``q`` equal-count quantile
    bins; one pick per bin. With ``diversity_key`` (e.g. matched_team), each
    bin prefers values picked least so far, so one robot can't fill the
    stratum for a match."""
    if q <= 0 or not cands:
        return []
    ordered = sorted(cands, key=lambda c: (c["frame"], c["robot_id"], c["job_id"]))
    if q >= len(ordered):
        return ordered
    n = len(ordered)
    used: Counter = Counter()
    picks = []
    for b in range(q):
        lo, hi = (b * n) // q, ((b + 1) * n) // q
        bin_ = ordered[lo:hi]
        if diversity_key:
            least = min(used[c.get(diversity_key)] for c in bin_)
            bin_ = [c for c in bin_ if used[c.get(diversity_key)] == least]
        pick = bin_[rng.randrange(len(bin_))]
        if diversity_key:
            used[pick.get(diversity_key)] += 1
        picks.append(pick)
    return picks


def select_bench(cands: list[dict], target: int, seed: int) -> list[dict]:
    """Dedupe, then stratify by stratum x match with time-spread picks.

    A stratum short of candidates hands its shortfall to the others (matched
    first, then null) so the bench still reaches ``target`` when possible."""
    deduped = dedupe_candidates(cands, seed)
    by_sm: dict[str, dict[str, list[dict]]] = {s: defaultdict(list) for s in STRATA}
    for c in deduped:
        by_sm[c["stratum"]][c["match_key"]].append(c)

    targets = stratum_targets(target)
    avail = {s: sum(len(v) for v in by_sm[s].values()) for s in STRATA}
    short = 0
    for s in STRATA:
        if targets[s] > avail[s]:
            short += targets[s] - avail[s]
            targets[s] = avail[s]
    for s in STRATA:  # hand any shortfall to strata with spare candidates
        take = min(short, avail[s] - targets[s])
        targets[s] += take
        short -= take

    picks: list[dict] = []
    for s in STRATA:
        rng = random.Random(_stable_int(seed, "alloc", s))
        alloc = allocate_quota({m: len(v) for m, v in by_sm[s].items()}, targets[s], rng)
        for m in sorted(alloc):
            prng = random.Random(_stable_int(seed, "pick", s, m))
            picks.extend(spread_pick(by_sm[s][m], alloc[m], prng,
                                     diversity_key="matched_team" if s == "matched" else None))
    picks.sort(key=lambda c: (c["match_key"], c["frame"], c["robot_id"]))
    return picks


def make_crop_id(match_key: str, frame: int, robot_id: int) -> str:
    """Stable, human-readable id; unique because it IS the dedupe key."""
    return f"{match_key}_f{int(frame):05d}_r{int(robot_id):03d}"


def manifest_row(c: dict, roster: dict, width: int, height: int,
                 sha1: str) -> dict:
    cid = make_crop_id(c["match_key"], c["frame"], c["robot_id"])
    return {
        "id": cid,
        "file": f"crops/{cid}.jpg",
        "job_id": c["job_id"],
        "match_key": c["match_key"],
        "frame": c["frame"],
        "t_ms": c["t_ms"],
        "robot_id": c["robot_id"],
        "width": width,
        "height": height,
        "sha1": sha1,
        "stratum": c["stratum"],
        "red_teams": list(roster["red"]),
        "blue_teams": list(roster["blue"]),
        "baseline_raw_read": c["raw_read"],
        "baseline_matched_team": c["matched_team"],
        "baseline_alliance_read": c["alliance_read"],
        "baseline_confidence": c["confidence"],
        "baseline_legibility": c["legibility"],
        "baseline_surface": c["surface"],
        "baseline_filter_reason": c["filter_reason"],
        "label_team": None,
        "label_alliance": None,
        "label_status": "unlabelled",
        "label_note": None,
    }


def is_labelled(row: dict) -> bool:
    return (row.get("label_status") or "unlabelled") != "unlabelled"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _job_id_from_path(p: Path) -> str:
    return p.name[len("diagnostic_"):-len(".jsonl")]


def harvest(uploads: Path, job_matches: dict[str, str], rosters: dict) -> tuple[list[dict], dict]:
    files = sorted(uploads.glob("diagnostic_*.jsonl"))
    stats = Counter()
    cands: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(files):
        jid = _job_id_from_path(p)
        mk = job_matches.get(jid)
        if mk is None:
            stats["skip_no_replay"] += 1
            continue
        if mk not in rosters:
            stats["skip_no_roster"] += 1
            continue
        n0 = len(cands)
        cands.extend(scan_diagnostic(p, jid, mk))
        stats["files_scanned"] += 1
        if len(cands) > n0:
            stats["files_with_reads"] += 1
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(files)} files, {len(cands)} candidates "
                  f"({time.time() - t0:.0f}s)", flush=True)
    stats["candidates"] = len(cands)
    return cands, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=200, help="crops to sample (default 200)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="bench dir (default data/eval/bumper_crop_bench)")
    ap.add_argument("--uploads", default=str(UPLOADS_DIR), help="diagnostics dir")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--force", action="store_true",
                    help="allow dropping labelled rows that the new selection no longer includes")
    args = ap.parse_args()

    out = Path(args.out)
    manifest_path = out / "manifest.jsonl"
    crops_dir = out / "crops"

    existing: dict[str, dict] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["id"]] = r

    rosters = load_rosters(Path(args.db))
    job_matches = load_job_matches(Path(args.db))
    print(f"rosters: {len(rosters)} matches; replays with match_key: {len(job_matches)}")

    cands, stats = harvest(Path(args.uploads), job_matches, rosters)
    print(f"scan: {dict(stats)}")
    raw_by_stratum = Counter(c["stratum"] for c in cands)
    print(f"raw candidates by stratum: {dict(raw_by_stratum)}")

    picks = select_bench(cands, args.target, args.seed)
    new_ids = {make_crop_id(c["match_key"], c["frame"], c["robot_id"]) for c in picks}
    dropped_labelled = [i for i, r in existing.items() if is_labelled(r) and i not in new_ids]
    if dropped_labelled and not args.force:
        sys.exit(f"refusing: {len(dropped_labelled)} labelled rows would be dropped "
                 f"(e.g. {dropped_labelled[:3]}); keep --seed/--target or pass --force")

    from PIL import Image  # local: keep the pure helpers importable without PIL
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in picks:
        jpg = base64.b64decode(read_crop_b64(c["path"], c["offset"]))
        with Image.open(io.BytesIO(jpg)) as im:
            w, h = im.size
        row = manifest_row(c, rosters[c["match_key"]], w, h, hashlib.sha1(jpg).hexdigest())
        prev = existing.get(row["id"])
        if prev is not None:  # keep human labels across rebuilds
            for f in LABEL_FIELDS:
                if f in prev:
                    row[f] = prev[f]
        (out / row["file"]).write_bytes(jpg)
        rows.append(row)

    stale = [p for p in crops_dir.glob("*.jpg") if p.stem not in new_ids]
    for p in stale:
        p.unlink()

    with open(manifest_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    by_stratum = Counter(r["stratum"] for r in rows)
    by_match = defaultdict(Counter)
    for r in rows:
        by_match[r["match_key"]][r["stratum"]] += 1
    total_bytes = sum((out / r["file"]).stat().st_size for r in rows)
    print(f"\nwrote {len(rows)} crops to {out} ({total_bytes / 1e6:.1f} MB, "
          f"{len(by_match)} matches, {len(stale)} stale crops removed)")
    print(f"by stratum: {dict(by_stratum)}")
    print(f"{'match':<16} {'matched':>7} {'null':>5} {'filtered':>8} {'total':>5}")
    for m in sorted(by_match):
        cnt = by_match[m]
        print(f"{m:<16} {cnt['matched']:>7} {cnt['null']:>5} {cnt['filtered']:>8} "
              f"{sum(cnt.values()):>5}")
    kept = sum(1 for r in rows if is_labelled(r))
    if kept:
        print(f"preserved labels on {kept} rows")


if __name__ == "__main__":
    main()
