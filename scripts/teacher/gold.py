#!/usr/bin/env python3
"""Gold set for the teacher-distillation project: sample blind items, then score labellers.

  sample : stratified sample of harvested crops -> <gold>/items/gNNNN.jpg (upscaled copies),
           <gold>/items.jsonl (what judges may see: id, image, roster) and
           <gold>/key.jsonl (hidden: source crop, tracklet, Jev decision, strata).
  score  : join judge answers (<gold>/answers/*.jsonl) with key.jsonl and report Jev tracklet-label
           precision/coverage per stratum, plus judge-vs-judge agreement on overlap items.

Judges must only open items/*.jpg (see JUDGE_BRIEF.md). Usage:
  python scripts/teacher/gold.py sample --events 2025cur,2026cur --n 480 --gold data/teacher/gold/b1
  python scripts/teacher/gold.py score --gold data/teacher/gold/b1
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

REPO_DATA = Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    from scripts.teacher.harvest import main_checkout  # noqa: WPS433
    return main_checkout() / "data" / "teacher"


def size_bucket(px_w: int) -> str:
    return "s" if px_w < 50 else ("m" if px_w < 90 else "l")


def load_match(mdir: Path):
    rows = [json.loads(l) for l in (mdir / "manifest.jsonl").open()]
    labels = {}
    lf = mdir / "labels_jev.jsonl"
    if lf.exists():
        for l in lf.open():
            r = json.loads(l)
            labels[(r["detector"], r["tracklet_id"])] = r
    return rows, labels


def render(src: Path, dst: Path, min_short: int = 160, max_long: int = 512) -> None:
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    s = max(1.0, min_short / min(h, w))
    s = min(s, max_long / max(h, w)) if max(h, w) * s > max_long else s
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def cmd_sample(a) -> None:
    rng = random.Random(a.seed)
    root = _data_root() / "harvest_v2"
    gold = Path(a.gold)
    (gold / "items").mkdir(parents=True, exist_ok=True)
    # one candidate per tracklet: a random crop of it
    pools = defaultdict(list)  # stratum -> [(row, label)]
    for ev in a.events.split(","):
        for mdir in sorted((root / ev).iterdir()):
            if not (mdir / "manifest.jsonl").exists():
                continue
            rows, labels = load_match(mdir)
            by_tr = defaultdict(list)
            for r in rows:
                by_tr[(r["detector"], r["tracklet_id"])].append(r)
            for key, crops in by_tr.items():
                r = rng.choice(crops)
                lab = labels.get(key)
                dec = lab["decision"] if lab else "none"
                if r["detector"] == "robot":
                    dec = "team" if lab and lab.get("team") else "noteam"
                stratum = (ev, r["detector"], dec, size_bucket(r["px_w"]))
                pools[stratum].append((r, lab))
    # quotas: robot share, then split evenly across strata of each (event, detector)
    n_ev = len(a.events.split(","))
    per_ev = a.n // n_ev
    want = {"bumper": int(per_ev * (1 - a.robot_frac)), "robot": per_ev - int(per_ev * (1 - a.robot_frac))}
    picked = []
    for ev in a.events.split(","):
        for det, n in want.items():
            strata = [k for k in pools if k[0] == ev and k[1] == det]
            for st in strata:
                rng.shuffle(pools[st])
            # round-robin over strata so small strata are fully represented
            i, got = 0, 0
            while got < n and any(pools[st] for st in strata):
                st = strata[i % len(strata)]
                if pools[st]:
                    picked.append((st, *pools[st].pop()))
                    got += 1
                i += 1
    rng.shuffle(picked)
    with (gold / "items.jsonl").open("w") as fi, (gold / "key.jsonl").open("w") as fk:
        for n, (st, r, lab) in enumerate(picked):
            gid = f"g{n:04d}"
            render(root / r["path"], gold / "items" / f"{gid}.jpg")
            fi.write(json.dumps({"id": gid, "image": f"items/{gid}.jpg", "roster": r["roster"]}) + "\n")
            fk.write(json.dumps({
                "id": gid, "stratum": list(st), "path": r["path"], "event": r["event"],
                "match_key": r["match_key"], "detector": r["detector"], "tracklet_id": r["tracklet_id"],
                "px_w": r["px_w"], "colour": r.get("alliance_colour"),
                "jev_decision": (lab or {}).get("decision"), "jev_team": (lab or {}).get("team"),
                "crop_was_jev_called": bool(lab) and any(c.get("path") == r["path"] for c in lab.get("calls", [])),
            }) + "\n")
    print(f"{len(picked)} items -> {gold}")
    print(Counter((s[0], s[1], s[2]) for s, _, _ in picked))


def _answers(gold: Path) -> dict:
    out = defaultdict(dict)  # id -> judge -> answer
    for f in sorted((gold / "answers").glob("*.jsonl")):
        for l in f.open():
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            out[r["id"]][f.stem] = r
    return out


MIN_CONF = 0.8


def gold_team(ans: dict, min_conf: float = None):
    """A judge's team answer, or None when unsure / not a robot / below the confidence floor."""
    t = ans.get("team")
    floor = MIN_CONF if min_conf is None else min_conf
    if not t or t in ("unsure", "null", "none"):
        return None
    try:
        conf = float(ans.get("conf") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return t if conf >= floor else None


def cmd_score(a) -> None:
    global MIN_CONF
    MIN_CONF = a.min_conf
    gold = Path(a.gold)
    kpath = Path(a.key) if a.key else gold / "key.jsonl"
    key = {json.loads(l)["id"]: json.loads(l) for l in kpath.open()}
    ans = _answers(gold)
    print(f"items {len(key)}, answered {sum(1 for k in key if k in ans)}")
    # judge-vs-judge on overlap
    agree = Counter()
    for gid, js in ans.items():
        if len(js) > 1:
            ts = [gold_team(v) for v in js.values()]
            named = [t for t in ts if t]
            agree["overlap"] += 1
            if len(set(named)) > 1:
                agree["conflict"] += 1
            elif len(named) == len(ts):
                agree["both_named_same"] += 1
    if agree:
        print("judge overlap:", dict(agree))
    tab = defaultdict(Counter)
    kinds = Counter()
    for gid, k in key.items():
        if gid not in ans:
            continue
        js = list(ans[gid].values())
        named = {gold_team(v) for v in js} - {None}
        g = named.pop() if len(named) == 1 else ("conflict" if named else None)
        kinds[(k["detector"], js[0].get("kind"))] += 1
        st = (k["event"], k["detector"], size_bucket(k["px_w"]))
        jt = k["jev_team"]
        c = tab[st]
        c["n"] += 1
        c["gold_named"] += g not in (None, "conflict")
        if jt:
            c["jev_named"] += 1
            if g in (None, "conflict"):
                c["jev_named_gold_unsure"] += 1
            elif g == jt:
                c["jev_right"] += 1
            else:
                c["jev_WRONG"] += 1
        elif g not in (None, "conflict"):
            c["gold_named_jev_not"] += 1
    print("\nper stratum (event, detector, size): n / gold named / jev named: right, WRONG, gold-unsure / gold named but Jev not")
    tot = Counter()
    for st in sorted(tab):
        c = tab[st]
        tot.update(c)
        print(f"  {st}: {c['n']} / {c['gold_named']} / {c['jev_named']}: {c['jev_right']}, {c['jev_WRONG']}, "
              f"{c['jev_named_gold_unsure']} / {c['gold_named_jev_not']}")
    print(f"  TOTAL: {dict(tot)}")
    print("\njudge kinds:", dict(kinds))
    wrong = [(gid, key[gid]["path"], key[gid]["jev_team"], [gold_team(v) for v in ans[gid].values()])
             for gid in key if gid in ans and key[gid]["jev_team"]
             and {gold_team(v) for v in ans[gid].values()} - {None, key[gid]["jev_team"]}]
    for w in wrong:
        print("  JEV≠GOLD", w)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--events", default="2025cur,2026cur")
    s.add_argument("--n", type=int, default=480)
    s.add_argument("--robot-frac", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gold", required=True)
    s.set_defaults(fn=cmd_sample)
    c = sub.add_parser("score")
    c.add_argument("--gold", required=True)
    c.add_argument("--min-conf", type=float, default=0.8)
    c.add_argument("--key", default=None, help="key.jsonl path if kept outside the judges' folder")
    c.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
