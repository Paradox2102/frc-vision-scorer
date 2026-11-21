"""Join our 2026cur scouting metrics (cached :8004 API responses) with Manta (3128) scouting
and (if available) Statbotics team_events. Writes data/teams.json + prints correlations.
Manta decoding reused from the Nov 16 manta_corr.py session."""
import json, glob, re, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EV = "2026cur"
SPEED = re.compile(
    r"\b(?P<a>very |really |rlly |kinda |decently |pretty )?(?P<w>fast|slow|average)\s+(?:speed\s+)?(?:dt|drivetrain|drive ?train|drivetrian|bot|robot|speed)\b"
    r"|\b(?:dt|drivetrain|drivetrian|bot|robot|speed)\s+(?:is\s+|was\s+)?(?:very |really |rlly |kinda |decently |pretty )?(?P<w2>fast|slow)\b"
    r"|\b(?P<w3>average) speed\b", re.I)
BEACH = re.compile(r"beach|stuck on (?:the )?bump|high.?cent", re.I)

# ---- Manta per (match, team)
manta = defaultdict(list)
for f in glob.glob(f"{REPO}/data/manta_cache/{EV}_qual_*.json"):
    d = json.load(open(f))
    for r in d.get("robots") or []:
        for s in r["scouts"]:
            manta[(d["match"]["number"], int(r["team_number"]))].append(s["data"])

def manta_row(recs):
    num = lambda k: float(np.mean([x.get(k) or 0 for x in recs]))
    txt = " ".join((x.get("qata") or "") + " " + (x.get("defense_qata") or "") for x in recs)
    sp = []
    for m in SPEED.finditer(txt):
        w = (m.group("w") or m.group("w2") or m.group("w3")).lower()
        sp.append({"fast": 1, "slow": -1, "average": 0}[w])
    return {"attacking": num("attacking_time"), "oof": num("oof_time"),
            "scoring": num("tele_scoring"), "feeding": num("tele_feeding"),
            "auto_scoring": num("auto_scoring"), "tele_climb": num("tele_climb"),
            "speed": float(np.mean(sp)) if sp else None, "beached": float(bool(BEACH.search(txt))),
            "notes": [((x.get("qata") or "") + " || " + (x.get("defense_qata") or "")).replace("\n", " / ") for x in recs]}
M = {k: manta_row(v) for k, v in manta.items()}

# ---- ours per (match, team), counted qm only
FIELDS = {
    "movement": {"avg_speed": "avg_speed_mps", "moving_frac": "moving_frac"},
    "cycles": {"cyc_per_min": "cycles_per_min", "cyc_tele": "cycles_teleop", "cyc_auto": "cycles_auto",
               "tele_s": "teleop_tracked_s"},
    "defense": {"opp_frac": "opp_side_frac"},
    "bump_stuck": {"stuck_s": "stuck_s"},
}
TEAMSCORE = {"movement": ["avg_speed_mps", "moving_frac", "coverage", "n_counted"],
             "cycles": ["cycles_per_min", "cycles_auto_per_match", "cycles_teleop_per_match", "coverage", "n_counted"],
             "defense": ["opp_side_frac", "coverage", "n_counted"],
             "bump_stuck": ["stuck_s_per_match", "coverage", "n_counted"]}
O, T = {}, defaultdict(dict)
for key, fields in FIELDS.items():
    env = json.load(open(f"{D}/ours_{key}.json"))
    for t, tm in env["teams"].items():
        for f in TEAMSCORE[key]:
            T[int(t)][f"{key}.{f}"] = tm.get(f)
        for r in tm["matches"]:
            if r.get("counted") and r.get("comp_level") == "qm":
                O.setdefault((r["match_number"], int(t)), {}).update({k: r.get(v) for k, v in fields.items()})

joined = {k: (O[k], M[k]) for k in O if k in M}
print(f"joined (match, team) pairs: {len(joined)}  (ours counted qm {len(O)}, manta {len(M)})")
print(f"teams: ours {len(T)}, manta {len({t for _, t in M})}, joined {len({t for _, t in joined})}")

def per_team(of, mf, min_n=3):
    per = defaultdict(lambda: ([], []))
    for (m, t), (o, mm) in joined.items():
        a, b = o.get(of), mm.get(mf)
        if a is None or b is None:
            continue
        per[t][0].append(a); per[t][1].append(b)
    return {t: (float(np.mean(a)), float(np.mean(b)), len(a)) for t, (a, b) in per.items() if len(a) >= min_n}

# ---- team table
teams = {}
for t in T:
    teams[t] = dict(T[t])
for of, mf, name in [("avg_speed", "speed", "speed"), ("moving_frac", "oof", "oof"),
                     ("cyc_per_min", "scoring", "scoring"), ("cyc_per_min", "feeding", "feeding"),
                     ("opp_frac", "attacking", "attacking"), ("stuck_s", "beached", "beached"),
                     ("cyc_auto", "auto_scoring", "auto_scoring")]:
    for t, (a, b, n) in per_team(of, mf).items():
        teams.setdefault(t, {})[f"joined.{of}|{name}"] = a
        teams[t][f"manta.{name}"] = b
        teams[t][f"manta.{name}.n"] = n

# ---- Statbotics
sbp = f"{D}/statbotics_team_events_2026cur.json"
if os.path.exists(sbp) and os.path.getsize(sbp) > 1000:
    sb = json.load(open(sbp))
    for e in sb:
        t = int(e["team"]); ep = e.get("epa") or {}
        bd = ep.get("breakdown") or {}
        rec = (e.get("record") or {}).get("qual") or e.get("record") or {}
        num = lambda v: v.get("mean") if isinstance(v, dict) else v
        teams.setdefault(t, {}).update({
            "sb.total": num(bd.get("total_points", ep.get("total_points"))), "sb.auto": num(bd.get("auto_points")),
            "sb.teleop": num(bd.get("teleop_points")), "sb.endgame": num(bd.get("endgame_points")),
            "sb.rank": rec.get("rank"), "sb.winrate": rec.get("winrate"),
            "sb.bd": bd})
json.dump({str(k): v for k, v in teams.items()}, open(f"{D}/teams.json", "w"), indent=1)
json.dump({f"{m}_{t}": M[(m, t)]["notes"] for (m, t) in M}, open(f"{D}/manta_notes.json", "w"), indent=0)

def rho(xk, yk):
    pts = [(v[xk], v[yk]) for v in teams.values() if v.get(xk) is not None and v.get(yk) is not None]
    if len(pts) < 5:
        return None, len(pts)
    a, b = zip(*pts)
    return spearmanr(a, b)[0], len(pts)

print("\n-- joined-match means (prior session method)")
for x, y in [("joined.avg_speed|speed", "manta.speed"), ("joined.moving_frac|oof", "manta.oof"),
             ("joined.cyc_per_min|scoring", "manta.scoring"), ("joined.cyc_per_min|feeding", "manta.feeding"),
             ("joined.opp_frac|attacking", "manta.attacking"), ("joined.stuck_s|beached", "manta.beached"),
             ("joined.cyc_auto|auto_scoring", "manta.auto_scoring")]:
    r, n = rho(x, y); print(f"  {x:32s} vs {y:20s} rho {r:+.2f} n {n}")
print("\n-- app team scores (x) vs Manta team means (y)")
for x, y in [("movement.avg_speed_mps", "manta.speed"), ("movement.moving_frac", "manta.oof"),
             ("cycles.cycles_per_min", "manta.scoring"), ("cycles.cycles_per_min", "manta.feeding"),
             ("defense.opp_side_frac", "manta.attacking"), ("bump_stuck.stuck_s_per_match", "manta.beached"),
             ("cycles.cycles_auto_per_match", "manta.auto_scoring")]:
    r, n = rho(x, y); print(f"  {x:32s} vs {y:20s} rho {r:+.2f} n {n}")
if any("sb.total" in v for v in teams.values()):
    print("\n-- vs Statbotics")
    for x in ["cycles.cycles_per_min", "cycles.cycles_teleop_per_match", "cycles.cycles_auto_per_match",
              "movement.avg_speed_mps", "movement.moving_frac", "defense.opp_side_frac", "bump_stuck.stuck_s_per_match",
              "manta.scoring", "manta.attacking"]:
        for y in ["sb.total", "sb.auto", "sb.teleop", "sb.endgame"]:
            r, n = rho(x, y)
            if r is not None:
                print(f"  {x:32s} vs {y:12s} rho {r:+.2f} n {n}")
