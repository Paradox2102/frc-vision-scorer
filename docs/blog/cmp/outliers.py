"""Rank-space disagreements per pairing, with coverage and n."""
import json, os, sys
import numpy as np
from scipy.stats import rankdata
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
T = json.load(open(f"{D}/teams.json"))
opr = json.load(open(f"{D}/tba_opr_2026cur.json"))
for t, v in opr.items():
    T.setdefault(t, {}).update({f"opr.{k}": x for k, x in v.items()})
P = [("joined.opp_frac|attacking", "manta.attacking"), ("joined.cyc_per_min|scoring", "manta.scoring"),
     ("joined.avg_speed|speed", "manta.speed"), ("joined.stuck_s|beached", "manta.beached"),
     ("joined.cyc_auto|auto_scoring", "manta.auto_scoring"),
     ("cycles.cycles_per_min", "opr.teleop"), ("cycles.cycles_auto_per_match", "opr.auto"), ("movement.avg_speed_mps", "opr.total")]
sel = sys.argv[1:] or None
for x, y in P:
    if sel and not any(s in x + y for s in sel):
        continue
    ts = [t for t, v in T.items() if v.get(x) is not None and v.get(y) is not None]
    rx = rankdata([T[t][x] for t in ts]) / len(ts); ry = rankdata([T[t][y] for t in ts]) / len(ts)
    d = rx - ry
    o = np.argsort(-np.abs(d))[:8]
    print(f"\n== {x} vs {y}  (n {len(ts)})   + = we rank higher")
    for i in o:
        t = ts[i]; v = T[t]
        print(f"  {t:>5} d {d[i]:+.2f}  ours {v[x]:.3f} (pct {rx[i]:.2f})  them {v[y]:.2f} (pct {ry[i]:.2f})  "
              f"cov {v.get('movement.coverage')}  n_counted {v.get('movement.n_counted')}  nmanta {v.get(y + '.n')}")
