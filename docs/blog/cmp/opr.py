"""Fallback when Statbotics is down: component OPR (least squares) from cached TBA qual matches."""
import json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ms = [m for m in json.load(open(f"{D}/tba_matches_2026cur.json")) if m["comp_level"] == "qm" and m.get("score_breakdown")]
teams = sorted({int(k[3:]) for m in ms for a in ("red", "blue") for k in m["alliances"][a]["team_keys"]})
idx = {t: i for i, t in enumerate(teams)}
comps = {"auto": lambda b: b["totalAutoPoints"], "teleop": lambda b: b["totalTeleopPoints"],
         "endgame": lambda b: b["endGameTowerPoints"], "total": lambda b: b["totalPoints"] - b["foulPoints"]}
A, Y = [], {k: [] for k in comps}
for m in ms:
    for a in ("red", "blue"):
        row = np.zeros(len(teams))
        for k in m["alliances"][a]["team_keys"]:
            row[idx[int(k[3:])]] = 1
        A.append(row)
        for k, f in comps.items():
            Y[k].append(f(m["score_breakdown"][a]))
A = np.array(A)
out = {t: {} for t in teams}
for k in comps:
    x, *_ = np.linalg.lstsq(A, np.array(Y[k], float), rcond=None)
    for t in teams:
        out[t][k] = float(x[idx[t]])
json.dump({str(t): v for t, v in out.items()}, open(f"{D}/tba_opr_2026cur.json", "w"), indent=1)
print(len(ms), "matches", len(teams), "teams; sample", list(out.items())[:2])
b = ms[0]["score_breakdown"]["red"]; print("check teleop+auto+endgame vs total:", b["totalAutoPoints"] + b["totalTeleopPoints"] + b["endGameTowerPoints"], b["totalPoints"] - b["foulPoints"])
