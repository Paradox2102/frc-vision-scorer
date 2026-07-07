"""Grade observe-only VLM equipment-adjudication diagnostics from one job.

Reads `adjudication` records (verdict + per-identity equipment-streak) and
summarizes per slot: verdict distribution and max equipment/person streak. A
slot the tracker still holds while the VLM repeatedly calls it equipment or
person is a candidate hijack the signal would flag — exactly the
detector-corroborated case the absence culls can't catch.

No ground-truth join here; this is the signal summary the analyst reads
alongside the Manta scores and the HUD frames.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    adj = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "adjudication":
            adj.append(r)
    return adj


def grade(path: Path) -> dict:
    adj = load(path)
    by_slot_verdicts: dict = defaultdict(Counter)
    max_streak: dict = defaultdict(int)
    for r in adj:
        sid = r["robot_id"]
        by_slot_verdicts[sid][r["verdict"]] += 1
        max_streak[sid] = max(max_streak[sid], int(r.get("streak", 0)))
    per_slot = {
        str(sid): {
            "verdicts": dict(by_slot_verdicts[sid]),
            "max_equipment_streak": max_streak[sid],
        }
        for sid in sorted(by_slot_verdicts)
    }
    flagged = [str(sid) for sid, s in max_streak.items() if s >= 2]
    return {
        "path": str(path),
        "records": len(adj),
        "slots_seen": len(by_slot_verdicts),
        "flagged_as_equipment_streak>=2": flagged,
        "per_slot": per_slot,
    }


if __name__ == "__main__":
    p = Path(sys.argv[1])
    print(json.dumps(grade(p), indent=2))
