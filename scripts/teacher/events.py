"""Default harvest scope + event-level splits for the teacher-distillation data
(design: docs/superpowers/specs/2026-11-10-teacher-distillation-design.md, §3).

Match lists were picked from TBA on 2026-11-10 (quals with a YouTube video;
local ``data/test_footage/<match_key>.mp4`` matches preferred; otherwise ~15
quals spread evenly over the schedule; 2025cur = qm1-qm15, the first batch).

Splits are by EVENT, never by crop:
  * test  - whole events never trained on (one 2026, one 2025).
  * excluded - our hand-GT matches; never trained on, validated on or harvested
    into a training set.
  * val   - ~10% of each training event's matches, spread over the schedule.
  * train - everything else in scope.
"""
from __future__ import annotations

from typing import Dict, List

# event key -> qual match suffixes ("qmN")
EVENTS: Dict[str, List[str]] = {
    # ---- 2026 (interlaced broadcast) --------------------------------------
    # Championship divisions, Houston TX. 2026cur = our local test_footage set
    # (qm8 / qm11 are GT matches and excluded).
    "2026cur": ["qm22", "qm23", "qm24", "qm26", "qm31", "qm36", "qm39", "qm41",
                "qm45", "qm48", "qm53", "qm54", "qm62", "qm67", "qm71"],
    "2026hop": ["qm1", "qm10", "qm19", "qm27", "qm36", "qm45", "qm54", "qm63",
                "qm71", "qm80", "qm89", "qm98", "qm106", "qm115", "qm124"],
    # Canadian Pacific Regional, Vancouver BC (week 0)
    "2026bcvi": ["qm1", "qm7", "qm12", "qm18", "qm23", "qm29", "qm34", "qm40",
                 "qm45", "qm48", "qm52", "qm57", "qm63", "qm68", "qm74"],
    # Hawaii Regional, Honolulu HI
    "2026hiho": ["qm1", "qm5", "qm10", "qm14", "qm19", "qm23", "qm28", "qm32",
                 "qm37", "qm41", "qm42", "qm47", "qm51", "qm56", "qm60"],
    # FMA District Bensalem, PA
    "2026paben": ["qm1", "qm6", "qm11", "qm16", "qm21", "qm26", "qm31", "qm37",
                  "qm42", "qm45", "qm48", "qm53", "qm58", "qm63", "qm68"],
    # FIRST Indiana State Championship (district champs), Indianapolis IN
    "2026incmp": ["qm1", "qm7", "qm12", "qm18", "qm23", "qm29", "qm35", "qm40",
                  "qm46", "qm48", "qm53", "qm59", "qm65", "qm70", "qm76"],
    # FIT District Amarillo, TX
    "2026txama": ["qm1", "qm6", "qm10", "qm15", "qm19", "qm24", "qm29", "qm33",
                  "qm38", "qm42", "qm44", "qm48", "qm53", "qm57", "qm62"],
    # Rocket City Regional, Huntsville AL  -> held-out TEST event
    "2026alhu": ["qm1", "qm7", "qm13", "qm19", "qm25", "qm31", "qm37", "qm43",
                 "qm49", "qm51", "qm56", "qm62", "qm68", "qm74", "qm80"],
    # ---- 2025 (progressive) ------------------------------------------------
    "2025cur": [f"qm{i}" for i in range(1, 16)],          # Curie, Houston TX
    # FIM District Milford, MI (week 0)
    "2025mimil": ["qm1", "qm7", "qm12", "qm17", "qm23", "qm29", "qm34", "qm39",
                  "qm45", "qm51", "qm56", "qm61", "qm67", "qm73", "qm78"],
    # ISR District Event #1, Hadera, Israel
    "2025isde1": ["qm1", "qm5", "qm9", "qm13", "qm17", "qm21", "qm25", "qm29",
                  "qm34", "qm38", "qm42", "qm46", "qm50", "qm54", "qm58"],
    # San Diego Regional, La Jolla CA  -> held-out TEST event
    "2025casd": ["qm1", "qm6", "qm12", "qm17", "qm23", "qm28", "qm34", "qm39",
                 "qm44", "qm50", "qm55", "qm61", "qm66", "qm72", "qm77"],
    # ---- 2024 (progressive) ------------------------------------------------
    "2024cur": ["qm1", "qm10", "qm19", "qm27", "qm36", "qm45", "qm54", "qm63",
                "qm71", "qm80", "qm89", "qm98", "qm106", "qm115", "qm124"],
    # FIM District Milford, MI
    "2024mimil": ["qm1", "qm6", "qm12", "qm17", "qm22", "qm28", "qm33", "qm39",
                  "qm44", "qm49", "qm55", "qm60", "qm65", "qm71", "qm76"],
    # Silicon Valley Regional, San Jose CA
    "2024casj": ["qm1", "qm6", "qm12", "qm17", "qm22", "qm28", "qm33", "qm39",
                 "qm44", "qm50", "qm56", "qm61", "qm66", "qm72", "qm77"],
}

# Whole events held out for testing: never trained on, never used for val.
TEST_EVENTS = frozenset({"2026alhu", "2025casd"})

# Hand-GT matches: excluded from training / val entirely.
GT_EXCLUDE = frozenset({"2026cur_qm8", "2026cur_qm11"})

VAL_FRACTION = 0.10


def match_keys(event: str) -> List[str]:
    return [f"{event}_{m}" for m in EVENTS[event]]


def all_match_keys() -> List[str]:
    return [k for ev in EVENTS for k in match_keys(ev)]


def val_matches(event: str) -> List[str]:
    """~10% of a training event's scope (>= 1), spread over the schedule."""
    if event in TEST_EVENTS or event not in EVENTS:
        return []
    keys = [k for k in match_keys(event) if k not in GT_EXCLUDE]
    n_val = max(1, round(VAL_FRACTION * len(keys)))
    step = len(keys) / n_val
    return [keys[int(step * i + step / 2)] for i in range(n_val)]


def split_of(match_key: str) -> str:
    """'excluded' | 'test' | 'val' | 'train' for any match key (events outside
    the default scope count as train unless GT/test)."""
    if match_key in GT_EXCLUDE:
        return "excluded"
    event = match_key.split("_", 1)[0]
    if event in TEST_EVENTS:
        return "test"
    if match_key in val_matches(event):
        return "val"
    return "train"
