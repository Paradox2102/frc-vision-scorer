"""Plain data the scouting metrics consume.

The loader (``loader.py``) turns a finished replay's meta.json + the TBA match
rows into one ``MatchData`` per match. Metrics never read replays or the DB
themselves — they take a ``list[MatchData]`` — so a metric is unit-testable on
synthetic matches and new metrics need no I/O code.

Coordinates: ``x, y`` are absolute WPILib field metres (blue-right corner origin,
+x toward red). ``ax, ay`` are the same point in ALLIANCE PERSPECTIVE: the
sample's own alliance wall at x=0 (see geometry.to_alliance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Sample:
    frame_num: int            # source-video frame
    t_match: float            # game-clock seconds since auto start (auto 0..auto_s,
                              # teleop auto_s..auto_s+teleop_s; the auto→teleop pause is removed)
    period: str               # "auto" | "teleop"
    remaining: float          # on-screen countdown within the period
    x: float
    y: float
    ax: float                 # alliance-perspective x
    ay: float                 # alliance-perspective y
    dt: float                 # seconds of match this sample stands for (frame period)
    confirmed: bool = False   # digit-confirmed label (vs. lineup / fill)
    slot: Optional[str] = None


@dataclass
class TeamMatch:
    team: int
    alliance: str                     # "red" | "blue"
    station: Optional[int]            # TBA station 1..3 (alliance_position), None = unknown
    samples: list = field(default_factory=list)       # list[Sample], labelled to this team
    # Auto path from summary.autonomous_paths (slot-continuous, smoothed by the
    # pipeline), attributed to this team by majority label. Samples are
    # Sample objects with period == "auto".
    auto_path: list = field(default_factory=list)
    auto_slot: Optional[str] = None
    auto_label_share: Optional[float] = None
    auto_runner_up_share: Optional[float] = None
    auto_source: Optional[str] = None  # "autonomous_paths" | "frames" | None


@dataclass
class MatchData:
    event_key: str
    match_key: str
    comp_level: str
    match_number: int
    set_number: int
    year: int
    replay_id: str
    replay_created_at: Optional[str]
    legacy_labels: bool                       # replay predates window labels
    alliances: dict                           # {"red": [team|None]*3, "blue": [...]} in station order
    hub_auto: dict                            # {"red": int|None, "blue": int|None} (2026: TBA hub auto fuel)
    frame_period_s: float                     # match seconds per stored frame (~0.2)
    labelled_s: dict = field(default_factory=dict)    # {"auto": s, "teleop": s} labelled robot-seconds
    tracked_s: dict = field(default_factory=dict)     # {"auto": s, "teleop": s} all tracked robot-seconds
    teams: dict = field(default_factory=dict)         # {team:int -> TeamMatch}
    off_roster_samples: int = 0       # labels naming a team not in the TBA lineup (dropped)
    duplicate_label_samples: int = 0  # same team on two slots in one frame (one kept)
    # For the export's per-match video timeline (timeline.py, built lazily):
    periods: list = field(default_factory=list)   # summary.periods (clock scan)
    video_path: Optional[str] = None              # source match video on disk
    youtube_key: Optional[str] = None             # TBA match video key

    def source_ref(self) -> dict:
        return {"match_key": self.match_key, "comp_level": self.comp_level,
                "set_number": self.set_number, "match_number": self.match_number,
                "replay_id": self.replay_id, "replay_created_at": self.replay_created_at,
                "legacy_labels": self.legacy_labels}

    def station_of(self, team: int) -> tuple:
        """(alliance, station) from the TBA lineup, or (None, None)."""
        for alliance in ("red", "blue"):
            row = self.alliances.get(alliance) or []
            for i, t in enumerate(row):
                if t is not None and int(t) == int(team):
                    return alliance, i + 1
        return None, None
