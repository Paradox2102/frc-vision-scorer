from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class CoOcclusionEvent:
    event_id: int
    alliance: str
    identity_ids: Set[int]
    start_frame: int
    last_occlusion_frame: int
    resolved: bool = False
    status: str = "collecting"
    resolved_mapping: Dict[int, int] = field(default_factory=dict)
    reappeared_slots: Set[int] = field(default_factory=set)
    reappeared_mapping: Dict[int, int] = field(default_factory=dict)
    occluded_times: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AssignmentCandidate:
    identity_id: int
    slot_id: int
    score: float


@dataclass(frozen=True)
class AmbiguityResolution:
    status: str
    mapping: Dict[int, int]
    best_score: float
    margin: float


def resolve_group_assignment(
    identity_ids: Iterable[int],
    slot_ids: Iterable[int],
    candidates: Iterable[AssignmentCandidate],
    decisive_margin: float = 0.3,
) -> AmbiguityResolution:
    identities = sorted(set(identity_ids))
    slots = sorted(set(slot_ids))
    if not identities or len(identities) != len(slots):
        return AmbiguityResolution("incomplete", {}, 0.0, 0.0)

    scores = {(c.identity_id, c.slot_id): float(c.score) for c in candidates}
    ranked = []
    for slot_perm in permutations(slots):
        mapping = dict(zip(identities, slot_perm))
        score = sum(scores.get((identity_id, slot_id), float("-inf"))
                    for identity_id, slot_id in mapping.items())
        ranked.append((score, mapping))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_score, best_mapping = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
    margin = float("inf") if second_score == float("-inf") else best_score - second_score
    status = "resolved" if margin >= decisive_margin else "ambiguous"
    return AmbiguityResolution(status, best_mapping, best_score, margin)


class CoOcclusionTracker:
    def __init__(
        self,
        re_emergence_window: int = 45,
        grouping_window: int = 15,
        decisive_margin: float = 0.3,
    ) -> None:
        self.events: Dict[int, CoOcclusionEvent] = {}
        self.re_emergence_window = re_emergence_window
        self.grouping_window = grouping_window
        self.decisive_margin = decisive_margin
        self._next_event_id = 0

    def record_occlusion(
        self, frame_idx: int, identity_id: int, alliance: str
    ) -> CoOcclusionEvent:
        existing = self._active_event_for_identity(identity_id)
        if existing is not None:
            existing.occluded_times.setdefault(identity_id, frame_idx)
            existing.last_occlusion_frame = frame_idx
            return existing

        event = next(
            (
                candidate
                for candidate in self.get_active_events()
                if candidate.alliance == alliance
                and frame_idx - candidate.last_occlusion_frame <= self.grouping_window
                and not candidate.reappeared_mapping
            ),
            None,
        )
        if event is None:
            event = CoOcclusionEvent(
                event_id=self._next_event_id,
                alliance=alliance,
                identity_ids=set(),
                start_frame=frame_idx,
                last_occlusion_frame=frame_idx,
            )
            self.events[event.event_id] = event
            self._next_event_id += 1

        event.identity_ids.add(identity_id)
        event.occluded_times[identity_id] = frame_idx
        event.last_occlusion_frame = frame_idx
        return event

    def handle_reemergence(
        self,
        frame_idx: int,
        identity_id: int,
        slot_id: int,
        margin: float,
    ) -> Optional[CoOcclusionEvent]:
        event = self._active_event_for_identity(identity_id)
        if event is None:
            return None
        occluded_frame = event.occluded_times.get(identity_id)
        if occluded_frame is None or frame_idx - occluded_frame > self.re_emergence_window:
            return None

        event.reappeared_slots.add(slot_id)
        event.reappeared_mapping[identity_id] = slot_id
        if len(event.identity_ids) < 2:
            self.resolve_event(event.event_id, event.reappeared_mapping)
            return None
        if margin < self.decisive_margin:
            event.status = "pending_vlm"
        elif event.status == "collecting":
            event.status = "reappearing"
        if (
            event.status != "pending_vlm"
            and set(event.reappeared_mapping) == event.identity_ids
        ):
            self.resolve_event(event.event_id, event.reappeared_mapping)
        return event if event.status == "pending_vlm" else None

    def ambiguous_identity_ids(self) -> Set[int]:
        result: Set[int] = set()
        for event in self.get_active_events():
            if event.status == "pending_vlm":
                result.update(event.identity_ids)
        return result

    def resolve_event(self, event_id: int, mapping: Dict[int, int]) -> None:
        event = self.events.get(event_id)
        if event is None:
            return
        event.resolved = True
        event.status = "resolved"
        event.resolved_mapping = dict(mapping)

    def get_active_events(self) -> List[CoOcclusionEvent]:
        return [event for event in self.events.values() if not event.resolved]

    def _active_event_for_identity(
        self, identity_id: int
    ) -> Optional[CoOcclusionEvent]:
        return next(
            (
                event
                for event in self.get_active_events()
                if identity_id in event.identity_ids
            ),
            None,
        )
