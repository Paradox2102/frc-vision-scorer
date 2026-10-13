"""Segment-level slot labels (knob ``identity_segment_resolve``).

Both whole-match ``global_identity_resolve`` and the swap-aware permutation
HMM (``identity_swap_resolve``) assume a slot follows ONE robot between rare,
paired swaps. qm8/qm11 diagnosis (2026-10-13) showed the opposite: a slot's
physical robot changes ~17 times per alliance per match (slot expiry +
``free_slots.pop(0)`` reseeds, wide-gate coasting re-acquisition, in-place
feed changes), slots hold duplicates (two slots on one robot) and junk, so
the 3-slot permutation state is not a valid model. What DOES hold: a slot
changes robot only at a tracker discontinuity -- consecutive same-slot reads
with no discontinuity between them agree 16/16 (qm8) and 30/30 (qm11), vs
6/14 and 6/15 across one.

So the unit of labelling is the **segment**: a maximal run of one slot's
presence with no discontinuity inside. Three stages:

1. **Cut.** Each slot's presence (identity-log rows with a position) is cut
   at (a) presence gaps (the slot expired and was later re-seeded), (b) a
   position step > ``step_m`` between consecutive samples, (c) a tracker /
   identity event that can change the physical robot -- exact live rows from
   ``web.field.slot_boundaries`` (seed; coasting re-acquisition whose step
   from the coasted prediction > step_m; identity birth on a disrupted slot
   or > ``birth_dist_m`` from its nearest candidate; feed changes only with
   ``cut_feed_change``) or, for jobs that predate them, the
   ``identity_reject`` diagnostic rows (the birth rule). (a)+(b)+birth is the
   rule validated by the diagnosis prototype.

2. **Anchor.** A segment is labelled from its OWN attributed reads (the same
   exact-frame attribution gir uses, in-alliance roster teams only): the top
   team when it reaches ``min_support`` and strictly beats the runner-up
   (a tie leaves it unlabelled, like gir's contested slots). Anchors are then
   made **exclusive**: two same-team anchors present at the same instants
   (>= ``min_overlap_frames`` common frames) more than ``dup_dist_m`` apart
   (median over the overlap) cannot both be right; strongest first, a losing
   anchor falls back to its next-best read team that is exclusive, else None.

3. **Stitch.** Unread (or contested / weak) segments inherit labels along
   physically feasible links. Per alliance, segment ENDS are matched one-to-
   one to segment STARTS (same or different slot) by a minimum-cost bipartite
   assignment (Hungarian, ``scipy.optimize.linear_sum_assignment``) over the
   feasible pairs: 0 < gap <= ``link_max_gap_s`` and end->start distance <=
   ``max_speed_mps`` * gap + ``link_slack_m``; cost = metres + seconds, minus
   ``same_label_bonus`` for two anchors of one team; two anchors of different
   teams are never linked. This is the standard optimal one-step tracklet
   association: every robot keeps existing across a discontinuity, so each
   end continues in at most one start, and the gate guarantees no physically
   impossible link is ever used. Links form chains (one physical robot's
   track pieces); labels propagate outward from each anchor along its chain,
   nearest hop first, until the next anchor. Between two anchors of different
   teams the chain is cut at its costliest link. Every propagated label must
   (i) not contradict the segment's own reads and (ii) stay exclusive at each
   instant (same rule as anchors); a refused segment stops that propagation
   run -- everything beyond it on that side was reached only through it.
   Segments nothing reaches get ``None``: no label is invented.

Duplicates -- two slots on one robot, co-located within ``dup_dist_m`` over
their overlap -- may share a label; they are flagged (``dup_with``) rather
than counted as exclusivity violations. A single shared frame (a handoff
frame, < ``min_overlap_frames``) is tolerated.

Defaults (scripts/identity_offline_check.py --segments, 9 stored full-match
jobs = 2 matches): ``link_slack_m`` 1.5 m was best or tied-best on held-out
reads for BOTH matches vs 1.0 / 2.0 / 2.5 (2.0 drops qm8 stitched precision
72% -> 43%); ``link_max_gap_s`` 2 s best on qm11 vs 1 / 3 s, tied on qm8;
step/birth thresholds, ``dup_dist_m``
and ``min_support`` are the prototype's / physical values (alternatives moved
held-out agreement < 1.5 pts). Tried and rejected: labelling a still-
unlabelled segment by elimination (the one roster team not claimed elsewhere
during it) -- on qm11 the extra labels were nearly all wrong (junk /
duplicate segments are not robots of the missing team).

Output segments are compatible with the ``slot_segments`` plumbing that
``identity_swap_resolve`` introduced (``reresolve_frames_by_slot``,
``InferenceJob._relabel_replay_events``, ``match_robot_identity_segment``):
``{"start_frame", "end_frame", "team_number", "confirmed", "score", ...}``
time-ordered per slot, plus ``source`` (read | stitched | none), ``support``
(the anchor's read weight), ``hops`` (links from the anchor) and
``dup_with``. When both knobs are on, the finalize step uses these segments
(segment resolve wins over swap resolve). The whole-match per-slot labels
(``match_robot_identity``) stay gir's either way.
"""
from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from web.field.identity_resolve import BLUE_SLOTS, SCORE_EPS, attribute_reads

__all__ = [
    "SegmentParams",
    "slot_presence",
    "presence_cuts",
    "boundary_cuts",
    "boundary_rows_from_rejects",
    "build_segments",
    "resolve_segment_labels",
    "labels_at",
    "exclusivity_violations",
    "format_segment_summary",
]

DEFAULT_FPS = 59.94
_SLOT_LABELS = ("B1", "B2", "B3", "R1", "R2", "R3")
_BIG = 1e6


@dataclass(frozen=True)
class SegmentParams:
    fps: float = DEFAULT_FPS
    # --- cutting ---
    step_m: float = 2.5          # consecutive slot positions this far apart = teleport
    birth_dist_m: float = 2.0    # identity birth this far from its nearest candidate
    cut_feed_change: bool = False  # exact rows only: cut where the fused feeder id changed
    # --- anchoring ---
    min_support: float = 0.5     # read weight a segment needs to label itself
    # --- stitching ---
    max_speed_mps: float = 5.0
    link_slack_m: float = 1.5    # position noise (fused/corner projections ~0.5-1.5 m)
    link_max_gap_s: float = 2.0
    same_label_bonus: float = 1.0
    # --- exclusivity / duplicates ---
    dup_dist_m: float = 1.0
    min_overlap_frames: int = 2
    confirm_margin: float = 2.0


@dataclass
class _Seg:
    idx: int
    slot: int
    frames: list[int]
    pos: dict[int, tuple[float, float]]
    reads: dict[str, float] = field(default_factory=dict)   # in-roster team -> weight
    n_reads: int = 0
    label: str | None = None
    source: str = "none"          # read | stitched | none
    support: float = 0.0
    margin: float = 0.0
    hops: int = 0
    anchor: int | None = None     # idx of the anchor a stitched label came from
    chain: int | None = None
    contested: bool = False
    demoted: bool = False         # anchor that lost an exclusivity conflict
    dup_with: set = field(default_factory=set)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def alliance(self) -> str:
        return "blue" if self.slot in BLUE_SLOTS else "red"

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.reads.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
# Presence + cuts
# --------------------------------------------------------------------------- #
def slot_presence(identity_log: list[dict]) -> dict[int, list[tuple[int, float, float]]]:
    """``{slot_id: [(frame, x, y), ...]}`` sorted by frame: one sample per
    (frame, slot) -- the first finite ``pos`` logged for it. Rows without a
    position (offline reconstruction over frames the replay dropped) are
    absence."""
    seen: dict[tuple[int, int], tuple[float, float]] = {}
    for row in identity_log:
        pos = row.get("pos")
        if pos is None or len(pos) < 2 or row.get("slot_id") is None:
            continue
        key = (int(row["frame_num"]), int(row["slot_id"]))
        if key in seen:
            continue
        x, y = float(pos[0]), float(pos[1])
        if math.isfinite(x) and math.isfinite(y):
            seen[key] = (x, y)
    out: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for (frame, slot), (x, y) in sorted(seen.items()):
        out[slot].append((frame, x, y))
    return dict(out)


def presence_cuts(presence: dict, grid: list[int], p: SegmentParams) -> dict[int, set[int]]:
    """Cut frames from presence alone: a slot present again after >= 1 grid
    frame of absence (expiry + reseed), or a position step > ``step_m``."""
    grid_idx = {f: k for k, f in enumerate(grid)}
    cuts: dict[int, set[int]] = defaultdict(set)
    for slot, samples in presence.items():
        for (f0, x0, y0), (f1, x1, y1) in zip(samples, samples[1:]):
            k0, k1 = grid_idx.get(f0), grid_idx.get(f1)
            gap = (k1 - k0 > 1) if (k0 is not None and k1 is not None) else False
            if gap or math.hypot(x1 - x0, y1 - y0) > p.step_m:
                cuts[slot].add(f1)
    return dict(cuts)


def boundary_rows_from_rejects(reject_rows: list[dict]) -> list[dict]:
    """``identity_reject`` diagnostic rows (identity births) as boundary rows
    (the offline stand-in for jobs that predate ``slot_boundary`` logging)."""
    out = []
    for r in reject_rows:
        if r.get("slot_id") is None or r.get("frame") is None:
            continue
        out.append({"frame": int(r["frame"]), "slot_id": int(r["slot_id"]),
                    "kinds": ["birth"],
                    "birth_disrupted": bool(r.get("slot_disrupted")),
                    "birth_distance_m": r.get("distance_m")})
    return out


def is_cut(row: dict, p: SegmentParams) -> bool:
    """Whether one boundary row (``web.field.slot_boundaries``) can change the
    slot's physical robot under ``p``."""
    kinds = set(row.get("kinds") or ())
    if "seed" in kinds:
        return True
    # Re-acquisition after coasting: the step from the slot's coasted
    # prediction, i.e. the same quantity the presence step rule measures
    # between consecutive logged positions (a coasting slot logs its
    # prediction). Measured from the last real measurement it would also
    # count the robot's own motion during the coast.
    jump = row.get("pred_jump_m")
    if jump is None:
        jump = row.get("jump_m")
    if "reacquire" in kinds and (jump or 0.0) > p.step_m:
        return True
    if "birth" in kinds and (row.get("birth_disrupted")
                             or (row.get("birth_distance_m") or 0.0) > p.birth_dist_m):
        return True
    if p.cut_feed_change and "feed_change" in kinds:
        return True
    return False


def boundary_cuts(rows: list[dict], p: SegmentParams) -> dict[int, set[int]]:
    cuts: dict[int, set[int]] = defaultdict(set)
    for r in rows or ():
        if r.get("slot_id") is not None and r.get("frame") is not None and is_cut(r, p):
            cuts[int(r["slot_id"])].add(int(r["frame"]))
    return dict(cuts)


def build_segments(presence: dict, cuts: dict[int, set[int]]) -> list[_Seg]:
    """Split each slot's presence at its cut frames. A cut frame where the
    slot is absent starts the segment at the slot's next present frame."""
    segs: list[_Seg] = []
    for slot in sorted(presence):
        samples = presence[slot]
        cut_sorted = sorted(cuts.get(slot, ()))
        cur: _Seg | None = None
        prev_f = None
        for f, x, y in samples:
            new = cur is None
            if not new and cut_sorted:
                # any cut in (prev_f, f]
                k = bisect.bisect_right(cut_sorted, prev_f)
                new = k < len(cut_sorted) and cut_sorted[k] <= f
            if new:
                cur = _Seg(idx=len(segs), slot=slot, frames=[], pos={})
                segs.append(cur)
            cur.frames.append(f)
            cur.pos[f] = (x, y)
            prev_f = f
    return segs


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _overlap_conflict(a: _Seg, b: _Seg, p: SegmentParams) -> tuple[bool, bool]:
    """(conflict, duplicate) for two segments claiming one team: present at
    >= min_overlap_frames common frames and apart (median) > dup_dist_m ->
    conflict; co-located over the overlap -> duplicate."""
    if a.end < b.start or b.end < a.start:
        return False, False
    common = [f for f in a.frames if f in b.pos] if len(a.frames) <= len(b.frames) else \
        [f for f in b.frames if f in a.pos]
    if not common:
        return False, False
    d = sorted(math.dist(a.pos[f], b.pos[f]) for f in common)
    med = d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])
    if med <= p.dup_dist_m:
        return False, True
    return len(common) >= p.min_overlap_frames, False


def _consistent(seg: _Seg, team: str) -> bool:
    """A label may not contradict the segment's own reads: no other team may
    out-weigh it there."""
    own = seg.reads.get(team, 0.0)
    return all(w <= own + SCORE_EPS for t, w in seg.reads.items() if t != team)


class _TeamIndex:
    """Accepted labelled segments per team, for exclusivity checks."""

    def __init__(self, p: SegmentParams):
        self.p = p
        self.by_team: dict[str, list[_Seg]] = defaultdict(list)

    def conflicts(self, seg: _Seg, team: str) -> bool:
        return any(o is not seg and _overlap_conflict(seg, o, self.p)[0]
                   for o in self.by_team.get(team, ()))

    def add(self, seg: _Seg, team: str) -> None:
        self.by_team[team].append(seg)


# --------------------------------------------------------------------------- #
# Resolve
# --------------------------------------------------------------------------- #
def _anchor(segs: list[_Seg], p: SegmentParams, index: _TeamIndex) -> int:
    """Own-read labels, then exclusivity strongest-first. Returns demotions."""
    cands = []
    for s in segs:
        ranked = s.ranked()
        if not ranked or ranked[0][1] < p.min_support - SCORE_EPS:
            continue
        if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] - SCORE_EPS:
            s.contested = True
            continue
        cands.append(s)
    cands.sort(key=lambda s: (-s.ranked()[0][1], -len(s.frames), s.idx))
    demoted = 0
    for s in cands:
        ranked = s.ranked()
        chosen = None
        for k, (team, w) in enumerate(ranked):
            if w < p.min_support - SCORE_EPS:
                break
            if k > 0 and len(ranked) > k + 1 and ranked[k + 1][1] >= w - SCORE_EPS:
                break     # fallback tied with the next one: ambiguous
            if not index.conflicts(s, team):
                chosen = (k, team, w)
                break
        if chosen is None:
            s.demoted = True
            demoted += 1
            continue
        k, team, w = chosen
        if k > 0:
            s.demoted = True
            demoted += 1
        nxt = next((ww for tt, ww in ranked if tt != team), 0.0)
        s.label, s.source, s.support, s.margin = team, "read", w, w - nxt
        index.add(s, team)
    return demoted


def _link_alliance(segs: list[_Seg], p: SegmentParams) -> list[tuple[int, int, float]]:
    """One-to-one min-cost end->start links among one alliance's segments."""
    from scipy.optimize import linear_sum_assignment

    n = len(segs)
    if n < 2:
        return []
    max_gap = p.link_max_gap_s * p.fps
    cost = np.full((n, n), _BIG)
    for i, a in enumerate(segs):
        ea = a.pos[a.end]
        for j, b in enumerate(segs):
            if i == j:
                continue
            gap = b.start - a.end
            if gap <= 0 or gap > max_gap:
                continue
            if a.label and b.label and a.label != b.label:
                continue
            gap_s = gap / p.fps
            d = math.dist(ea, b.pos[b.start])
            if d > p.max_speed_mps * gap_s + p.link_slack_m:
                continue
            c = d + gap_s
            if a.label and a.label == b.label:
                c -= p.same_label_bonus
            cost[i, j] = c
    rows, cols = linear_sum_assignment(cost)
    return [(segs[i].idx, segs[j].idx, float(cost[i, j]))
            for i, j in zip(rows, cols) if cost[i, j] < _BIG / 2]


def _chains(segs: list[_Seg], links) -> list[tuple[list[int], list[float]]]:
    """Chains (segment idx lists, time order) and their link costs."""
    nxt = {a: (b, c) for a, b, c in links}
    has_prev = {b for _, b, _ in links}
    out = []
    for s in segs:
        if s.idx in has_prev:
            continue
        chain, costs = [s.idx], []
        while chain[-1] in nxt:
            b, c = nxt[chain[-1]]
            chain.append(b)
            costs.append(c)
        out.append((chain, costs))
    return out


def _propagate(segs: list[_Seg], chains, p: SegmentParams, index: _TeamIndex) -> dict:
    """Spread anchor labels along chains, nearest hop first, exclusively."""
    by_idx = {s.idx: s for s in segs}
    runs = []      # (anchor idx, team, [chain positions in order], chain no)
    for cno, (chain, costs) in enumerate(chains):
        for i in chain:
            by_idx[i].chain = cno
        anchors = [k for k, i in enumerate(chain) if by_idx[i].source == "read"]
        if not anchors:
            continue
        for a_pos_k, k in enumerate(anchors):
            team = by_idx[chain[k]].label
            # left run: back to the previous anchor (or chain start)
            prev_k = anchors[a_pos_k - 1] if a_pos_k > 0 else None
            nxt_k = anchors[a_pos_k + 1] if a_pos_k + 1 < len(anchors) else None
            lo = 0 if prev_k is None else prev_k + 1
            hi = len(chain) - 1 if nxt_k is None else nxt_k - 1
            if prev_k is not None and by_idx[chain[prev_k]].label != team:
                # costliest link between the two anchors: links prev_k..k-1
                cut = max(range(prev_k, k), key=lambda q: (costs[q], q))
                lo = cut + 1
            if nxt_k is not None and by_idx[chain[nxt_k]].label != team:
                cut = max(range(k, nxt_k), key=lambda q: (costs[q], -q))
                hi = cut
            runs.append((chain[k], team, [chain[q] for q in range(k - 1, lo - 1, -1)], cno))
            runs.append((chain[k], team, [chain[q] for q in range(k + 1, hi + 1)], cno))
    heap = []
    for r, (anc, team, members, _c) in enumerate(runs):
        if members:
            heapq.heappush(heap, (1, -by_idx[anc].support, r, 0))
    stats = {"stitched": 0, "refused_reads": 0, "refused_exclusive": 0}
    while heap:
        hops, _neg, r, pos = heapq.heappop(heap)
        anc, team, members, _c = runs[r]
        s = by_idx[members[pos]]
        ok = True
        if s.label is not None:
            ok = s.label == team         # already labelled (other run / anchor)
        elif not _consistent(s, team):
            stats["refused_reads"] += 1
            ok = False
        elif index.conflicts(s, team):
            stats["refused_exclusive"] += 1
            ok = False
        else:
            s.label, s.source, s.hops, s.anchor = team, "stitched", hops, anc
            s.support = by_idx[anc].support
            s.margin = s.reads.get(team, 0.0)
            index.add(s, team)
            stats["stitched"] += 1
        if ok and pos + 1 < len(members):
            heapq.heappush(heap, (hops + 1, _neg, r, pos + 1))
    return stats


def resolve_segment_labels(
    vlm_read_rows: list[dict] | None,
    identity_log: list[dict],
    blue_roster: list[str],
    red_roster: list[str],
    boundary_rows: list[dict] | None = None,
    params: SegmentParams | None = None,
    grid: list[int] | None = None,
) -> dict:
    """Segment-level per-slot labels (see module doc).

    ``identity_log`` rows carry ``frame_num``/``identity_id``/``slot_id`` (read
    attribution) and ``pos`` (presence + geometry). ``boundary_rows``: exact
    ``slot_boundary`` rows, or ``boundary_rows_from_rejects`` for old jobs.
    ``grid``: the processed-frame grid (default: every identity-log frame).

    Returns ``{"segments": {slot: [seg dict]}, "links", "stats"}``; segment
    dicts carry ``start_frame``/``end_frame`` (inclusive presence frames),
    ``team_number``, ``confirmed`` (own-read label with margin >=
    confirm_margin), ``score`` (= ``support``), ``source``, ``hops``,
    ``reads`` (own in-roster read weights), ``n_reads``, ``chain``,
    ``dup_with`` (slot ids co-located with it under the same label),
    ``track`` (= the slot; swap-plumbing compatibility).
    """
    p = params or SegmentParams()
    presence = slot_presence(identity_log)
    grid = sorted(set(grid or ()) | {f for v in presence.values() for f, _x, _y in v})
    cuts = presence_cuts(presence, grid, p)
    for slot, fs in boundary_cuts(boundary_rows or [], p).items():
        cuts.setdefault(slot, set()).update(fs)
    segs = build_segments(presence, cuts)

    rosters = {"blue": {str(t) for t in (blue_roster or [])},
               "red": {str(t) for t in (red_roster or [])}}
    by_slot: dict[int, list[_Seg]] = defaultdict(list)
    for s in segs:
        by_slot[s.slot].append(s)
    starts = {slot: [s.start for s in v] for slot, v in by_slot.items()}
    attributed, dropped = (attribute_reads(vlm_read_rows, identity_log)
                           if vlm_read_rows else ([], 0))
    used = 0
    for frame, slot, team, w in attributed:
        v = by_slot.get(slot)
        if not v:
            continue
        k = max(0, bisect.bisect_right(starts[slot], frame) - 1)
        s = v[k]
        team = str(team)
        if team not in rosters[s.alliance]:
            continue          # cross-alliance / off-roster: ignored, as in gir
        s.reads[team] = s.reads.get(team, 0.0) + float(w)
        s.n_reads += 1
        used += 1

    index = _TeamIndex(p)
    demoted = _anchor(segs, p, index)
    links = []
    for alliance in ("blue", "red"):
        links += _link_alliance([s for s in segs if s.alliance == alliance], p)
    chains = _chains(segs, links)
    pstats = _propagate(segs, chains, p, index)

    # Duplicates: same-team segments co-located over their overlap.
    for team, members in index.by_team.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a.slot != b.slot and _overlap_conflict(a, b, p)[1]:
                    a.dup_with.add(b.slot)
                    b.dup_with.add(a.slot)

    segments: dict[int, list[dict]] = {}
    for slot, v in sorted(by_slot.items()):
        segments[slot] = [{
            "start_frame": s.start, "end_frame": s.end,
            "team_number": s.label,
            "confirmed": bool(s.source == "read" and s.margin >= p.confirm_margin - SCORE_EPS),
            "score": round(s.support, 3), "support": round(s.support, 3),
            "source": s.source, "hops": s.hops, "n_reads": s.n_reads,
            "reads": {t: round(w, 3) for t, w in s.ranked()},
            "contested": s.contested, "demoted": s.demoted,
            "chain": s.chain, "dup_with": sorted(s.dup_with), "track": slot,
            "frames": len(s.frames),
        } for s in v]
    n_frames = sum(len(s.frames) for s in segs)
    lab_frames = sum(len(s.frames) for s in segs if s.label is not None)
    stats = {
        "segments": len(segs),
        "cuts": sum(len(v) for v in cuts.values()),
        "read": sum(1 for s in segs if s.source == "read"),
        "stitched": sum(1 for s in segs if s.source == "stitched"),
        "unlabelled": sum(1 for s in segs if s.label is None),
        "contested": sum(1 for s in segs if s.contested),
        "demoted": demoted,
        "links": len(links),
        "chains": len(chains),
        "reads_used": used,
        "reads_dropped_misattributed": dropped,
        "coverage": round(lab_frames / n_frames, 4) if n_frames else 0.0,
        "dup_segments": sum(1 for s in segs if s.dup_with),
        **pstats,
    }
    return {"segments": segments, "links": links, "stats": stats,
            "params": p}


def labels_at(segments: dict[int, list[dict]], slot: int, frame: int) -> str | None:
    """Label of ``slot`` at ``frame`` (the segment covering it: last start <=
    frame; frames before the first segment map to the first)."""
    segs = segments.get(slot)
    if not segs:
        return None
    k = bisect.bisect_right([s["start_frame"] for s in segs], frame) - 1
    return segs[max(k, 0)]["team_number"]


def exclusivity_violations(segments: dict[int, list[dict]], identity_log: list[dict],
                           dup_dist_m: float = 1.0) -> dict:
    """Instants (frame, team) where >= 2 present slots carry the same label:
    ``violations`` when some pair is > ``dup_dist_m`` apart, ``duplicates``
    when all are co-located (flagged, allowed)."""
    pres = slot_presence(identity_log)
    at: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for slot, samples in pres.items():
        for f, x, y in samples:
            at[f].append((slot, (x, y)))
    viol = dup = 0
    for f, rows in at.items():
        by_team: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for slot, xy in rows:
            t = labels_at(segments, slot, f)
            if t is not None:
                by_team[t].append(xy)
        for t, xys in by_team.items():
            if len(xys) < 2:
                continue
            far = any(math.dist(a, b) > dup_dist_m
                      for i, a in enumerate(xys) for b in xys[i + 1:])
            viol += far
            dup += not far
    return {"violations": viol, "duplicates": dup}


def format_segment_summary(res: dict, slot_label=None) -> str:
    """One finalize line: counts + each slot's label sequence (runs merged)."""
    lab = slot_label or (lambda s: _SLOT_LABELS[s] if 0 <= s < 6 else str(s))
    st = res["stats"]
    parts = []
    for slot, segs in sorted(res["segments"].items()):
        runs: list[list] = []
        for g in segs:
            t = g["team_number"]
            if runs and runs[-1][0] == t:
                runs[-1][2] = g["end_frame"]
            else:
                runs.append([t, g["start_frame"], g["end_frame"]])
        parts.append(f"{lab(slot)}=" + ">".join(f"{t}[{a}-{b}]" for t, a, b in runs))
    return (f"[segment-resolve] segments={st['segments']} read={st['read']} "
            f"stitched={st['stitched']} none={st['unlabelled']} links={st['links']} "
            f"demoted={st['demoted']} coverage={st['coverage']:.0%} "
            f"dups={st['dup_segments']} | " + " ".join(parts))
