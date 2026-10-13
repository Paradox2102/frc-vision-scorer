"""Swap-aware, time-varying slot->team resolve (knob ``identity_swap_resolve``).

``global_identity_resolve`` (``identity_resolve.resolve_slot_labels``) gives
each slot ONE whole-match team. Slots physically swap robots mid-match (the
tracker re-binds two slots across a crossing or a lost/re-seeded track), so a
whole-match label is wrong for part of the match whichever way it is chosen
(qm11 18a8fd23: R3 reads 4256 x11 until f5656 and 6800 after; R1 reads 4256
x17 from f8336 -> gir sums both halves and labels R1=4256, R3=6800, both
wrong for the auto period).

Two stages, per alliance:

1. **Swap structure (Viterbi).** The 3 slots carry one of the 3! = 6
   slot->team permutations at every processed frame (the roster is padded
   with placeholder teams when it has < 3 entries). A read attributed to
   (slot, frame) -- the exact-frame attribution ``build_slot_evidence`` uses
   (``attribute_reads``) -- adds its ``vote_weight`` to every permutation
   that puts the read's team on that slot; reads naming a team outside this
   alliance's roster are ignored, as in gir. Staying costs 0; one
   transposition of two slots costs ``swap_lambda`` (a 3-cycle is two
   transpositions on consecutive frames, 2*lambda). A transposition of
   slots (a, b) is only POSSIBLE at a frame where it is physically plausible
   (``pair_open_masks``): within ``window_s`` the two slots' positions came
   within ``prox_m``, or BOTH slots had a track discontinuity (a position
   jump > ``jump_m``, or no position for >= ``gap_s`` -- the slot lost its
   robot). A real swap moves both slots onto new robots, so both must show
   it unless they were touching. Ties prefer fewer swaps. Transpositions
   with no read between them form one move (a 3-cycle); each move is then
   centred in the read-free window the evidence leaves it
   (``_centre_moves``) instead of the Viterbi's earliest-possible frame.

2. **Labels (unchanged gir semantics).** The swap structure splits the match
   into *virtual tracks* -- one per slot, each following one permutation
   entry through the swaps, named by the slot it ends the match on. The
   identity log is relabelled onto virtual tracks and the SAME
   ``resolve_slot_labels`` (min_score, ties -> contested/unassigned,
   collision sweep, merge into the final identities' slot_meta) labels them.
   Every slot segment takes its virtual track's label. With no swaps the
   virtual log IS the identity log, so the result equals today's
   whole-match gir output exactly (``test_no_swap_equals_gir``).

Per-slot positions come from the identity log's ``pos`` (live: the identity's
last-seen field position; offline: the replay's per-slot ``field_pos``). A
slot without any position sample is "lost" for the whole match (gate open),
and so is an alliance without positions at all -- lambda alone then governs.
"""
from __future__ import annotations

import bisect
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

from web.field.identity_resolve import (
    BLUE_SLOTS,
    RED_SLOTS,
    SCORE_EPS,
    attribute_reads,
    resolve_slot_labels,
)

__all__ = [
    "SwapParams",
    "slot_position_samples",
    "discontinuity_intervals",
    "pair_open_masks",
    "resolve_slot_segments",
    "segment_at",
    "format_swap_summary",
]

DEFAULT_SWAP_LAMBDA = 3.0   # > the heaviest single read seen (2.5): one stray read never flips
DEFAULT_PROX_M = 1.5        # slot centres closer than this = robots touching (pos noise ~0.5-1 m)
WINDOW_S = 2.0              # a plausible-swap cue opens the gate +- this long
GAP_S = 1.0                 # no position this long = the slot lost its robot
JUMP_M = 3.0                # consecutive positions this far apart = a teleport (re-bind)
DEFAULT_FPS = 59.94

_SLOT_LABELS = ("B1", "B2", "B3", "R1", "R2", "R3")


@dataclass(frozen=True)
class SwapParams:
    swap_lambda: float = DEFAULT_SWAP_LAMBDA
    prox_m: float = DEFAULT_PROX_M
    window_s: float = WINDOW_S
    gap_s: float = GAP_S
    jump_m: float = JUMP_M
    fps: float = DEFAULT_FPS


# --------------------------------------------------------------------------- #
# Plausibility gate
# --------------------------------------------------------------------------- #
def slot_position_samples(identity_log: list[dict]) -> dict[int, list[tuple[int, float, float]]]:
    """``{slot_id: [(frame, x, y), ...]}`` sorted by frame, one sample per
    (frame, slot) -- the first finite ``pos`` logged for it."""
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


def discontinuity_intervals(samples, lo: int, hi: int, p: SwapParams) -> list[tuple[int, int]]:
    """Frame intervals over which a slot may have changed robot: a position
    gap >= ``gap_s`` (including before its first / after its last sample
    inside ``[lo, hi]``) or a jump > ``jump_m`` between consecutive samples.
    No samples at all -> the whole range."""
    if not samples:
        return [(lo, hi)]
    gap_f = p.gap_s * p.fps
    out = []
    if samples[0][0] - lo >= gap_f:
        out.append((lo, samples[0][0]))
    for (f0, x0, y0), (f1, x1, y1) in zip(samples, samples[1:]):
        if f1 - f0 >= gap_f or math.hypot(x1 - x0, y1 - y0) > p.jump_m:
            out.append((f0, f1))
    if hi - samples[-1][0] >= gap_f:
        out.append((samples[-1][0], hi))
    return out


def _within(sorted_frames: list[int], f: float, w: float) -> bool:
    i = bisect.bisect_left(sorted_frames, f - w)
    return i < len(sorted_frames) and sorted_frames[i] <= f + w


def _covered_mask(grid: list[int], intervals: list[tuple[int, int]], w: float) -> list[bool]:
    """Per grid frame: inside some interval widened by ``w`` on both sides."""
    mask = [False] * len(grid)
    for a, b in intervals:
        for k in range(bisect.bisect_left(grid, a - w), bisect.bisect_right(grid, b + w)):
            mask[k] = True
    return mask


def pair_open_masks(grid: list[int], slots: tuple[int, ...], positions: dict,
                    p: SwapParams) -> dict[tuple[int, int], list[bool]]:
    """``{(slot_a, slot_b): [bool per grid frame]}`` -- whether a swap of the
    two slots is physically plausible at that frame (see module doc)."""
    if not grid:
        return {pair: [] for pair in itertools.combinations(slots, 2)}
    w = p.window_s * p.fps
    lo, hi = grid[0], grid[-1]
    disc = {s: discontinuity_intervals(positions.get(s, []), lo, hi, p) for s in slots}
    lost = {s: _covered_mask(grid, disc[s], w) for s in slots}
    masks = {}
    for a, b in itertools.combinations(slots, 2):
        pa = {f: (x, y) for f, x, y in positions.get(a, [])}
        close = [f for f, x, y in positions.get(b, [])
                 if f in pa and math.hypot(x - pa[f][0], y - pa[f][1]) < p.prox_m]
        masks[(a, b)] = [(la and lb) or _within(close, f, w)
                         for f, la, lb in zip(grid, lost[a], lost[b])]
    return masks


# --------------------------------------------------------------------------- #
# Viterbi over slot->team permutations
# --------------------------------------------------------------------------- #
def _better(sc: float, sw: int, best_sc: float, best_sw: int) -> bool:
    """Lexicographic (score, fewer swaps) with float tolerance."""
    return sc > best_sc + SCORE_EPS or (abs(sc - best_sc) <= SCORE_EPS and sw < best_sw)


def _viterbi(n: int, states: list[tuple], emis: list[list[float]],
             masks_by_pos: dict[tuple[int, int], list[bool]], lam: float) -> list[int]:
    """Best state index per grid step. ``masks_by_pos`` is keyed by slot
    POSITION pairs (i < j) within the alliance."""
    index = {st: k for k, st in enumerate(states)}
    neigh: list[list[tuple[int, tuple[int, int]]]] = []
    for st in states:
        row = []
        for i, j in masks_by_pos:
            sw = list(st)
            sw[i], sw[j] = sw[j], sw[i]
            row.append((index[tuple(sw)], (i, j)))
        neigh.append(row)
    best = [(emis[0][s], 0) for s in range(len(states))]
    back: list[list[int]] = []
    for k in range(1, n):
        cur, bk = [], []
        for s in range(len(states)):
            c_sc, c_sw, c_prev = best[s][0], best[s][1], s      # stay first: ties keep
            for t, pair in neigh[s]:
                if not masks_by_pos[pair][k]:
                    continue
                sc, sw = best[t][0] - lam, best[t][1] + 1
                if _better(sc, sw, c_sc, c_sw):
                    c_sc, c_sw, c_prev = sc, sw, t
            cur.append((c_sc + emis[k][s], c_sw))
            bk.append(c_prev)
        best = cur
        back.append(bk)
    last = 0
    for s in range(1, len(states)):
        if _better(best[s][0], best[s][1], best[last][0], best[last][1]):
            last = s
    path = [last]
    for k in range(n - 1, 0, -1):
        path.append(back[k - 1][path[-1]])
    path.reverse()
    return path


def _transposition(a: tuple, b: tuple) -> tuple[int, int]:
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return diff[0], diff[1]


def _reads_on(read_idx_by_pos: dict[int, list[int]], positions) -> list[int]:
    return sorted({k for i in positions for k in read_idx_by_pos.get(i, [])})


def _swap_moves(path: list[int], states: list[tuple],
                read_idx_by_pos: dict[int, list[int]]) -> list[list[tuple[int, tuple[int, int]]]]:
    """Group the path's transpositions into MOVES: consecutive transpositions
    with no read on their slots in between are one composite move (a
    3-cycle is two transpositions; the permutation between them has no
    evidence of its own). Each move is ``[(grid_index, (pos_i, pos_j)), ...]``."""
    events = [(k, _transposition(states[path[k - 1]], states[path[k]]))
              for k in range(1, len(path)) if path[k] != path[k - 1]]
    moves: list[list[tuple[int, tuple[int, int]]]] = []
    for k, pair in events:
        if moves:
            pk, ppair = moves[-1][-1]
            reads = _reads_on(read_idx_by_pos, set(pair) | set(ppair))
            r = bisect.bisect_left(reads, pk)
            if not (r < len(reads) and reads[r] < k):
                moves[-1].append((k, pair))
                continue
        moves.append([(k, pair)])
    return moves


def _centre_moves(path: list[int], grid: list[int],
                  moves: list[list[tuple[int, tuple[int, int]]]],
                  read_idx_by_pos: dict[int, list[int]],
                  masks_by_pos: dict[tuple[int, int], list[bool]]):
    """Slide each move to the open frame nearest the middle of the read-free
    window it can occupy without changing the path score: after the last
    read on its slots (or the previous move), up to the first read on them
    (or the next move). A composite move lands atomically on ONE frame
    where all its pairs are open (no one-frame transient label); when no
    such frame exists it keeps the Viterbi's frames. The Viterbi alone puts
    every swap at the EARLIEST open frame after the last old-label read.
    Any two transpositions of 3 slots share a slot, so move order -- hence
    the permutation sequence -- is preserved. Returns ``(path, moves)``
    with the moves' new grid indices."""
    if not moves:
        return path, moves
    placed: list[list[tuple[int, tuple[int, int]]]] = []
    for e, mv in enumerate(moves):
        reads = _reads_on(read_idx_by_pos, {i for _, pr in mv for i in pr})
        r0 = bisect.bisect_left(reads, mv[0][0])
        r1 = bisect.bisect_left(reads, mv[-1][0])
        lo = reads[r0 - 1] + 1 if r0 > 0 else 1
        hi = reads[r1] if r1 < len(reads) else len(grid) - 1
        if placed:
            lo = max(lo, placed[-1][-1][0] + 1)
        if e + 1 < len(moves):
            hi = min(hi, moves[e + 1][0][0] - 1)
        cands = [m for m in range(lo, hi + 1)
                 if all(masks_by_pos[pr][m] for _, pr in mv)]
        if cands:
            mid = (grid[lo] + grid[hi]) / 2.0
            m = min(cands, key=lambda c: (abs(grid[c] - mid), c))
            placed.append([(m, pr) for _, pr in mv])
        else:
            placed.append(list(mv))
    after = {pk: path[ok] for mv, pmv in zip(moves, placed)
             for (ok, _), (pk, _) in zip(mv, pmv)}
    out, cur = [], path[0]
    for k in range(len(path)):
        cur = after.get(k, cur)
        out.append(cur)
    return out, placed


# --------------------------------------------------------------------------- #
# Public resolve
# --------------------------------------------------------------------------- #
def resolve_slot_segments(
    vlm_read_rows: list[dict] | None,
    identity_log: list[dict],
    votes_by_slot: dict[int, dict[str, float]],
    gallery_by_slot: dict[int, dict[str, float]],
    slot_alliance,
    blue_roster: list[str],
    red_roster: list[str],
    slot_meta: dict,
    min_score: float = 0.5,
    confirm_margin: float = 2.0,
    params: SwapParams | None = None,
) -> dict:
    """Swap-aware per-slot team segments. Same inputs as
    :func:`resolve_slot_labels` plus ``params``.

    Returns ``{"segments", "swaps", "slot_meta", "virtual", "stats"}``:

    - ``segments``: ``{slot_id: [{"start_frame", "end_frame", "team_number",
      "confirmed", "score", "track"}]}`` covering the identity log's frame
      range; ``track`` is the virtual track (= the slot that virtual track
      ends the match on), ``score`` the summed weight of reads inside the
      segment naming its team.
    - ``swaps``: one entry per move, ``{"alliance", "frame", "slots",
      "moves": [(a, b), ...], "support"}`` (``moves`` = its transpositions,
      all applied at ``frame`` -- a composite move whose pairs share no open
      frame keeps the Viterbi's consecutive frames; ``slots`` = the slots
      involved); ``support`` = min(old-labelling advantage before, new-
      labelling advantage after), in vote-weight units (> lambda by
      construction).
    - ``slot_meta``: virtual-track labels (keyed by final slot) -- equal to
      the whole-match gir ``slot_meta`` when no swap was found.
    - ``virtual``: the ``resolve_slot_labels`` result on the virtual log.
    """
    p = params or SwapParams()
    grid = sorted({int(r["frame_num"]) for r in identity_log})
    k_of = {f: k for k, f in enumerate(grid)}
    attributed, _ = attribute_reads(vlm_read_rows, identity_log) if vlm_read_rows else ([], 0)
    positions = slot_position_samples(identity_log)

    vt_by_slot: dict[int, list[int]] = {}       # slot -> virtual track per grid step
    swaps: list[dict] = []
    gate_open: dict[str, float] = {}
    for alliance, slots, roster in (("blue", BLUE_SLOTS, blue_roster),
                                    ("red", RED_SLOTS, red_roster)):
        roster = [str(t) for t in (roster or [])]
        if not grid or not roster:
            for s in slots:
                vt_by_slot[s] = [s] * len(grid)
            continue
        padded = roster + [f"\x00pad{i}" for i in range(max(0, len(slots) - len(roster)))]
        states = list(itertools.permutations(padded, len(slots)))
        pos_of = {s: i for i, s in enumerate(slots)}
        roster_set = set(roster)
        emis = [[0.0] * len(states) for _ in grid]
        read_idx_by_pos: dict[int, list[int]] = defaultdict(list)
        for frame, slot, team, w in attributed:
            if slot not in pos_of or team not in roster_set or frame not in k_of:
                continue
            k, i = k_of[frame], pos_of[slot]
            read_idx_by_pos[i].append(k)
            for s_idx, st in enumerate(states):
                if st[i] == team:
                    emis[k][s_idx] += w
        masks = pair_open_masks(grid, slots, positions, p)
        masks_by_pos = {(pos_of[a], pos_of[b]): m for (a, b), m in masks.items()}
        for (a, b), m in masks.items():
            gate_open[f"{_SLOT_LABELS[a]}-{_SLOT_LABELS[b]}"] = round(
                sum(m) / len(m), 3) if m else 0.0
        path = _viterbi(len(grid), states, emis, masks_by_pos, p.swap_lambda)
        moves = _swap_moves(path, states, read_idx_by_pos)
        path, moves = _centre_moves(path, grid, moves, read_idx_by_pos, masks_by_pos)

        final = states[path[-1]]
        final_slot_of = {team: slots[i] for i, team in enumerate(final)}
        for i, s in enumerate(slots):
            vt_by_slot[s] = [final_slot_of[states[path[k]][i]] for k in range(len(grid))]
        for e, mv in enumerate(moves):
            k0, k1 = mv[0][0], mv[-1][0]
            o_idx, n_idx = path[k0 - 1], path[k1]
            prev_k = moves[e - 1][-1][0] if e > 0 else 0
            next_k = moves[e + 1][0][0] if e + 1 < len(moves) else len(grid)
            before = sum(emis[q][o_idx] - emis[q][n_idx] for q in range(prev_k, k0))
            after = sum(emis[q][n_idx] - emis[q][o_idx] for q in range(k1, next_k))
            pairs = [(slots[i], slots[j]) for _, (i, j) in mv]
            swaps.append({"alliance": alliance, "frame": grid[k0],
                          "slots": tuple(sorted({s for pr in pairs for s in pr})),
                          "moves": pairs,
                          "support": round(min(before, after), 3)})

    # Relabel the identity log onto virtual tracks and run the unchanged gir.
    virtual_log = []
    for row in identity_log:
        sid, f = row["slot_id"], int(row["frame_num"])
        track = vt_by_slot.get(sid)
        new_sid = track[k_of[f]] if track else sid
        virtual_log.append(row if new_sid == sid else {**row, "slot_id": new_sid})
    vres = resolve_slot_labels(
        vlm_read_rows, virtual_log, votes_by_slot, gallery_by_slot, slot_alliance,
        list(blue_roster or []), list(red_roster or []), slot_meta,
        min_score=min_score, confirm_margin=confirm_margin)
    vmeta = vres["slot_meta"]

    reads_by_slot: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for frame, slot, team, w in attributed:
        reads_by_slot[slot].append((frame, str(team), w))
    segments: dict[int, list[dict]] = {}
    for sid in sorted(vt_by_slot):
        track = vt_by_slot[sid]
        segs: list[dict] = []
        for k, vt in enumerate(track):
            if segs and segs[-1]["track"] == vt:
                segs[-1]["end_frame"] = grid[k]
                continue
            meta = vmeta.get(vt) or {}
            segs.append({"start_frame": grid[k], "end_frame": grid[k],
                         "team_number": meta.get("team_number"),
                         "confirmed": bool(meta.get("confirmed")), "track": vt})
        for seg in segs:
            team = seg["team_number"]
            seg["score"] = round(sum(
                w for f, t, w in reads_by_slot.get(sid, [])
                if team is not None and t == str(team)
                and seg["start_frame"] <= f <= seg["end_frame"]), 3)
        segments[sid] = segs

    stats = {
        "grid_frames": len(grid),
        "reads_attributed": len(attributed),
        "slots_with_positions": sorted(positions),
        "gate_open_frac": gate_open,
        "swap_lambda": p.swap_lambda,
        "prox_m": p.prox_m,
    }
    return {"segments": segments, "swaps": swaps, "slot_meta": vmeta,
            "virtual": vres, "stats": stats}


def segment_at(segments: list[dict] | None, frame: int) -> dict | None:
    """The segment covering ``frame``: the last one starting at or before
    it (frames before the first segment map to the first)."""
    if not segments:
        return None
    starts = [s["start_frame"] for s in segments]
    k = bisect.bisect_right(starts, frame) - 1
    return segments[max(k, 0)]


def format_swap_summary(res: dict, slot_label=None) -> str:
    """One finalize line: per-alliance swaps and the multi-segment slots."""
    lab = slot_label or (lambda s: _SLOT_LABELS[s] if 0 <= s < 6 else str(s))
    parts = []
    for alliance in ("blue", "red"):
        sw = [s for s in res["swaps"] if s["alliance"] == alliance]
        desc = ", ".join(
            "+".join(f"{lab(a)}<->{lab(b)}" for a, b in s.get("moves") or [s["slots"]])
            + f" @f{s['frame']} (support {s['support']:.1f})" for s in sw) or "none"
        parts.append(f"{alliance}: {desc}")
    segs = res["segments"]
    multi = " ".join(
        f"{lab(sid)}=" + ">".join(f"{sg['team_number']}[{sg['start_frame']}-{sg['end_frame']}]"
                                  for sg in segs[sid])
        for sid in sorted(segs) if len(segs[sid]) > 1)
    total = sum(len(v) for v in segs.values())
    st = res["stats"]
    return (f"[swap-resolve] lambda={st['swap_lambda']:g} prox={st['prox_m']:g}m | "
            + " | ".join(parts) + f" | segments={total}" + (f" {multi}" if multi else ""))
