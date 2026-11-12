"""Score pipeline replays against hand-annotated ground-truth robot tracks.

Pure functions (no DB / filesystem): the CLI ``scripts/score_ground_truth.py``
loads a replay's ``meta.json`` and the GT docs (web/field/ground_truth.py) and
hands them here.

Time base
---------
Everything is evaluated in *replay source frames* (``rf``): the 0-based frame
index of the video the replay was made from (``meta.frames[].frame_num``).
Each GT doc is mapped GT frame -> rf by :func:`align_gt` (identity when both
were made from byte-identical videos, else by match time via the GT
``match_t0_frame`` and the replay's auto-period ``start_frame``).

Pipeline position at a time
---------------------------
Bracket by replay frames. If the bracket spans <= 2x the median replay frame
spacing, slots present on both sides are linearly interpolated (team label from
the nearer side); a slot present on only one side is used if that side is
within half a spacing. Otherwise the nearest replay frame is used if within
half a spacing; else there is no pipeline data (the GT sample stays unmatched).
GT times outside the replay's ``processing_ranges`` are "unprocessed" and are
excluded from every rate (reported as a count).

Which GT samples are scored
---------------------------
``eval_step_s > 0`` (default 0.25 s): a regular time grid; at each grid point
the nearest *positioned* GT keyframe within half a step is scored (else the
nearest unknown one is counted as unknown). Metrics then don't depend on
annotation density, so dense "follow" passes and sparse "click" passes are
comparable. ``eval_step_s <= 0`` scores every keyframe.

Per-time matching
-----------------
At each scored time, every GT robot with a position (its own keyframe, or
interpolated between its positioned keyframes <= ``interp_max_gap_s`` apart)
takes part in a Hungarian assignment against the pipeline robots at that time,
cost = distance, gated at ``match_radius_m``. Only the robot owning the
keyframe is scored at that time; the others are there so two GT robots can't
both claim the same pipeline robot.

``lag_ms`` shifts ``src == "follow"`` keyframes earlier by ``lag_ms * rate`` ms
of video time (a human cursor trails the robot; ``rate`` = playback rate).
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

POSITIONED = ("visible", "estimated")
STATE_GROUPS = ("visible", "estimated", "all")   # "all" = visible + estimated
DEFAULT_SRC_FPS = 60.0
AUTO_SECONDS = 20.0          # 2026; a match_key's season overrides it (auto_seconds_for)
_BIG = 1e9
_EPS = 1e-6


class AlignmentError(ValueError):
    """GT video can't be placed on the replay's time base."""


def auto_seconds_for(match_key: Optional[str]) -> float:
    """Length of the autonomous period for ``match_key``'s season (2026: 20 s,
    2025 and earlier: 15 s); AUTO_SECONDS when there is no key."""
    if not match_key:
        return AUTO_SECONDS
    from web.field.season import season_for_event
    return float(season_for_event(str(match_key).split("_")[0]).auto_s)


def field_length_for(match_key: Optional[str], default: float = 16.541) -> float:
    """Field length (m) of ``match_key``'s season, ``default`` when unknown."""
    if not match_key:
        return default
    from web.field.season import season_for_event
    s = season_for_event(str(match_key).split("_")[0])
    return float(s.field_m[0]) if s.field_m else default


def norm_team(t) -> Optional[str]:
    if t is None:
        return None
    s = str(t).strip().lower()
    if s.startswith("frc"):
        s = s[3:]
    return s or None


# --------------------------------------------------------------------------- #
# Replay side
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ReplayTrack:
    frame_nums: np.ndarray            # (F,) sorted source frames
    slots: list                       # (K,) slot keys ("B1", ..., dupes "B1#2")
    xy: np.ndarray                    # (F, K, 2) field metres, NaN = absent
    teams: np.ndarray                 # (F, K) object: normalised team or None
    spacing: float                    # median frame spacing
    ranges: list                      # [(start, end)] processed source frames
    auto: Optional[tuple]             # (start_frame, end_frame) of auto, or None
    src_fps: float                    # fps of the replay's source video
    fingerprint: Optional[str] = None


def replay_from_meta(meta: dict, *, src_fps: Optional[float] = None,
                     fingerprint: Optional[str] = None) -> ReplayTrack:
    """Build a :class:`ReplayTrack` from a replay ``meta.json`` dict.

    ``src_fps`` defaults to ``summary.fps * median frame spacing`` (the replay
    is rendered at one frame per sampled source frame), else 60."""
    summary = meta.get("summary") or {}
    by_frame: dict[int, list] = {}
    for fr in meta.get("frames") or []:
        if fr.get("frame_num") is None:
            continue
        by_frame[int(fr["frame_num"])] = fr.get("robots") or []
    fnums = sorted(by_frame)
    rows = []
    slot_index: dict[str, int] = {}
    for fn in fnums:
        seen: Counter = Counter()
        row = {}
        for r in by_frame[fn]:
            pos = r.get("field_pos")
            if not pos or len(pos) < 2 or pos[0] is None or pos[1] is None:
                continue
            base = str(r.get("slot") or "?")
            seen[base] += 1
            key = base if seen[base] == 1 else f"{base}#{seen[base]}"
            slot_index.setdefault(key, len(slot_index))
            row[key] = (float(pos[0]), float(pos[1]), norm_team(r.get("team_number")))
        rows.append(row)
    F, K = len(fnums), len(slot_index)
    xy = np.full((F, K, 2), np.nan)
    teams = np.full((F, K), None, dtype=object)
    for i, row in enumerate(rows):
        for key, (x, y, t) in row.items():
            k = slot_index[key]
            xy[i, k] = (x, y)
            teams[i, k] = t
    frame_nums = np.asarray(fnums, dtype=float)
    spacing = float(np.median(np.diff(frame_nums))) if F >= 2 else 1.0

    ranges = [(float(a), float(b)) for a, b in (summary.get("processing_ranges") or [])]
    if not ranges and F:
        ranges = [(float(frame_nums[0]), float(frame_nums[-1]))]
    auto = None
    for p in summary.get("periods") or []:
        if p.get("label") == "auto" and p.get("start_frame") is not None:
            end = p.get("end_frame")
            auto = (float(p["start_frame"]), float(end) if end is not None else None)
            break
    if src_fps is None:
        out_fps = summary.get("fps")
        src_fps = float(out_fps) * spacing if out_fps and F >= 2 else DEFAULT_SRC_FPS
    return ReplayTrack(frame_nums, list(slot_index), xy, teams, spacing, ranges,
                       auto, float(src_fps), fingerprint)


def in_processed(rt: ReplayTrack, T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.zeros(T.shape, dtype=bool)
    for a, b in rt.ranges:
        out |= (T >= a - _EPS) & (T <= b + _EPS)
    return out


def pipeline_at(rt: ReplayTrack, T):
    """Vectorised pipeline positions at source-frame times ``T``.

    Returns ``(has_data (E,), xy (E, K, 2) NaN=absent, teams (E, K))``."""
    T = np.asarray(T, dtype=float)
    E, K, F = len(T), len(rt.slots), len(rt.frame_nums)
    xy = np.full((E, K, 2), np.nan)
    teams = np.full((E, K), None, dtype=object)
    has = np.zeros(E, dtype=bool)
    if F == 0 or E == 0:
        return has, xy, teams
    Fn, sp = rt.frame_nums, rt.spacing
    hi = np.searchsorted(Fn, T, side="left")
    hi_c = np.clip(hi, 0, F - 1)
    lo_c = np.clip(hi - 1, 0, F - 1)
    exact = (hi < F) & (np.abs(Fn[hi_c] - T) < _EPS)
    inside = (hi > 0) & (hi < F) & ~exact
    span = Fn[hi_c] - Fn[lo_c]
    bracket = inside & (span <= 2 * sp + _EPS)

    # exact frame
    if exact.any():
        xy[exact] = rt.xy[hi_c[exact]]
        teams[exact] = rt.teams[hi_c[exact]]
        has |= exact

    # interpolated bracket
    if bracket.any():
        b = bracket
        A, B = rt.xy[lo_c[b]], rt.xy[hi_c[b]]                 # (n, K, 2)
        tA, tB = rt.teams[lo_c[b]], rt.teams[hi_c[b]]
        da = (T[b] - Fn[lo_c[b]])[:, None]                     # (n, 1)
        db = (Fn[hi_c[b]] - T[b])[:, None]
        w = (da / (da + db))[..., None]                        # (n, 1, 1)
        pa, pb = ~np.isnan(A[..., 0]), ~np.isnan(B[..., 0])    # (n, K)
        both = pa & pb
        only_a = pa & ~pb & (da <= sp / 2 + _EPS)
        only_b = pb & ~pa & (db <= sp / 2 + _EPS)
        out = np.full(A.shape, np.nan)
        lerp = A + (B - A) * w
        out[both] = lerp[both]
        out[only_a] = A[only_a]
        out[only_b] = B[only_b]
        tout = np.full(tA.shape, None, dtype=object)
        nearer_a = np.broadcast_to(da <= db, both.shape)
        tout[both & nearer_a] = tA[both & nearer_a]
        tout[both & ~nearer_a] = tB[both & ~nearer_a]
        tout[only_a] = tA[only_a]
        tout[only_b] = tB[only_b]
        xy[b], teams[b] = out, tout
        has |= b

    # nearest frame within half a spacing
    rest = ~(exact | bracket)
    if rest.any():
        d_lo = np.where(hi > 0, np.abs(T - Fn[lo_c]), np.inf)
        d_hi = np.where(hi < F, np.abs(Fn[hi_c] - T), np.inf)
        near = np.where(d_lo <= d_hi, lo_c, hi_c)
        ok = rest & (np.minimum(d_lo, d_hi) <= sp / 2 + _EPS)
        if ok.any():
            xy[ok] = rt.xy[near[ok]]
            teams[ok] = rt.teams[near[ok]]
            has |= ok
    return has, xy, teams


# --------------------------------------------------------------------------- #
# GT side
# --------------------------------------------------------------------------- #
def align_gt(doc: dict, *, replay_fingerprint: Optional[str],
             replay_auto_start: Optional[float], replay_src_fps: float,
             force: bool = False) -> dict:
    """GT frame -> replay source frame: ``rf = scale * g + offset``.

    Returns ``{mode, scale, offset, warning}``; ``mode`` is ``fingerprint``,
    ``match_t0`` or ``forced``. Raises :class:`AlignmentError` otherwise."""
    v = doc.get("video") or {}
    gfp = v.get("fingerprint")
    if gfp and replay_fingerprint and gfp == replay_fingerprint:
        return {"mode": "fingerprint", "scale": 1.0, "offset": 0.0, "warning": None}
    t0 = doc.get("match_t0_frame")
    gt_fps = float(v.get("fps") or DEFAULT_SRC_FPS)
    if t0 is not None and replay_auto_start is not None:
        scale = float(replay_src_fps) / gt_fps
        return {"mode": "match_t0", "scale": scale,
                "offset": float(replay_auto_start) - float(t0) * scale, "warning": None}
    why = (f"GT video fingerprint {gfp!r} != replay source {replay_fingerprint!r}, and "
           f"match-time alignment needs GT match_t0_frame ({t0!r}) and the replay's "
           f"auto start_frame ({replay_auto_start!r})")
    if force:
        return {"mode": "forced", "scale": 1.0, "offset": 0.0,
                "warning": f"team {doc.get('team')}: FORCED identity alignment — {why}"}
    raise AlignmentError(f"team {doc.get('team')}: cannot align — {why}")


@dataclasses.dataclass
class GTTrack:
    team: str
    alliance: Optional[str]
    station: Optional[int]
    g: np.ndarray          # (N,) GT frames after lag shift (float, sorted)
    rf: np.ndarray         # (N,) replay source frames
    states: np.ndarray     # (N,) str
    xy: np.ndarray         # (N, 2), NaN unless positioned with field_xy
    gt_fps: float
    alignment: dict
    n_no_xy: int           # positioned keyframes without field_xy (unscorable)
    t0_rf: Optional[float]  # match_t0_frame mapped to rf


def gt_track(doc: dict, rt: ReplayTrack, *, lag_ms: float = 0.0,
             force: bool = False) -> GTTrack:
    al = align_gt(doc, replay_fingerprint=rt.fingerprint,
                  replay_auto_start=rt.auto[0] if rt.auto else None,
                  replay_src_fps=rt.src_fps, force=force)
    gt_fps = float((doc.get("video") or {}).get("fps") or DEFAULT_SRC_FPS)
    rows, n_no_xy = [], 0
    for kf in doc.get("keyframes") or []:
        g = float(kf["frame"])
        if lag_ms and kf.get("src") == "follow":
            g -= lag_ms / 1000.0 * float(kf.get("rate") or 1.0) * gt_fps
        st = kf.get("state")
        fxy = kf.get("field_xy")
        if st in POSITIONED and not (fxy and len(fxy) == 2 and None not in fxy):
            n_no_xy += 1
            fxy = None
        rows.append((g, st, fxy if st in POSITIONED else None))
    rows.sort(key=lambda r: r[0])
    g = np.array([r[0] for r in rows], dtype=float)
    xy = np.full((len(rows), 2), np.nan)
    for i, r in enumerate(rows):
        if r[2] is not None:
            xy[i] = r[2]
    t0 = doc.get("match_t0_frame")
    return GTTrack(
        team=norm_team(doc.get("team")) or str(doc.get("team")),
        alliance=doc.get("alliance"), station=doc.get("station"),
        g=g, rf=al["scale"] * g + al["offset"],
        states=np.array([r[1] for r in rows], dtype=object), xy=xy, gt_fps=gt_fps,
        alignment=al, n_no_xy=n_no_xy,
        t0_rf=(al["scale"] * float(t0) + al["offset"]) if t0 is not None else None)


def gt_position_at(tr: GTTrack, T, max_gap_s: float = 1.0) -> np.ndarray:
    """(E, 2) GT positions at rf times ``T``: the keyframe itself, or linear
    interpolation between two consecutive positioned keyframes <= max_gap_s
    apart (GT seconds). NaN where the robot has no position."""
    T = np.asarray(T, dtype=float)
    out = np.full((len(T), 2), np.nan)
    N = len(tr.rf)
    if N == 0 or len(T) == 0:
        return out
    hi = np.searchsorted(tr.rf, T, side="left")
    hi_c = np.clip(hi, 0, N - 1)
    lo_c = np.clip(hi - 1, 0, N - 1)
    exact = (hi < N) & (np.abs(tr.rf[hi_c] - T) < _EPS)
    out[exact] = tr.xy[hi_c[exact]]
    inside = ~exact & (hi > 0) & (hi < N)
    gap_s = (tr.g[hi_c] - tr.g[lo_c]) / tr.gt_fps
    pos = ~np.isnan(tr.xy[:, 0])
    ok = inside & pos[lo_c] & pos[hi_c] & (gap_s <= max_gap_s + _EPS)
    if ok.any():
        A, B = tr.xy[lo_c[ok]], tr.xy[hi_c[ok]]
        w = ((T[ok] - tr.rf[lo_c[ok]]) / (tr.rf[hi_c[ok]] - tr.rf[lo_c[ok]]))[:, None]
        out[ok] = A + (B - A) * w
    return out


def select_samples(tr: GTTrack, eval_step_s: float, anchor_rf: float,
                   src_fps: float) -> np.ndarray:
    """Indices of the keyframes to score (see module doc). Sorted, unique."""
    N = len(tr.rf)
    if N == 0:
        return np.zeros(0, dtype=int)
    if not eval_step_s or eval_step_s <= 0:
        return np.arange(N)
    step = eval_step_s * src_fps
    k0 = np.floor((tr.rf[0] - anchor_rf) / step)
    k1 = np.ceil((tr.rf[-1] - anchor_rf) / step)
    grid = anchor_rf + step * np.arange(k0, k1 + 1)
    half = step / 2 + _EPS

    def nearest(cand_idx: np.ndarray):
        if len(cand_idx) == 0:
            return np.full(len(grid), -1), np.full(len(grid), np.inf)
        r = tr.rf[cand_idx]
        j = np.searchsorted(r, grid)
        jl, jh = np.clip(j - 1, 0, len(r) - 1), np.clip(j, 0, len(r) - 1)
        dl, dh = np.abs(grid - r[jl]), np.abs(r[jh] - grid)
        pick = np.where(dl <= dh, jl, jh)
        return cand_idx[pick], np.minimum(dl, dh)

    positioned = np.flatnonzero(~np.isnan(tr.xy[:, 0]))
    unpositioned = np.flatnonzero(np.isnan(tr.xy[:, 0]))
    ip, dp = nearest(positioned)
    iu, du = nearest(unpositioned)
    chosen = np.where(dp <= half, ip, np.where(du <= half, iu, -1))
    return np.unique(chosen[chosen >= 0])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _stats(vals) -> dict:
    if not len(vals):
        return {"mean": None, "median": None, "p90": None}
    a = np.asarray(vals, dtype=float)
    return {"mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}


def _rate(num, den):
    return round(num / den, 4) if den else None


def group_metrics(recs: list, team: str, radius: float) -> dict:
    """Metrics over one robot's scored samples (already filtered to a state
    group). Rates are over *processed* samples: coverage = label_correct +
    label_wrong + unlabeled."""
    proc = [r for r in recs if r["processed"]]
    matched = [r for r in proc if r["slot"] is not None]
    correct = sum(1 for r in matched if r["team"] == team)
    wrong = sum(1 for r in matched if r["team"] is not None and r["team"] != team)
    unlabeled = sum(1 for r in matched if r["team"] is None)
    lp = [r for r in proc if r["label_present"]]
    ld = [r["label_dist"] for r in lp]
    seq = [r["slot"] for r in sorted(matched, key=lambda r: r["rf"])]
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    main_slot, main_n = (Counter(seq).most_common(1)[0] if seq else (None, 0))
    return {
        "n": len(recs), "n_processed": len(proc), "n_unprocessed": len(recs) - len(proc),
        "n_matched": len(matched),
        "coverage": _rate(len(matched), len(proc)),
        "err": _stats([r["dist"] for r in matched]),
        "label_correct": _rate(correct, len(proc)),
        "label_wrong": _rate(wrong, len(proc)),
        "unlabeled": _rate(unlabeled, len(proc)),
        "label_present": _rate(len(lp), len(proc)),
        "label_err": _stats(ld),
        "label_on_wrong_robot": _rate(sum(1 for d in ld if d > radius), len(lp)),
        "slot_switches": switches,
        "main_slot": main_slot, "main_slot_share": _rate(main_n, len(seq)),
        "n_correct": correct, "n_wrong": wrong, "n_unlabeled": unlabeled,
    }


def aggregate(robot_recs: dict, radius: float) -> dict:
    """Across robots for one state group: mean of per-robot coverage, pooled
    errors and label rates, total slot switches."""
    covs, dists, ld = [], [], []
    n_proc = n_corr = n_wrong = n_unl = n_lp = n_lwr = sw = n = 0
    for team, recs in robot_recs.items():
        m = group_metrics(recs, team, radius)
        n += m["n"]
        if m["n_processed"]:
            covs.append(m["coverage"])
        n_proc += m["n_processed"]
        n_corr += m["n_correct"]
        n_wrong += m["n_wrong"]
        n_unl += m["n_unlabeled"]
        sw += m["slot_switches"]
        dists += [r["dist"] for r in recs if r["processed"] and r["slot"] is not None]
        lpr = [r["label_dist"] for r in recs if r["processed"] and r["label_present"]]
        n_lp += len(lpr)
        n_lwr += sum(1 for d in lpr if d > radius)
        ld += lpr
    return {
        "n_robots": len(covs), "n": n, "n_processed": n_proc,
        "coverage": round(float(np.mean(covs)), 4) if covs else None,
        "err": _stats(dists),
        "label_correct": _rate(n_corr, n_proc), "label_wrong": _rate(n_wrong, n_proc),
        "unlabeled": _rate(n_unl, n_proc), "label_present": _rate(n_lp, n_proc),
        "label_err": _stats(ld), "label_on_wrong_robot": _rate(n_lwr, n_lp),
        "slot_switches": sw,
    }


def identity_metrics(recs: list, team: str, alliance: Optional[str],
                     team_alliance: dict, length_m: float, step_s: float) -> dict:
    """Identity-focused view of one robot's processed samples (time order):

    * lost: no pipeline robot within the radius — seconds, episodes, longest.
    * misid: matched robot labelled with ANOTHER team — seconds, split into
      cross-alliance (label belongs to the other alliance) vs same-alliance.
    * wrong_colour: matched pipeline SLOT belongs to the other alliance
      (B* vs R*) — a tracking error independent of bumper reads; reported
      overall and split by whether the robot is in its own or the opponent's
      half (``x`` = GT x)."""
    proc = sorted((r for r in recs if r["processed"]), key=lambda r: r["rf"])
    lost_eps, run = [], 0
    for r in proc:
        if r["slot"] is None:
            run += 1
        elif run:
            lost_eps.append(run); run = 0
    if run:
        lost_eps.append(run)
    matched = [r for r in proc if r["slot"] is not None]
    mis = [r for r in matched if r["team"] is not None and r["team"] != team]
    cross = sum(1 for r in mis if alliance and team_alliance.get(r["team"])
                and team_alliance[r["team"]] != alliance)
    def _slot_al(slot):
        return "blue" if str(slot).startswith("B") else "red" if str(slot).startswith("R") else None
    wc = [r for r in matched if alliance and _slot_al(r["slot"]) not in (None, alliance)]
    def _half(r):
        if r.get("x") is None or not alliance:
            return None
        return "own" if ("blue" if r["x"] < length_m / 2 else "red") == alliance else "opp"
    own = [r for r in matched if _half(r) == "own"]
    opp = [r for r in matched if _half(r) == "opp"]
    wcs = set(id(r) for r in wc)
    return {
        "lost_s": round(sum(lost_eps) * step_s, 2), "lost_episodes": len(lost_eps),
        "longest_lost_s": round(max(lost_eps, default=0) * step_s, 2),
        "misid_s": round(len(mis) * step_s, 2),
        "misid_cross_s": round(cross * step_s, 2),
        "misid_same_s": round((len(mis) - cross) * step_s, 2),
        "wrong_colour_rate": _rate(len(wc), len(matched)),
        "wrong_colour_own_half": _rate(sum(1 for r in own if id(r) in wcs), len(own)),
        "wrong_colour_opp_half": _rate(sum(1 for r in opp if id(r) in wcs), len(opp)),
        "n_matched": len(matched), "n_wrong_colour": len(wc),
        "n_own": len(own), "n_own_wc": sum(1 for r in own if id(r) in wcs),
        "n_opp": len(opp), "n_opp_wc": sum(1 for r in opp if id(r) in wcs),
    }


def aggregate_identity(rows: list) -> dict:
    """Pool identity_metrics across robots (seconds summed, rates pooled)."""
    tot = lambda k: round(sum(r[k] for r in rows), 2)
    nm, nwc = sum(r["n_matched"] for r in rows), sum(r["n_wrong_colour"] for r in rows)
    return {
        "lost_s": tot("lost_s"), "lost_episodes": sum(r["lost_episodes"] for r in rows),
        "longest_lost_s": max((r["longest_lost_s"] for r in rows), default=0),
        "misid_s": tot("misid_s"), "misid_cross_s": tot("misid_cross_s"),
        "misid_same_s": tot("misid_same_s"),
        "wrong_colour_rate": _rate(nwc, nm),
        "wrong_colour_own_half": _rate(sum(r["n_own_wc"] for r in rows), sum(r["n_own"] for r in rows)),
        "wrong_colour_opp_half": _rate(sum(r["n_opp_wc"] for r in rows), sum(r["n_opp"] for r in rows)),
    }


def _in_group(state: str, group: str) -> bool:
    return state in POSITIONED if group == "all" else state == group


def score_match(gt_docs: Iterable[dict], rt: ReplayTrack, *, match_radius_m: float = 1.0,
                period: str = "full", eval_step_s: float = 0.25, lag_ms: float = 0.0,
                interp_max_gap_s: float = 1.0, force: bool = False,
                details: bool = False, roster: Optional[dict] = None,
                field_length_m: Optional[float] = None,
                match_key: Optional[str] = None) -> dict:
    """Score one replay against a match's GT docs. See module doc.

    Returns ``{params, window, alignment, warnings, robots: [...],
    aggregate: {visible, estimated, all}, headline}``; ``headline`` is the
    visible-keyframe aggregate. ``details`` adds per-sample records.
    ``match_key`` picks the season's auto length (auto window when the replay has no
    auto period) and, unless ``field_length_m`` is given, its field length."""
    auto_seconds = auto_seconds_for(match_key)
    if field_length_m is None:
        field_length_m = field_length_for(match_key)
    docs = [d for d in gt_docs if d and d.get("team") is not None]
    tracks = [gt_track(d, rt, lag_ms=lag_ms, force=force) for d in docs]
    warnings = [t.alignment["warning"] for t in tracks if t.alignment["warning"]]
    R = len(tracks)

    window = None
    if period == "auto":
        if rt.auto and rt.auto[1] is not None:
            window = (rt.auto[0], rt.auto[1])
        else:
            t0s = [t for t in tracks if t.t0_rf is not None]
            if not t0s:
                raise AlignmentError("period=auto: replay has no auto period and no GT "
                                     "doc has match_t0_frame")
            t = t0s[0]
            window = (t.t0_rf, t.t0_rf + auto_seconds * t.gt_fps * t.alignment["scale"])
    elif period != "full":
        raise ValueError(f"period must be 'auto' or 'full', not {period!r}")
    anchor = rt.auto[0] if rt.auto else 0.0

    # scored samples per robot
    samples = []                       # (robot, kf idx, rf, state)
    for ri, tr in enumerate(tracks):
        for i in select_samples(tr, eval_step_s, anchor, rt.src_fps):
            rfv = float(tr.rf[i])
            if window and not (window[0] - _EPS <= rfv <= window[1] + _EPS):
                continue
            samples.append((ri, int(i), rfv, tr.states[i]))
    pos_samples = [s for s in samples if not np.isnan(tracks[s[0]].xy[s[1], 0])]

    events = np.unique(np.array([s[2] for s in pos_samples], dtype=float))
    ev_index = {float(e): j for j, e in enumerate(events)}
    processed = in_processed(rt, events)
    has, P, PT = pipeline_at(rt, events)
    G = np.stack([gt_position_at(tr, events, interp_max_gap_s) for tr in tracks], axis=1) \
        if R else np.zeros((len(events), 0, 2))
    for ri, i, rfv, _ in pos_samples:              # owner's own keyframe wins
        G[ev_index[rfv], ri] = tracks[ri].xy[i]

    # Hungarian per event (only events with scored samples, processed + data)
    assign = {}                                     # (event j, robot) -> (k, dist)
    for j in range(len(events)):
        if not (processed[j] and has[j]):
            continue
        gi = np.flatnonzero(~np.isnan(G[j, :, 0]))
        pk = np.flatnonzero(~np.isnan(P[j, :, 0]))
        if not len(gi) or not len(pk):
            continue
        D = np.linalg.norm(G[j, gi][:, None, :] - P[j, pk][None, :, :], axis=2)
        C = np.where(D <= match_radius_m, D, _BIG)
        rows, cols = linear_sum_assignment(C)
        for a, b in zip(rows, cols):
            if D[a, b] <= match_radius_m:
                assign[(j, int(gi[a]))] = (int(pk[b]), float(D[a, b]))

    recs_by_robot: list[list] = [[] for _ in range(R)]
    unknown = [0] * R
    for ri, i, rfv, st in samples:
        tr = tracks[ri]
        if np.isnan(tr.xy[i, 0]):
            if st == "unknown":
                unknown[ri] += 1
            continue
        j = ev_index[rfv]
        rec = {"frame": int(round(tr.g[i])), "rf": round(rfv, 2), "state": st,
               "x": float(tr.xy[i][0]),
               "processed": bool(processed[j]), "slot": None, "team": None,
               "dist": None, "label_present": False, "label_dist": None}
        if processed[j]:
            hit = assign.get((j, ri))
            if hit:
                k, d = hit
                rec.update(slot=rt.slots[k], team=PT[j, k], dist=round(d, 4))
            lk = [k for k in range(len(rt.slots))
                  if PT[j, k] == tr.team and not np.isnan(P[j, k, 0])]
            if lk:
                rec["label_present"] = True
                rec["label_dist"] = round(float(min(np.linalg.norm(P[j, k] - tr.xy[i])
                                                    for k in lk)), 4)
        recs_by_robot[ri].append(rec)

    robots = []
    for ri, tr in enumerate(tracks):
        recs = recs_by_robot[ri]
        row = {"team": tr.team, "alliance": tr.alliance, "station": tr.station,
               "alignment": tr.alignment["mode"], "n_keyframes": len(tr.rf),
               "n_unknown": unknown[ri], "n_no_xy": tr.n_no_xy}
        for grp in STATE_GROUPS:
            row[grp] = group_metrics([r for r in recs if _in_group(r["state"], grp)],
                                     tr.team, match_radius_m)
        if details:
            row["samples"] = recs
        robots.append(row)
    # identity view (visible samples): team -> alliance from the GT docs plus
    # an optional roster {team: alliance} for teams without GT (no-shows etc.)
    team_alliance = {str(t): a for t, a in (roster or {}).items()}
    team_alliance.update({tr.team: tr.alliance for tr in tracks if tr.alliance})
    step = eval_step_s if eval_step_s and eval_step_s > 0 else (1.0 / tracks[0].gt_fps if R else 0.0)
    id_rows = []
    for ri, tr in enumerate(tracks):
        im = identity_metrics([r for r in recs_by_robot[ri] if r["state"] == "visible"],
                              tr.team, tr.alliance, team_alliance, field_length_m, step)
        robots[ri]["identity"] = im
        id_rows.append(im)
    agg = {grp: aggregate({tr.team: [r for r in recs_by_robot[ri] if _in_group(r["state"], grp)]
                           for ri, tr in enumerate(tracks)}, match_radius_m)
           for grp in STATE_GROUPS}
    agg["visible"]["n_unknown"] = sum(unknown)
    return {
        "params": {"match_radius_m": match_radius_m, "period": period,
                   "eval_step_s": eval_step_s, "lag_ms": lag_ms,
                   "interp_max_gap_s": interp_max_gap_s,
                   "replay_src_fps": rt.src_fps, "replay_spacing": rt.spacing},
        "window": [round(window[0], 2), round(window[1], 2)] if window else None,
        "warnings": warnings,
        "robots": robots,
        "aggregate": agg,
        "identity": aggregate_identity(id_rows),
        "headline": agg["visible"],
    }
