"""Shared motion cleaning for the 2026 ranking metrics (movement, cycles,
defense, bump_stuck): raw labelled samples -> per-sample smoothed position,
smoothed speed and a validity flag.

Why every ranking metric goes through this instead of reading ``Sample.ax``:

* HELD POSITIONS. When tracking stalls, the pipeline repeats the exact same
  position for many frames (2026hop_qm84: every robot frozen at identical
  coords from t=0.51 to 1.31 s). Real observations jitter 0.3–0.5 m frame to
  frame, so a run of bit-identical (ax, ay) is never a measurement. A run that
  lasts longer than ``held_s`` keeps its first sample (the last real
  observation). What the repeats mean depends on how the run ENDS: on
  2026cur the next real sample is a median 0.05 m from the held position, i.e.
  the robot sat still while tracking stalled (stalls happen mostly on slow /
  parked robots). So a run whose next real sample arrives within ``gap_s``
  and within ``held_settle_m`` of the held position is KEPT as stationary
  observations (a beached robot whose track froze on the bump still counts);
  a run that ends in a jump, a gap or the end of the track is INVALID: those
  seconds are unknown and count toward neither moving, stationary, bump nor
  zone dwell.
* JITTER. Frame-to-frame speed is useless (a parked robot reads ~2 m/s).
  Positions are median-smoothed over ``smooth_s`` (centred; 1.0 s = 5 samples
  at 0.2 s frames — 0.5 s / 3 samples let ~10% of an i.i.d. 0.2 m-sigma
  parked robot read >= 0.5 m/s, 5 samples ~1.5%), and speed is the smoothed
  displacement across a centred ``speed_window_s`` window divided by the
  actual time between its two end samples. Speed is only defined where both
  window ends exist inside one SEGMENT.
* SEGMENTS. A segment is a run of valid samples with no gap > ``gap_s``
  between neighbours and no tracker-slot change (a label moving to another
  slot may be another robot: never difference positions across it).

Speeds above ``max_speed_mps`` (no FRC robot sustains them; they come from a
label jumping between robots) are set to unknown and counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from web.scouting.geometry import MIRROR, field_dims, field_symmetry

DEFAULTS = {"held_s": 0.5, "gap_s": 0.6, "smooth_s": 1.0, "speed_window_s": 1.0,
            "max_speed_mps": 6.0, "held_settle_m": 0.3}
# Bounds shared by every metric that exposes the motion params.
BOUNDS = {"held_s": (0.1, 5.0), "gap_s": (0.25, 5.0), "smooth_s": (0.0, 3.0),
          "speed_window_s": (0.4, 5.0), "max_speed_mps": (2.0, 20.0),
          "held_settle_m": (0.0, 2.0)}
# A speed window's two end samples must span at least this share of it (with
# 0.2 s frames a 1.0 s window spans 0.8 s: ends at t-0.4 and t+0.4).
_MIN_SPAN = 0.7


@dataclass
class Track:
    """One team in one match, cleaned. All arrays are per labelled sample, time-sorted."""
    t: np.ndarray            # t_match (s)
    dt: np.ndarray           # seconds each sample stands for
    teleop: np.ndarray       # bool: period == "teleop"
    valid: np.ndarray        # bool: a real observation (not a held repeat)
    seg: np.ndarray          # int segment id (-1 = invalid sample)
    ax: np.ndarray           # smoothed alliance-perspective x (nan when invalid)
    ay: np.ndarray
    x: np.ndarray            # smoothed absolute x / y (nan when invalid)
    y: np.ndarray
    speed: np.ndarray        # smoothed speed m/s (nan = unknown)
    held_s: float            # seconds of held (excluded) samples
    n_held_runs: int
    held_kept_s: float       # seconds of held samples kept as stationary (settled)
    n_implausible: int       # samples whose speed exceeded max_speed_mps (set unknown)

    def __len__(self) -> int:
        return len(self.t)


def motion_params(params: Optional[dict]) -> dict:
    return {k: float((params or {}).get(k, v)) for k, v in DEFAULTS.items()}


def _held_mask(ax: np.ndarray, ay: np.ndarray, t: np.ndarray, dt: np.ndarray,
               held_s: float, gap_s: float = 0.6, settle_m: float = 0.3) -> tuple:
    """(invalid mask, held seconds, runs, kept seconds): repeats of
    bit-identical runs lasting > held_s, unless the run SETTLES (the next
    sample comes within gap_s and within settle_m of the held position: the
    robot sat still), in which case the repeats stay valid (kept seconds)."""
    n = len(ax)
    bad = np.zeros(n, dtype=bool)
    if n < 2:
        return bad, 0.0, 0, 0.0
    same = (ax[1:] == ax[:-1]) & (ay[1:] == ay[:-1])
    # run starts: i where sample i differs from i-1
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    ends = np.concatenate((starts[1:], [n])) - 1
    secs, runs, kept = 0.0, 0, 0.0
    for a, b in zip(starts, ends):
        if b > a and (t[b] - t[a] + dt[a]) > held_s + 1e-9:
            e = b + 1
            if (e < n and t[e] - t[b] <= gap_s + 1e-9
                    and math.hypot(ax[e] - ax[a], ay[e] - ay[a]) <= settle_m):
                kept += float(dt[a + 1:b + 1].sum())
                continue
            bad[a + 1:b + 1] = True
            secs += float(dt[a + 1:b + 1].sum())
            runs += 1
    return bad, secs, runs, kept


def _rolling_median(v: np.ndarray, k: int) -> np.ndarray:
    """Centred median over k samples (k odd), shrinking at the ends."""
    if k <= 1 or len(v) < 2:
        return v.copy()
    h = k // 2
    pad = np.concatenate((np.full(h, np.nan), v, np.full(h, np.nan)))
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    return np.nanmedian(win, axis=1)


def _to_absolute(ax: np.ndarray, ay: np.ndarray, alliance: str, year: int) -> tuple:
    """Alliance perspective -> absolute (the transform is its own inverse)."""
    if alliance != "red":
        return ax.copy(), ay.copy()
    L, W = field_dims(year)
    if field_symmetry(year) == MIRROR:
        return L - ax, ay.copy()
    return L - ax, W - ay


def clean_track(samples: list, alliance: str, year: int,
                params: Optional[dict] = None) -> Track:
    """Clean one TeamMatch's samples (see the module docstring)."""
    p = motion_params(params)
    ss = sorted(samples, key=lambda s: s.t_match)
    n = len(ss)
    t = np.fromiter((s.t_match for s in ss), float, n)
    dt = np.fromiter((s.dt for s in ss), float, n)
    rax = np.fromiter((s.ax for s in ss), float, n)
    ray = np.fromiter((s.ay for s in ss), float, n)
    teleop = np.fromiter((s.period == "teleop" for s in ss), bool, n)
    slots = [s.slot for s in ss]

    invalid, held_secs, held_runs, held_kept = _held_mask(
        rax, ray, t, dt, p["held_s"], p["gap_s"], p["held_settle_m"])
    valid = ~invalid
    seg = np.full(n, -1, dtype=int)
    sax, say = np.full(n, np.nan), np.full(n, np.nan)
    speed = np.full(n, np.nan)
    vi = np.flatnonzero(valid)
    n_implausible = 0
    if len(vi):
        tv = t[vi]
        cut = np.zeros(len(vi), dtype=bool)
        cut[0] = True
        if len(vi) > 1:
            cut[1:] = np.diff(tv) > p["gap_s"]
            for j in range(1, len(vi)):
                if slots[vi[j]] != slots[vi[j - 1]]:
                    cut[j] = True
        sid = np.cumsum(cut) - 1
        seg[vi] = sid
        med_dt = float(np.median(dt[vi])) if len(vi) else 0.2
        k = max(1, int(round(p["smooth_s"] / med_dt)) if med_dt > 0 else 1)
        k += (k + 1) % 2                        # odd
        bounds = np.flatnonzero(np.concatenate((cut, [True])))
        hw = p["speed_window_s"] / 2.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = vi[a:b]
            mx, my = _rolling_median(rax[idx], k), _rolling_median(ray[idx], k)
            sax[idx], say[idx] = mx, my
            ts = t[idx]
            lo = np.searchsorted(ts, ts - hw - 1e-9, side="left")
            hi = np.searchsorted(ts, ts + hw + 1e-9, side="right") - 1
            span = ts[hi] - ts[lo]
            ok = span >= _MIN_SPAN * p["speed_window_s"]
            sp = np.full(len(idx), np.nan)
            if ok.any():
                d = np.hypot(mx[hi[ok]] - mx[lo[ok]], my[hi[ok]] - my[lo[ok]])
                sp[ok] = d / span[ok]
            wild = sp > p["max_speed_mps"]
            n_implausible += int(wild.sum())
            sp[wild] = np.nan
            speed[idx] = sp
    sx, sy = _to_absolute(sax, say, alliance, year)
    return Track(t=t, dt=dt, teleop=teleop, valid=valid, seg=seg, ax=sax, ay=say,
                 x=sx, y=sy, speed=speed, held_s=held_secs, n_held_runs=held_runs, held_kept_s=held_kept,
                 n_implausible=n_implausible)


def runs(mask: np.ndarray) -> list:
    """[(start, end_inclusive)] of consecutive True runs."""
    if not len(mask):
        return []
    m = np.concatenate(([False], mask.astype(bool), [False]))
    d = np.diff(m.astype(int))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def sustained_max(track: Track, sustain_s: float, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Highest speed the robot stayed at or above for ``sustain_s`` straight:
    max over windows of consecutive samples (one segment, every speed known,
    window spanning >= sustain_s) of the window's minimum speed. A label jump
    spikes the speed for about one speed window only, so a sustain longer than
    ``speed_window_s`` keeps such spikes out. ``mask`` restricts the samples."""
    best = None
    sel = ~np.isnan(track.speed) & (track.seg >= 0)
    if mask is not None:
        sel &= mask
    for a, b in runs(sel):
        idx = np.arange(a, b + 1)
        # split where the segment changes inside the run
        cuts = np.flatnonzero(np.diff(track.seg[idx]) != 0) + 1
        for part in np.split(idx, cuts):
            ts, sp = track.t[part], track.speed[part]
            if len(part) < 2 or ts[-1] - ts[0] < sustain_s - 1e-9:
                continue
            hi = np.searchsorted(ts, ts + sustain_s - 1e-9, side="left")
            for i in np.flatnonzero(hi < len(ts)):
                v = float(sp[i:hi[i] + 1].min())
                if best is None or v > best:
                    best = v
    return best
