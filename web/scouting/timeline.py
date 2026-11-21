"""Per-match phase timestamps on the SOURCE VIDEO (the TBA / YouTube match video).

The scouting export carries one ``timeline`` per match so another system can
seek the match video to any phase without re-running our clock scan:

  marks   points in video seconds: match_start, auto_end, teleop_start,
          endgame_start, match_end
  phases  intervals: auto, between (the auto->teleop pause), teleop, endgame,
          and for 2026 the teleop hub shifts (transition, shift1..4)

VIDEO TIME is ``frame / fps``, fps = the source file's average frame rate
(ffprobe). The 2026cur videos are constant-rate, some at odd rates (59.716,
59.94); OpenCV's seek lands exactly on frame / that rate, which is how the
pipeline numbers its frames.

WHERE THE BOUNDARIES COME FROM (2026cur, 138 matches, Nov 20):
  * The broadcast clock shows ceil(remaining) and ticks once a second, so each
    observed tick d+1 -> d (clock OCR is sampled every 0.5 s and cached per
    video as ``web/uploads/clockscan_*.json``) brackets the instant the
    displayed countdown hits 0 to within the sample gap. Teleop has ~138 ticks,
    so the countdown's zero is pinned to well under 0.25 s.
  * That on-screen clock trails the field: the frame-accurate scoreboard start
    indicator lights a median DISPLAY_LAG_S = 0.53 s before the fitted auto
    clock start (135 matches, p10..p90 0.17..0.83). Robots also start moving
    ~0.3 s before it (clock.py). Field time = display time - DISPLAY_LAG_S.
  * match_start = the indicator frame (``true_start_frame``) when present;
    every other boundary (auto_end, teleop_start, each 2026 shift, endgame,
    match_end) = the display time of that countdown value, fitted from the
    clock ticks within 6 s of it, minus the lag. NOT start + rule length: the
    broadcast clock runs ~1% slow against these videos (teleop median 141.5 s
    of video, qm20/25/31/110 143-144.5 s). Measured: auto 20.08 s, the
    auto->teleop pause ("between") 4.0 s (4.0-4.6).
  * Without a clock-scan cache, teleop_start falls back to the clock-scan
    ``start_frame`` + TELEOP_START_FROM_LAST_START_S (measured median -0.38 s,
    p10..p90 -0.75..-0.18) and auto to ``start_frame`` - CLOCK_LEAD_S.
    ``marks[*].source`` says which anchor each value used.
"""
from __future__ import annotations

import functools
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Optional

from web.scouting.clock import CLOCK_LEAD_S, MatchClock

TIMELINE_VERSION = 1
REPO = Path(__file__).resolve().parents[2]

DISPLAY_LAG_S = 0.5
TELEOP_START_FROM_LAST_START_S = -0.4
# Must match web/app.py (clock_scan_cache_path / OVERHEAD_CROP_RATIO /
# MATCH_CLOCK_SAMPLE_SECONDS) - the cache key the job wrote.
_OVERHEAD_CROP_RATIO = 0.67
_CLOCK_SAMPLE_S = 0.5

# Period lengths by rule (the clock scan occasionally misreads the start value).
PERIOD_S = {2026: {"auto": 20, "teleop": 140}, 2025: {"auto": 15, "teleop": 135},
            2024: {"auto": 15, "teleop": 135}, 2023: {"auto": 15, "teleop": 135},
            2022: {"auto": 15, "teleop": 135}}
# Countdown value (teleop seconds remaining) at which endgame starts, per season.
ENDGAME_REMAINING_S = {2026: 30.0, 2025: 20.0, 2024: 20.0, 2023: 30.0, 2022: 30.0}
DEFAULT_ENDGAME_REMAINING_S = 30.0


# ----------------------------------------------------------------- video probe
@functools.lru_cache(maxsize=1024)
def _probe(path: str, size: int, mtime: float) -> Optional[dict]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,height", "-of", "json",
             path], capture_output=True, text=True, timeout=30).stdout
        st = (json.loads(out).get("streams") or [{}])[0]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    def rate(s):
        try:
            n, d = str(s).split("/")
            return float(n) / float(d) if float(d) else None
        except (TypeError, ValueError):
            return None

    def num(v, cast):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    if not fps or fps <= 0:
        return None
    return {"fps": fps, "frame_count": num(st.get("nb_frames"), int),
            "duration_s": num(st.get("duration"), float), "height": num(st.get("height"), int)}


def video_info(path) -> Optional[dict]:
    """{fps, frame_count, duration_s, height} of a video file, or None."""
    if not path:
        return None
    try:
        s = Path(path).stat()
    except OSError:
        return None
    return _probe(str(path), int(s.st_size), float(s.st_mtime))


# ----------------------------------------------------------------- clock scan
def _clock_scan_samples(video_path, fps: float, height: Optional[int]) -> Optional[list]:
    """The job's cached clock OCR [(frame, seconds|None)], or None."""
    if not video_path or not height:
        return None
    try:
        st = Path(video_path).stat()
    except OSError:
        return None
    out = _clock_scan_cached(str(video_path), int(st.st_size), float(st.st_mtime),
                             float(fps), int(height))
    return list(out) if out is not None else None


@functools.lru_cache(maxsize=1024)
def _clock_scan_cached(video_path: str, size: int, mtime: float, fps: float,
                       height: int) -> Optional[tuple]:
    p = Path(video_path)
    step = max(1, int(round(fps * _CLOCK_SAMPLE_S)))
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
        h.update(f"{p.stat().st_size}:{step}:{int(height * _OVERHEAD_CROP_RATIO)}".encode())
    except OSError:
        return None
    name = f"clockscan_{h.hexdigest()}.json"
    for d in (p.parent, REPO / "web" / "uploads"):
        try:
            return tuple((int(f), None if v is None else int(v))
                         for f, v in json.loads((d / name).read_text()))
        except (OSError, ValueError, TypeError):
            continue
    return None


def countdown_time_at(samples: list, fps: float, lo_frame: int, hi_frame: int,
                      start_value: int, remaining: float, window: int = 6,
                      min_ticks: int = 3) -> Optional[tuple]:
    """(video second the DISPLAYED countdown passes ``remaining``, ticks used).

    The display shows ceil(remaining) and ticks once a second, so a tick
    d+1 -> d seen between samples at t0 and t1 puts the countdown's zero ``c``
    in (t0 + d, t1 + d]. Only ticks within ``window`` s of ``remaining`` are
    used: the broadcast clock is NOT real-time against every video (2026cur
    qm25: teleop spans ~143 s of video), so each boundary is fitted locally.
    ``c`` = the intersection of the brackets (their midpoints' median if OCR
    misreads make them disagree); the answer is ``c - remaining``."""
    pts = [(f / fps, v) for f, v in samples if v is not None and lo_frame <= f <= hi_frame]
    los, his = [], []
    gap = 1.5 * _CLOCK_SAMPLE_S
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if (v0 == v1 + 1 and 0 <= v1 < start_value and abs(v1 - remaining) <= window
                and 0 < t1 - t0 <= gap):
            los.append(t0 + v1)
            his.append(t1 + v1)
    if len(los) < min_ticks:
        return None
    lo, hi = max(los), min(his)
    c = (lo + hi) / 2.0 if lo <= hi else statistics.median(
        (x + y) / 2.0 for x, y in zip(los, his))
    return c - remaining, len(los)


# ----------------------------------------------------------------- timeline
def _clock_str(seconds: float) -> str:
    """152.25 -> '2:32.25' (video position, for humans)."""
    m, s = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(m)}:{s:05.2f}"


def _shifts(year: int) -> list:
    """[(name, remaining_hi, remaining_lo)] teleop sub-phases for the season."""
    if int(year) == 2026:
        from web.scouting.games.rebuilt2026 import SHIFTS
        return [(n, hi, max(lo, 0.0)) for n, hi, lo in SHIFTS]
    return []


def for_match(md) -> Optional[dict]:
    """``build_timeline`` for a loader ``MatchData``."""
    return build_timeline(md.periods, md.year, md.video_path, md.youtube_key)


def build_timeline(periods: list, year: int, video_path=None,
                   youtube_key: Optional[str] = None) -> Optional[dict]:
    """The ``timeline`` block for one match, or None without a match clock."""
    clock = MatchClock(periods or [])
    if not clock.ok:
        return None
    info = video_info(video_path)
    if info:
        fps, fps_source = info["fps"], "video"
    else:
        # No source file: the clock-scan rate (period frames / clock seconds)
        # is within ~2% of the real rate.
        fps, fps_source, info = clock.frames_per_s(), "clock_estimate", {}
    if not fps or fps <= 0:
        return None
    samples = _clock_scan_samples(video_path, fps, info.get("height")) if info else None
    raw = {q.get("label"): q for q in periods or []}
    year = int(year)
    endgame_r = ENDGAME_REMAINING_S.get(year, DEFAULT_ENDGAME_REMAINING_S)

    def length(label):
        rule = PERIOD_S.get(year, {}).get(label)
        return float(rule) if rule else clock.periods[label].length_s

    def on_clock(label, remaining):
        """(field video second at which ``label`` has ``remaining`` s left, how)."""
        if not samples or label not in clock.periods:
            return None
        p, q = clock.periods[label], raw[label]
        lo = int(q.get("clock_first_frame", p.start_frame)) - int(fps)
        hit = countdown_time_at(samples, fps, lo, p.end_frame + int(2 * fps),
                                int(round(length(label))), remaining)
        if hit is None:
            return None
        return hit[0] - DISPLAY_LAG_S, f"clock_ticks({hit[1]})"

    marks: dict = {}

    def mark(name, t, how):
        t = max(0.0, float(t))
        marks[name] = {"t": round(t, 3), "frame": int(round(t * fps)),
                       "video_time": _clock_str(t), "source": how}

    a = clock.periods.get("auto")
    if a is not None:
        a_len = length("auto")
        if a.true_start_frame is not None:
            start = (a.true_start_frame / fps,
                     raw["auto"].get("start_method") or "true_start_frame")
        else:
            start = on_clock("auto", a_len) or (a.start_frame / fps - CLOCK_LEAD_S, "clock_lead")
        mark("match_start", *start)
        end = on_clock("auto", 0.0) or (start[0] + a_len, f"{start[1]}+{a_len:g}s")
        mark("auto_end", *end)

    t = clock.periods.get("teleop")
    tele_at = {}
    if t is not None:
        t_len = length("teleop")
        if t.true_start_frame is not None:
            start = (t.true_start_frame / fps, "true_start_frame")
        else:
            start = (on_clock("teleop", t_len) or
                     (t.start_frame / fps + TELEOP_START_FROM_LAST_START_S, "clock_lead"))

        def at(r):
            """Teleop boundary with r s remaining: local clock fit, else start + rule."""
            if r >= t_len:
                return start
            return on_clock("teleop", r) or (start[0] + t_len - r,
                                             f"{start[1]}+{t_len - r:g}s")

        tele_at = {r: at(r) for r in {t_len, endgame_r, 0.0} |
                   {hi for _, hi, _ in _shifts(year)} | {lo for _, _, lo in _shifts(year)}}
        mark("teleop_start", *tele_at[t_len])
        mark("endgame_start", *tele_at[endgame_r])
        mark("match_end", *tele_at[0.0])

    phases = []

    def phase(name, ta, tb, parent=None):
        row = {"phase": name, "start_s": round(ta, 3), "end_s": round(tb, 3),
               "duration_s": round(tb - ta, 3),
               "start_frame": int(round(ta * fps)), "end_frame": int(round(tb * fps))}
        if parent:
            row["within"] = parent
        phases.append(row)

    def between(name, ma, mb, parent=None):
        if ma in marks and mb in marks:
            phase(name, marks[ma]["t"], marks[mb]["t"], parent)

    between("auto", "match_start", "auto_end")
    between("between", "auto_end", "teleop_start")
    between("teleop", "teleop_start", "match_end")
    if tele_at:
        subs = _shifts(year) or [("endgame", endgame_r, 0.0)]
        for name, hi, lo in subs:
            phase(name, max(0.0, tele_at[hi][0]), max(0.0, tele_at[lo][0]), parent="teleop")
    phases.sort(key=lambda r: (r["start_s"], "within" in r))

    out = {"timeline_version": TIMELINE_VERSION,
           "video": {"fps": round(fps, 4), "fps_source": fps_source,
                     "frame_count": info.get("frame_count"),
                     "duration_s": (round(info["duration_s"], 3)
                                    if info.get("duration_s") else None),
                     "youtube_key": youtube_key or None, "youtube_url": None},
           "marks": marks, "phases": phases}
    if youtube_key and "match_start" in marks:
        out["video"]["youtube_url"] = (f"https://www.youtube.com/watch?v={youtube_key}"
                                       f"&t={int(marks['match_start']['t'])}s")
    return out
