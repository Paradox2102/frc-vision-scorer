"""Blind-spot figures for the blog: where the cameras can and can't see (2026cur, 2025hop).

Measured, not modelled: each replay's draw.jsonl.gz radar track has one record per
rendered frame, each robot ``visible`` or ``coasting`` and, when visible, ``corner_only``
(no main-camera box). A robot missing from the radar (the tracker dropped it after coasting
too long) counts as unseen at the spot where it was lost until it reappears. Per 0.25 m cell we count robot-time seen by the main camera, by the
side cameras only (2026: corner cams; 2025 Hopper: barge cams), or by nothing, and draw
three tiers on the AdvantageScope field.

    .venv/bin/python docs/blog/blindspots.py [2026cur|2025hop]            # aggregate (cached) + render
    .venv/bin/python docs/blog/blindspots.py [2026cur|2025hop] --refresh  # re-read the replays
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CELL = 0.25
DATA_DIR = Path(__file__).with_name("cmp") / "data"
IMG_DIR = Path(__file__).with_name("img")


def _batch_replays(event, window, skip):
    """Latest complete qual replay per match created inside ``window``."""
    db = sqlite3.connect(ROOT / "frc_scorer.db")
    rows = db.execute(
        "select match_key, id, created_at from replays where event_key=? and status='complete' "
        "and match_key like ? and created_at between ? and ? order by created_at",
        (event, f"{event}_qm%", *window)).fetchall()
    latest = {}
    for mk, rid, _ in rows:
        latest[mk] = rid
    return {mk: rid for mk, rid in latest.items() if mk not in skip}


EVENTS = {
    "2026cur": dict(
        field2d="2026_field2d", L=16.541, W=8.069, blur=1.0, min_samples=20,
        # The Nov 15-16 full-event batch (cur-local swarm run), same replays as the post's Curie numbers.
        # README "Where the numbers come from": qm37 robot POV, qm32/qm72 wrong video, qm76 broadcast cut.
        replays=lambda: _batch_replays("2026cur", ("2026-11-15T12:00", "2026-11-16T08:00"),
                                       {"2026cur_qm32", "2026cur_qm37", "2026cur_qm72", "2026cur_qm76"}),
        subtitle="Every robot position from all {n} Curie 2026 qualification matches, colored by which camera had it.",
        side_name="corner cameras",
    ),
    "2025hop": dict(
        field2d="2025_field2d", L=17.5482, W=8.0518, blur=2.0, min_samples=2, sparse=True, green_alpha=0.26,
        # Every 2025 Hopper qual we've processed. qm2 pinned to the replay the post's GT chart uses
        # (it has ~36 re-runs, many from fps sweeps); qm3/qm4 latest complete.
        replays=lambda: {"2025hop_qm2": "b1fc98da-f359-4ebb-b0a0-a1733c66c71c",
                         "2025hop_qm3": "4ca89189-454b-463a-b0f5-a9ccfe64ac75",
                         "2025hop_qm4": "568d7ea8-90f3-48f3-a30e-b1bcf78a8885"},
        subtitle="Every robot position from the {n} Hopper 2025 qualification matches we've run, colored by which camera had it.",
        side_name="barge cameras",
    ),
}


def cache_path(event):
    return DATA_DIR / f"blindspots_{event}.json"


def aggregate(event):
    cfg = EVENTS[event]
    L, W = cfg["L"], cfg["W"]
    nx, ny = int(round(L / CELL)), int(round(W / CELL))
    total = np.zeros((ny, nx))
    unseen = np.zeros((ny, nx))
    corner_only = np.zeros((ny, nx))
    dropped = np.zeros((ny, nx))
    reps = cfg["replays"]()
    used = []
    for mk, rid in sorted(reps.items()):
        p = ROOT / "web" / "replays" / rid / "draw.jsonl.gz"
        if not p.is_file():
            continue
        used.append(mk)
        # A slot missing from the radar was dropped by the tracker after coasting too long. Count
        # that time as unseen at the spot where it was lost, until it comes back (or the match ends).
        last = {}
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                present = set()
                for r in d.get("radar") or []:
                    pos = r.get("pos")
                    if not pos:
                        continue
                    x, y = float(pos[0]), float(pos[1])
                    present.add(r.get("slot"))
                    last[r.get("slot")] = (x, y)
                    i, j = int(x / CELL), int(y / CELL)
                    if not (0 <= i < nx and 0 <= j < ny):
                        continue
                    total[j, i] += 1
                    if r.get("state") != "visible":
                        unseen[j, i] += 1
                    elif r.get("corner_only"):
                        corner_only[j, i] += 1
                for slot, (x, y) in last.items():
                    if slot in present:
                        continue
                    i, j = int(x / CELL), int(y / CELL)
                    if 0 <= i < nx and 0 <= j < ny:
                        total[j, i] += 1
                        unseen[j, i] += 1
                        dropped[j, i] += 1
    cache = cache_path(event)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "cell_m": CELL, "event": event, "matches": used, "replays": [reps[m] for m in used],
        "total": total.tolist(), "unseen": unseen.tolist(), "corner_only": corner_only.tolist(),
        "dropped": dropped.tolist(),
    }))
    return json.loads(cache.read_text())


# Tiers (measured, per cell after a Gaussian blur of the counts, sigma = cfg["blur"] cells):
MAIN_MIN = 0.50     # main camera boxes the robot >= 50% of the time -> "main camera"
CORNER_MIN = 0.15   # else corner cameras alone carry >= 15% -> "corner cameras only"

BG, TEXT, DIM, MUTED = "#0d1220", "#e8ecf5", "#9aa5c0", "#6f7a96"
GREEN, YELLOW, RED = "#3fbf6f", "#f2c230", "#e5483b"


def tiers(data, cfg):
    from scipy.ndimage import gaussian_filter
    t, u, c = (gaussian_filter(np.array(data[k], float), cfg["blur"]) for k in ("total", "unseen", "corner_only"))
    ok = t > cfg["min_samples"]
    tt = np.maximum(t, 1e-9)
    main, corner = (t - u - c) / tt, c / tt
    tier = np.where(main >= MAIN_MIN, 0, np.where(corner >= CORNER_MIN, 1, 2))
    return np.where(ok, tier, -1), main, corner, u / tt, ok


def fill_sparse(event, tier, ok):
    """Cells with too few robot positions take the tier of the nearest measured cell, but only
    inside the main camera's frame (calibration frustum + overlay masks) and outside the
    hub/reef footprints. Everything else stays unshaded. Returns (filled_tier, filled_mask)."""
    from scipy.ndimage import distance_transform_edt
    from web.field.camera_view import load_views
    from web.field.coverage import _in_any_mask, _in_frustum
    from web.field.field_model import FieldModel
    from web.field.occlusion import _point_in_poly, hub_prisms

    views = load_views(event)
    main = next(v for v in views if v.name == "main").calibration
    prisms = hub_prisms(FieldModel.load(main.field_season))
    ny, nx = tier.shape
    _, (jj, ii) = distance_transform_edt(~ok, return_indices=True)
    filled, fmask = tier.copy(), np.zeros_like(ok)
    for j in range(ny):
        for i in range(nx):
            if ok[j, i]:
                continue
            x, y = (i + .5) * CELL, (j + .5) * CELL
            if any(_point_in_poly((x, y), p.footprint) for p in prisms):
                continue
            if _in_frustum(main, x, y) and not _in_any_mask(main, x, y):
                filled[j, i], fmask[j, i] = tier[jj[j, i], ii[j, i]], True
    return filled, fmask


def stats(data, tier):
    t, u, c = (np.array(data[k], float) for k in ("total", "unseen", "corner_only"))
    out = {}
    ny, nx = t.shape
    Y = np.repeat(((np.arange(ny) + .5) * CELL)[:, None], nx, 1)
    for k, m in {"main": tier == 0, "corner": tier == 1, "blind": tier == 2,
                 "blind_near_corners": (tier == 2) & (Y < 3.0)}.items():
        n = t[m].sum()
        out[k] = {"area_m2": float(m.sum() * CELL * CELL), "time": float(n / t.sum()),
                  "unseen": float(u[m].sum() / max(n, 1)), "corner_only": float(c[m].sum() / max(n, 1)),
                  "main": float((t - u - c)[m].sum() / max(n, 1))}
    out["all"] = {"unseen": float(u.sum() / t.sum())}
    return out


def _cam_marker(ax):
    def cam(x, y, label, ha, dx, ty=-0.7):
        ax.plot(x, y, marker="o", ms=12, mfc=CONE_SIDE, mec=BG, mew=2, zorder=7)
        ax.text(x + dx, ty, label, ha=ha, va="top", fontsize=12.5, color=CONE_SIDE, zorder=7, fontweight="bold")
    return cam


def _main_arrow(ax, L, text):
    ax.annotate("", xy=(L / 2, -0.15), xytext=(L / 2, -1.35),
                arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.2, mutation_scale=20), zorder=6)
    ax.text(L / 2, -1.45, text, ha="center", va="top", fontsize=13, color=TEXT, fontweight="bold", zorder=6)


def _callouts(fig, *items):
    xs = (0.035, 0.53) if len(items) == 2 else (0.035, 0.36, 0.685)
    fs = 12.5 if len(items) == 2 else 12
    for x, (title, body, color) in zip(xs, items):
        fig.text(x, 0.205, title, fontsize=15, fontweight="bold", color=color, va="top")
        fig.text(x, 0.178, body, fontsize=fs, color=DIM, va="top", linespacing=1.45)


def _annotate_cur(fig, ax, st, L, W, data):
    cam = _cam_marker(ax)
    cam(0.60, -0.25, "Left corner camera\n0.6 m up", "left", -0.25)
    cam(15.73, -0.14, "Right corner camera\n0.6 m up", "right", 0.25)
    _main_arrow(ax, L, "Main camera: 5.6 m back from this wall, 4.3 m up")
    b, c = st["blind_near_corners"], st["corner"]
    _callouts(fig,
              ("Behind the hubs",
               f"The hub blocks the main camera. Only the corner cameras see in, low and\n"
               f"from far away, past other robots: robots here are unseen {c['unseen']:.0%} of the time.\n"
               f"The bump and trench lane just behind each hub (red) is worse.", YELLOW),
              ("Near corners",
               f"Below the bottom of the main camera's frame, and too close to the\n"
               f"corner cameras for them to see either: robots here are unseen {b['unseen']:.0%}\n"
               f"of the time. The blue outpost and red depot are in these corners.", RED))


def region_stats(data, L, W, pred):
    """Pooled shares (main / side-only / unseen) over cells whose centre (X, Y) satisfies pred."""
    t, u, c = (np.array(data[k], float) for k in ("total", "unseen", "corner_only"))
    ny, nx = t.shape
    X, Y = np.meshgrid((np.arange(nx) + .5) * CELL, (np.arange(ny) + .5) * CELL)
    m = pred(X, Y)
    n = max(t[m].sum(), 1)
    return {"time": t[m].sum() / t.sum(), "main": (t - u - c)[m].sum() / n,
            "side": c[m].sum() / n, "unseen": u[m].sum() / n}


def _annotate_hop25(fig, ax, st, L, W, data):
    cam = _cam_marker(ax)
    # Both "corner" insets at Hopper 2025 are barge-mounted fisheyes at midfield, one facing each end.
    for x, y, tx in ((8.78, 4.02, 6.9), (9.46, 3.95, 11.35)):
        ax.annotate("", xy=(tx, y), xytext=(x, y), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=CONE_SIDE, lw=2.2, mutation_scale=18))
        ax.plot(x, y, marker="o", ms=12, mfc=CONE_SIDE, mec=BG, mew=2, zorder=7)
    ax.text(9.12, 4.55, "Barge cameras, 2 m up,\none facing each end", ha="center", va="bottom", fontsize=12.5,
            color=CONE_SIDE, fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec="none", alpha=0.85))
    _main_arrow(ax, L, "Main camera: 6.7 m back from this wall, 4.2 m up")
    from web.field.field_model import FieldModel
    from web.field.occlusion import hub_prisms, occluded_by_hub
    prisms = hub_prisms(FieldModel.load("2025-reefscape-welded"))
    main_cam = (8.667, -6.685, 4.192)          # 2025hop main calibration camera_center
    in_shadow = np.vectorize(lambda x, y: any(occluded_by_hub(main_cam, p, (x, y)) for p in prisms))
    tier = tiers(data, EVENTS["2025hop"])[0]
    near = region_stats(data, L, W, lambda X, Y: ((X < 3.5) | (X > L - 3.5)) & (Y < 1.6))
    shadow = region_stats(data, L, W, in_shadow)
    rest = region_stats(data, L, W, lambda X, Y: ~in_shadow(X, Y))
    streak = region_stats(data, L, W, lambda X, Y: (tier == 2) & (Y > 4.5) & ((X < 7) | (X > 10.5)))
    barge = region_stats(data, L, W, lambda X, Y: (X > 6.8) & (X < 10.7) & (Y > 4.6))
    on_barge = region_stats(data, L, W, lambda X, Y: (X >= 8.25) & (X < 9.3) & (Y > 4.6))
    _callouts(fig,
              ("Near coral stations",
               f"Below the main camera's frame. The barge\n"
               f"cameras look straight at them, but from\n"
               f"7-8 m away, past other robots: robots here\n"
               f"are unseen {near['unseen']:.0%} of the time.", YELLOW),
              ("Behind the reefs",
               f"Only partly blocked: the reef is an open\n"
               f"frame the main camera sees through.\n"
               f"Unseen {shadow['unseen']:.0%} across its shadow ({rest['unseen']:.0%} elsewhere),\n"
               f"{streak['unseen']:.0%} in the streak right behind it.", RED),
              ("Along the far half of the barge",
               f"The main camera looks straight down the\n"
               f"barge, so it hides the strip beside the far\n"
               f"half, and the barge cameras face away.\n"
               f"Unseen {barge['unseen']:.0%}, {on_barge['unseen']:.0%} right against the barge.", RED))


EVENTS["2026cur"]["annotate"] = _annotate_cur
EVENTS["2025hop"]["annotate"] = _annotate_hop25


CONE_MAIN, CONE_SIDE = "#ffffff", "#7cc4ff"


def draw_cones(ax, event, L, W):
    """Each camera's field of view on the ground, from its calibration: the outline of the ground
    points that project inside its image frame (solid, clipped to the field), plus the two cone
    edges traced back to the camera (dashed, up to where the frame first reaches the ground)."""
    from matplotlib.patches import Rectangle
    from scipy.ndimage import gaussian_filter, zoom
    from web.field.camera_view import load_views
    from web.field.coverage import _in_frustum

    step = 0.1
    xs = np.arange(-2.5, L + 2.5, step)
    ys = np.arange(-2.5, W + 2.5, step)
    clip = Rectangle((0, 0), L, W, transform=ax.transData)
    clip_rays = Rectangle((-0.6, -1.1), L + 1.2, W + 1.6, transform=ax.transData)  # keep off the labels
    for v in load_views(event):
        cal = v.calibration
        cc = getattr(cal, "camera_center", None)
        if cc is None:
            continue
        cx, cy = float(cc[0]), float(cc[1])
        col = CONE_MAIN if v.name == "main" else CONE_SIDE
        m = np.array([[_in_frustum(cal, x, y) for x in xs] for y in ys])
        f = 4
        mz = zoom(gaussian_filter(m.astype(float), 1.5), f, order=1)
        zx = xs[0] + (np.arange(mz.shape[1]) + .5) * step / f - step / 2
        zy = ys[0] + (np.arange(mz.shape[0]) + .5) * step / f - step / 2
        lw = 2.2 if v.name == "main" else 1.8
        cs = ax.contour(zx, zy, mz, levels=[0.5], colors=[col], linewidths=lw, alpha=0.9, zorder=5)
        cs.set_clip_path(clip)
        # Cone edges: extreme bearings of in-frame points around the camera's mean look direction.
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X[m] - cx, Y[m] - cy
        look = np.arctan2(dy.mean(), dx.mean())
        rel = (np.arctan2(dy, dx) - look + np.pi) % (2 * np.pi) - np.pi
        for k in (rel.argmin(), rel.argmax()):
            a = look + rel[k]
            r = np.arange(0.0, 30.0, 0.05)
            px, py = cx + r * np.cos(a), cy + r * np.sin(a)
            ii = np.clip(np.round((px - xs[0]) / step).astype(int), 0, len(xs) - 1)
            jj = np.clip(np.round((py - ys[0]) / step).astype(int), 0, len(ys) - 1)
            inside = (px >= xs[0]) & (px <= xs[-1]) & (py >= ys[0]) & (py <= ys[-1])
            hit = np.nonzero(inside & m[jj, ii])[0]
            if hit.size:
                r0 = r[hit[0]]
                ln, = ax.plot([cx, cx + r0 * np.cos(a)], [cy, cy + r0 * np.sin(a)], color=col, lw=1.5,
                              ls=(0, (4, 3)), alpha=0.8, zorder=5)
                ln.set_clip_path(clip_rays)


def render(event, data):
    cfg = EVENTS[event]
    L, W = cfg["L"], cfg["W"]
    out = IMG_DIR / f"blindspots_{event}.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch, Patch
    from scipy.ndimage import zoom
    from web.field.field2d import _load_meta

    tier, *_ , ok = tiers(data, cfg)
    st = stats(data, tier)
    fmask = np.zeros_like(ok)
    if cfg.get("sparse"):
        tier, fmask = fill_sparse(event, tier, ok)
        ok = ok | fmask
    meta = _load_meta(cfg["field2d"])
    img = plt.imread(meta["image_path"]).astype(float) / 255.0
    (x0, y0), (x1, y1) = meta["field_top_left_px"], meta["field_bottom_right_px"]
    sx, sy = (x1 - x0) / L, (y1 - y0) / W
    H, Wp = img.shape[:2]
    ext = (-x0 / sx, (Wp - x0) / sx, (y1 - H) / sy, y1 / sy)   # left, right, bottom, top in metres

    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(13.8, 12.0), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.02, 0.215, 0.96, 0.62])
    ax.set_facecolor(BG)
    ax.imshow(img * 0.62 + 0.03, extent=ext, origin="upper", interpolation="lanczos")

    # Smooth tier boundaries: upsample each tier's indicator and take the argmax.
    f = 8
    ind = np.stack([zoom((tier == k).astype(float), f, order=1) for k in (0, 1, 2)])
    okz = zoom(ok.astype(float), f, order=1) > 0.5
    cls = np.where(okz, ind.argmax(0), -1)
    rgba = np.zeros(cls.shape + (4,))
    fz = np.clip(zoom(fmask.astype(float), f, order=1), 0, 1)   # filled (sparse) cells: lighter, soft edge
    for k, (col, a) in enumerate([(GREEN, cfg.get("green_alpha", 0.16)), (YELLOW, 0.62), (RED, 0.66)]):
        m = cls == k
        rgba[m] = to_rgba(col, a)
        rgba[m, 3] = a * (1 - (0.3 if k == 0 else 0.5) * fz[m])
    ax.imshow(rgba, extent=(0, L, 0, W), origin="lower", interpolation="bilinear", zorder=2)
    zx = (np.arange(cls.shape[1]) + .5) * L / cls.shape[1]
    zy = (np.arange(cls.shape[0]) + .5) * W / cls.shape[0]
    for k, col in ((1, YELLOW), (2, RED)):
        ax.contour(zx, zy, (cls == k).astype(float), levels=[0.5], colors=[col], linewidths=1.6, zorder=3)

    draw_cones(ax, event, L, W)
    cfg["annotate"](fig, ax, st, L, W, data)

    ax.set_xlim(-0.6, L + 0.6)
    ax.set_ylim(-2.0, W + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.035, 0.968, "Where the cameras can see", fontsize=27, fontweight="bold", color=TEXT, va="top")
    fig.text(0.035, 0.928, cfg["subtitle"].format(n=len(data["matches"])), fontsize=14, color=DIM, va="top")
    hs = [Patch(fc=to_rgba(GREEN, .55), ec=GREEN, label="Main camera sees it"),
          Patch(fc=to_rgba(YELLOW, .8), ec=YELLOW, label=f"Only the {cfg['side_name']} see it"),
          Patch(fc=to_rgba(RED, .8), ec=RED, label="No camera sees it")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.03, 0.9), ncol=3, frameon=False,
               fontsize=13.5, labelcolor=TEXT, handlelength=1.3, handleheight=1.1, columnspacing=2.2)
    from matplotlib.lines import Line2D
    side = cfg["side_name"].replace(" cameras", "")
    ls = [Line2D([], [], color=CONE_MAIN, lw=2.2, label="Main camera's view of the floor"),
          Line2D([], [], color=CONE_SIDE, lw=1.8, label=f"{side.capitalize()} cameras' view"),
          Line2D([], [], color=DIM, lw=1.5, ls=(0, (4, 3)), label="Traced back to the camera")]
    fig.legend(handles=ls, loc="upper left", bbox_to_anchor=(0.03, 0.866), ncol=3, frameon=False,
               fontsize=12.5, labelcolor=DIM, handlelength=2.2, columnspacing=2.2)
    m = st["main"]
    fig.text(0.035, 0.085,
             f"Green: the main camera has a box on the robot at least {MAIN_MIN:.0%} of the time. Yellow: it doesn't, but the "
             f"{cfg['side_name']} carry at least {CORNER_MIN:.0%}.\nRed: neither. "
             f"Overall, robots go unseen {st['all']['unseen']:.0%} of the time ({m['unseen']:.0%} in green areas, "
             f"mostly robots blocking each other). Unseen includes\nrobots the tracker coasted, and robots it dropped after coasting too long (counted where it lost them)."
             + ("\nLighter shading: few robot positions there in 3 matches, so it's filled in from the nearest "
               "measured area inside the main camera's frame." if cfg.get("sparse") else ""),
             fontsize=12, color=MUTED, va="top", linespacing=1.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event", nargs="?", default="2026cur", choices=sorted(EVENTS))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print numbers only")
    a = ap.parse_args()
    cache = cache_path(a.event)
    data = aggregate(a.event) if a.refresh or not cache.is_file() else json.loads(cache.read_text())
    t, u, c = (np.array(data[k]) for k in ("total", "unseen", "corner_only"))
    print(f"{len(data['matches'])} matches, {int(t.sum())} robot-samples, "
          f"unseen {u.sum() / t.sum():.1%}, corner-only {c.sum() / t.sum():.1%}")
    st = stats(data, tiers(data, EVENTS[a.event])[0]) if a.stats else render(a.event, data)
    print(json.dumps(st, indent=1))
