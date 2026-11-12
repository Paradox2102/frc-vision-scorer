"""HUD drawing shared by the live analysis loop and the post-finalize renderer
(docs/superpowers/specs/2026-11-10-hud-post-render-design.md section 1).

Pure functions: explicit inputs, no job state. The live loop passes live labels
and live flags; web/hud_render.py passes final labels and the flags recorded in
``meta.summary.hud_static``. tests/test_hud_golden.py pins the live pixels.
Import-light (cv2 + numpy + field geometry): the render subprocess imports it
without booting web.app.
"""
from __future__ import annotations

import base64
import dataclasses
import os
from typing import Optional

import cv2
import numpy as np

from web.field.bumper import bumper_band_box
from web.field.occlusion import hub_prisms
from web.field.overlay_geometry import (bump_wireframe_edges, hub_wireframe_edges,
                                        tower_wireframe_edges, trench_wireframe_edges)
from web.hud_util import draw_dashed_box

# Mask tint per slot (BGR): B1 green, B2 cyan, B3 yellow, R1 red, R2 magenta, R3 orange.
SLOT_COLORS = {
    0: (0, 220, 0),
    1: (255, 220, 0),
    2: (0, 220, 220),
    3: (30, 30, 255),
    4: (225, 50, 225),
    5: (0, 165, 255),
}
PANEL_MAX_WIDTH = 720


@dataclasses.dataclass(frozen=True)
class HudFlags:
    """The HUD_DRAW_* toggles (web/app.py). Defaults = the env defaults."""
    robot_mask: bool = True
    bumper_band: bool = True
    robot_label: bool = True
    robot_box: bool = False
    ground_plane: bool = False
    velocity_arrow: bool = False
    ball_boxes: bool = False
    coasting_ghost: bool = True


# (field, env var, env default) — the same names and defaults app.py reads.
FLAG_ENV = (
    ("robot_mask", "HUD_DRAW_ROBOT_MASK", "1"),
    ("bumper_band", "HUD_DRAW_BUMPER_BAND", "1"),
    ("robot_label", "HUD_DRAW_ROBOT_LABEL", "1"),
    ("robot_box", "HUD_DRAW_ROBOT_BOX", "0"),
    ("ground_plane", "HUD_DRAW_GROUND_PLANE", "0"),
    ("velocity_arrow", "HUD_DRAW_VELOCITY_ARROW", "0"),
    ("ball_boxes", "HUD_DRAW_BALL_BOXES", "0"),
    ("coasting_ghost", "HUD_DRAW_COASTING_GHOST", "1"),
)


def draw_flags(env=None) -> HudFlags:
    """Snapshot of the HUD_DRAW_* environment flags ('0'/'false' = off)."""
    env = os.environ if env is None else env
    return HudFlags(**{f: env.get(name, d) not in ("0", "false", "False")
                       for f, name, d in FLAG_ENV})


def flags_to_dict(flags: HudFlags) -> dict:
    """JSON-ready snapshot (meta.summary.hud_static.flags)."""
    return dataclasses.asdict(flags)


def flags_from_dict(d: Optional[dict]) -> HudFlags:
    """HudFlags from a flags_to_dict() snapshot (unknown keys
    ignored, missing ones default)."""
    names = {f.name for f in dataclasses.fields(HudFlags)}
    return HudFlags(**{k: bool(v) for k, v in (d or {}).items() if k in names})


def alliance_color(alliance) -> tuple:
    return (255, 80, 20) if alliance == "blue" else (30, 30, 255) if alliance == "red" else (190, 190, 190)


def slot_label(slot_id: int) -> str:
    """B1..B3 / R1..R3 (slots 0-2 blue, 3-5 red)."""
    return f"{'B' if slot_id < 3 else 'R'}{slot_id % 3 + 1}"


def slot_id_from_label(label) -> Optional[int]:
    """Inverse of slot_label ("B1" -> 0 ... "R3" -> 5); None when not a slot."""
    if not isinstance(label, str) or len(label) != 2 or label[0] not in "BR" \
            or label[1] not in "123":
        return None
    return (0 if label[0] == "B" else 3) + int(label[1]) - 1


def draw_robot(img: np.ndarray, box, alliance: str, label: str, flags: HudFlags,
               slot_id: Optional[int] = None, cal=None, field_pos=None, vel=None,
               mask: Optional[np.ndarray] = None) -> None:
    """Box evidence for one robot: mask tint, bumper band, optional box, the
    label (prefixed with the slot when ``slot_id`` is given) and the optional
    ground plane / velocity arrow."""
    color = alliance_color(alliance)
    mask_color = SLOT_COLORS.get(slot_id, color) if slot_id is not None else color
    x1, y1, x2, y2 = map(int, box)

    if mask is not None and flags.robot_mask:
        overlay = img.copy()
        overlay[mask] = mask_color
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    if flags.bumper_band:
        band = bumper_band_box(img, box)
        if band is not None:
            bx1, by1, bx2, by2 = band
            overlay = img.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            cv2.line(img, (bx1, by2), (bx2, by2), (255, 255, 255), 2)

    if flags.robot_box:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if flags.robot_label:
        text = f"{slot_label(slot_id)} {label}" if slot_id is not None else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 9)), (x1 + tw + 7, y1), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    if cal is not None and field_pos is not None and (flags.ground_plane or flags.velocity_arrow):
        draw_projected_robot_plane(img, cal, field_pos, vel, color, flags, box)


def draw_projected_robot_plane(img: np.ndarray, cal, field_pos, vel, color,
                               flags: HudFlags, box=None) -> None:
    """Bumper-ground strip (anchored to the detected bumper bottom edge) and a
    separate motion arrow. Best-effort: any projection failure draws nothing."""
    try:
        x, y = float(field_pos[0]), float(field_pos[1])
        vx, vy = (float(vel[0]), float(vel[1])) if vel is not None else (0.0, 0.0)

        if box is not None and flags.ground_plane:
            band = bumper_band_box(img, box)
            if band is not None:
                bx1, by1, bx2, by2 = band
                bottom_y = by2
                height = max(10, int((by2 - by1) * 0.45))
                skew = int((bx2 - bx1) * 0.12)
                poly = np.array([
                    (bx1, bottom_y),
                    (bx2, bottom_y),
                    (bx2 - skew, bottom_y + height),
                    (bx1 + skew, bottom_y + height),
                ], np.int32)
                h, w = img.shape[:2]
                poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly], color)
                cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
                cv2.polylines(img, [poly], True, color, 2)

        speed = float(np.hypot(vx, vy))
        wh = (img.shape[1], img.shape[0])
        center = _drawable_point(cal, (x, y, 0.0), wh)
        if flags.velocity_arrow and speed >= 0.15 and center is not None:
            n = max(speed, 1e-6)
            arrow_len = min(1.0, 0.35 + speed * 0.20)
            tip = _drawable_point(cal, (x + arrow_len * vx / n, y + arrow_len * vy / n, 0.0),
                                  wh)
            if tip is None:
                return
            cv2.arrowedLine(img, (int(center[0]), int(center[1])),
                            (int(tip[0]), int(tip[1])), (255, 255, 255), 2,
                            tipLength=0.35)
    except Exception:
        return


def draw_ghost(img: np.ndarray, box, alliance: str, label: str) -> None:
    """Coasting track: dashed last-known box + ``"<label> coast"``."""
    gcolor = (153, 60, 0) if alliance == "blue" else (0, 30, 153)
    draw_dashed_box(img, box, gcolor, thickness=2)
    gx1, gy1 = int(box[0]), int(box[1])
    cv2.putText(img, f"{label} coast", (gx1, max(12, gy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, gcolor, 1, cv2.LINE_AA)


def draw_phantom(img: np.ndarray, box, text: str) -> None:
    """A recorded robot finalize rejected (phantom / ref / static slot): thin
    grey dashed box labelled with the slot only (renderer only)."""
    color = (150, 150, 150)
    draw_dashed_box(img, box, color, thickness=1)
    cv2.putText(img, text, (int(box[0]), max(12, int(box[1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _det_box(img: np.ndarray, box, color, label) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_hub_box(img: np.ndarray, box) -> None:
    """Hub-opening detection box (green, "hub")."""
    _det_box(img, box, (0, 255, 0), "hub")


def draw_ball_box(img: np.ndarray, box) -> None:
    """Fuel-ball detection box (yellow, "ball")."""
    _det_box(img, box, (0, 255, 255), "ball")


def draw_hub(img: np.ndarray, x: float, y: float, alliance: str) -> None:
    """Hub circle + ``"<alliance> hub"``."""
    color = (255, 0, 0) if alliance == "blue" else (0, 0, 255)
    cv2.circle(img, (int(x), int(y)), 20, color, 3)
    cv2.putText(img, f"{alliance} hub", (int(x) - 30, int(y) - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_ocr_regions(img: np.ndarray, regions: dict) -> None:
    """Score / clock OCR rectangles, so placement can be checked by eye."""
    for region_key, (rx, ry, rw, rh) in regions.items():
        if "blue" in region_key:
            rect_color = (255, 0, 0)
        elif "red" in region_key:
            rect_color = (0, 0, 255)
        else:
            rect_color = (0, 255, 255)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), rect_color, 2)
        cv2.putText(img, region_key, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect_color, 1)


# BGR colour per AdvantageScope wireframe group (non-2026 seasons). Groups not listed
# draw grey; WIREFRAME_SKIP_GROUPS are not drawn (floor decals / loose bits).
WIREFRAME_GROUP_COLORS = {
    "reef": (255, 220, 0),
    "barge": (255, 0, 255),
    "coral_station": (0, 140, 255),
    "processor": (180, 80, 255),
    "perimeter": (200, 200, 200),
    "tape": (255, 255, 255),
}
WIREFRAME_DEFAULT_COLOR = (160, 160, 160)
# BGR colours of the field outline / 2 m carpet grid (every season) and the 2026
# hand-coded structures.
FIELD_OUTLINE_COLOR = (0, 255, 0)
FIELD_GRID_COLOR = (0, 180, 0)
STRUCTURE_COLORS_2026 = {
    "hubs": (255, 220, 0),
    "bumps": (0, 140, 255),
    "towers": (255, 0, 255),
    "trenches": (180, 80, 255),
}
WIREFRAME_SKIP_GROUPS = frozenset({"floor", "other"})
_WIRE_SAMPLE_M = 0.25        # sample spacing along a GLB edge (lens distortion)
_WIRE_MAX_SAMPLES = 24


def _project_many(cal, pts) -> np.ndarray:
    """(N, 3) field points -> (N, 2) pixels, the batch form of cal.field_to_image
    (same model per mode; homography ignores z, as field_to_image does)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2))
    if cal.mode == "homography":
        Hinv = np.linalg.inv(cal.H)
        return cv2.perspectiveTransform(pts[:, :2].reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    obj = pts.reshape(-1, 1, 3)
    if cal.mode == "pinhole":
        D = np.asarray(cal.D, float).reshape(-1, 1)
        img, _ = cv2.projectPoints(obj, cal.rvec, cal.tvec, cal.K, D)
    else:
        img, _ = cv2.fisheye.projectPoints(obj, cal.rvec, cal.tvec, cal.K, cal.D)
    return img.reshape(-1, 2)


_DRAW_GUARD_PX = 10000      # keep int32 polyline coords sane off-screen


def _project_drawable(cal, pts, frame_wh=None):
    """((N, 2) pixels, (N,) bool drawable) for field points. Drawable = visible to
    the camera with the frame test off (cal.project_checked(margin_px=None): in
    front of the near limit and inside the lens's fold radius — past it a point far
    outside the view folds back into the image and streaks lines through it), finite,
    and within _DRAW_GUARD_PX of the frame so the int32 polyline can't overflow.
    A duck-typed calibration without project_checked is projected point by point
    (finite + guard only). ``frame_wh``: the canvas the guard is measured from
    (default cal.image_size)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    if hasattr(cal, "project_checked"):
        uv, ok = cal.project_checked(pts, margin_px=None)
    else:
        uv = np.array([cal.field_to_image(tuple(p)) for p in pts], np.float64).reshape(-1, 2)
        ok = np.all(np.isfinite(uv), axis=1)
    w, h = frame_wh or getattr(cal, "image_size", None) or (0, 0)
    g = _DRAW_GUARD_PX
    with np.errstate(invalid="ignore"):
        ok = ok & ((uv[:, 0] >= -g) & (uv[:, 0] <= w + g)
                   & (uv[:, 1] >= -g) & (uv[:, 1] <= h + g))
    return uv, ok


def _drawable_point(cal, p, frame_wh=None):
    """(u, v) for one field point, or None when it is not drawable (_project_drawable)."""
    uv, ok = _project_drawable(cal, [p], frame_wh)
    return (float(uv[0, 0]), float(uv[0, 1])) if ok[0] else None


def _runs(uv, ok, bounds):
    """Polylines (int32 (k, 2)) of consecutive drawable samples: each [a, b) in
    ``bounds`` is one sampled edge, broken wherever a sample is not drawable."""
    polys = []
    for a, b in bounds:
        run = []
        for j in range(a, b):
            if ok[j]:
                run.append((int(uv[j, 0]), int(uv[j, 1])))
            else:
                if len(run) >= 2:
                    polys.append(np.array(run, np.int32))
                run = []
        if len(run) >= 2:
            polys.append(np.array(run, np.int32))
    return polys


def _draw_glb_wireframe(frame, cal, groups) -> None:
    """Draw AdvantageScope wireframe groups ({name: (N, 6) segments}) as sampled
    polylines, broken wherever a sample is not drawable (_project_drawable: behind /
    at the camera, past the lens fold radius, non-finite, or far off-screen) — the
    same rule as draw_field_wireframe's seg3."""
    H, Wd = frame.shape[:2]
    for name, segs in groups.items():
        if name in WIREFRAME_SKIP_GROUPS or segs is None or len(segs) == 0:
            continue
        color = WIREFRAME_GROUP_COLORS.get(name, WIREFRAME_DEFAULT_COLOR)
        segs = np.asarray(segs, float).reshape(-1, 6)
        a, b = segs[:, :3], segs[:, 3:]
        n = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) / _WIRE_SAMPLE_M),
                    1, _WIRE_MAX_SAMPLES).astype(int)
        P = np.vstack([a[i] + (b[i] - a[i]) * np.linspace(0.0, 1.0, k + 1)[:, None]
                       for i, k in enumerate(n)])
        uv, good = _project_drawable(cal, P, (Wd, H))
        ends = np.cumsum(n + 1)
        polys = _runs(uv, good, zip(ends - (n + 1), ends))
        if polys:
            cv2.polylines(frame, polys, False, color, 1)


def draw_field_wireframe(frame, cal, fm):
    """Draw the reprojected field and its structures using cal.field_to_image.
    Field lines are SAMPLED into polylines (a straight field line projects to a curved
    image line under lens distortion, so a 2-point segment can't trace it).
    Every season gets the perimeter + 2 m grid. 2026 (web.field.season.is_2026) adds
    the hand-coded hubs, bumps, towers and trenches; other seasons add their
    AdvantageScope GLB wireframe groups (data/field/wireframes/<season>.json), and a
    season without one draws just the perimeter and grid.
    Returns fp(x,y)->(int,int) so callers can add their own markers."""
    from web.field.season import is_2026, wireframe_for
    L, W = fm.length_m, fm.width_m
    H, Wd = frame.shape[:2]
    def fp(x, y):
        u, v = cal.field_to_image((x, y, 0.0))
        return (int(u), int(v))
    def seg3(p0, p1, color, thick, n=24):
        # Sampled polyline, broken at samples that are not drawable: behind or at
        # the camera, past the lens fold radius (a wide lens maps far off-axis
        # points back into the image), non-finite, or blown far off-screen.
        a, b = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
        t = np.arange(n + 1, dtype=np.float64)[:, None]
        uv, ok = _project_drawable(cal, a + (b - a) * t / n, (Wd, H))
        for run in _runs(uv, ok, [(0, n + 1)]):
            cv2.polylines(frame, [run], False, color, thick)
    def seg(p0, p1, color, thick, n=24):
        seg3((p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0),
             color, thick, n)
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))]:
        seg(a, b, FIELD_OUTLINE_COLOR, 2)
    x = 0.0
    while x <= L + 1e-6:
        seg((x, 0), (x, W), FIELD_GRID_COLOR, 1); x += 2.0
    y = 0.0
    while y <= W + 1e-6:
        seg((0, y), (L, y), FIELD_GRID_COLOR, 1); y += 2.0
    if not is_2026(fm):
        groups = wireframe_for(fm)
        if groups:
            _draw_glb_wireframe(frame, cal, groups)
        return fp
    prisms = hub_prisms(fm)
    sc = STRUCTURE_COLORS_2026
    for a, b in hub_wireframe_edges(prisms):
        seg3(a, b, sc["hubs"], 2, n=12)
    for a, b in bump_wireframe_edges(prisms, W):
        seg3(a, b, sc["bumps"], 2, n=8)
    for a, b in tower_wireframe_edges(fm):
        seg3(a, b, sc["towers"], 2, n=8)
    for a, b in trench_wireframe_edges(fm):
        seg3(a, b, sc["trenches"], 2, n=8)
    return fp


def _hex(bgr) -> str:
    b, g, r = (int(c) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def wireframe_legend(fm) -> list:
    """[{"label", "color" ("#rrggbb")}] for what draw_field_wireframe draws on
    ``fm``'s field: the outline + carpet grid, then the 2026 hand-coded structures
    or the season's AdvantageScope wireframe groups (in WIREFRAME_GROUP_COLORS
    order, unlisted groups last in the default grey)."""
    from web.field.season import is_2026, wireframe_for
    out = [{"label": "carpet grid", "color": _hex(FIELD_OUTLINE_COLOR)}]
    if is_2026(fm):
        return out + [{"label": k, "color": _hex(c)} for k, c in STRUCTURE_COLORS_2026.items()]
    groups = [g for g in (wireframe_for(fm) or {}) if g not in WIREFRAME_SKIP_GROUPS]
    known = [g for g in WIREFRAME_GROUP_COLORS if g in groups]
    rest = sorted(g for g in groups if g not in WIREFRAME_GROUP_COLORS)
    return out + [{"label": g.replace("_", " "),
                   "color": _hex(WIREFRAME_GROUP_COLORS.get(g, WIREFRAME_DEFAULT_COLOR))}
                  for g in known + rest]


def extract_view(composite_frame: np.ndarray, crop, masks=()) -> np.ndarray:
    """A camera's working frame: its crop of the composite (a copy) with the
    overlay masks blacked out (= web.field.camera_view.CameraView.extract)."""
    x, y, w, h = (int(v) for v in crop)
    out = composite_frame[y:y + h, x:x + w].copy()
    for (mx, my, mw, mh) in masks or ():
        out[my:my + mh, mx:mx + mw] = 0
    return out


def camera_panel(name: str, panel: np.ndarray, entries, flags: HudFlags,
                 max_width: int = PANEL_MAX_WIDTH) -> np.ndarray:
    """One corner-camera panel: ``panel`` (the view's working frame, drawn on
    in place) with each entry's robot evidence, scaled to ``max_width`` and
    given a name header. Entry keys: box, alliance, label, and optional
    slot_id / cal / field_pos / vel (as for draw_robot); an entry with
    ``phantom`` set is drawn with draw_phantom (renderer only)."""
    for e in entries:
        if e.get("phantom"):
            draw_phantom(panel, e["box"], e["label"])
            continue
        draw_robot(panel, e["box"], e["alliance"], e["label"], flags,
                   slot_id=e.get("slot_id"), cal=e.get("cal"),
                   field_pos=e.get("field_pos"), vel=e.get("vel"))
    h, w = panel.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        panel = cv2.resize(panel, (max_width, int(h * scale)))
    header_h = 26
    out = np.full((panel.shape[0] + header_h, panel.shape[1], 3), 18, np.uint8)
    out[header_h:, :] = panel
    cv2.putText(out, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    return out


def compose(annotated_frame, radar_img, camera_panels) -> np.ndarray:
    """Stack a larger field/corner-camera dashboard below the main view."""
    ah, aw = annotated_frame.shape[:2]
    dash_h = max(300, min(560, int(ah * 0.48)))
    dash = np.full((dash_h, aw, 3), 16, np.uint8)
    pad = 12
    title_h = 30
    body_y = pad + title_h
    body_h = dash_h - body_y - pad

    def paste_fit(dst, src, x, y, w, h, max_upscale=1.8):
        sh, sw = src.shape[:2]
        if sh <= 0 or sw <= 0 or w <= 0 or h <= 0:
            return
        scale = min(w / sw, h / sh, max_upscale)
        out = cv2.resize(src, (max(1, int(sw * scale)), max(1, int(sh * scale))))
        ox = x + max(0, (w - out.shape[1]) // 2)
        oy = y + max(0, (h - out.shape[0]) // 2)
        dst[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out

    map_w = int(aw * 0.34) if camera_panels else aw - 2 * pad
    cv2.putText(dash, "FIELD MAP", (pad, pad + 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (235, 235, 235), 1, cv2.LINE_AA)
    paste_fit(dash, radar_img, pad, body_y, map_w - pad, body_h, max_upscale=2.0)

    if camera_panels:
        cams_x = map_w + pad
        cams_w = aw - cams_x - pad
        cv2.putText(dash, "CORNER CAMERA HUD", (cams_x, pad + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
        n = len(camera_panels)
        gap = 10
        cell_w = max(1, (cams_w - gap * (n - 1)) // n)
        for i, panel in enumerate(camera_panels):
            x = cams_x + i * (cell_w + gap)
            cv2.rectangle(dash, (x, body_y), (x + cell_w, body_y + body_h),
                          (55, 55, 55), 1)
            paste_fit(dash, panel, x + 4, body_y + 4, cell_w - 8, body_h - 8,
                      max_upscale=1.7)

    cv2.line(dash, (0, 0), (aw, 0), (80, 80, 80), 1)
    return np.vstack([annotated_frame, dash])


THUMB_W, THUMB_H = 32, 18


def main_thumb(img: np.ndarray) -> str:
    """The decode-alignment fingerprint of a main view: a 32x18 grey
    (INTER_AREA) uint8 thumbnail, base64. The same decode of the same frame
    gives the same string; see thumb_diff."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(grey, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    return base64.b64encode(np.ascontiguousarray(small, np.uint8).tobytes()).decode("ascii")


def thumb_array(thumb: str) -> np.ndarray:
    """main_thumb() back to an (18, 32) uint8 array; ValueError when malformed."""
    raw = base64.b64decode(thumb, validate=True)
    if len(raw) != THUMB_W * THUMB_H:
        raise ValueError(f"thumbnail is {len(raw)} bytes, not {THUMB_W * THUMB_H}")
    return np.frombuffer(raw, np.uint8).reshape(THUMB_H, THUMB_W)


def thumb_diff(a, b) -> int:
    """The largest per-pixel difference between two main_thumb() strings
    (0-255); 255 when either is missing or malformed. The max, not the mean:
    on a fixed camera one robot moving a few pixels changes a handful of
    thumbnail pixels by 10+ levels while the mean barely moves."""
    try:
        return int(np.abs(thumb_array(a).astype(np.int16)
                          - thumb_array(b).astype(np.int16)).max())
    except (TypeError, ValueError):
        return 255


MASK_POLY_EPS_PX = 1.5
MASK_POLY_MAX_POINTS = 64


def mask_poly(mask: Optional[np.ndarray]) -> Optional[list]:
    """The mask's largest external contour, approxPolyDP(eps 1.5 px), at most
    64 points (evenly thinned when longer), as [[x, y], ...]; None for no mask
    or an empty one. Other mask components are dropped."""
    if mask is None:
        return None
    m = np.asarray(mask).astype(np.uint8)
    if not m.any():
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    pts = cv2.approxPolyDP(c, MASK_POLY_EPS_PX, True).reshape(-1, 2)
    if len(pts) > MASK_POLY_MAX_POINTS:
        step = -(-len(pts) // MASK_POLY_MAX_POINTS)
        pts = pts[::step]
    return [[int(x), int(y)] for x, y in pts]


def poly_mask(poly, shape) -> Optional[np.ndarray]:
    """Boolean mask of ``shape[:2]`` filled from a mask_poly() polygon."""
    if not poly or len(poly) < 3:
        return None
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    return m.astype(bool)


def draw_note(img: np.ndarray, text: str) -> None:
    """Small note in the bottom-right corner (renderer: "approx boxes")."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    h, w = img.shape[:2]
    x, y = w - tw - 10, h - 10
    cv2.rectangle(img, (x - 4, y - th - 6), (w - 4, h - 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                cv2.LINE_AA)
