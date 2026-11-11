"""Round trip (HUD post-render spec sections 3-5): a draw-list line recorded
through the live HUD path (InferenceJob._live_hud on the golden scene) and
redrawn by hud_render.draw_frame gives the live pixels. With labels the final
tags can spell exactly as the live ones, the only difference is the SAM mask
outline (the recorded polygon is approxPolyDP at 1.5 px); with a tentative live
label ("2637? (99%)"), the differences stay inside the tag areas."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import web.app as app_mod
from web import hud_draw, hud_render
from web.field.field_model import FieldModel
from web.field.radar import Radar
from tests.test_hud_golden import DEFAULT_FLAGS, scene


@pytest.fixture(autouse=True)
def _default_flags(monkeypatch):
    for k, v in DEFAULT_FLAGS.items():
        monkeypatch.setattr(app_mod, k, v)


def _ctx(job):
    """The renderer context the job's hud_static would give."""
    fm = FieldModel.load("2026-rebuilt-welded")
    radar = Radar(fm.length_m, fm.width_m, field_model=fm)
    radar.set_reference_paths([SimpleNamespace(
        alliance="blue", team_number="3128", path_m=[(2.0, 2.0), (3.0, 2.5), (4.0, 3.0)])])
    views = [{"name": v.name, "crop": [int(c) for c in v.crop],
              "masks": [[int(c) for c in m] for m in v.masks]}
             for v in job.views if v.name != "main"]
    return SimpleNamespace(flags=app_mod._live_hud_flags(), overhead_height=240, views=views,
                           ocr_regions=job.ocr_regions, wire_cal=None, field_model=fm,
                           radar=radar, approx=False, phantoms=0,
                           panel_max_width=hud_draw.PANEL_MAX_WIDTH)


def _rows(r2_team, r2_confirmed):
    """meta rows whose final tags equal the live ones (B1 confirmed 3128, B3
    ghost and corner-only R1 unlabelled)."""
    return {"B1": {"slot": "B1", "team_number": "3128", "confirmed": True},
            "R2": {"slot": "R2", "team_number": r2_team, "confirmed": r2_confirmed},
            "B3": {"slot": "B3", "team_number": None, "confirmed": False},
            "R1": {"slot": "R1", "team_number": None, "confirmed": False}}


def _live_and_line(r2_confirmed):
    job, kw = scene()
    if r2_confirmed:
        job.robot_tracks[4].team_number = "2637"
        job.robot_tracks[4].team_number_confirmed = True
    line = {}
    live = job._live_hud(**kw, draw_line=line)
    return job, kw, live, line


def _mask_outline_band(line, shape, px=3) -> np.ndarray:
    """Pixels within ``px`` of a recorded mask polygon's outline."""
    band = np.zeros(shape[:2], bool)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    for e in line["main"]:
        m = hud_draw.poly_mask(e.get("mask_poly"), shape)
        if m is not None:
            m8 = m.astype(np.uint8)
            band |= (cv2.dilate(m8, k) - cv2.erode(m8, k)).astype(bool)
    return band


def _diff(a, b) -> np.ndarray:
    assert a.shape == b.shape and a.dtype == b.dtype
    return np.any(a != b, axis=2)


def test_equal_labels_redraw_the_live_frame():
    job, kw, live, line = _live_and_line(r2_confirmed=True)
    out = hud_render.draw_frame(kw["frame"], line, _rows("2637", True), {}, _ctx(job))
    diff = _diff(live, out)
    band = _mask_outline_band(line, live.shape)
    assert diff.any() and band.any()
    assert not (diff & ~band).any(), np.argwhere(diff & ~band)[:10]


def test_a_tentative_live_tag_differs_only_in_the_tag_areas():
    job, kw, live, line = _live_and_line(r2_confirmed=False)       # live "2637? (99%) S1"
    out = hud_render.draw_frame(kw["frame"], line, _rows("2637", False), {}, _ctx(job))
    # The tag areas: where R2's tag changes the live frame and the redrawn
    # frame (vs R2 confirmed), padded a little.
    job_c, kw_c, live_c, line_c = _live_and_line(r2_confirmed=True)
    out_c = hud_render.draw_frame(kw_c["frame"], line_c, _rows("2637", True), {}, _ctx(job_c))
    tags = cv2.dilate((_diff(live, live_c) | _diff(out, out_c)).astype(np.uint8),
                      np.ones((7, 7), np.uint8)).astype(bool)
    diff = _diff(live, out)
    outside = diff & ~tags & ~_mask_outline_band(line, live.shape)
    assert diff.any() and not outside.any(), np.argwhere(outside)[:10]
