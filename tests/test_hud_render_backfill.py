"""Back-fill (HUD post-render spec section 6): a replay recorded before the
draw list existed gets an approximate one from meta.frames, with boxes
projected through the main calibration (a fake one here) and no score."""
import asyncio

import pytest

import web.database as database
from web import hud_draw, hud_render
from tests.test_hud_render import _frame_count, make_replay


class FakeCal:
    """u = 40 x, v = 150 - 10 y - 100 z; optional camera centre for the push."""
    field_season = "2026-rebuilt-welded"

    def __init__(self, camera_center=None):
        self.camera_center = camera_center

    def field_to_image(self, p):
        x, y, z = p
        return (40.0 * x, 150.0 - 10.0 * y - 100.0 * z)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    return tmp_path / "t.db"


def test_box_is_the_projected_robot_volume():
    box = hud_render.backfill_box(FakeCal(), (2.0, 3.0), 0.0)
    # x 1.525..2.475 -> u 61..99; y 2.525..3.475, z 0..0.5 -> v 65.25..124.75
    assert box == pytest.approx([61.0, 65.25, 99.0, 124.75])


def test_box_centre_is_pushed_away_from_the_camera_by_the_remaining_offset():
    cal = FakeCal(camera_center=(0.0, 3.0, 5.0))       # camera on the y=3 line at x=0
    box = hud_render.backfill_box(cal, (2.0, 3.0), 0.2)  # push 0.475 - 0.2 = 0.275 m in +x
    assert box[0] == pytest.approx(40.0 * (2.275 - 0.475))
    assert hud_render.backfill_box(cal, (2.0, 3.0), 0.6) == \
        hud_render.backfill_box(FakeCal(), (2.0, 3.0), 0.0)   # offset spent: no push


def test_unprojectable_positions_have_no_box():
    class Bad(FakeCal):
        def field_to_image(self, p):
            return (float("nan"), 0.0)
    assert hud_render.backfill_box(Bad(), (2.0, 3.0), 0.0) is None


def test_lines_have_boxes_trails_and_no_score():
    frames = [{"i": i, "frame_num": 10 + i, "robots": [
        {"slot": "B2", "alliance": "blue", "team_number": "6773", "field_pos": [2.0 + i, 3.0]},
        {"slot": "R1", "alliance": "red", "team_number": None, "field_pos": [40.0, 3.0]}]}
        for i in range(20)]
    frames[5]["robots"] = frames[5]["robots"][1:]              # B2 vanishes once
    lines = hud_render.build_backfill_lines(frames, FakeCal(), 0.0, (320, 160))
    assert sorted(lines) == list(range(20))
    b2 = lines[3]["main"][0]
    assert b2["slot"] == "B2" and b2["slot_id"] == 1 and b2["score"] is None
    assert b2["mask_poly"] is None and b2["ghost"] is False
    assert all(e["slot"] != "R1" for ln in lines.values() for e in ln["main"])  # off-view
    assert [e["slot"] for e in lines[3]["radar"]] == ["B2", "R1"]
    assert len(lines[4]["radar"][0]["trail"]) == 5
    assert len(lines[6]["radar"][0]["trail"]) == 1             # re-seeded after the gap
    assert len(lines[19]["radar"][0]["trail"]) == 14
    assert len(lines[19]["radar"][1]["trail"]) == 16           # capped
    assert all("main_hash" not in ln and ln["hubs"] == [] and ln["corners"] == {}
               for ln in lines.values())


def test_backfill_render_without_score_suffix_and_with_the_note(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path, draw=False, live_video=True, robots=[
        {"slot": "B1", "alliance": "blue", "team_number": "111", "confirmed": True,
         "field_pos": [3.0, 3.0]},
        {"slot": "R2", "alliance": "red", "team_number": "222", "confirmed": False,
         "field_pos": [6.0, 4.0]}])
    monkeypatch.setattr(hud_render, "_main_calibration",
                        lambda ev, cam, ver, mk, warnings: FakeCal())
    monkeypatch.setattr(hud_render, "_corner_views",
                        lambda ev, mk: [{"name": "corner_left", "crop": [0, 150, 160, 90],
                                         "masks": []}])
    texts, notes = [], []
    real_robot, real_note = hud_draw.draw_robot, hud_draw.draw_note
    monkeypatch.setattr(hud_draw, "draw_robot", lambda img, box, al, label, flags, **kw:
                        (texts.append(label), real_robot(img, box, al, label, flags, **kw))[1])
    monkeypatch.setattr(hud_draw, "draw_note", lambda img, text:
                        (notes.append(text), real_note(img, text))[1])
    stats = hud_render.render_replay(rid)
    assert stats["source"] == "approx" and stats["decode_checked"] == 0
    assert set(texts) == {"111", "222?"}
    assert notes == ["approx boxes"] * 12
    assert _frame_count(d / "hud.mp4") == 12 and stats["kept_live_video"] is True


def test_backfill_without_a_calibration_fails(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path, draw=False)
    monkeypatch.setattr(hud_render, "_main_calibration", lambda *a: None)
    with pytest.raises(hud_render.RenderError, match="no main calibration"):
        hud_render.render_replay(rid)


def test_gco_comes_from_the_replay_config_else_the_knob_default():
    assert hud_render._ground_center_offset({"config": {"ground_center_offset_m": 0.3}}) == 0.3
    from web.job_config import KNOBS
    default = next(k.default for k in KNOBS if k.name == "ground_center_offset_m")
    assert hud_render._ground_center_offset({}) == default
