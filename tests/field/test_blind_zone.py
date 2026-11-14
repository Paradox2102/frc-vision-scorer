"""Camera blind zones (job knob blind_zone_coast): zone geometry on synthetic
calibrations, the per-camera out-of-view occlusion + ALL-cameras merge, and the
tracker's blind hold (long coast, no drift, emerge gate). Off = unchanged."""
import numpy as np
import pytest

from tests._field_synth import look_at
from web.field.blind_zone import BlindZone, compute_blind_zone, field_grid, view_useful_grid
from web.field.calibration import CameraCalibration
from web.field.camera_view import CameraView
from web.field.occlusion import _merge_occlusion, resolve_occlusion
from web.field.tracker import FieldDetection, FieldTracker

DIMS = (16.541, 8.069)
K_MAIN = np.array([[1200., 0., 960.], [0., 1200., 360.], [0., 0., 1.]])
K_CORNER = np.array([[600., 0., 480.], [0., 600., 190.], [0., 0., 1.]])


def _cam(eye, target, size, K):
    rvec, tvec = look_at(eye, target)
    return CameraCalibration(mode="pinhole", K=K.copy(), D=np.zeros(5), rvec=rvec,
                             tvec=tvec, image_size=size, crop_region=(0, 0) + tuple(size),
                             field_season="2026-rebuilt-welded")


# Main: behind the scoring table, looking across the field (cuts off the near
# corners). Corner: just outside the near-left corner, looking into the field.
MAIN = _cam((8.27, -5.0, 3.3), (8.27, 4.5, 0.0), (1920, 720), K_MAIN)
CORNER = _cam((0.7, -0.5, 2.0), (3.0, 2.0, 0.0), (960, 380), K_CORNER)
VIEWS = [CameraView("main", (0, 0, 1920, 720), (), MAIN),
         CameraView("corner_left", (0, 720, 960, 380), (), CORNER)]


def _useful(cal, pt, **kw):
    g = view_useful_grid(cal, [pt[0]], [pt[1]], **kw)
    return bool(g[0, 0])


# ---------------------------------------------------------------- zone geometry

def test_point_outside_every_frame_is_blind_and_main_view_is_not():
    zone = compute_blind_zone(VIEWS, DIMS, corner_min_range_m=0.0)
    assert zone.contains((0.4, 0.4))           # near-left corner: no frame shows it
    assert zone.contains((16.2, 0.4))          # near-right corner: main cuts it off
    assert not zone.contains((8.27, 4.0))      # field centre: main sees it
    assert not zone.contains((8.27, 7.0))
    assert 0 < zone.area_m2 < 0.25 * DIMS[0] * DIMS[1]


def test_corner_min_range_blinds_points_right_at_the_corner_camera():
    pt = (2.375, 1.375)                        # a cell centre 2.5 m from the corner camera
    assert _useful(CORNER, pt, min_range_m=0.0)
    assert not _useful(CORNER, pt, min_range_m=3.0)
    assert not _useful(MAIN, pt)               # main cuts this corner off
    assert not compute_blind_zone(VIEWS, DIMS, corner_min_range_m=0.0).contains(pt)
    assert compute_blind_zone(VIEWS, DIMS, corner_min_range_m=3.0).contains(pt)


def test_min_range_never_applies_to_main():
    near_main = (8.27, 1.5)
    a = compute_blind_zone(VIEWS, DIMS, corner_min_range_m=0.0)
    b = compute_blind_zone(VIEWS, DIMS, corner_min_range_m=50.0)
    assert np.array_equal(a.useful["main"], b.useful["main"])
    assert not b.contains(near_main)


def test_ground_point_and_optional_top_probe_must_be_in_frame():
    # Camera looking down the field: near its frame bottom a robot's 0.4 m
    # point is in frame while its ground point is not (cut off at the bottom).
    cam = _cam((5.0, -1.0, 3.0), (5.0, 3.0, 0.0), (800, 400), np.array(
        [[500., 0., 400.], [0., 500., 200.], [0., 0., 1.]]))
    xs, ys = field_grid(DIMS)
    g = view_useful_grid(cam, xs, ys, margin_px=0.0, top_probe_m=0.4)
    g0 = view_useful_grid(cam, xs, ys, margin_px=0.0)
    c = int(np.argmin(np.abs(xs - 5.0)))
    seen = {"cut": 0, "full": 0, "lean": 0}
    for r, y in enumerate(ys):
        _, ok_g = cam.project_checked([(xs[c], y, 0.0)], margin_px=0.0)
        _, ok_t = cam.project_checked([(xs[c], y, 0.4)], margin_px=0.0)
        if ok_t[0] and not ok_g[0]:
            assert not g[r, c] and not g0[r, c]
            seen["cut"] += 1
        elif ok_t[0] and ok_g[0]:
            assert g[r, c] and g0[r, c]
            seen["full"] += 1
        elif ok_g[0] and not ok_t[0]:      # body leaves the frame: probe only
            assert not g[r, c] and g0[r, c]
            seen["lean"] += 1
    assert seen["cut"] >= 1 and seen["full"] >= 1


def test_masks_blank_cells_under_them():
    pt = (8.27, 4.0)
    uv = MAIN.field_to_image((pt[0], pt[1], 0.2))
    mask = (int(uv[0]) - 50, int(uv[1]) - 50, 100, 100)
    assert _useful(MAIN, pt)
    assert not _useful(MAIN, pt, masks=(mask,))


def test_summary_components_and_out_of_view_margin():
    zone = compute_blind_zone(VIEWS, DIMS, corner_min_range_m=3.0, margin_m=0.3)
    s = zone.summary()
    assert s["views"] == ["corner_left", "main"]
    assert s["area_m2"] == pytest.approx(zone.area_m2)
    assert sum(c["area_m2"] for c in s["components"]) == pytest.approx(zone.area_m2, abs=0.01)
    for c in s["components"]:
        x0, y0, x1, y1 = c["bbox"]
        assert 0 <= x0 < x1 <= DIMS[0] + 0.25 and 0 <= y0 < y1 <= DIMS[1] + 0.25
        assert len(c["polygon"]) >= 1
    assert len(s["grid"]["rows"]) == s["grid"]["ny"]
    # Inside the zone: out of view for EVERY camera; field centre: not for main.
    assert zone.out_of_view("main", (0.4, 0.4))
    assert zone.out_of_view("corner_left", (0.4, 0.4))
    assert not zone.out_of_view("main", (8.27, 4.0))
    assert not zone.out_of_view("unknown_view", (0.4, 0.4))
    assert zone.distance_m((0.4, 0.4)) == 0.0
    assert zone.distance_m((8.27, 7.0)) > 2.0


def test_empty_zone_distance_is_inf():
    xs, ys = field_grid(DIMS)
    zone = BlindZone(xs=xs, ys=ys, cell_m=0.25,
                     useful={"main": np.ones((len(ys), len(xs)), bool)})
    assert zone.area_m2 == 0.0
    assert zone.components() == []
    assert zone.distance_m((1.0, 1.0)) == float("inf")
    assert not zone.contains((1.0, 1.0))


# ---------------------------------------------------------------- occlusion

TRACKS = [{"slot_id": 0, "pos": (0.4, 0.4), "box": None},
          {"slot_id": 1, "pos": (8.27, 4.0), "box": (100, 100, 200, 200)},
          {"slot_id": 2, "pos": (0.5, 0.5), "box": (10, 10, 50, 50)}]


def _per_cam(zone, with_blind=True):
    return [resolve_occlusion(v.calibration, TRACKS, [], [],
                              out_of_view=((lambda p, n=v.name: zone.out_of_view(n, p))
                                           if with_blind else None))
            for v in VIEWS]


def test_off_output_is_unchanged():
    legacy = [resolve_occlusion(v.calibration, TRACKS, [], []) for v in VIEWS]
    zone = compute_blind_zone(VIEWS, DIMS)
    assert _per_cam(zone, with_blind=False) == legacy
    assert _merge_occlusion(legacy) == _merge_occlusion(_per_cam(zone, with_blind=False))
    assert all("blind" not in e for e in _merge_occlusion(legacy).values())


def test_slot_in_zone_is_occluded_in_every_camera():
    zone = compute_blind_zone(VIEWS, DIMS)
    per = _per_cam(zone)
    merged = _merge_occlusion(per)
    for sid in (0, 2):      # box-less coasting track and a boxed one, both in the zone
        assert all(p[sid]["out_of_view"] and p[sid]["view_occluded"] for p in per)
        assert merged[sid]["view_occluded"] and merged[sid]["blind"]
        assert merged[sid]["cause"] == "blind"
        assert merged[sid]["bumper_occluded"] is False   # VLM read gate untouched
    assert merged[1]["view_occluded"] is False and merged[1]["blind"] is False


def test_merge_blind_needs_every_camera_and_real_cause_wins():
    a = {0: {"view_occluded": True, "bumper_occluded": False, "cause": "blind",
             "out_of_view": True}}
    b = {0: {"view_occluded": True, "bumper_occluded": True, "cause": "hub",
             "out_of_view": False}}
    m = _merge_occlusion([a, b])[0]
    assert m["view_occluded"] is True and m["blind"] is False and m["cause"] == "hub"
    c = {0: {"view_occluded": False, "bumper_occluded": False, "cause": None,
             "out_of_view": False}}
    m2 = _merge_occlusion([a, c])[0]
    assert m2["view_occluded"] is False and m2["blind"] is False


# ---------------------------------------------------------------- tracker

class _Zone:
    """distance_m stub: the blind zone is the disc of radius 1 m at (1, 1)."""
    def distance_m(self, pos):
        return max(0.0, float(np.hypot(pos[0] - 1.0, pos[1] - 1.0)) - 1.0)


BLIND = {0: {"view_occluded": True, "bumper_occluded": False, "cause": "blind",
             "blind": True}}


def _drive_in(tr, occ_after):
    """Slot 0 (blue) drives toward the corner at ~2 m/s, then is lost."""
    for k in range(6):
        tr.update([FieldDetection(pos=(3.0 - 0.2 * k, 2.0 - 0.1 * k), color="blue",
                                  box=(0, 0, 10, 10))], {}, 0.1, frame_num=k)
    outs = []
    for k in range(6, 200):
        outs.append(tr.update([], occ_after, 0.1, frame_num=k))
    return outs


def _slot0(outs):
    return [next((t for t in o if t.slot_id == 0), None) for o in outs]


def test_blind_hold_coasts_long_without_drift():
    tr = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone=_Zone())
    s0 = _slot0(_drive_in(tr, BLIND))
    alive = [t for t in s0 if t is not None]
    assert len(alive) == tr.max_occ          # long budget, then expiry
    assert all(t.state == "coasting" for t in alive)
    # No drift: after the first coasted predict the position is frozen.
    p1 = alive[1].pos
    assert all(np.allclose(t.pos, p1) for t in alive[1:])
    assert np.hypot(p1[0] - 2.0, p1[1] - 1.5) < 0.5


def test_blind_max_coast_overrides_budget():
    tr = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone=_Zone(),
                      blind_zone_max_coast=150)
    alive = [t for t in _slot0(_drive_in(tr, BLIND)) if t is not None]
    assert len(alive) == 150


def test_off_ignores_blind_flag_and_matches_default_tracker():
    a = _drive_in(FieldTracker(DIMS[0]), BLIND)
    b = _drive_in(FieldTracker(DIMS[0], blind_zone_hold=False, blind_zone=_Zone(),
                               blind_zone_max_coast=150), BLIND)
    assert [[(t.slot_id, t.pos, t.state) for t in o] for o in a] == \
           [[(t.slot_id, t.pos, t.state) for t in o] for o in b]
    # Without the hold an occluded coast drifts (velocity decays, not zeroed).
    s0 = [t for t in _slot0(a) if t is not None]
    assert not np.allclose(s0[1].pos, s0[-1].pos)


def test_no_flag_means_short_coast_even_when_on():
    tr = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone=_Zone())
    alive = [t for t in _slot0(_drive_in(tr, {})) if t is not None]
    assert len(alive) == tr.max_open


def _held(tr):
    for k in range(6):
        tr.update([FieldDetection(pos=(3.0 - 0.2 * k, 2.0 - 0.1 * k), color="blue",
                                  box=(0, 0, 10, 10))], {}, 0.1, frame_num=k)
    for k in range(6, 40):
        tr.update([], BLIND, 0.1, frame_num=k)
    t0 = tr.tracks[0]
    assert t0.active and t0.blind_coasting
    return t0


def test_held_slot_reacquires_a_robot_emerging_from_the_zone():
    tr = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone=_Zone())
    t0 = _held(tr)
    out = tr.update([FieldDetection(pos=(3.2, 2.6), color="blue", box=(0, 0, 10, 10))],
                    BLIND, 0.1, frame_num=40)
    assert tr.last_det_to_slot == {0: 0}
    assert not t0.blind_coasting and not t0.coasting
    assert next(t for t in out if t.slot_id == 0).state == "visible"


def test_held_slot_does_not_steal_a_detection_elsewhere():
    tr = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone=_Zone())
    t0 = _held(tr)
    # 6 m from the held slot (inside its widened coast gate) but 6.6 m from the zone.
    far = (7.5, 3.0)
    tr.update([FieldDetection(pos=far, color="blue", box=(0, 0, 10, 10))],
              BLIND, 0.1, frame_num=40)
    assert tr.last_det_to_slot.get(0) != 0
    assert t0.blind_coasting
    assert any(e["event"] == "blind_emerge_reject" for e in tr.slot_events)
    # Same detection without the hold: the coasting slot takes it.
    tr2 = FieldTracker(DIMS[0])
    for k in range(6):
        tr2.update([FieldDetection(pos=(3.0 - 0.2 * k, 2.0 - 0.1 * k), color="blue",
                                   box=(0, 0, 10, 10))], {}, 0.1, frame_num=k)
    for k in range(6, 12):
        tr2.update([], BLIND, 0.1, frame_num=k)
    tr2.update([FieldDetection(pos=far, color="blue", box=(0, 0, 10, 10))],
               BLIND, 0.1, frame_num=12)
    assert tr2.last_det_to_slot == {0: 0}


def test_config_dict_unchanged_when_off():
    assert "blind_zone_hold" not in FieldTracker(DIMS[0]).config_dict()
    on = FieldTracker(DIMS[0], blind_zone_hold=True, blind_zone_max_coast=7).config_dict()
    assert on["blind_zone_hold"] is True and on["blind_zone_max_coast"] == 7


# ---------------------------------------------------------------- job wiring

def test_job_knob_off_is_a_noop_and_on_records_zone(monkeypatch):
    import web.app as app_mod
    from web.job_config import JobConfig
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = JobConfig({})
    job.field_tracker = FieldTracker(DIMS[0])
    job.field_dims = DIMS
    job.views = list(VIEWS)
    job._blind_zones = {}
    job._blind_zone = None
    job._blind_zone_record = None
    job._init_blind_zone()
    assert job._blind_zone_record is None
    assert job._blind_out_of_view(VIEWS, VIEWS[0]) is None
    assert job.field_tracker.blind_zone is None

    job.config = JobConfig({"blind_zone_coast": True})
    job._init_blind_zone()
    rec = job._blind_zone_record
    assert rec and len(rec["phases"]) == 1 and rec["phases"][0]["area_m2"] > 0
    assert job.field_tracker.blind_zone is job._blind_zone
    f = job._blind_out_of_view(VIEWS, VIEWS[0])
    assert f((0.4, 0.4)) is True and f((8.27, 4.0)) is False
    # A view going inactive (endgame cutoff) switches the tracker's zone.
    first = job._blind_zone
    job._blind_out_of_view(VIEWS[:1], VIEWS[0])
    assert job.field_tracker.blind_zone is not first
    assert job.field_tracker.blind_zone.views == ["main"]


def test_job_records_a_second_phase_for_an_endgame_cutoff():
    import web.app as app_mod
    from web.job_config import JobConfig
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.config = JobConfig({"blind_zone_coast": True})
    job.field_tracker = FieldTracker(DIMS[0])
    job.field_dims = DIMS
    corner = CameraView("corner_left", (0, 720, 960, 380), (), CORNER,
                        active_until_frame=5000)
    job.views = [VIEWS[0], corner]
    job._init_blind_zone()
    ph = job._blind_zone_record["phases"]
    assert [p["from_frame"] for p in ph] == [None, 5000]
    assert ph[1]["views"] == ["main"] and ph[1]["area_m2"] >= ph[0]["area_m2"]
