"""Position-stability behaviors: corner-only downweight, coast velocity
decay, teleport suppression. See docs/superpowers/specs/2026-06-12-field-map-v2-design.md."""
import numpy as np

from web.field.tracker import FieldTracker, FieldDetection


MAIN_BOX = (0, 0, 10, 10)


def _seed(tracker, pos=(2.0, 2.0), frames=5):
    """Settle one main-backed track at `pos` (stationary, small covariance)."""
    for _ in range(frames):
        tracker.update([FieldDetection(pos=pos, color="blue", box=MAIN_BOX)], {}, 0.1)
    return tracker


def _only_track(tracker, occ=None):
    outs = tracker.update([], occ or {}, 1e-9)
    assert len(outs) == 1
    return outs[0]


def test_corner_only_measurement_moves_track_less_than_main_backed():
    main = _seed(FieldTracker(field_length=16.0))
    corner = _seed(FieldTracker(field_length=16.0))

    # Same 0.4 m innovation (4 m/s — under the speed cap, so this isolates
    # the corner-only downweight); one main-backed, one corner-only.
    main.update([FieldDetection(pos=(2.4, 2.0), color="blue", box=MAIN_BOX)], {}, 0.1)
    corner.update([FieldDetection(pos=(2.4, 2.0), color="blue", box=None)], {}, 0.1)

    main_moved = _only_track(main).pos[0] - 2.0
    corner_moved = _only_track(corner).pos[0] - 2.0
    assert main_moved > 0.12              # main-backed follows the measurement
    assert corner_moved < main_moved / 2  # corner-only leans on the motion model
    assert corner_moved > 0.0             # ...but still gets gently pulled


def test_corner_only_flag_emitted_on_track_out():
    tr = _seed(FieldTracker(field_length=16.0))
    outs = tr.update([FieldDetection(pos=(2.0, 2.0), color="blue", box=None)], {}, 0.1)
    assert outs[0].corner_only is True
    outs = tr.update([FieldDetection(pos=(2.0, 2.0), color="blue", box=MAIN_BOX)], {}, 0.1)
    assert outs[0].corner_only is False


def test_coasting_track_velocity_decays_to_rest():
    tr = FieldTracker(field_length=16.0)
    # Build up rightward velocity ~4 m/s.
    for i in range(6):
        tr.update([FieldDetection(pos=(2.0 + 0.5 * i, 2.0), color="blue",
                                  box=MAIN_BOX)], {}, 0.1)
    moving = _only_track(tr)
    assert abs(moving.vel[0]) > 2.0

    # Occluded coast (long budget): the dot must ease to a stop, not sail away.
    occ = {moving.slot_id: {"view_occluded": True}}
    for _ in range(20):
        tr.update([], occ, 0.1)
    coasted = _only_track(tr, occ)
    assert abs(coasted.vel[0]) < 0.3
    # Constant-velocity coasting would travel ~8 m here; decay keeps it local.
    assert coasted.pos[0] - moving.pos[0] < 3.5


def test_teleport_measurement_does_not_yank_position():
    calm = _seed(FieldTracker(field_length=16.0))
    wild = _seed(FieldTracker(field_length=16.0, max_speed_m_s=1e9))

    # 2.4 m jump in 0.1 s = 24 m/s — impossible; inside the 2.5 m gate though.
    det = [FieldDetection(pos=(4.4, 2.0), color="blue", box=MAIN_BOX)]
    calm.update(det, {}, 0.1)
    wild.update(det, {}, 0.1)

    calm_moved = _only_track(calm).pos[0] - 2.0
    wild_moved = _only_track(wild).pos[0] - 2.0
    assert wild_moved > 1.0               # baseline: filter follows the teleport
    assert calm_moved < wild_moved / 3    # suppression: mostly stays put


def test_plausible_speed_measurement_unaffected_by_suppression():
    tr = _seed(FieldTracker(field_length=16.0))
    # 0.3 m in 0.1 s = 3 m/s, a normal FRC speed: full-trust update.
    tr.update([FieldDetection(pos=(2.3, 2.0), color="blue", box=MAIN_BOX)], {}, 0.1)
    assert _only_track(tr).pos[0] - 2.0 > 0.1
