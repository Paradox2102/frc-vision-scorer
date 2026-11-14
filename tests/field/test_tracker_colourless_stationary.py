"""colourless_stationary_gate: colourless ghosts parked on field structure.

On 2025hop_qm2 (full SAM3) main-only colourless fused observations sat on the
processor / reef for most of the match and held roster slots of either
alliance. With the gate on, a colourless detection whose track_id has stayed
within colourless_stationary_span_m for colourless_stationary_s seconds of
tracker time is excluded from the update: it claims no slot and seeds none.
Coloured and track_id-less detections are never gated.
"""
import json
import random

from web.field.tracker import FieldDetection, FieldTracker

L = 16.541
BOX = (0, 0, 10, 10)


def _det(x, y, color="unknown", tid=None, box=BOX):
    return FieldDetection(pos=(x, y), color=color, box=box, track_id=tid)


def _tracker(**kw):
    kw.setdefault("alliance_by_territory", True)
    return FieldTracker(L, zones=[], **kw)


def _snap(outs, tr):
    return ([(o.slot_id, o.alliance, round(o.pos[0], 9), round(o.pos[1], 9),
              round(o.vel[0], 9), round(o.vel[1], 9), o.state, o.corner_only)
             for o in outs], dict(tr.last_det_to_slot))


def _scripted(seed=0, n=120):
    """A deterministic mixed scene: a parked colourless ghost, a parked
    coloured robot, a moving colourless robot, id-less dets and dropouts."""
    rng = random.Random(seed)
    frames = []
    for k in range(n):
        dets = []
        if k % 7 != 3:
            dets.append(_det(6.1 + rng.uniform(-0.05, 0.05), 0.4, tid=7))
        dets.append(_det(3.0, 6.0, color="blue", tid=1))
        dets.append(_det(10.0 + 0.05 * k, 2.0 + rng.uniform(-0.1, 0.1), tid=2))
        if k % 5 == 0:
            dets.append(_det(12.0, 5.0, color="red"))
        if k % 11 == 0:
            dets.append(_det(14.5, 6.2, tid=6, box=None))
        frames.append(dets)
    return frames


# ------------------------------------------------------------------ off = legacy

def test_defaults_off_and_config_round_trip():
    tr = FieldTracker(L)
    assert tr.colourless_stationary_gate is False
    assert tr.colourless_stationary_s == 10.0
    assert tr.colourless_stationary_span_m == 1.0
    tr = FieldTracker(L, zones=[(1, 1, 2, 2)], colourless_stationary_gate=True,
                      colourless_stationary_s=7.5, colourless_stationary_span_m=0.6)
    cfg = tr.config_dict()
    json.dumps(cfg)
    assert cfg["colourless_stationary_gate"] is True
    assert cfg["colourless_stationary_s"] == 7.5
    assert cfg["colourless_stationary_span_m"] == 0.6
    again = FieldTracker(**cfg)
    assert again.config_dict() == cfg


def test_old_config_without_keys_still_builds_gate_off():
    cfg = FieldTracker(L).config_dict()
    for k in ("colourless_stationary_gate", "colourless_stationary_s",
              "colourless_stationary_span_m"):
        cfg.pop(k)
    assert FieldTracker(**cfg).colourless_stationary_gate is False


def test_off_is_identical_to_never_firing_gate():
    """Off (default) vs explicit off vs on-but-unreachable-window must give
    identical outputs on a scripted sequence, in both alliance modes."""
    for territory in (True, False):
        arms = [_tracker(alliance_by_territory=territory),
                _tracker(alliance_by_territory=territory,
                         colourless_stationary_gate=False),
                _tracker(alliance_by_territory=territory,
                         colourless_stationary_gate=True,
                         colourless_stationary_s=1e9)]
        for dets in _scripted():
            snaps = [_snap(tr.update(list(dets), {}, 0.2), tr) for tr in arms]
            assert snaps[0] == snaps[1] == snaps[2]
        assert all(tr.last_gated_det_indices == [] for tr in arms)


# ------------------------------------------------------------------ on

def test_stationary_colourless_stops_claiming_after_window():
    tr = _tracker(colourless_stationary_gate=True)        # 10 s window
    held = []
    for k in range(80):                                    # 16 s at dt 0.2
        tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
        held.append(tr.last_det_to_slot.get(0))
    # Bound while its history is shorter than the window...
    assert all(s is not None for s in held[:50])
    # ...gated from the update whose history first spans the full 10 s on.
    first = next(k for k, s in enumerate(held) if s is None)
    assert first == 50                                     # 50 intervals x 0.2 s
    assert all(s is None for s in held[first:])
    assert tr.last_gated_det_indices == [0]
    assert any(e["event"] == "colourless_stationary" and e["det_idx"] == [0]
               for e in tr.slot_events)
    assert tr.gated_det_total == 80 - first
    # The slot it held coasts out and expires.
    assert not any(t.active for t in tr.tracks)


def test_gated_ghost_cannot_seed_a_free_slot():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for _ in range(11):                                    # 10 x 0.2 = 2 s history
        tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
    for _ in range(12):                                    # let its slot expire
        tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
    assert not any(t.active for t in tr.tracks)
    for _ in range(5):
        outs = tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
        assert outs == [] and tr.last_det_to_slot == {}


def test_moving_colourless_never_gated():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for k in range(60):
        tr.update([_det(1.0 + 0.12 * k, 3.0, tid=3)], {}, 0.2)   # 0.6 m/s: 1.2 m per 2 s
        assert tr.last_gated_det_indices == []
        assert tr.last_det_to_slot == {0: tr.last_det_to_slot[0]}


def test_coloured_stationary_never_gated():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for _ in range(60):
        tr.update([_det(6.1, 0.4, color="blue", tid=7)], {}, 0.2)
        assert tr.last_gated_det_indices == []
        assert 0 in tr.last_det_to_slot


def test_trackid_less_never_gated():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for _ in range(60):
        tr.update([_det(6.1, 0.4, tid=None)], {}, 0.2)
        assert tr.last_gated_det_indices == []
        assert 0 in tr.last_det_to_slot


def test_colour_flicker_shares_history():
    """History accrues on every det with the id, coloured or not; only the
    colourless frames are gated."""
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for k in range(30):
        col = "blue" if k % 2 else "unknown"
        tr.update([_det(6.1, 0.4, color=col, tid=7)], {}, 0.2)
        if k >= 10:
            assert tr.last_gated_det_indices == ([] if col == "blue" else [0])


def test_gate_releases_when_id_moves_and_rearms_when_it_stops():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0,
                  colourless_stationary_span_m=1.0)
    for _ in range(12):
        tr.update([_det(6.0, 4.0, tid=7)], {}, 0.2)
    assert tr.last_gated_det_indices == [0]
    # Drives off: 0.5 m in one update -> window spans >= 1 m within 2 steps.
    gated = []
    for k in range(1, 6):
        tr.update([_det(6.0 + 0.5 * k, 4.0, tid=7)], {}, 0.2)
        gated.append(tr.last_gated_det_indices)
    assert gated[1:] == [[], [], [], []]
    assert 0 in tr.last_det_to_slot
    # Parks again at x=8.5: re-gated once a full window passes at rest.
    re = []
    for _ in range(20):
        tr.update([_det(8.5, 4.0, tid=7)], {}, 0.2)
        re.append(bool(tr.last_gated_det_indices))
    first = re.index(True)
    assert 8 <= first <= 11 and all(re[first:])


def test_window_is_in_seconds_not_updates():
    """Same 3 s window gates after 3 s at both 5 fps and 15 fps."""
    for dt in (0.2, 1 / 15):
        tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=3.0)
        t = 0.0
        first_t = None
        for _ in range(200):
            tr.update([_det(6.1, 0.4, tid=7)], {}, dt)
            t += dt
            if tr.last_gated_det_indices and first_t is None:
                first_t = t
        # gated once the history (first entry at t=dt) spans the full window
        assert first_t is not None
        assert 3.0 + dt - 1e-6 <= first_t <= 3.0 + 2 * dt + 1e-6


def test_long_gap_restarts_history():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0)
    for _ in range(12):
        tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
    assert tr.last_gated_det_indices == [0]
    for _ in range(15):                                   # 3 s unseen (> window)
        tr.update([], {}, 0.2)
    tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)
    assert tr.last_gated_det_indices == []                # fresh evidence needed


def test_history_memory_is_bounded():
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=1.0)
    for k in range(500):
        tr.update([_det(2.0 + (k % 10), 3.0, tid=1000 + k)], {}, 0.2)
    assert len(tr._cs_hist) <= 12
    assert all(len(h) <= 7 for h in tr._cs_hist.values())


def test_det_to_slot_keeps_original_indices():
    """With the ghost at index 0 gated, the robots after it keep their
    original detection indices in last_det_to_slot (both alliance modes)."""
    for territory in (True, False):
        tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0,
                      alliance_by_territory=territory)
        for _ in range(20):
            dets = [_det(6.1, 0.4, tid=7),
                    _det(3.0, 6.0, color="blue", tid=1),
                    _det(12.0, 2.0, color="red", tid=2)]
            tr.update(dets, {}, 0.2)
        assert tr.last_gated_det_indices == [0]
        assert set(tr.last_det_to_slot) == {1, 2}
        slots = {t.slot_id: t for t in tr.tracks}
        assert slots[tr.last_det_to_slot[1]].alliance == "blue"
        assert slots[tr.last_det_to_slot[2]].alliance == "red"


def test_territory_free_ghost_cannot_hold_existing_slot():
    """alliance_by_territory=False: unknown dets extend slots of either
    alliance; a gated ghost must stop extending the slot it rode."""
    tr = _tracker(colourless_stationary_gate=True, colourless_stationary_s=2.0,
                  alliance_by_territory=False)
    tr.update([_det(6.1, 0.4, color="red", tid=7)], {}, 0.2)   # seeds a red slot
    sid = tr.last_det_to_slot[0]
    bound = []
    for _ in range(30):
        tr.update([_det(6.1, 0.4, tid=7)], {}, 0.2)             # now colourless
        bound.append(tr.last_det_to_slot.get(0))
    assert bound[0] == sid
    assert bound[-1] is None
    assert not next(t for t in tr.tracks if t.slot_id == sid).active


# ------------------------------------------------------------------ plumbing

def test_job_config_knobs():
    from web.job_config import JobConfig
    d = JobConfig()
    assert d.colourless_stationary_gate is False
    assert d.colourless_stationary_s == 10.0
    assert d.colourless_stationary_span_m == 1.0
    c = JobConfig({"colourless_stationary_gate": True,
                   "colourless_stationary_s": 6.0,
                   "colourless_stationary_span_m": 0.5})
    assert (c.colourless_stationary_gate, c.colourless_stationary_s,
            c.colourless_stationary_span_m) == (True, 6.0, 0.5)


def test_app_tracker_constructor_passes_knobs():
    import inspect
    import web.app as app
    src = inspect.getsource(app)
    for k in ("colourless_stationary_gate", "colourless_stationary_s",
              "colourless_stationary_span_m"):
        assert f"{k}=self.config.{k}" in src
