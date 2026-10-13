"""read_keep_evicted (knob, default OFF): a bumper read whose identity is
evicted (registry caps) while the read is in flight still writes its
vlm_read row, flagged ``"evicted": true``, as evidence for the finalize slot
resolvers — and nothing else (no vote, no live identity / backoff / census
state change).

qm11 full match: reads.dropped_evicted = 84 of 328 paid reads (26%) with
read_census on (41/239, 49/172 in other arms) left no row, so the
whole-match resolvers (global_identity_resolve, identity_swap_resolve) never
saw them although they credit rows to SLOTS via the identity log at the
crop frame and do not need the identity to be alive.
"""
import asyncio
import copy
import json
import threading
import types
from unittest.mock import MagicMock

import pytest

import web.app as app_module
from web.app import InferenceJob
from web.diagnostic import DiagnosticRecorder
from web.field.identity import IdentityRegistry, RobotIdentity
from web.field.identity_resolve import (
    build_slot_evidence,
    resolve_slot_labels,
    vlm_read_rows_from_jsonl,
)
from web.field.identity_swap import resolve_slot_segments
from web.job_config import JobConfig

_RESULT = {"team_number": "4174", "confidence": 0.9, "alliance": "blue",
           "raw_read": "4174", "surface": "bumper",
           "clef_probs": {"4174": 0.9, "none": 0.1}}
CROP_FRAME = 100


def _alliance(slot):
    return "blue" if slot < 3 else "red"


def _make_job(diag_path, *, keep=False, current_frame=92):
    """Bare InferenceJob with a REAL IdentityRegistry, the REAL scheduler
    snapshot (_schedule_vlm_pass) and read pass; only the reader is faked.
    Identity 7 holds slot 1 (blue), identity 9 slot 4 (red)."""
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig({"read_keep_evicted": True} if keep else {})
    job._camera_centers_xy = {}
    job.match_teams = ["4174", "1538"]
    job.match_blue_teams = ["4174"]
    job.match_red_teams = ["1538"]
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = current_frame
    job._evidence_frame = current_frame
    job.video_fps = 30.0
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={0: 1})
    reg = IdentityRegistry()
    reg.identities[7] = RobotIdentity(identity_id=7, alliance="blue", active_slot=1)
    reg.identities[9] = RobotIdentity(identity_id=9, alliance="red", active_slot=4)
    reg._next_id = 12
    job.identity_registry = reg
    job.robot_tracks = {}
    job._last_bindings = {}
    job._slot_source = {}
    job._slot_crops = {}
    job.scoring_events = []
    job.user_seed_zones_raw = []
    job._occ = {}
    job.confidence_config = types.SimpleNamespace(backoff_base_sec=2.0)
    job.team_reader = "local"
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = MagicMock()
    job.timings = {}
    job.diagnostic = DiagnosticRecorder(diag_path, ws_send=None)
    job._deposit_to_gallery = MagicMock()
    job._due_triggers = {7: "re_attachment"}
    return job


def _fused_obs():
    return [types.SimpleNamespace(
        evidence=[("main", "fakecropb64", 1.0, (4.0, 3.0))],
        detection=types.SimpleNamespace(pos=(4.0, 3.0)),
    )]


def _rows(job, diag_path):
    job.diagnostic.flush()
    if not diag_path.exists():
        return []
    rows = [json.loads(line) for line in diag_path.read_text().splitlines()]
    return [r for r in rows if r["type"] == "vlm_read"]


def _install_blocking_read(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def blocking_read(crops, teams, blue, red, expected_alliances=None,
                      provider="local", **_kw):
        entered.set()
        release.wait(timeout=5.0)  # bounded: a bug can't hang the suite
        return [dict(_RESULT) for _ in crops]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", blocking_read)
    return entered, release


def _run_evict_scenario(monkeypatch, diag_path, *, keep, live_slot_owner_confirmed=False):
    """Pass scheduled at frame 100 for identity 7 (slot 1). While the read is
    in flight the registry evicts 7 and a new identity 12 takes slot 1 (the
    tracker advanced to frame 108). Optionally identity 12's track already
    holds a confirmed 4174 (the gallery-deposit condition of the live path)."""
    entered, release = _install_blocking_read(monkeypatch)

    async def run():
        job = _make_job(diag_path, keep=keep)
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0}, crop_frame=CROP_FRAME)
        await asyncio.sleep(0.05)
        assert await asyncio.to_thread(entered.wait, 5.0), "read never started"
        assert not job._vlm_task.done(), "pass should still be in flight"
        job._current_frame = 108
        job.identity_registry.identities.pop(7)          # cap eviction
        job.identity_registry.identities[12] = RobotIdentity(
            identity_id=12, alliance="blue", active_slot=1)
        if live_slot_owner_confirmed:
            t = job._get_or_create_slot_track(1)
            t.team_number = "4174"
            t.team_number_confirmed = True
        release.set()
        await job._vlm_task
        return job

    return asyncio.run(run())


def _live_state(job):
    """Everything the read pass may mutate on the live side (excludes the
    diagnostic sink and the keep-evicted tally itself)."""
    idents = {iid: {k: v for k, v in vars(i).items()}
              for iid, i in job.identity_registry.identities.items()}
    tracks = {rid: {k: (vars(v) if k == "consecutive_tracker" and v is not None else v)
                    for k, v in vars(t).items()}
              for rid, t in job.robot_tracks.items()}
    return copy.deepcopy({
        "identities": idents, "tracks": tracks,
        "slot_crops": job._slot_crops, "slot_source": job._slot_source,
        "vlm_status": job.vlm_status,
        "trigger_counts": getattr(job, "_read_trigger_counts", {}),
        "dropped": getattr(job, "_read_dropped_counts", {}),
        "slot_attempts": getattr(job, "_slot_last_read_attempt", {}),
        "census_votes": getattr(job, "_census_slot_votes", {}),
        "immediate": getattr(job, "_immediate_reasons", {}),
        "gallery_calls": job._deposit_to_gallery.call_count,
    })


IDENTITY_LOG = [
    {"frame_num": 92, "identity_id": 7, "slot_id": 1},
    {"frame_num": 92, "identity_id": 9, "slot_id": 4},
    {"frame_num": 100, "identity_id": 7, "slot_id": 1},   # crop frame: 7 on slot 1
    {"frame_num": 100, "identity_id": 9, "slot_id": 4},
    {"frame_num": 108, "identity_id": 12, "slot_id": 1},  # 7 evicted, 12 rebinds
    {"frame_num": 108, "identity_id": 9, "slot_id": 4},
]


# ------------------------------------------------------------------ knob

def test_knob_declared_default_off():
    assert JobConfig().read_keep_evicted is False
    assert JobConfig({"read_keep_evicted": True}).read_keep_evicted is True


# ------------------------------------------------------- read pass: rows

def test_knob_off_evicted_read_leaves_no_row(monkeypatch, tmp_path):
    diag = tmp_path / "diag.jsonl"
    job = _run_evict_scenario(monkeypatch, diag, keep=False)
    assert _rows(job, diag) == []
    assert job._read_dropped_counts == {"re_attachment": 1}
    assert not getattr(job, "_read_kept_evicted_counts", None)


def test_knob_on_writes_the_row_flagged_evicted(monkeypatch, tmp_path):
    diag = tmp_path / "diag.jsonl"
    job = _run_evict_scenario(monkeypatch, diag, keep=True)
    (row,) = _rows(job, diag)
    # The row the live read would have written: crop frame, the identity
    # that held the crop's slot at that frame, team, weights, trigger.
    assert row["frame"] == CROP_FRAME
    assert row["robot_id"] == 7
    assert row["matched_team"] == "4174"
    assert row["confidence"] == pytest.approx(0.9)
    assert row["vote_weight"] == pytest.approx(1.0)       # conf >= 0.85 x cam 1.0
    assert row["alliance_read"] == "blue" and row["surface"] == "bumper"
    assert row["clef_probs"] == {"4174": 0.9, "none": 0.1}
    assert row["trigger"] == "re_attachment"
    assert row["crop_b64"] == "fakecropb64"
    assert row["evicted"] is True
    # Still counted as churn loss; the kept row tallied separately.
    assert job._read_dropped_counts == {"re_attachment": 1}
    assert job._read_kept_evicted_counts == {"re_attachment": 1}
    assert not getattr(job, "_read_trigger_counts", None)  # reads.total = live rows


def test_live_rows_carry_no_evicted_field(monkeypatch, tmp_path):
    """Knob ON, no eviction: the ordinary row is unchanged."""
    monkeypatch.setattr(app_module, "_vision_read_bumpers",
                        lambda crops, *a, **k: [dict(_RESULT) for _ in crops])
    diag = tmp_path / "diag.jsonl"

    async def run():
        job = _make_job(diag, keep=True)
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0}, crop_frame=CROP_FRAME)
        await job._vlm_task
        return job

    job = asyncio.run(run())
    (row,) = _rows(job, diag)
    assert row["robot_id"] == 7 and "evicted" not in row
    assert not getattr(job, "_read_kept_evicted_counts", None)


def test_knob_on_casts_no_vote_and_changes_no_live_state(monkeypatch, tmp_path):
    """Live side byte-identical to knob OFF: no vote on the slot's new owner
    (even one whose confirmed team matches the read — the gallery-deposit
    condition), no stamps/backoff on any identity, no crop capture."""
    off = _run_evict_scenario(monkeypatch, tmp_path / "off.jsonl", keep=False,
                              live_slot_owner_confirmed=True)
    on = _run_evict_scenario(monkeypatch, tmp_path / "on.jsonl", keep=True,
                             live_slot_owner_confirmed=True)
    assert len(_rows(on, tmp_path / "on.jsonl")) == 1
    assert _live_state(on) == _live_state(off)
    live = on.robot_tracks[12]
    assert not live.team_ocr_votes
    ct = live.consecutive_tracker
    assert ct is None or (ct.current_value is None and ct.consecutive_count == 0)
    on._deposit_to_gallery.assert_not_called()
    assert on._slot_crops == {}
    ident = on.identity_registry.identities[12]
    assert ident.last_vlm_time == -999.0 and ident.vlm_fail_count == 0
    assert ident.backoff_until == -999.0 and not ident.has_vlm_read


def test_scheduled_crop_slot_without_owner_writes_no_row(monkeypatch, tmp_path):
    """A crop slot with no owner at the crop frame (slot_identity lacks it)
    cannot be attributed: no robot_id is guessed, no row is written."""
    diag = tmp_path / "diag.jsonl"
    job = _make_job(diag, keep=True)

    def evicting_reader(crops, *a, **k):
        job.identity_registry.identities.pop(7)
        return [dict(_RESULT) for _ in crops]
    monkeypatch.setattr(app_module, "_vision_read_bumpers", evicting_reader)
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused_obs(), {7: 1.0}, det_to_slot={0: 1}, crop_frame=CROP_FRAME,
        slot_identity={4: 9}, read_triggers={7: "census"}))
    assert _rows(job, diag) == []
    assert job._read_dropped_counts == {"census": 1}
    assert not getattr(job, "_read_kept_evicted_counts", None)


def test_direct_caller_row_names_the_read_identity(monkeypatch, tmp_path):
    """No scheduler snapshot (direct caller): the crop was cut at the live
    frame with the read identity on its slot."""
    diag = tmp_path / "diag.jsonl"
    job = _make_job(diag, keep=True, current_frame=CROP_FRAME)

    def evicting_reader(crops, *a, **k):
        job.identity_registry.identities.pop(7)
        job.identity_registry.identities[12] = RobotIdentity(
            identity_id=12, alliance="blue", active_slot=1)
        return [dict(_RESULT) for _ in crops]
    monkeypatch.setattr(app_module, "_vision_read_bumpers", evicting_reader)
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused_obs(), {7: 1.0}, det_to_slot={0: 1}))
    (row,) = _rows(job, diag)
    assert row["robot_id"] == 7 and row["frame"] == CROP_FRAME and row["evicted"]
    assert row["trigger"] == "unspecified"


# ------------------------------------------------- finalize: attribution

def test_resolvers_attribute_the_evicted_row_to_its_crop_slot(monkeypatch, tmp_path):
    """global_identity_resolve + identity_swap_resolve read the rows back
    from the diagnostic JSONL and credit them via (frame, robot_id) in the
    identity log — identity 7 is gone from the registry and robot_tracks
    but its crop-frame binding (slot 1) is in the append-only log."""
    diag = tmp_path / "diag.jsonl"
    job = _run_evict_scenario(monkeypatch, diag, keep=True)
    job.diagnostic.flush()
    rows = vlm_read_rows_from_jsonl(diag.read_text().splitlines())
    assert len(rows) == 1 and rows[0]["evicted"] is True
    assert 7 not in job.identity_registry.identities and 7 not in job.robot_tracks

    evidence, stats = build_slot_evidence(rows, IDENTITY_LOG, {}, {}, _alliance)
    assert stats == {"reads_used": 1, "reads_dropped_misattributed": 0,
                     "fallback_used": False}
    assert evidence[1].read_scores == {"4174": pytest.approx(1.0)}
    assert 4 not in evidence

    meta = {s: {"team_number": None, "confirmed": False, "alliance": _alliance(s)}
            for s in range(6)}
    res = resolve_slot_labels(rows, IDENTITY_LOG, {}, {}, _alliance,
                              ["4174"], ["1538"], meta)
    assert res["slot_meta"][1]["team_number"] == "4174"
    sres = resolve_slot_segments(rows, IDENTITY_LOG, {}, {}, _alliance,
                                 ["4174"], ["1538"], meta)
    assert [s["team_number"] for s in sres["segments"][1]] == ["4174"]

    # Knob OFF: the same pass leaves the resolvers nothing.
    off_diag = tmp_path / "off.jsonl"
    _run_evict_scenario(monkeypatch, off_diag, keep=False)
    off_rows = vlm_read_rows_from_jsonl(
        off_diag.read_text().splitlines() if off_diag.exists() else [])
    assert off_rows == []


def test_evicted_row_off_the_identity_log_is_dropped_not_guessed(monkeypatch, tmp_path):
    """If (crop frame, robot_id) has no identity-log binding the existing
    misattribution rule drops (and counts) the row; nothing is credited."""
    diag = tmp_path / "diag.jsonl"
    job = _run_evict_scenario(monkeypatch, diag, keep=True)
    job.diagnostic.flush()
    rows = vlm_read_rows_from_jsonl(diag.read_text().splitlines())
    log = [r for r in IDENTITY_LOG if not (r["identity_id"] == 7 and r["frame_num"] == 100)]
    evidence, stats = build_slot_evidence(rows, log, {}, {}, _alliance)
    assert stats["reads_used"] == 0
    assert stats["reads_dropped_misattributed"] == 1
    assert stats["fallback_used"] is True
    assert all(not ev.read_scores for ev in evidence.values())


# ------------------------------------------------------------ telemetry

def test_finalize_reports_kept_evicted(capsys):
    job = InferenceJob.__new__(InferenceJob)
    job.timings = {}
    job.identity_diagnostics = MagicMock()
    job.team_reader = "clef"
    job._count_read_trigger("census", 10.0)
    job._count_read_trigger("census", dropped=True)
    job._count_read_trigger("re_attachment", dropped=True)
    job._read_kept_evicted_counts = {"census": 1}
    assert job._read_trigger_suffix().endswith(
        "; 2 dropped (identity evicted in flight), 1 kept as evidence rows "
        "(read_keep_evicted)")
    job._finalize_read_triggers()
    assert job.timings["reads.total"] == 1
    assert job.timings["reads.dropped_evicted"] == 2
    assert job.timings["reads.kept_evicted"] == 1
    job.identity_diagnostics.gauge.assert_any_call("read_kept_evicted", {"census": 1})


def test_knob_off_telemetry_unchanged():
    job = InferenceJob.__new__(InferenceJob)
    job.timings = {}
    job.identity_diagnostics = MagicMock()
    job.team_reader = "clef"
    job._count_read_trigger("census", dropped=True)
    assert job._read_trigger_suffix().endswith("; 1 dropped (identity evicted in flight)")
    job._finalize_read_triggers()
    assert "reads.kept_evicted" not in job.timings


# ------------------------------------------------------ offline consumers

def test_offline_check_counts_and_can_exclude_evicted_rows():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "identity_offline_check", root / "scripts" / "identity_offline_check.py")
    ioc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ioc)
    frames = list(range(0, 80, 8))
    binds = [(10 + s, s, 0, None) for s in range(6)]
    truth = {"B1": "4786", "B2": "3181", "B3": "2992",
             "R1": "6800", "R2": "1477", "R3": "4256"}
    reads = [{"type": "vlm_read", "frame": 16, "robot_id": 10 + s,
              "matched_team": t, "vote_weight": 1.0}
             for s, t in enumerate(truth.values()) if s != 2]
    reads.append({"type": "vlm_read", "frame": 24, "robot_id": 12,
                  "matched_team": "2992", "vote_weight": 1.0, "evicted": True})
    res = ioc.run_check(binds, frames, reads, truth)
    assert res["correct"] == 6 and res["evicted_team_reads"] == 1
    assert "(1 evicted-in-flight)" in ioc.format_report(res)
    res = ioc.run_check(binds, frames, [r for r in reads if not r.get("evicted")], truth)
    assert res["correct"] == 5 and res["slots"]["B3"]["status"] == "BLANK"
    assert "evicted-in-flight" not in ioc.format_report(res)


def test_sim_keep_evicted_recovers_rows_without_changing_live_reads():
    """qm11 timeline through the real scheduler + read pass: the knob leaves
    every live row identical and turns every evicted-in-flight read into a
    row attributable to a slot at its crop frame."""
    import scripts.sim_read_census as sim
    fx = sim.load_fixture()
    off = sim.simulate(fx, census=False)
    on = sim.simulate(fx, census=False, keep_evicted=True)
    assert on.reads == off.reads and on.reader_items == off.reader_items
    lost = off.reader_items - len(off.reads)
    assert lost > 0 and not off.kept
    assert len(on.kept) == lost == on.timings["reads.kept_evicted"]
    assert on.timings["reads.dropped_evicted"] == lost
    assert all(slot is not None for _f, slot, *_ in on.kept)
    summ = sim.summarize(on, fx["fps"])
    assert summ["kept"]["rows"] == lost and summ["kept"]["named"] > 0
