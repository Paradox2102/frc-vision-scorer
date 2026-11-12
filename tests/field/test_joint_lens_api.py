"""The calibrate page's lens choice through web.app: /api/calibration/lens-options
(the server-side default), the near-fold warning on /api/calibration/solve-tags and
the joint-lens solve (``joint_with``) that saves a version for BOTH cameras.
Versions live in a tmp UPLOAD_DIR (never web/uploads); 2025hop values are copied
from the real versions (hop2025_fixtures)."""
import pytest
from fastapi.testclient import TestClient

from tests.field.hop2025_fixtures import (HOP_LEFT, HOP_LEFT_AUTO, HOP_LEFT_MANUAL,
                                          HOP_LEFT_SIZE, HOP_LEFT_SOURCE,
                                          HOP_RIGHT_AUTO, HOP_RIGHT_MANUAL2,
                                          HOP_RIGHT_SIZE)
from tests.field.test_lens_borrow_api import app_mod  # noqa: F401 (fixture)
from web.field import store as field_store
from web.field.calibration import CameraCalibration


def _client(mod):
    return TestClient(mod.app)


def _save_left(start_match="2025hop_qm1", source=HOP_LEFT_SOURCE):
    return field_store.save_version("2025hop", "corner_left",
                                    CameraCalibration.from_json(HOP_LEFT),
                                    start_match=start_match, source=source)


def _right_body(**kw):
    b = {"event": "2025hop", "camera": "corner_right", "image_size": list(HOP_RIGHT_SIZE),
         "auto_tags": HOP_RIGHT_AUTO, "manual_tags": HOP_RIGHT_MANUAL2,
         "start_match": "2025hop_qm1", "timestamp": 21.3, "match_key": "2025hop_qm1"}
    b.update(kw)
    return b


def _left_body(**kw):
    b = {"event": "2025hop", "camera": "corner_left", "image_size": list(HOP_LEFT_SIZE),
         "auto_tags": HOP_LEFT_AUTO, "manual_tags": HOP_LEFT_MANUAL,
         "start_match": "2025hop_qm1", "timestamp": 125.8, "match_key": "2025hop_qm1"}
    b.update(kw)
    return b


# ------------------------------------------------------------- lens-options

def test_lens_options_defaults_a_corner_to_joint_with_its_sibling(app_mod):
    vid = _save_left()
    j = _client(app_mod).get("/api/calibration/lens-options", params={
        "event": "2025hop", "camera": "corner_right", "match": "2025hop_qm3"}).json()
    assert j["default"] == "joint:corner_left"
    assert {o["value"] for o in j["options"]} == {"", "joint:corner_left", "corner_left"}
    assert all(o.get("version_id") in (None, vid) for o in j["options"])
    assert "share a lens model" in j["hint"]


def test_lens_options_borrows_when_the_sibling_has_no_stored_tags(app_mod):
    _save_left(source=None)
    j = _client(app_mod).get("/api/calibration/lens-options", params={
        "event": "2025hop", "camera": "corner_right"}).json()
    assert j["default"] == "corner_left"


def test_lens_options_main_and_lonely_corner_solve_their_own(app_mod):
    _save_left()
    c = _client(app_mod)
    j = c.get("/api/calibration/lens-options",
              params={"event": "2025hop", "camera": "main"}).json()
    assert j["default"] == "" and [o["kind"] for o in j["options"]] == ["own"]
    j = c.get("/api/calibration/lens-options",
              params={"event": "2025hop", "camera": "corner_left"}).json()
    assert j["default"] == ""              # corner_right has no version yet


# ------------------------------------------------------- near-fold warning

def test_overfitted_own_lens_solve_warns_and_suggests_the_joint_lens(app_mod):
    _save_left()
    j = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_right_body(dry_run=True)).json()
    assert j["model"] == "pp_k1"
    assert j["fold_ratio"] == pytest.approx(0.856, abs=0.01)
    w = j["warning"]
    assert "tag 2" in w and "r 0.45 of 0.53" in w and "over-fitted" in w
    assert w.endswith("try the joint lens with corner_left")


def test_near_fold_warning_names_no_sibling_when_none_is_available(app_mod):
    j = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_right_body(dry_run=True)).json()
    assert j["warning"].endswith("over-fitted")


def test_well_conditioned_own_lens_solve_has_no_warning(app_mod):
    j = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_left_body(dry_run=True)).json()
    assert j["fold_ratio"] == pytest.approx(0.62, abs=0.01)
    assert j["warning"] is None


def test_borrowed_lens_solve_reports_its_fold_ratio(app_mod):
    _save_left()
    j = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        lens_from="corner_left", dry_run=True)).json()
    assert j["fold_ratio"] == pytest.approx(0.69, abs=0.02)
    assert j["warning"] is None


# --------------------------------------------------------------- joint lens

def test_joint_preview_saves_nothing(app_mod):
    _save_left()
    r = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        joint_with="corner_left", dry_run=True))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "joint"
    jt = j["joint"]
    assert jt["partner"] == "corner_left" and jt["partner_version_id"] is None
    assert jt["f_px"] == pytest.approx(692, rel=0.15)
    for cam in ("corner_left", "corner_right"):
        assert 8.2 < jt["cameras"][cam]["center"][0] < 9.4
    assert jt["rms_inlier_px"] < jt["rms_px"]
    assert j["fold_ratio"] < 0.5 and j["warning"] is None
    assert field_store.load_versions("2025hop", "corner_right") == []
    assert len(field_store.load_versions("2025hop", "corner_left")) == 1


def test_joint_save_writes_a_version_for_both_cameras(app_mod):
    old = _save_left(start_match="2025hop_qm1")
    r = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        joint_with="corner_left", start_match="2025hop_qm4"))
    assert r.status_code == 200, r.text
    j = r.json()
    (rv,) = field_store.load_versions("2025hop", "corner_right")
    lv = [v for v in field_store.load_versions("2025hop", "corner_left")
          if v["version_id"] != old]
    assert len(lv) == 1
    lv = lv[0]
    assert j["joint"]["partner_version_id"] == lv["version_id"]
    assert j["joint"]["partner_start_match"] == "2025hop_qm1"
    # same lens on both (same crop height), poses on the barge
    rc, lc = rv["calibration"], lv["calibration"]
    assert rc.K[0, 0] == pytest.approx(lc.K[0, 0]) == pytest.approx(692, rel=0.15)
    assert rc.D[0] == pytest.approx(lc.D[0])
    assert 8.2 < rc.camera_center[0] < 9.4 and 8.2 < lc.camera_center[0] < 9.4
    assert rv["start_match"] == "2025hop_qm4"
    assert lv["start_match"] == "2025hop_qm1"        # the partner keeps its epoch
    rs, ls = rv["source"], lv["source"]
    assert rs["lens_kind"] == ls["lens_kind"] == "joint"
    assert rs["lens_partner"] == "corner_left" and ls["lens_partner"] == "corner_right"
    assert rs["lens_partner_version_id"] == lv["version_id"]
    assert rs["joint_id"] == ls["joint_id"]
    assert rs["manual_tags"] == HOP_RIGHT_MANUAL2
    assert ls["auto_tags"] == HOP_LEFT_AUTO and ls["manual_tags"] == HOP_LEFT_MANUAL
    assert ls["match_key"] == "2025hop_qm1" and ls["crop"] == HOP_LEFT_SOURCE["crop"]
    assert "lens_from" not in rs


def test_joint_repreview_pins_the_partner_version(app_mod):
    old = _save_left(start_match=None)
    _save_left()
    j = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        joint_with="corner_left", joint_version_id=old, dry_run=True)).json()
    assert j["joint"]["partner_input_version_id"] == old
    r = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        joint_with="corner_left", joint_version_id="nope", dry_run=True))
    assert r.status_code == 400


@pytest.mark.parametrize("camera,partner,needle", [
    ("main", "corner_left", "main"),
    ("corner_right", "main", "main"),
    ("corner_left", "corner_left", "own lens"),
])
def test_joint_rejects_main_and_self(app_mod, camera, partner, needle):
    _save_left()
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_right_body(camera=camera, joint_with=partner))
    assert r.status_code == 400 and needle in r.json()["error"]
    assert field_store.load_versions("2025hop", "corner_right") == []


def test_joint_needs_the_partners_stored_tags(app_mod):
    _save_left(source=None)
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_right_body(joint_with="corner_left"))
    assert r.status_code == 400 and "stored tags" in r.json()["error"]
    r = _client(app_mod).post("/api/calibration/solve-tags",
                              json=_right_body(joint_with="corner_far"))
    assert r.status_code == 400 and "no calibration version" in r.json()["error"]


def test_joint_preview_renders_the_partners_overlay_on_its_own_frame(app_mod, monkeypatch):
    import numpy as np
    _save_left()
    asked = []

    def frame(body):
        asked.append((body.get("match_key"), body.get("timestamp")))
        return np.zeros((1080, 1920, 3), np.uint8), None
    monkeypatch.setattr(app_mod, "_read_composite_frame", frame)
    j = _client(app_mod).post("/api/calibration/solve-tags", json=_right_body(
        joint_with="corner_left", dry_run=True)).json()
    assert j["overlay"] and j["joint"]["partner_overlay"]
    assert ("2025hop_qm1", 125.8) in asked          # corner_left's stored frame
