from fastapi.testclient import TestClient
import importlib

def _client():
    app_mod = importlib.import_module("web.app")
    return TestClient(app_mod.app)

def test_get_calibration_empty():
    r = _client().get("/api/calibration", params={"event": "test_evt_xyz", "camera": "main"})
    assert r.status_code == 200
    assert r.json()["has_calibration"] in (False, True)

def test_manual_without_intrinsics_returns_clear_error():
    r = _client().post("/api/calibration/manual", json={
        "event": "no_such_evt", "camera": "main", "correspondences": []})
    assert r.status_code == 400
    assert "auto" in r.json().get("error", "").lower()
