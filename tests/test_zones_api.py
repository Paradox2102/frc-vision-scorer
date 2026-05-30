from fastapi.testclient import TestClient
import importlib


def _client():
    return TestClient(importlib.import_module("web.app").app)


def test_zones_get_empty_for_unknown_event(tmp_path, monkeypatch):
    from web.field import zones as zmod
    monkeypatch.setattr(zmod, "UPLOAD_DIR", tmp_path)
    r = _client().get("/api/calibration/zones", params={"event": "evt_none"})
    assert r.status_code == 200
    assert r.json()["zones"] == []


def test_zones_post_then_get_round_trip(tmp_path, monkeypatch):
    from web.field import zones as zmod
    monkeypatch.setattr(zmod, "UPLOAD_DIR", tmp_path)
    rects = [[1.0, 1.0, 3.0, 3.0], [10.0, 5.0, 12.5, 7.0]]
    r = _client().post("/api/calibration/zones",
                       json={"event": "evt_z", "zones": rects})
    assert r.status_code == 200, r.text
    r2 = _client().get("/api/calibration/zones", params={"event": "evt_z"})
    assert r2.json()["zones"] == rects


def test_zones_post_rejects_malformed_rect(tmp_path, monkeypatch):
    from web.field import zones as zmod
    monkeypatch.setattr(zmod, "UPLOAD_DIR", tmp_path)
    r = _client().post("/api/calibration/zones",
                       json={"event": "evt_bad", "zones": [[1.0, 2.0, 3.0]]})
    assert r.status_code == 400


def test_zones_post_requires_an_event(monkeypatch):
    import web.app as app_mod
    monkeypatch.setattr(app_mod, "current_event", None)
    r = _client().post("/api/calibration/zones", json={"zones": []})
    assert r.status_code == 400
