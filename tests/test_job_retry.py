"""POST /api/job/{job_id}/retry — continue an aborted match analysis.

A calibration-drift abort stops the job before any tracking. Once the operator
fixes the calibration (approve the auto-fix or recalibrate manually), retry re-runs
the SAME video + config + match under a fresh job id, without re-uploading. The new
run re-scans calibration against the now-fixed epoch and proceeds normally.

InferenceJob is monkeypatched to a light fake so the endpoint logic is tested
without loading YOLO models.
"""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import web.app as app


def _client():
    return TestClient(app.app)


class _FakeJob:
    def __init__(self, job_id, video_path, match_key=None, config=None):
        self.job_id = job_id
        self.video_path = video_path
        self.match_key = match_key
        self.config = config
        self.status = "initializing"
        self._run_claimed = False
        self.match_teams = self.match_blue_teams = self.match_red_teams = []


def _install_fake(monkeypatch, roster=None):
    monkeypatch.setattr(app, "InferenceJob", _FakeJob)
    async def _roster(_mk):
        return roster
    monkeypatch.setattr(app, "_load_db_roster", _roster)


def test_retry_reuses_video_config_and_match(tmp_path, monkeypatch):
    vid = tmp_path / "match.mp4"; vid.write_bytes(b"x")
    cfg = object()
    old = SimpleNamespace(status="error", video_path=str(vid),
                          match_key="2026hop_qm84", config=cfg,
                          error="Calibration drift ...",
                          needs_calibration={"autofix_pending": True})
    app.jobs["OLD"] = old
    _install_fake(monkeypatch)
    created_ids = []
    try:
        r = _client().post("/api/job/OLD/retry")
        assert r.status_code == 200, r.text
        new_id = r.json()["job_id"]
        created_ids.append(new_id)
        assert new_id != "OLD"
        assert r.json()["previous_job_id"] == "OLD"
        nj = app.jobs[new_id]
        assert str(nj.video_path) == str(vid)      # video reused, not re-uploaded
        assert nj.match_key == "2026hop_qm84"
        assert nj.config is cfg                     # same knobs
        assert nj._run_claimed is False            # fresh -> the ws will run it
    finally:
        app.jobs.pop("OLD", None)
        for jid in created_ids:
            app.jobs.pop(jid, None)


def test_retry_seeds_roster_from_db(tmp_path, monkeypatch):
    vid = tmp_path / "m.mp4"; vid.write_bytes(b"x")
    old = SimpleNamespace(status="error", video_path=str(vid),
                          match_key="2026hop_qm84", config=object())
    app.jobs["OLDR"] = old
    _install_fake(monkeypatch, roster=(["11", "22"], ["11"], ["22"]))
    created_ids = []
    try:
        r = _client().post("/api/job/OLDR/retry")
        assert r.status_code == 200, r.text
        nj = app.jobs[r.json()["job_id"]]; created_ids.append(nj.job_id)
        assert nj.match_blue_teams == ["11"] and nj.match_red_teams == ["22"]
    finally:
        app.jobs.pop("OLDR", None)
        for jid in created_ids:
            app.jobs.pop(jid, None)


def test_retry_unknown_job_is_404():
    r = _client().post("/api/job/does-not-exist/retry")
    assert r.status_code == 404


def test_retry_missing_video_is_400(tmp_path, monkeypatch):
    old = SimpleNamespace(status="error", video_path=str(tmp_path / "gone.mp4"),
                          match_key=None, config=object())
    app.jobs["OLDG"] = old
    _install_fake(monkeypatch)
    try:
        r = _client().post("/api/job/OLDG/retry")
        assert r.status_code == 400
        assert "video" in r.json()["error"].lower()
    finally:
        app.jobs.pop("OLDG", None)
