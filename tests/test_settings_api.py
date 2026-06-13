"""Runtime settings: knob registry resolution + /api/settings endpoints."""
import pytest
from fastapi.testclient import TestClient

import web.job_config as job_config
from web.job_config import JobConfig


@pytest.fixture(autouse=True)
def clean_runtime_overrides():
    job_config.reset_runtime_overrides()
    yield
    job_config.reset_runtime_overrides()


@pytest.fixture()
def client():
    from web.app import app
    return TestClient(app)


class TestRuntimeOverrideResolution:
    def test_runtime_override_used_by_new_jobconfig(self):
        job_config.set_runtime_override("reprompt_seconds", 3.5)
        assert JobConfig().reprompt_seconds == 3.5

    def test_per_job_override_beats_runtime(self):
        job_config.set_runtime_override("reprompt_seconds", 3.5)
        cfg = JobConfig({"reprompt_seconds": 1.5})
        assert cfg.reprompt_seconds == 1.5

    def test_runtime_beats_env(self, monkeypatch):
        monkeypatch.setenv("REPROMPT_SECONDS", "4.0")
        job_config.set_runtime_override("reprompt_seconds", 2.5)
        assert JobConfig().reprompt_seconds == 2.5

    def test_clear_returns_to_env(self, monkeypatch):
        monkeypatch.setenv("REPROMPT_SECONDS", "4.0")
        job_config.set_runtime_override("reprompt_seconds", 2.5)
        job_config.set_runtime_override("reprompt_seconds", None)
        assert JobConfig().reprompt_seconds == 4.0

    def test_invalid_value_rejected_and_not_stored(self):
        with pytest.raises(ValueError):
            job_config.set_runtime_override("reprompt_seconds", 0.0)  # below min
        assert JobConfig().reprompt_seconds == 2.0  # default intact

    def test_unknown_knob_rejected(self):
        with pytest.raises(ValueError):
            job_config.set_runtime_override("nonsense", 1)

    def test_string_coercion_like_env(self):
        job_config.set_runtime_override("corner_track_stride", "3")
        assert JobConfig().corner_track_stride == 3

    def test_runtime_override_shows_in_non_defaults(self):
        job_config.set_runtime_override("corner_track_stride", 2)
        assert JobConfig().non_defaults()["corner_track_stride"] == 2


class TestDescribeKnobs:
    def test_covers_every_knob_with_metadata(self):
        desc = {d["name"]: d for d in job_config.describe_knobs()}
        assert set(desc) == {k.name for k in job_config.KNOBS}
        tb = desc["tracking_backend"]
        assert tb["choices"] == ["sam2", "sam3"]
        assert tb["type"] == "str"

    def test_source_tracks_resolution(self, monkeypatch):
        def get(name):
            return next(d for d in job_config.describe_knobs()
                        if d["name"] == name)
        assert get("reprompt_seconds")["source"] == "default"
        monkeypatch.setenv("REPROMPT_SECONDS", "4.0")
        d = get("reprompt_seconds")
        assert d["source"] == "env" and d["effective"] == 4.0
        job_config.set_runtime_override("reprompt_seconds", 2.5)
        d = get("reprompt_seconds")
        assert d["source"] == "runtime" and d["effective"] == 2.5
        # launch_value still shows what a restart returns to
        assert d["launch_value"] == 4.0


class TestSettingsEndpoints:
    def test_get_lists_knobs(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        names = {k["name"] for k in r.json()["knobs"]}
        assert "tracking_backend" in names and "reprompt_seconds" in names

    def test_post_sets_and_reports(self, client):
        r = client.post("/api/settings",
                        json={"name": "reprompt_seconds", "value": 3.0})
        assert r.status_code == 200
        knob = r.json()["knob"]
        assert knob["effective"] == 3.0 and knob["source"] == "runtime"
        assert JobConfig().reprompt_seconds == 3.0

    def test_post_null_clears(self, client):
        client.post("/api/settings",
                    json={"name": "reprompt_seconds", "value": 3.0})
        r = client.post("/api/settings",
                        json={"name": "reprompt_seconds", "value": None})
        assert r.status_code == 200
        assert r.json()["knob"]["source"] in ("env", "default")

    def test_post_invalid_400(self, client):
        r = client.post("/api/settings",
                        json={"name": "tracking_backend", "value": "sam9"})
        assert r.status_code == 400
        assert "sam9" in r.json()["error"]
        r = client.post("/api/settings", json={"value": 1})
        assert r.status_code == 400

    def test_reset_clears_all(self, client):
        client.post("/api/settings",
                    json={"name": "reprompt_seconds", "value": 3.0})
        client.post("/api/settings",
                    json={"name": "corner_track_stride", "value": 2})
        r = client.post("/api/settings/reset")
        assert r.status_code == 200
        assert JobConfig().reprompt_seconds == 2.0
        assert JobConfig().corner_track_stride == 1
