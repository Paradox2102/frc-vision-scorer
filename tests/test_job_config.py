"""JobConfig: one typed registry for per-job knobs.

Defaults come from the same env vars the module constants use; per-job
overrides (UI form / headless JSON) are validated at job creation so a bad
knob fails the submit with a clear error instead of failing mid-inference.
"""
import pytest

from web.job_config import JobConfig, KNOBS


def test_defaults_without_env(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.tracking_backend == "sam2"
    assert cfg.sam2_model_size == "small"
    assert cfg.sam2_track_fps == 8.0
    assert cfg.sam2_export_scale == 0.5
    assert cfg.reprompt_seconds == 2.0
    assert cfg.cull_hijack_enabled is False
    assert cfg.reanchor_enabled is False
    assert cfg.frame_stream_jpeg_quality == 70


def test_env_provides_defaults(monkeypatch):
    monkeypatch.setenv("SAM2_TRACK_FPS", "4.0")
    monkeypatch.setenv("CULL_HIJACK_ENABLED", "1")
    cfg = JobConfig()
    assert cfg.sam2_track_fps == 4.0
    assert cfg.cull_hijack_enabled is True


def test_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("SAM2_TRACK_FPS", "4.0")
    cfg = JobConfig({"sam2_track_fps": 12})
    assert cfg.sam2_track_fps == 12.0


def test_string_overrides_are_coerced():
    # Form fields arrive as strings; JSON may send real types. Accept both.
    cfg = JobConfig({
        "sam2_track_fps": "6.5",
        "cull_hijack_enabled": "1",
        "reanchor_enabled": "false",
        "sam2_max_gpu_frames": "300",
    })
    assert cfg.sam2_track_fps == 6.5
    assert cfg.cull_hijack_enabled is True
    assert cfg.reanchor_enabled is False
    assert cfg.sam2_max_gpu_frames == 300


def test_unknown_key_rejected_with_name():
    with pytest.raises(ValueError) as exc:
        JobConfig({"sam2_trak_fps": 6.0})
    assert "sam2_trak_fps" in str(exc.value)


def test_choice_validation():
    with pytest.raises(ValueError) as exc:
        JobConfig({"tracking_backend": "sam9"})
    assert "tracking_backend" in str(exc.value)
    assert "sam2" in str(exc.value)  # error lists valid choices


def test_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"sam2_export_scale": 0})
    with pytest.raises(ValueError):
        JobConfig({"frame_stream_jpeg_quality": 101})
    with pytest.raises(ValueError):
        JobConfig({"sam2_track_fps": -1})


def test_bad_value_names_the_knob():
    with pytest.raises(ValueError) as exc:
        JobConfig({"sam2_track_fps": "fast"})
    assert "sam2_track_fps" in str(exc.value)


def test_non_defaults_for_ab_provenance(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig({"sam2_track_fps": 12, "reanchor_enabled": True})
    nd = cfg.non_defaults()
    assert nd == {"sam2_track_fps": 12.0, "reanchor_enabled": True}
    # Env-sourced values count as non-default too: they differ from code
    # defaults and matter for cross-run comparisons.
    monkeypatch.setenv("REPROMPT_SECONDS", "1.0")
    assert JobConfig().non_defaults() == {"reprompt_seconds": 1.0}


def test_to_dict_round_trip():
    cfg = JobConfig({"sam2_model_size": "large"})
    d = cfg.to_dict()
    assert d["sam2_model_size"] == "large"
    assert JobConfig(d).to_dict() == d
