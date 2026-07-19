"""JobConfig: one typed registry for per-job knobs.

Defaults come from the same env vars the module constants use; per-job
overrides (UI form / headless JSON) are validated at job creation so a bad
knob fails the submit with a clear error instead of failing mid-inference.
"""
import pytest

from web.job_config import JobConfig, KNOBS, describe_knobs


def test_defaults_without_env(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.tracking_backend == "sam3"
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


def test_field_wireframe_overlay_default_and_override(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().field_wireframe_overlay is False
    # Form checkboxes arrive as strings; "1" must coerce to True.
    assert JobConfig({"field_wireframe_overlay": "1"}).field_wireframe_overlay is True


def test_seed_gate_knobs_default_on(monkeypatch):
    # qm12 fix landed as the default 2026-07-06 after cross-match validation
    # (qm8/10/11/12): seed-box gate on, motion-gated eviction on, seed at the
    # period open, and the opening-window dense reprompt on.
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.seed_box_max_aspect == 3.0
    assert cfg.evict_requires_motion is True
    assert cfg.sam2_prompt_offset_seconds == 0.0
    assert cfg.opening_reprompt_seconds == 0.5
    assert cfg.opening_window_seconds == 6.0


def test_seed_gate_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("SEED_BOX_MAX_ASPECT", "3.0")
    monkeypatch.setenv("EVICT_REQUIRES_MOTION", "1")
    cfg = JobConfig()
    assert cfg.seed_box_max_aspect == 3.0
    assert cfg.evict_requires_motion is True
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"seed_box_max_aspect": "4.47",
                      "evict_requires_motion": "false"})
    assert cfg2.seed_box_max_aspect == 4.47
    assert cfg2.evict_requires_motion is False


def test_seed_box_max_aspect_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"seed_box_max_aspect": -1})
    with pytest.raises(ValueError):
        JobConfig({"seed_box_max_aspect": 21})


def test_size_prior_cull_knobs_default_off(monkeypatch):
    # Geometric hijack cull: equipment hijacks are YOLO-corroborated, so only a
    # size violation catches them. DEFAULT OFF pending A/B.
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.size_prior_cull is False
    assert cfg.size_prior_hi == 3.0
    assert cfg.size_prior_lo == 0.25
    assert cfg.size_prior_boundaries == 3


def test_size_prior_cull_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("SIZE_PRIOR_CULL_ENABLED", "1")
    monkeypatch.setenv("SIZE_PRIOR_HI", "4.0")
    monkeypatch.setenv("SIZE_PRIOR_LO", "0.2")
    monkeypatch.setenv("SIZE_PRIOR_BOUNDARIES", "2")
    cfg = JobConfig()
    assert cfg.size_prior_cull is True
    assert cfg.size_prior_hi == 4.0
    assert cfg.size_prior_lo == 0.2
    assert cfg.size_prior_boundaries == 2
    # Per-job override (form fields arrive as strings) beats env.
    cfg2 = JobConfig({"size_prior_cull": "0", "size_prior_hi": "5.0",
                      "size_prior_lo": "0.1", "size_prior_boundaries": "4"})
    assert cfg2.size_prior_cull is False
    assert cfg2.size_prior_hi == 5.0
    assert cfg2.size_prior_lo == 0.1
    assert cfg2.size_prior_boundaries == 4


def test_size_prior_cull_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"size_prior_hi": 0.5})   # min 1.0
    with pytest.raises(ValueError):
        JobConfig({"size_prior_lo": 1.5})   # max 1.0
    with pytest.raises(ValueError):
        JobConfig({"size_prior_boundaries": 0})  # min 1


def test_mask_leak_cleanup_default_and_override(monkeypatch):
    # qm11/qm10/qm12 gated leak cleanup: drop a mask component only if it is
    # BOTH small AND detached, so a leak onto adjacent structure can't stretch
    # the box while a robot arm / occlusion-split half is preserved. DEFAULT
    # OFF pending re-validation.
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().mask_leak_cleanup is False
    # Env var flips the default on.
    monkeypatch.setenv("SAM2_MASK_LEAK_CLEANUP", "1")
    assert JobConfig().mask_leak_cleanup is True
    # Per-job override (form checkbox arrives as a string) beats env.
    assert JobConfig({"mask_leak_cleanup": "1"}).mask_leak_cleanup is True
    assert JobConfig({"mask_leak_cleanup": "0"}).mask_leak_cleanup is False


def test_mask_footprint_pos_default_and_override(monkeypatch):
    # Field position from the mask's carpet-contact band instead of the box
    # bottom-center, so an extended intake/arm doesn't shift the ray origin.
    # DEFAULT OFF (byte-identical default path).
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().mask_footprint_pos is False
    monkeypatch.setenv("MASK_FOOTPRINT_POS", "1")
    assert JobConfig().mask_footprint_pos is True
    assert JobConfig({"mask_footprint_pos": "1"}).mask_footprint_pos is True
    assert JobConfig({"mask_footprint_pos": "0"}).mask_footprint_pos is False


def test_vote_ray_gate_knobs_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.vote_ray_gate is False
    assert cfg.vote_ray_cross_m == 1.5
    assert cfg.vote_stale_max_m == 2.0


def test_vote_ray_gate_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("VOTE_RAY_GATE_ENABLED", "1")
    monkeypatch.setenv("VOTE_RAY_CROSS_M", "2.0")
    monkeypatch.setenv("VOTE_STALE_MAX_M", "3.0")
    cfg = JobConfig()
    assert cfg.vote_ray_gate is True
    assert cfg.vote_ray_cross_m == 2.0
    assert cfg.vote_stale_max_m == 3.0
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"vote_ray_gate": "false", "vote_ray_cross_m": "1.0",
                      "vote_stale_max_m": "1.0"})
    assert cfg2.vote_ray_gate is False
    assert cfg2.vote_ray_cross_m == 1.0
    assert cfg2.vote_stale_max_m == 1.0


def test_vote_ray_gate_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"vote_ray_cross_m": 0.0})
    with pytest.raises(ValueError):
        JobConfig({"vote_stale_max_m": 0.0})


def test_kin_gate_knobs_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.kin_gate is False
    assert cfg.kin_max_speed_mps == 6.5


def test_kin_gate_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("KIN_GATE_ENABLED", "1")
    monkeypatch.setenv("KIN_MAX_SPEED_MPS", "7.0")
    cfg = JobConfig()
    assert cfg.kin_gate is True
    assert cfg.kin_max_speed_mps == 7.0
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"kin_gate": "false", "kin_max_speed_mps": "5.0"})
    assert cfg2.kin_gate is False
    assert cfg2.kin_max_speed_mps == 5.0


def test_kin_gate_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"kin_max_speed_mps": 0.5})


def test_occlusion_portal_knob_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.occlusion_portal is False


def test_occlusion_portal_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("OCCLUSION_PORTAL_ENABLED", "1")
    cfg = JobConfig()
    assert cfg.occlusion_portal is True
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"occlusion_portal": "false"})
    assert cfg2.occlusion_portal is False


def test_start_zone_seed_knobs_default_off(monkeypatch):
    # Autonomous starting-zone seed priors: admit sub-threshold bumper reads
    # inside alliance-wall starting bands during the opening window. OFF by
    # default pending A/B.
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.start_zone_seed is False
    assert cfg.start_zone_depth_m == 2.0
    assert cfg.start_zone_window_s == 6.0


def test_start_zone_seed_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("START_ZONE_SEED_ENABLED", "1")
    monkeypatch.setenv("START_ZONE_DEPTH_M", "3.0")
    monkeypatch.setenv("START_ZONE_WINDOW_S", "8.0")
    cfg = JobConfig()
    assert cfg.start_zone_seed is True
    assert cfg.start_zone_depth_m == 3.0
    assert cfg.start_zone_window_s == 8.0
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"start_zone_seed": "0", "start_zone_depth_m": "1.5",
                      "start_zone_window_s": "4.0"})
    assert cfg2.start_zone_seed is False
    assert cfg2.start_zone_depth_m == 1.5
    assert cfg2.start_zone_window_s == 4.0


def test_start_zone_seed_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"start_zone_depth_m": 0.1})   # min 0.5
    with pytest.raises(ValueError):
        JobConfig({"start_zone_depth_m": 9.0})   # max 8.0
    with pytest.raises(ValueError):
        JobConfig({"start_zone_window_s": -1.0})  # min 0.0


def test_corrob_conf_floor_knob_default_off(monkeypatch):
    # Corroboration confidence floor: 0 disables the sub-seed-confidence
    # corroborate-only detection band (single-threshold behavior).
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.corrob_conf_floor == 0.0


def test_corrob_conf_floor_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("CORROB_CONF_FLOOR", "0.2")
    cfg = JobConfig()
    assert cfg.corrob_conf_floor == 0.2
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"corrob_conf_floor": "0.35"})
    assert cfg2.corrob_conf_floor == 0.35


def test_corrob_conf_floor_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"corrob_conf_floor": -0.1})   # min 0.0
    with pytest.raises(ValueError):
        JobConfig({"corrob_conf_floor": 1.1})    # max 1.0


def test_sam3_multiplex_knob_default_off(monkeypatch):
    # SAM 3.1 Object Multiplex builder: DEFAULT OFF pending qm8-qm12 A/B (the
    # multiplex builder bakes in different tracking heuristics, not just a
    # perf change -- see web/job_config.py comment).
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.sam3_multiplex is False


def test_sam3_multiplex_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("SAM3_MULTIPLEX", "1")
    cfg = JobConfig()
    assert cfg.sam3_multiplex is True
    # Form fields arrive as strings; per-job override beats env.
    cfg2 = JobConfig({"sam3_multiplex": "false"})
    assert cfg2.sam3_multiplex is False


def test_selective_mask_prop_knob_default_off(monkeypatch):
    monkeypatch.delenv("SELECTIVE_MASK_PROP", raising=False)
    cfg = JobConfig()
    assert cfg.selective_mask_prop is False
    assert cfg.selective_margin_entry == 0.05
    assert cfg.selective_margin_exit == 0.10
    assert cfg.selective_gap_trigger == 7
    assert cfg.selective_min_streak == 5
    assert cfg.selective_window_max_frames == 120


def test_selective_mask_prop_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_MASK_PROP", "1")
    assert JobConfig().selective_mask_prop is True
    assert JobConfig({"selective_mask_prop": "false"}).selective_mask_prop is False
    cfg = JobConfig({"selective_margin_entry": "0.2", "selective_window_max_frames": 60})
    assert cfg.selective_margin_entry == 0.2
    assert cfg.selective_window_max_frames == 60


def test_selective_knob_bounds():
    import pytest
    with pytest.raises(ValueError):
        JobConfig({"selective_margin_entry": 1.5})
    with pytest.raises(ValueError):
        JobConfig({"selective_window_max_frames": 2})


def test_to_dict_round_trip():
    cfg = JobConfig({"sam2_model_size": "large"})
    d = cfg.to_dict()
    assert d["sam2_model_size"] == "large"
    assert JobConfig(d).to_dict() == d


def test_every_knob_has_a_non_empty_section():
    # Settings UI groups knobs by section; an unset section silently falls
    # back to "misc" (see Knob.section default) which would hide the knob's
    # real grouping instead of failing loudly. Assert every knob was tagged.
    for knob in KNOBS:
        assert isinstance(knob.section, str) and knob.section.strip(), (
            f"{knob.name} has no section")
        assert knob.section != "misc", (
            f"{knob.name} left at the 'misc' fallback; assign a real section")


def test_describe_knobs_carries_section():
    by_name = {k.name: k for k in KNOBS}
    for entry in describe_knobs():
        knob = by_name[entry["name"]]
        assert entry["section"] == knob.section
        assert entry["section"].strip()
