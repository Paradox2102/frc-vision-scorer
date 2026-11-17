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
    assert cfg.sam3_track_fps == 10.0      # 5 -> 10 Nov 17 2026
    assert cfg.sam3_export_scale == 0.5
    assert cfg.reprompt_seconds == 2.0
    assert cfg.cull_hijack_enabled is False
    assert cfg.reanchor_enabled is False
    assert cfg.frame_stream_jpeg_quality == 70


def test_env_provides_defaults(monkeypatch):
    monkeypatch.setenv("SAM3_TRACK_FPS", "4.0")
    monkeypatch.setenv("CULL_HIJACK_ENABLED", "1")
    cfg = JobConfig()
    assert cfg.sam3_track_fps == 4.0
    assert cfg.cull_hijack_enabled is True


def test_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("SAM3_TRACK_FPS", "4.0")
    cfg = JobConfig({"sam3_track_fps": 12})
    assert cfg.sam3_track_fps == 12.0


def test_string_overrides_are_coerced():
    # Form fields arrive as strings; JSON may send real types. Accept both.
    cfg = JobConfig({
        "sam3_track_fps": "6.5",
        "cull_hijack_enabled": "1",
        "reanchor_enabled": "false",
        "sam3_max_gpu_frames": "300",
    })
    assert cfg.sam3_track_fps == 6.5
    assert cfg.cull_hijack_enabled is True
    assert cfg.reanchor_enabled is False
    assert cfg.sam3_max_gpu_frames == 300


def test_unknown_key_rejected_with_name():
    with pytest.raises(ValueError) as exc:
        JobConfig({"sam3_trak_fps": 6.0})
    assert "sam3_trak_fps" in str(exc.value)


def test_choice_validation():
    with pytest.raises(ValueError) as exc:
        JobConfig({"tracking_backend": "sam9"})
    assert "tracking_backend" in str(exc.value)
    assert "sam2" in str(exc.value)  # error lists valid choices


def test_range_validation():
    with pytest.raises(ValueError):
        JobConfig({"sam3_export_scale": 0})
    with pytest.raises(ValueError):
        JobConfig({"frame_stream_jpeg_quality": 101})
    with pytest.raises(ValueError):
        JobConfig({"sam3_track_fps": -1})


def test_bad_value_names_the_knob():
    with pytest.raises(ValueError) as exc:
        JobConfig({"sam3_track_fps": "fast"})
    assert "sam3_track_fps" in str(exc.value)


def test_non_defaults_for_ab_provenance(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig({"sam3_track_fps": 12, "reanchor_enabled": True})
    nd = cfg.non_defaults()
    assert nd == {"sam3_track_fps": 12.0, "reanchor_enabled": True}
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
    assert cfg.sam3_prompt_offset_seconds == 0.0
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
    monkeypatch.setenv("SAM3_MASK_LEAK_CLEANUP", "1")
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


def test_selective_mask_prop_knob_default_on(monkeypatch):
    monkeypatch.delenv("SELECTIVE_MASK_PROP", raising=False)
    cfg = JobConfig()
    assert cfg.selective_mask_prop is True
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


def test_selective_sam_budget_frac_knob_default(monkeypatch):
    monkeypatch.delenv("SELECTIVE_SAM_BUDGET_FRAC", raising=False)
    cfg = JobConfig()
    assert cfg.selective_sam_budget_frac == 1.0


def test_selective_sam_budget_frac_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_SAM_BUDGET_FRAC", "0.8")
    assert JobConfig().selective_sam_budget_frac == 0.8
    cfg = JobConfig({"selective_sam_budget_frac": "1.5"})
    assert cfg.selective_sam_budget_frac == 1.5


def test_selective_sam_budget_frac_knob_bounds():
    import pytest
    with pytest.raises(ValueError):
        JobConfig({"selective_sam_budget_frac": 0.05})
    with pytest.raises(ValueError):
        JobConfig({"selective_sam_budget_frac": 11})


def test_selective_mask_overlap_exit_knob_default(monkeypatch):
    monkeypatch.delenv("SELECTIVE_MASK_OVERLAP_EXIT", raising=False)
    cfg = JobConfig()
    assert cfg.selective_mask_overlap_exit == 0.90


def test_selective_mask_overlap_exit_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_MASK_OVERLAP_EXIT", "0.95")
    assert JobConfig().selective_mask_overlap_exit == 0.95
    cfg = JobConfig({"selective_mask_overlap_exit": "0.97"})
    assert cfg.selective_mask_overlap_exit == 0.97


def test_selective_mask_overlap_exit_knob_bounds():
    import pytest
    with pytest.raises(ValueError):
        JobConfig({"selective_mask_overlap_exit": 0.4})
    with pytest.raises(ValueError):
        JobConfig({"selective_mask_overlap_exit": 1.1})


def test_selective_kill_coast_knob_default(monkeypatch):
    monkeypatch.delenv("SELECTIVE_KILL_COAST", raising=False)
    monkeypatch.delenv("SELECTIVE_KILL_COAST_FRAMES", raising=False)
    cfg = JobConfig()
    assert cfg.selective_kill_coast is False
    assert cfg.selective_kill_coast_frames == 24


def test_selective_kill_coast_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_KILL_COAST", "true")
    assert JobConfig().selective_kill_coast is True
    cfg = JobConfig({"selective_kill_coast": False,
                     "selective_kill_coast_frames": 48})
    assert cfg.selective_kill_coast is False
    assert cfg.selective_kill_coast_frames == 48


def test_selective_kill_coast_frames_knob_bounds():
    import pytest
    with pytest.raises(ValueError):
        JobConfig({"selective_kill_coast_frames": 0})
    with pytest.raises(ValueError):
        JobConfig({"selective_kill_coast_frames": 241})


def test_selective_app_knobs_default(monkeypatch):
    monkeypatch.delenv("SELECTIVE_APP_WEIGHT", raising=False)
    monkeypatch.delenv("SELECTIVE_APP_EMA", raising=False)
    cfg = JobConfig()
    assert cfg.selective_app_weight == 0.3
    assert cfg.selective_app_ema == 0.9


def test_selective_app_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_APP_WEIGHT", "0.6")
    monkeypatch.setenv("SELECTIVE_APP_EMA", "0.7")
    assert JobConfig().selective_app_weight == 0.6
    assert JobConfig().selective_app_ema == 0.7
    cfg = JobConfig({"selective_app_weight": 1.2, "selective_app_ema": 0.5})
    assert cfg.selective_app_weight == 1.2
    assert cfg.selective_app_ema == 0.5


def test_selective_app_knobs_bounds():
    import pytest
    with pytest.raises(ValueError):
        JobConfig({"selective_app_weight": -0.1})
    with pytest.raises(ValueError):
        JobConfig({"selective_app_weight": 5.1})
    with pytest.raises(ValueError):
        JobConfig({"selective_app_ema": -0.1})
    with pytest.raises(ValueError):
        JobConfig({"selective_app_ema": 1.1})


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


def test_global_identity_resolve_default_on_env_off(monkeypatch):
    # identity-binding-fix: whole-match per-slot final labels are the default;
    # GLOBAL_IDENTITY_RESOLVE=0 (or a per-job override) restores the
    # last-identity labels.
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    assert JobConfig().global_identity_resolve is True
    monkeypatch.setenv("GLOBAL_IDENTITY_RESOLVE", "0")
    assert JobConfig().global_identity_resolve is False
    assert JobConfig({"global_identity_resolve": "true"}).global_identity_resolve is True


def test_identity_continuity_knobs_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.identity_continuity is False
    assert cfg.identity_continuity_max_m == 1.0
    assert cfg.identity_warmup_frames == 5


def test_identity_continuity_knobs_env_override_and_bounds(monkeypatch):
    monkeypatch.setenv("IDENTITY_CONTINUITY", "1")
    monkeypatch.setenv("IDENTITY_CONTINUITY_MAX_M", "0.75")
    cfg = JobConfig()
    assert cfg.identity_continuity is True and cfg.identity_continuity_max_m == 0.75
    assert JobConfig({"identity_warmup_frames": "0"}).identity_warmup_frames == 0
    with pytest.raises(ValueError):
        JobConfig({"identity_continuity_max_m": 0.0})
    with pytest.raises(ValueError):
        JobConfig({"identity_continuity_max_m": 2.0})
    with pytest.raises(ValueError):
        JobConfig({"identity_warmup_frames": 99})


def test_identity_registry_built_from_knobs(monkeypatch):
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    # 5 fps = the frame-knob reference rate: warm-up passes through unscaled.
    job.config = JobConfig({"identity_continuity": True,
                            "identity_continuity_max_m": 0.8,
                            "identity_warmup_frames": 3,
                            "sam3_track_fps": 5.0})
    from web.field.identity import IdentityRegistry
    # Mirrors the constructor wiring in InferenceJob.__init__.
    reg = InferenceJob._build_identity_registry(job)
    assert isinstance(reg, IdentityRegistry)
    assert reg.continuity is True and reg.continuity_max_m == 0.8
    assert reg.warmup_frames == 3
    # At the 10 fps default the same knob is 6 tracked frames (same 0.6 s).
    job.config = JobConfig({"identity_continuity": True,
                            "identity_continuity_max_m": 0.8,
                            "identity_warmup_frames": 3})
    assert InferenceJob._build_identity_registry(job).warmup_frames == 6


def test_identity_per_cam_appearance_knob_default_on_and_wired(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    from web.app import InferenceJob
    from web.field.identity import IdentityRegistry
    cfg = JobConfig()
    assert cfg.identity_per_cam_appearance is True
    assert "identity_per_cam_appearance" not in cfg.non_defaults()
    job = InferenceJob.__new__(InferenceJob)
    job.config = cfg
    reg = InferenceJob._build_identity_registry(job)
    assert isinstance(reg, IdentityRegistry) and reg.per_cam_appearance is True
    monkeypatch.setenv("IDENTITY_PER_CAM_APPEARANCE", "0")
    assert JobConfig().identity_per_cam_appearance is False
    job.config = JobConfig({"identity_per_cam_appearance": "false"})
    assert InferenceJob._build_identity_registry(job).per_cam_appearance is False


def test_fused_per_cam_relink_knob_default_off_wired_and_gauged(monkeypatch, capsys):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    from web.app import InferenceJob
    from web.field.fused_track_id import FusedTrackRegistry
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    cfg = JobConfig()
    assert cfg.fused_per_cam_relink is False
    assert "fused_per_cam_relink" not in cfg.non_defaults()
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig({"relink_max_dist_m": "2.5"})
    reg = InferenceJob._build_fused_registry(job)
    assert isinstance(reg, FusedTrackRegistry) and reg.per_cam_relink is False
    assert reg.relink_max_dist_m == 2.5
    assert reg.app_floor == job._emb_thr().relink_floor
    job.config = JobConfig({"fused_per_cam_relink": "true"})
    assert InferenceJob._build_fused_registry(job).per_cam_relink is True
    monkeypatch.setenv("FUSED_PER_CAM_RELINK", "1")
    assert JobConfig().fused_per_cam_relink is True

    # End-of-job gauge + log line.
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job.fused_registry = InferenceJob._build_fused_registry(job)
    job.fused_registry.stats.update({"continued": 40, "created_with_emb": 3,
                                     "created_crossview_only": 2})
    job._record_fused_ids_summary()
    gauge = job.identity_diagnostics.summary()["gauges"]["fused_ids"]
    assert gauge["continued"] == 40 and gauge["created_with_emb"] == 3
    assert gauge["created_crossview_only"] == 2 and gauge["relinked"] == 0
    assert gauge["per_cam_relink"] is True
    assert "[fused-ids] continued 40" in capsys.readouterr().out
    # No registry (Kalman path): no gauge, no crash.
    job2 = InferenceJob.__new__(InferenceJob)
    job2.identity_diagnostics = IdentityDiagnosticRecorder()
    job2._record_fused_ids_summary()
    assert "fused_ids" not in job2.identity_diagnostics.summary()["gauges"]


def test_identity_swap_resolve_knobs_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.identity_swap_resolve is False
    assert cfg.identity_swap_lambda == 3.0
    assert cfg.identity_swap_prox_m == 1.5
    assert "identity_swap_resolve" not in cfg.non_defaults()
    on = JobConfig({"identity_swap_resolve": "true", "identity_swap_lambda": "4"})
    assert on.identity_swap_resolve is True and on.identity_swap_lambda == 4.0


def test_tracker_reacquire_knobs_defaults(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.assign_partial_fix is True
    assert cfg.coast_gate_phys is True
    assert cfg.reseed_identity is False
    assert cfg.corner_dup_suppress is True
    assert cfg.phys_reach_mps == 5.0
    assert cfg.phys_reach_slack_m == 1.0
    assert cfg.phys_corner_slack_m == 2.0
    assert cfg.reseed_hold_s == 1.5
    assert cfg.corner_dup_radius_m == 2.0


def test_tracker_reacquire_knobs_env_override_and_range(monkeypatch):
    monkeypatch.setenv("ASSIGN_PARTIAL_FIX", "0")
    monkeypatch.setenv("COAST_GATE_PHYS", "0")
    cfg = JobConfig({"reseed_identity": "true", "corner_dup_suppress": "0",
                     "reseed_hold_s": "2.5"})
    assert not cfg.assign_partial_fix and not cfg.coast_gate_phys
    assert cfg.reseed_identity and not cfg.corner_dup_suppress
    assert cfg.reseed_hold_s == 2.5
    with pytest.raises(ValueError):
        JobConfig({"corner_dup_radius_m": 0.1})
    with pytest.raises(ValueError):
        JobConfig({"phys_reach_mps": 0.5})


def test_window_labels_knobs_default_on(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.window_labels is True
    assert cfg.window_labels_lineup_min_p == 0.8
    assert "window_labels" not in cfg.non_defaults()
    assert "window_labels_lineup_min_p" not in cfg.non_defaults()


def test_window_labels_knobs_env_override_and_range(monkeypatch):
    monkeypatch.setenv("WINDOW_LABELS", "1")
    cfg = JobConfig({"window_labels_lineup_min_p": "0.65"})
    assert cfg.window_labels is True
    assert cfg.window_labels_lineup_min_p == 0.65
    assert JobConfig({"window_labels": "false"}).window_labels is False
    assert JobConfig({"window_labels_lineup_min_p": 1.01}).window_labels_lineup_min_p == 1.01
    with pytest.raises(ValueError):
        JobConfig({"window_labels_lineup_min_p": 1.5})
    with pytest.raises(ValueError):
        JobConfig({"window_labels_lineup_min_p": -0.1})


def test_event_lineup_knob_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.event_lineup is False and "event_lineup" not in cfg.non_defaults()
    assert JobConfig({"event_lineup": True}).non_defaults() == {"event_lineup": True}
    assert JobConfig({"event_lineup": "false"}).event_lineup is False
    monkeypatch.setenv("EVENT_LINEUP", "1")
    assert JobConfig().event_lineup is True
    assert next(k for k in KNOBS if k.name == "event_lineup").section == "Identity & votes"


def test_hud_post_render_knob_default_on(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.hud_post_render is True and "hud_post_render" not in cfg.non_defaults()
    assert JobConfig({"hud_post_render": "0"}).hud_post_render is False
    monkeypatch.setenv("HUD_POST_RENDER", "0")
    assert JobConfig().hud_post_render is False
    assert next(k for k in KNOBS if k.name == "hud_post_render").section == "Streaming & output"


def test_window_labels_deinterlace_knob_default_on(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.window_labels_deinterlace is True
    assert "window_labels_deinterlace" not in cfg.non_defaults()
    assert JobConfig({"window_labels_deinterlace": "false"}).window_labels_deinterlace is False
    monkeypatch.setenv("WINDOW_LABELS_DEINTERLACE", "0")
    assert JobConfig().window_labels_deinterlace is False


def test_window_labels_fill_knob_default_on(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.window_labels_fill is True
    assert "window_labels_fill" not in cfg.non_defaults()
    assert JobConfig({"window_labels_fill": "false"}).window_labels_fill is False
    assert next(k for k in KNOBS if k.name == "window_labels_fill").section == "Identity & votes"
    monkeypatch.setenv("WINDOW_LABELS_FILL", "0")
    assert JobConfig().window_labels_fill is False


def test_window_labels_eliminate_knob_default_off(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.window_labels_eliminate is False
    assert "window_labels_eliminate" not in cfg.non_defaults()
    assert JobConfig({"window_labels_eliminate": "true"}).window_labels_eliminate is True
    assert next(k for k in KNOBS if k.name == "window_labels_eliminate").section == \
        "Identity & votes"
    monkeypatch.setenv("WINDOW_LABELS_ELIMINATE", "1")
    assert JobConfig().window_labels_eliminate is True


def test_window_labels_reader_knob_default_student_only(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.window_labels_reader == "student_only"
    assert "window_labels_reader" not in cfg.non_defaults()
    assert JobConfig({"window_labels_reader": "jev"}).window_labels_reader == "jev"
    knob = next(k for k in KNOBS if k.name == "window_labels_reader")
    assert knob.choices == ("jev", "student", "student_only")
    monkeypatch.setenv("WINDOW_LABELS_READER", "jev")
    assert JobConfig().window_labels_reader == "jev"


def test_sam3_skip_detector_heads_knob_default_on(monkeypatch):
    # Backbone-only partial path: byte-identical offline; DEFAULT ON since the
    # 2026-11-12 live A/B (see web/job_config.py comment).
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.sam3_skip_detector_heads is True
    assert "sam3_skip_detector_heads" not in cfg.non_defaults()


def test_sam3_skip_detector_heads_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("SAM3_SKIP_DETECTOR_HEADS", "1")
    cfg = JobConfig()
    assert cfg.sam3_skip_detector_heads is True
    cfg2 = JobConfig({"sam3_skip_detector_heads": "false"})
    assert cfg2.sam3_skip_detector_heads is False


def test_sam3_retire_culled_knob_default_on(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.sam3_retire_culled is True


def test_sam3_retire_culled_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("SAM3_RETIRE_CULLED", "1")
    cfg = JobConfig()
    assert cfg.sam3_retire_culled is True
    cfg2 = JobConfig({"sam3_retire_culled": "false"})
    assert cfg2.sam3_retire_culled is False


def test_sam3_frame_load_workers_knob_default_auto(monkeypatch):
    # Byte-identical threaded frame loader: DEFAULT ON via auto (0).
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.sam3_frame_load_workers == 0
    assert "sam3_frame_load_workers" not in cfg.non_defaults()
    knob = next(k for k in KNOBS if k.name == "sam3_frame_load_workers")
    assert knob.type is int and knob.section == "Tracking backend"


def test_sam3_frame_load_workers_knob_env_and_override(monkeypatch):
    monkeypatch.setenv("SAM3_FRAME_LOAD_WORKERS", "8")
    cfg = JobConfig()
    assert cfg.sam3_frame_load_workers == 8
    cfg2 = JobConfig({"sam3_frame_load_workers": "1"})
    assert cfg2.sam3_frame_load_workers == 1


def test_selective_segmented_knobs_defaults(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.selective_retire_closed is True
    assert cfg.selective_lazy_seed is False   # never live A/B'd


def test_selective_segmented_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("SELECTIVE_RETIRE_CLOSED", "1")
    monkeypatch.setenv("SELECTIVE_LAZY_SEED", "true")
    cfg = JobConfig()
    assert cfg.selective_retire_closed is True
    assert cfg.selective_lazy_seed is True
    cfg2 = JobConfig({"selective_retire_closed": "false",
                      "selective_lazy_seed": False})
    assert cfg2.selective_retire_closed is False
    assert cfg2.selective_lazy_seed is False


# --- sam2_* -> sam3_* rename (Nov 2026): legacy names keep working ---------

_RENAMED = {
    "sam2_track_fps": ("sam3_track_fps", 7.0),
    "sam2_export_scale": ("sam3_export_scale", 0.75),
    "sam2_max_gpu_frames": ("sam3_max_gpu_frames", 900),
    "sam2_prompt_offset_seconds": ("sam3_prompt_offset_seconds", 1.5),
}


def _clear_knob_env(monkeypatch):
    for knob in KNOBS:
        for var in (knob.env, *knob.env_aliases):
            monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("old", sorted(_RENAMED))
def test_legacy_override_key_resolves_to_new_knob(monkeypatch, old):
    _clear_knob_env(monkeypatch)
    new, value = _RENAMED[old]
    cfg = JobConfig({old: value})            # persisted config_json / A/B arm
    assert getattr(cfg, new) == value
    assert not hasattr(cfg, old)
    assert cfg.non_defaults() == {new: value}
    assert old not in cfg.to_dict() and new in cfg.to_dict()


def test_new_override_key_wins_over_legacy_alias(monkeypatch):
    _clear_knob_env(monkeypatch)
    cfg = JobConfig({"sam2_track_fps": 4.0, "sam3_track_fps": 9.0})
    assert cfg.sam3_track_fps == 9.0
    cfg = JobConfig({"sam3_track_fps": 9.0, "sam2_track_fps": 4.0})
    assert cfg.sam3_track_fps == 9.0


def test_legacy_alias_value_is_validated_under_new_name(monkeypatch):
    _clear_knob_env(monkeypatch)
    with pytest.raises(ValueError) as exc:
        JobConfig({"sam2_track_fps": -1})
    assert "sam3_track_fps" in str(exc.value)


def test_legacy_env_var_still_read(monkeypatch):
    _clear_knob_env(monkeypatch)
    monkeypatch.setenv("SAM2_TRACK_FPS", "4.0")
    monkeypatch.setenv("SAM2_DARK_REANCHOR_ENABLED", "0")
    cfg = JobConfig()
    assert cfg.sam3_track_fps == 4.0
    assert cfg.dark_reanchor_enabled is False


def test_new_env_var_wins_over_legacy(monkeypatch):
    _clear_knob_env(monkeypatch)
    monkeypatch.setenv("SAM2_TRACK_FPS", "4.0")
    monkeypatch.setenv("SAM3_TRACK_FPS", "6.0")
    assert JobConfig().sam3_track_fps == 6.0
    # An empty canonical var counts as unset (same rule as before the rename).
    monkeypatch.setenv("SAM3_TRACK_FPS", "")
    assert JobConfig().sam3_track_fps == 4.0


def test_settings_api_lists_new_names_only(monkeypatch):
    _clear_knob_env(monkeypatch)
    monkeypatch.setenv("SAM2_EVICT_DARK_FRAMES", "99999")
    entries = {e["name"]: e for e in describe_knobs()}
    for old, (new, _) in _RENAMED.items():
        assert new in entries and old not in entries
    assert not any(e["name"].startswith("sam2_") and e["name"] not in (
        "sam2_model_size", "sam2_vos_optimized") for e in entries.values())
    assert not any(e["env"].startswith("SAM2_") and e["name"] not in (
        "sam2_model_size", "sam2_vos_optimized") for e in entries.values())
    # The launch value still honours the legacy env var.
    ev = entries["evict_dark_frames"]
    assert ev["env"] == "SAM3_EVICT_DARK_FRAMES"
    assert ev["launch_value"] == 99999 and ev["source"] == "env"


def test_runtime_override_accepts_legacy_name(monkeypatch):
    from web import job_config as jc
    _clear_knob_env(monkeypatch)
    try:
        assert jc.set_runtime_override("sam2_track_fps", "3") == 3.0
        assert JobConfig().sam3_track_fps == 3.0
        entry = next(e for e in describe_knobs() if e["name"] == "sam3_track_fps")
        assert entry["source"] == "runtime"
        jc.set_runtime_override("sam2_track_fps", None)   # clear via old name
        assert JobConfig().sam3_track_fps == 10.0
    finally:
        jc.reset_runtime_overrides()


def test_normalize_overrides_and_helpers():
    from web.job_config import canonical_knob_name, knob_default, normalize_overrides
    assert canonical_knob_name("sam2_export_scale") == "sam3_export_scale"
    assert canonical_knob_name("reprompt_seconds") == "reprompt_seconds"
    assert canonical_knob_name("not_a_knob") == "not_a_knob"
    assert knob_default("sam2_track_fps") == knob_default("sam3_track_fps") == 10.0
    assert normalize_overrides({"sam2_track_fps": 8, "bogus": 1}) == {
        "sam3_track_fps": 8, "bogus": 1}
    assert normalize_overrides(None) == {}
    # Unknown keys still fail JobConfig validation.
    with pytest.raises(ValueError):
        JobConfig({"sam2_bogus": 1})


def test_only_sam2_backend_knobs_keep_sam2_names():
    # sam2_model_size / sam2_vos_optimized only reach the SAM2 predictor build.
    sam2_named = sorted(k.name for k in KNOBS if "sam2" in k.name)
    assert sam2_named == ["sam2_model_size", "sam2_vos_optimized"]
    for knob in KNOBS:
        if knob.name not in sam2_named:
            assert not knob.env.startswith("SAM2_"), knob.name
        for alias in knob.env_aliases:
            assert alias.startswith("SAM2_") and alias[5:] == knob.env[5:]


def test_module_env_constants_read_legacy_names(monkeypatch):
    from web.env_compat import env_get, sam3_env
    monkeypatch.delenv("SAM3_X_TEST", raising=False)
    monkeypatch.delenv("SAM2_X_TEST", raising=False)
    assert sam3_env("X_TEST", "d") == "d"
    monkeypatch.setenv("SAM2_X_TEST", "old")
    assert sam3_env("X_TEST", "d") == "old"
    monkeypatch.setenv("SAM3_X_TEST", "new")
    assert sam3_env("X_TEST", "d") == "new"
    assert env_get("NOPE_X_TEST", "z", legacy=("SAM3_X_TEST",)) == "new"
