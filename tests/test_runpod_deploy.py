"""deploy/runpod: profiles, secrets, API body, bundle manifest (no network, no pods)."""
from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNPOD = REPO / "deploy" / "runpod"


def _load(name):
    sys.path.insert(0, str(RUNPOD))
    spec = importlib.util.spec_from_file_location(name, RUNPOD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


profiles = _load("profiles")
cli = _load("runpod_cli")


# ----------------------------------------------------------------- profiles
@pytest.mark.parametrize("mib,tier", [
    (24576, "t24"),      # RTX 3090
    (46068, "t48"),      # A40 / A6000 / L40S report ~45 GiB
    (49140, "t48"),
    (81559, "t80"),      # H100 80GB
    (97887, "t80"),      # RTX PRO 6000 96GB
    (143771, "t141"),    # H200
])
def test_tier_selection(mib, tier):
    assert profiles.tier_for_vram(mib).name == tier


def test_tier_bounds_are_contiguous():
    t = profiles.TIERS
    assert t[0].min_gib == 0
    for a, b in zip(t, t[1:]):
        assert a.max_gib == b.min_gib
        assert b.chunk_cap >= a.chunk_cap


def test_knob_env_matches_job_config_registry():
    from web.job_config import KNOBS
    by_name = {k.name: k for k in KNOBS}
    for name, env in profiles.KNOB_ENV.items():
        assert by_name[name].env == env
        assert by_name[name].default == profiles.REPO_DEFAULTS[name], name
    # every BASE_ENV knob var is a real knob env (except TEAM_READER's own knob, also real)
    envs = {k.env for k in KNOBS}
    for var in ("VLLM_PHASE_EXCLUSIVE", "TEAM_READER"):
        assert var in envs and var in profiles.BASE_ENV


def test_base_env_mirrors_start_all_sh():
    text = (REPO / "web" / "start_all.sh").read_text()
    import re
    exports = dict(re.findall(r'^export (\w+)="\$\{\w+:-([^}]*)\}"', text, re.M))
    assert exports, "start_all.sh export format changed"
    for k, v in exports.items():
        assert profiles.BASE_ENV.get(k) == v, k


def test_ladder_is_cumulative_and_starts_at_repo_defaults():
    tier = profiles.tier_for_vram(81559)
    rungs = profiles.ladder(tier)
    assert [r.name for r in rungs] == ["R0", "R1", "R2", "R3", "R4", "R5"]
    assert rungs[0].knobs == profiles.REPO_DEFAULTS
    for prev, cur in zip(rungs, rungs[1:]):
        assert cur.knobs == {**prev.knobs, **cur.delta}
        assert cur.delta and cur.rationale
    top = rungs[-1].knobs
    assert top["corner_track_stride"] == 1 and top["sam3_track_fps"] == 15.0
    assert top["sam3_chunk_frames"] == tier.chunk_cap
    # quality only goes up: fps never drops, stride never rises, chunk never shrinks
    for prev, cur in zip(rungs, rungs[1:]):
        assert cur.knobs["sam3_track_fps"] >= prev.knobs["sam3_track_fps"]
        assert cur.knobs["corner_track_stride"] <= prev.knobs["corner_track_stride"]
        assert cur.knobs["sam3_chunk_frames"] >= prev.knobs["sam3_chunk_frames"]


def test_rung_config_validates_against_jobconfig():
    from web.job_config import JobConfig
    for tier in profiles.TIERS:
        for r in profiles.ladder(tier):
            cfg = JobConfig({**r.config(), "auto_period_only": False})
            assert cfg.sam3_track_fps == r.knobs["sam3_track_fps"]
            assert cfg.sam3_chunk_frames == r.knobs["sam3_chunk_frames"]


def test_profile_env_render_roundtrip():
    tier = profiles.tier_for_vram(46068)
    r = profiles.rung_by_name(tier, "r3")
    env = profiles.profile_env(tier, r, extra={"YTDLP_COOKIES_FILE": "/workspace/youtube_cookies.txt"})
    parsed = profiles.parse_env_file(profiles.render_profile_env(env, "hdr"))
    assert parsed == {k: str(v) for k, v in env.items()}
    assert parsed["SAM3_CHUNK_FRAMES"] == str(tier.chunk_cap)
    assert parsed["SAM3_TRACK_FPS"] == "10.0" and parsed["CORNER_TRACK_STRIDE"] == "2"
    assert parsed["VLLM_PHASE_EXCLUSIVE"] == "0" and parsed["FRC_PROFILE_RUNG"] == "R3"
    with pytest.raises(ValueError):
        profiles.render_profile_env({"X": "a b"})


def test_profiles_cli_writes_profile(tmp_path):
    out = tmp_path / "profile.env"
    rc = profiles.main(["--write-profile", str(out), "--vram-mib", "81559"])
    assert rc == 0
    env = profiles.parse_env_file(out.read_text())
    assert env["FRC_GPU_TIER"] == "t80" and env["FRC_PROFILE_RUNG"] == "R0"
    out.write_text("KEEP=1\n")
    profiles.main(["--write-profile", str(out), "--vram-mib", "81559"])
    assert out.read_text() == "KEEP=1\n"       # never clobbers an applied profile


# ----------------------------------------------------------------- secrets
LOCAL = {"JEV_TYPESAFE_API_TOKEN": "jev-secret-1234", "HF_TOKEN": "hf_abcdefgh5678",
         "TBA_API_KEY": "tba-key-9999", "RUNPOD_API": "rp-should-never-leave",
         "CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acct", "MANTA_API": "m",
         "ROBOFLOW_API": "rf"}


def test_mask_shows_last_four_only():
    assert cli.mask("abcdefgh1234") == "****1234"
    assert cli.mask("short") == "****"
    assert cli.mask("") == "(unset)"


def test_select_secrets_noninteractive_never_sends_denied():
    got = cli.select_secrets(LOCAL, interactive=False)
    assert set(got) == {"JEV_TYPESAFE_API_TOKEN", "HF_TOKEN", "TBA_API_KEY"}
    for bad in ("RUNPOD_API", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "ROBOFLOW_API", "MANTA_API"):
        assert bad not in got


def test_select_secrets_interactive_keep_replace_skip():
    answers = iter(["", "hf_newtoken_0000", "", "-", "deepseek-xyz"])
    prompts = []

    def fake_input(p):
        prompts.append(p)
        return next(answers)
    got = cli.select_secrets(LOCAL, interactive=True, input_fn=fake_input)
    assert got["JEV_TYPESAFE_API_TOKEN"] == LOCAL["JEV_TYPESAFE_API_TOKEN"]   # Enter = keep
    assert got["HF_TOKEN"] == "hf_newtoken_0000"                              # pasted
    assert "ANTHROPIC_API_KEY" not in got                                     # '-' = skip
    assert got["DEEPSEEK_API"] == "deepseek-xyz"
    joined = "\n".join(prompts)
    assert "****1234" in joined and "jev-secret-1234" not in joined          # masked, never echoed
    assert "RUNPOD" not in joined and "CLOUDFLARE" not in joined


def test_select_secrets_required_missing_raises():
    with pytest.raises(SystemExit):
        cli.select_secrets({"HF_TOKEN": "x" * 12}, interactive=False)


def test_render_dotenv_refuses_denied_and_quotes():
    text = cli.render_dotenv({"HF_TOKEN": "hf_abc", "TBA_API_KEY": "a b#c"})
    parsed = cli.parse_dotenv(text)
    assert parsed == {"HF_TOKEN": "hf_abc", "TBA_API_KEY": "a b#c"}
    with pytest.raises(ValueError):
        cli.render_dotenv({"RUNPOD_API": "x"})
    with pytest.raises(ValueError):
        cli.render_dotenv({"CLOUDFLARE_API_TOKEN": "x"})


def test_secret_specs_cover_server_env_reads():
    """Every credential-looking env var the server reads is either a spec or
    deliberately denied (scan of web/ and src/)."""
    import re
    names = set()
    for p in list((REPO / "web").rglob("*.py")) + list((REPO / "src").rglob("*.py")):
        for m in re.finditer(r"""environ(?:\.get)?\(?\[?\s*["']([A-Z0-9_]+)["']|getenv\(\s*["']([A-Z0-9_]+)["']""",
                             p.read_text(errors="ignore")):
            names.add(m.group(1) or m.group(2))
    secretish = {n for n in names if re.search(r"(KEY|TOKEN|SECRET|_API$|PASSWORD)", n)}
    specs = {s.name for s in cli.SECRET_SPECS}
    for n in secretish:
        assert n in specs or cli.never_send(n), n


# ----------------------------------------------------------------- API body
def test_create_body_dry_run_shape():
    body = cli.build_create_body(name="frc", gpus=["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
                                 cloud="SECURE", image=cli.DEFAULT_IMAGE, volume_gb=150,
                                 container_gb=40, pubkey="ssh-ed25519 AAAAkey me@host\n",
                                 min_vcpu=8, min_ram=32)
    assert body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
    assert body["gpuTypePriority"] == "custom" and body["gpuCount"] == 1
    assert body["ports"] == ["22/tcp"]                      # never expose 8004 over http
    assert body["env"] == {"PUBLIC_KEY": "ssh-ed25519 AAAAkey me@host"}   # no secrets in RunPod metadata
    assert body["volumeMountPath"] == "/workspace" and body["volumeInGb"] == 150
    assert all(float(v) >= 12.8 for v in body["allowedCudaVersions"])
    assert body["interruptible"] is False and body["supportPublicIp"] is True
    nv = cli.build_create_body(name="frc", gpus=["NVIDIA A40"], cloud="COMMUNITY", image="x",
                               volume_gb=150, container_gb=40, pubkey="k", min_vcpu=8, min_ram=32,
                               network_volume_id="vol123")
    assert nv["networkVolumeId"] == "vol123" and "volumeInGb" not in nv
    with pytest.raises(ValueError):
        cli.build_create_body(name="x", gpus=["g"], cloud="CHEAP", image="x", volume_gb=1,
                              container_gb=1, pubkey="k", min_vcpu=1, min_ram=1)


def test_default_gpus_are_valid_api_ids_and_in_target_range():
    import json
    # enum from the RunPod REST OpenAPI (PodCreateInput.gpuTypeIds), Nov 2026
    known_80_plus = {"NVIDIA H100 80GB HBM3", "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                     "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"}
    assert set(cli.DEFAULT_GPUS) <= known_80_plus
    assert json.dumps(cli.redact_body({"env": {"PUBLIC_KEY": "ssh-ed25519 " + "A" * 60}}))


def test_parse_ts_variants():
    for s in ("2026-11-12T10:00:00Z", "2026-11-12T10:00:00.123Z", "2026-11-12 10:00:00.123 +0000 UTC"):
        t = cli.parse_ts(s)
        assert t is not None and t.tzinfo is not None and t.hour == 10, s
    assert cli.parse_ts(None) is None and cli.parse_ts("garbage") is None


def test_recommend_picks_highest_in_budget():
    rows = {"R0": {"status": "done", "wall_final_s": 300}, "R1": {"status": "done", "wall_final_s": 420},
            "R2": {"status": "done", "wall_final_s": 530}}
    order = ["R0", "R1", "R2"]
    assert cli.recommend(rows, order, 8.0) == "R1"
    assert cli.recommend(rows, order, 9.0) == "R2"
    assert cli.recommend(rows, order, 4.0) is None
    assert cli.over_budget({"wall_job_s": 400, "wall_final_s": 500}, 8.0, on="job") is False
    assert cli.over_budget({"wall_job_s": 400, "wall_final_s": 500}, 8.0, on="final") is True


# ----------------------------------------------------------------- bundle
MAIN = cli.main_checkout()


def test_weights_manifest_resolves_in_main_checkout():
    rows = cli.weights_manifest(MAIN, include_optional=True)
    rels = [arc for _, arc, _ in rows]
    for w in cli.WEIGHTS:
        if w.required:
            assert w.rel in rels
    for p, arc, kind in rows:
        assert p.is_file() and p.stat().st_size > 1_000_000, arc     # real weights, not LFS pointers
        assert kind == "weights" and not arc.startswith("/")
    # no unused variants
    assert not any(("s1_nosynth" in a or "s4_r" in a or "gapfill" in a or "hiera_large" in a) for a in rels)


def test_weights_manifest_matches_code_paths():
    """The manifest paths are the ones the code resolves (PROJECT_ROOT / main checkout relative)."""
    from web.students import paths as sp
    rels = {w.rel for w in cli.WEIGHTS}
    assert str(sp.BUMPER_WEIGHTS.relative_to(sp.MAIN)) in rels
    assert "data/teacher/models/s1_digits.pt" in rels and "data/teacher/models/s4_embedder.pt" in rels
    app_src = (REPO / "web" / "app.py").read_text()
    for rel in ("runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt",
                "runs/train/frc_bumpers_equipment_hardneg/weights/best.pt", '"yolo26n.pt"'):
        assert rel in app_src


def test_code_filter_excludes_runtime_and_training_dirs():
    for bad in ("web/uploads/video_x.mp4", "web/replays/abc/meta.json", "web/logs/x.log",
                "data/field/FE-2026-rev-rebuilt-playing-field.step",
                "data/field/reference_frames/a.jpg", "web/__pycache__/app.cpython-312.pyc",
                "deploy/runpod/tune_results/pod.json"):
        assert not cli.code_member_ok(bad), bad
    for ok in ("web/app.py", "web/field/season.py", "data/field/2026-rebuilt-welded.json",
               "data/field/wireframes/2026-rebuilt-welded.json", "deploy/runpod/bootstrap.sh"):
        assert cli.code_member_ok(ok), ok
    assert all(not p.startswith(("data/teacher", "data/eval", "runs", "scripts", "labeling"))
               for p in cli.CODE_PATHS)


def test_event_state_filter():
    assert cli.event_state_ok("calibration_2026cur_main__20261107-123640-609667.json")
    assert cli.event_state_ok("composite_layout_2026cur.jpg")
    assert cli.event_state_ok("zones_2026cur.json") and cli.event_state_ok("clockscan_abc.json")
    for bad in ("diagnostic_x.jsonl", "detections_x.jsonl", "video_abc.mp4", "frame_1.jpg",
                "calib_frame_x.jpg", "ff4be1d9-b8e5-49ff-ac78-4a6a0e77a697.mp4"):
        assert not cli.event_state_ok(bad), bad


def test_filtered_code_tar_from_head():
    data, names = cli.filtered_code_tar("HEAD")
    assert "web/app.py" in names and "web/field/season.py" in names
    assert "data/field/2026-rebuilt-welded.json" in names
    assert not any(n.startswith(("web/uploads/", "web/replays/")) or n.endswith(".step") for n in names)
    assert len(data) < 20_000_000
    tf = tarfile.open(fileobj=io.BytesIO(data))
    assert set(tf.getnames()) >= set(names)


def test_shell_scripts_parse():
    for sh in ("bootstrap.sh", "serve.sh", "pod_env.sh"):
        subprocess.run(["bash", "-n", str(RUNPOD / sh)], check=True)


def test_requirements_pins():
    req = (RUNPOD / "requirements-runpod.txt").read_text()
    assert "download.pytorch.org/whl/cu128" in req
    assert "torch==2.10.0+cu128" in req and "torchvision==0.25.0+cu128" in req
    assert "opencv-python-headless==" in req and "\nopencv-python==" not in req
    assert not any(l.strip().lower().startswith("vllm") for l in req.splitlines())
    assert "yt-dlp==2026.10.26" in req
    git_req = (RUNPOD / "requirements-runpod-git.txt").read_text()
    pyproject = (REPO / "pyproject.toml").read_text()
    for commit in ("2b90b9f5ceec907a1c18123530e92e794ad901a4", "8e451d5eb43c817b64ae7577fb7b9ae223db88a9"):
        assert commit in git_req and commit in pyproject


def test_yes_alone_never_reads_local_env(monkeypatch):
    """--yes skips prompts but must not silently ship the operator's .env keys."""
    import argparse
    monkeypatch.setattr(cli, "local_env", lambda: dict(LOCAL))
    with pytest.raises(SystemExit, match="secrets-from-env"):
        cli.gather_secrets(argparse.Namespace(yes=True, secrets_from_env=False))
    got = cli.gather_secrets(argparse.Namespace(yes=True, secrets_from_env=True))
    assert got["TBA_API_KEY"] == "tba-key-9999" and "RUNPOD_API" not in got


def test_repo_ships_no_key_defaults():
    """No secret spec carries a baked-in default value."""
    for s in cli.SECRET_SPECS:
        assert not getattr(s, "default", None)


def test_multiplex_arm_layers_on_base_rung():
    tier = profiles.tier_for_vram(97887)
    base = profiles.rung_by_name(tier, "R1")
    arms = {a.tag: a for a in profiles.ARMS}
    assert arms["M"].knobs == {"sam3_multiplex": True}
    r = profiles.arm_rung(base, arms["M"])
    assert r.name == "R1M"
    assert r.knobs == {**base.knobs, "sam3_multiplex": True}
    assert r.delta == {"sam3_multiplex": True}
    # arms are never ladder rungs (never recommended / applied)
    assert all(not x.name.endswith("M") for x in profiles.ladder(tier))


def test_parallel_views_and_qwen_reader_arms(monkeypatch):
    monkeypatch.delenv("VLM_SERVER_GPU_GB", raising=False)
    tier = profiles.tier_for_vram(97887)
    base = profiles.rung_by_name(tier, "R1")
    arms = {a.tag: a for a in profiles.ARMS}
    assert [a.tag for a in profiles.ARMS][:3] == ["M", "P", "Q"]
    assert arms["P"].knobs == {"sam3_parallel_views": True}
    assert arms["Q"].knobs == {"team_reader": "vllm", "vlm_model": "qwen3.5-4b",
                               "vllm_phase_exclusive": False}
    for tag in ("P", "Q"):
        r = profiles.arm_rung(base, arms[tag])
        assert r.name == "R1" + tag
        assert r.knobs == {**base.knobs, **arms[tag].knobs}
        assert r.delta == arms[tag].knobs
        assert arms[tag].rationale
    # Q's reader really is the escape-prompt / thinking-off Qwen3.5-4B entry,
    # and on the 96 GB card its default absolute budget is the 12 GiB the pod
    # bench ran (util 0.126), leaving the card to SAM3.
    from web import vlm_models
    assert vlm_models.read_prompt("qwen3.5-4b") == "escape"
    assert vlm_models.extra_body("qwen3.5-4b") == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert vlm_models.get("qwen3.5-4b").hf_repo == "Qwen/Qwen3.5-4B"
    assert vlm_models.serve_util("qwen3.5-4b", 95.6) == pytest.approx(12.0 / 95.6, abs=1e-3)


def test_every_arm_knob_is_a_real_job_config_knob():
    """A knob rename (e.g. sam2 -> sam3) must not silently turn an arm into a
    no-op: arm knobs use canonical names (not aliases), and the per-job
    config the pod submits resolves to the arm's values."""
    from web.job_config import KNOBS, JobConfig
    names = {k.name for k in KNOBS}
    tier = profiles.tier_for_vram(97887)
    base = profiles.rung_by_name(tier, "R1")
    tags = [a.tag for a in profiles.ARMS]
    assert len(tags) == len(set(tags))
    for arm in profiles.ARMS:
        assert arm.knobs, arm.tag
        for k in arm.knobs:
            assert k in names, f"arm {arm.tag}: {k!r} is not a web.job_config knob"
        cfg = JobConfig({**profiles.arm_rung(base, arm).config(), "auto_period_only": False})
        for k, v in arm.knobs.items():
            assert getattr(cfg, k) == v, (arm.tag, k)


def test_arm_base_prefers_recommendation_then_r1():
    import runpod_cli as cli
    rungs = profiles.ladder(profiles.tier_for_vram(97887))
    done = {"status": "done"}
    assert cli.arm_base(rungs, {"R0": done, "R1": done, "R2": done}, "R2").name == "R2"
    assert cli.arm_base(rungs, {"R0": done, "R1": done, "R2": done}, None).name == "R1"
    assert cli.arm_base(rungs, {"R0": done}, None).name == "R0"
    assert cli.arm_base(rungs, {}, None) is None
