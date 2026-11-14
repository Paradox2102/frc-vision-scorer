"""deploy/runpod vast.ai provider: offer search body, instance -> endpoint
mapping, provider dispatch, `up --provider vast --dry-run` (no network: every
HTTP call is mocked; nothing is rented)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
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


cli = _load("runpod_cli")
vast = cli.vast

FAKE_KEY = "vast-test-key-0123456789"


@pytest.fixture
def no_network(monkeypatch):
    """Any real HTTP call fails the test."""
    def boom(*a, **k):
        raise AssertionError(f"unexpected network call: {a[:2]}")
    monkeypatch.setattr(vast.urllib.request, "urlopen", boom)
    monkeypatch.setattr(cli.urllib.request, "urlopen", boom)


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "HERE", tmp_path)
    monkeypatch.setattr(cli, "STATE_PATH", tmp_path / ".state.json")
    monkeypatch.setattr(cli, "PULL_ROOT", tmp_path / "pull")
    return tmp_path


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------- offers
def test_offer_query_defaults():
    q = vast.offer_query()
    assert q["num_gpus"] == {"eq": 1}
    assert q["gpu_name"] == {"in": ["RTX 5090", "RTX 4090"]}
    assert q["rentable"] == {"eq": True} and q["rented"] == {"eq": False}
    assert q["verified"] == {"eq": True} and q["external"] == {"eq": False}
    assert q["cuda_max_good"] == {"gte": 12.8}
    assert q["direct_port_count"] == {"gte": 1}
    assert q["disk_space"] == {"gte": 120.0}
    assert q["reliability"] == {"gte": 0.97}
    assert q["cpu_cores_effective"] == {"gte": 8.0}
    assert "cpu_ghz" not in q and "dph_total" not in q      # 0 / None = no filter
    assert q["order"] == [["cpu_ghz", "desc"], ["dph_total", "asc"]]
    assert q["type"] == "on-demand" and q["allocated_storage"] == 120.0


def test_offer_query_options():
    q = vast.offer_query(gpus=["RTX_5090"], verified=False, min_ghz=5.0, min_cores=12,
                         max_dph=0.9, min_disk=200, min_cuda=13.0, limit=7)
    assert "verified" not in q
    assert q["gpu_name"] == {"in": ["RTX 5090"]}
    assert q["cpu_ghz"] == {"gte": 5.0}
    assert q["cpu_cores_effective"] == {"gte": 12.0}
    assert q["dph_total"] == {"lte": 0.9}
    assert q["disk_space"] == {"gte": 200.0} and q["allocated_storage"] == 200.0
    assert q["cuda_max_good"] == {"gte": 13.0} and q["limit"] == 7


def test_search_offers_posts_bundles_and_sorts(monkeypatch):
    monkeypatch.setenv("VASTAI_API", FAKE_KEY)
    seen = {}
    offers = [{"id": 1, "cpu_ghz": 4.0, "dph_total": 0.3},
              {"id": 2, "cpu_ghz": 5.7, "dph_total": 0.6},
              {"id": 3, "cpu_ghz": 5.7, "dph_total": 0.5}]

    def fake(req, timeout=None):
        seen.update(method=req.get_method(), url=req.full_url, body=json.loads(req.data),
                    auth=req.get_header("Authorization"))
        return _Resp(json.dumps({"offers": offers}).encode())

    monkeypatch.setattr(vast.urllib.request, "urlopen", fake)
    q = vast.offer_query()
    got = vast.search_offers(q)
    assert seen["method"] == "POST" and seen["url"] == "https://console.vast.ai/api/v0/bundles/"
    assert seen["body"] == q and seen["auth"] == f"Bearer {FAKE_KEY}"
    assert [o["id"] for o in got] == [3, 2, 1]


def test_format_offers_table_units():
    o = {"id": 42, "gpu_name": "RTX 5090", "cpu_name": "AMD Ryzen 9 9950X", "cpu_ghz": 5.7,
         "cpu_cores_effective": 16.0, "cpu_ram": 64000, "dph_total": 0.512, "inet_down": 900.0,
         "inet_down_cost": 0.002, "inet_up_cost": 0.001, "cuda_max_good": 12.9,
         "reliability": 0.991, "geolocation": "Texas, US"}
    out = vast.format_offers([o])
    row = out.splitlines()[1]
    assert "RTX 5090" in row and "9950X" in row and " 5.7 " in row and "64G" in row
    assert "2.05" in row and "1.02" in row          # $/GB x1024 = $/TB
    assert "Texas, US" in row


def test_vast_offers_command_json(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(vast, "_request", lambda m, p, b=None, **k: calls.append((m, p, b)) or
                        {"offers": [{"id": 9, "cpu_ghz": 5.0}]})
    assert cli.main(["vast-offers", "--gpu", "RTX 4090", "--min-ghz", "5", "--json"]) == 0
    assert calls[0][0] == "POST" and calls[0][1] == "/bundles/"
    assert calls[0][2]["gpu_name"] == {"in": ["RTX 4090"]} and calls[0][2]["cpu_ghz"] == {"gte": 5.0}
    assert json.loads(capsys.readouterr().out)[0]["id"] == 9


# ----------------------------------------------------------------- instances
DIRECT = {"id": 777, "actual_status": "running", "dph_total": 0.61, "gpu_name": "RTX 5090",
          "public_ipaddr": "203.0.113.7 ", "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]},
          "ssh_host": "ssh4.vast.ai", "ssh_port": 31777, "start_date": 1_800_000_000.0,
          "image_uuid": cli.DEFAULT_IMAGE}
PROXY = {"id": 778, "actual_status": "loading", "dph_total": 0.55, "public_ipaddr": "203.0.113.8",
         "ports": {}, "ssh_host": "ssh5.vast.ai", "ssh_port": 31778}


def test_endpoint_prefers_direct():
    assert vast.ssh_endpoint(DIRECT) == ("203.0.113.7", 41022, "direct")
    assert vast.instance_endpoint(DIRECT) == ("RUNNING", 0.61, "203.0.113.7", 41022)


def test_endpoint_falls_back_to_proxy():
    assert vast.ssh_endpoint(PROXY) == ("ssh5.vast.ai", 31778, "proxy")
    assert vast.instance_endpoint(PROXY) == ("LOADING", 0.55, "ssh5.vast.ai", 31778)
    assert vast.ssh_endpoint({"actual_status": "created"}) == (None, None, None)


@pytest.mark.parametrize("raw,want", [
    ("running", "RUNNING"), ("stopped", "EXITED"), ("exited", "EXITED"),
    ("loading", "LOADING"), ("created", "CREATED"), ("offline", "OFFLINE"), (None, "UNKNOWN"),
])
def test_status_mapping(raw, want):
    assert vast.map_status(raw) == want


def test_pod_view_is_runpod_shaped():
    pv = vast.pod_view(DIRECT)
    assert pv["desiredStatus"] == "RUNNING" and pv["publicIp"] == "203.0.113.7"
    assert pv["portMappings"] == {"22": 41022} and pv["costPerHr"] == 0.61
    assert pv["gpu"]["displayName"] == "RTX 5090" and pv["lastStartedAt"].startswith("2027-01-15")


def test_create_body_has_no_secrets_and_ssh_direct():
    b = vast.build_create_body(image=cli.DEFAULT_IMAGE, disk_gb=120, label="frc-x")
    assert b["client_id"] == "me" and b["image"] == cli.DEFAULT_IMAGE and b["disk"] == 120
    assert b["runtype"] == "ssh_direc ssh_proxy" and b["env"] == {}
    assert b["cancel_unavail"] is True and b["price"] is None
    blob = json.dumps(b)
    for name in ("RUNPOD_API", "VASTAI_API", "HF_TOKEN", "JEV", "TBA", "PUBLIC_KEY"):
        assert name not in blob


def test_create_instance_strips_instance_key(monkeypatch):
    calls = []

    def fake(m, p, b=None, **k):
        calls.append((m, p, b))
        return {"success": True, "new_contract": 31337, "instance_api_key": "secret-xyz"}
    monkeypatch.setattr(vast, "_request", fake)
    made = vast.create_instance(555, image="img", disk_gb=100, label="l")
    assert calls == [("PUT", "/asks/555/", vast.build_create_body(image="img", disk_gb=100, label="l"))]
    assert made["id"] == 31337 and "secret-xyz" not in json.dumps(made)


def test_show_stop_destroy_paths(monkeypatch):
    calls = []

    def fake(m, p, b=None, **k):
        calls.append((m, p, b, k.get("query")))
        return {"instances": DIRECT} if m == "GET" else {"success": True}
    monkeypatch.setattr(vast, "_request", fake)
    assert vast.show_instance("777")["id"] == 777
    vast.stop_instance("777")
    vast.start_instance(777)
    vast.destroy_instance("777")
    assert calls == [("GET", "/instances/777/", None, {"owner": "me"}),
                     ("PUT", "/instances/777/", {"state": "stopped"}, None),
                     ("PUT", "/instances/777/", {"state": "running"}, None),
                     ("DELETE", "/instances/777/", {}, None)]


def test_key_registered():
    keys = ["ssh-ed25519 AAAAbody other@host"]
    assert vast.key_registered("ssh-ed25519 AAAAbody nick@box", keys)
    assert not vast.key_registered("ssh-ed25519 AAAAdifferent nick@box", keys)
    assert not vast.key_registered("", keys)


def test_api_key_from_env_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("VASTAI_API", FAKE_KEY)
    assert vast.api_key() == FAKE_KEY
    assert FAKE_KEY not in capsys.readouterr().out


# ----------------------------------------------------------------- secrets
def test_never_send_vast_key():
    assert cli.never_send("VASTAI_API")
    assert cli.never_send("RUNPOD_API")
    local = {"VASTAI_API": FAKE_KEY, "JEV_TYPESAFE_API_TOKEN": "j", "HF_TOKEN": "h", "TBA_API_KEY": "t"}
    specs = cli.SECRET_SPECS + (cli.SecretSpec("VASTAI_API", False, "x"),)
    got = cli.select_secrets(local, interactive=False, specs=specs)
    assert "VASTAI_API" not in got
    with pytest.raises(ValueError):
        cli.render_dotenv({"VASTAI_API": FAKE_KEY})


# ----------------------------------------------------------------- dispatch
def test_runpod_default_when_state_has_no_provider(monkeypatch, tmp_state, no_network):
    st = {"pod_id": "abc123"}
    assert cli.provider(st) == "runpod"
    monkeypatch.setattr(cli, "get_pod", lambda pid: {
        "id": pid, "desiredStatus": "RUNNING", "costPerHr": 2.69, "publicIp": "198.51.100.2",
        "portMappings": {"22": 22011}, "gpu": {"displayName": "RTX PRO 6000"}})
    monkeypatch.setattr(vast, "show_instance", lambda *a: pytest.fail("vast called for a RunPod pod"))
    pod = cli.refresh_endpoint(st)
    assert pod["desiredStatus"] == "RUNNING"
    assert st["ssh_host"] == "198.51.100.2" and st["ssh_port"] == 22011 and st["cost_per_hr"] == 2.69
    assert "provider" not in st
    assert json.loads(cli.STATE_PATH.read_text())["pod_id"] == "abc123"


def test_vast_state_refreshes_from_vast(monkeypatch, tmp_state, no_network):
    st = {"provider": "vast", "pod_id": "777"}
    monkeypatch.setattr(cli, "get_pod", lambda pid: pytest.fail("RunPod called for a vast instance"))
    monkeypatch.setattr(vast, "show_instance", lambda iid: dict(DIRECT))
    pod = cli.refresh_endpoint(st)
    assert pod["desiredStatus"] == "RUNNING"
    assert st["ssh_host"] == "203.0.113.7" and st["ssh_port"] == 41022 and st["cost_per_hr"] == 0.61
    assert st["gpu"] == "RTX 5090"
    assert cli.ssh_base(st)[-1] == "root@203.0.113.7"


def _stub_bundle(monkeypatch):
    monkeypatch.setattr(cli, "warn_dirty", lambda ref: None)
    monkeypatch.setattr(cli, "plan_bundle", lambda ref, **k: cli.BundlePlan(
        ref=ref, commit="deadbee", code_bytes=b"", code_files=[]))
    monkeypatch.setattr(cli, "local_env", lambda: {
        "VASTAI_API": FAKE_KEY, "RUNPOD_API": "rp-secret-zzzz", "JEV_TYPESAFE_API_TOKEN": "jev-1234567890",
        "HF_TOKEN": "hf-1234567890", "TBA_API_KEY": "tba-1234567890"})


def test_up_vast_dry_run_creates_nothing(monkeypatch, tmp_state, no_network, capsys):
    _stub_bundle(monkeypatch)
    monkeypatch.setattr(vast, "_request", lambda *a, **k: pytest.fail(f"vast API called in dry run: {a[:2]}"))
    monkeypatch.setattr(cli, "_request", lambda *a, **k: pytest.fail(f"RunPod API called in dry run: {a[:2]}"))
    assert cli.main(["up", "--provider", "vast", "--offer", "123", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "https://console.vast.ai/api/v0/asks/123/" in out
    assert '"runtype": "ssh_direc ssh_proxy"' in out and '"disk": 120' in out
    assert FAKE_KEY not in out and "rp-secret-zzzz" not in out
    assert "VASTAI_API" not in out.split("secrets that would be installed")[1]
    assert not cli.STATE_PATH.exists()


def test_up_vast_requires_offer(monkeypatch, tmp_state, no_network):
    _stub_bundle(monkeypatch)
    with pytest.raises(SystemExit, match="--offer"):
        cli.main(["up", "--provider", "vast", "--dry-run"])


def test_up_refuses_provider_mismatch(monkeypatch, tmp_state, no_network):
    _stub_bundle(monkeypatch)
    cli.save_state({"pod_id": "abc123"})                       # a RunPod pod (no provider)
    with pytest.raises(SystemExit, match="runpod pod"):
        cli.main(["up", "--provider", "vast", "--offer", "1", "--dry-run"])
    cli.save_state({"provider": "vast", "pod_id": "777"})
    with pytest.raises(SystemExit, match="vast pod"):
        cli.main(["up", "--provider", "runpod", "--dry-run"])


def test_up_runpod_dry_run_unchanged(monkeypatch, tmp_state, no_network, capsys):
    _stub_bundle(monkeypatch)
    assert cli.main(["up", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "POST https://rest.runpod.io/v1/pods" in out and "vast" not in out.lower()


def test_terminate_vast_destroys_and_archives(monkeypatch, tmp_state, no_network):
    cli.save_state({"provider": "vast", "pod_id": "777"})
    monkeypatch.setattr(vast, "show_instance", lambda iid: dict(DIRECT))
    destroyed = []
    monkeypatch.setattr(vast, "destroy_instance", lambda iid: destroyed.append(iid) or {"success": True})
    monkeypatch.setattr(cli, "_request", lambda *a, **k: pytest.fail("RunPod DELETE for a vast instance"))
    prompts = []
    monkeypatch.setattr(cli, "confirm", lambda msg, word: prompts.append((msg, word)) or True)
    assert cli.main(["terminate"]) == 0
    assert destroyed == ["777"]
    assert prompts[0][1] == "terminate 777" and "NOT run `pull`" in prompts[0][0]
    assert not cli.STATE_PATH.exists() and (tmp_state / ".state.777.terminated.json").exists()


def test_terminate_vast_declined(monkeypatch, tmp_state, no_network):
    cli.save_state({"provider": "vast", "pod_id": "777"})
    monkeypatch.setattr(vast, "show_instance", lambda iid: dict(DIRECT))
    monkeypatch.setattr(vast, "destroy_instance", lambda iid: pytest.fail("destroyed without confirm"))
    monkeypatch.setattr(cli, "confirm", lambda msg, word: False)
    assert cli.main(["terminate"]) == 1
    assert cli.STATE_PATH.exists()


def test_stop_vast(monkeypatch, tmp_state, no_network):
    cli.save_state({"provider": "vast", "pod_id": "777"})
    monkeypatch.setattr(vast, "show_instance", lambda iid: dict(DIRECT))
    monkeypatch.setattr(cli, "remote_job_active", lambda st: None)
    stopped = []
    monkeypatch.setattr(vast, "stop_instance", lambda iid: stopped.append(iid))
    monkeypatch.setattr(cli, "_request", lambda *a, **k: pytest.fail("RunPod stop for a vast instance"))
    monkeypatch.setattr(cli, "confirm", lambda msg, word: word == "stop" and "vast instance" in msg)
    assert cli.main(["stop"]) == 0 and stopped == ["777"]


def test_status_vast(monkeypatch, tmp_state, no_network, capsys):
    cli.save_state({"provider": "vast", "pod_id": "777", "cost_per_hr": 0.61, "pushed_ref": "main"})
    monkeypatch.setattr(vast, "show_instance", lambda iid: {**DIRECT, "cpu_name": "AMD Ryzen 9 9950X",
                                                           "start_date": cli.time.time() - 7200})
    monkeypatch.setattr(vast, "current_user", lambda: {"credit": 12.5})
    remote = []
    monkeypatch.setattr(cli, "ssh_out", lambda st, cmd, **k: remote.append(cmd) or "GPU OK\n")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "vast instance 777  RUNNING  RTX 5090  $0.610/h" in out
    assert "9950X" in out and "up 2.0 h" in out and "~$1.22" in out and "$12.50" in out
    assert "ssh root@203.0.113.7 -p 41022 (direct)" in out and "GPU OK" in out
    assert remote and remote[0].startswith(cli.STATUS_REMOTE)
