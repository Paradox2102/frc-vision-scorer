"""web.cpu_budget: affinity x cgroup-quota CPU count and torch thread clamp."""

import pytest

from web import cpu_budget


def _fake_files(monkeypatch, files: dict, affinity: int = 8):
    def fake_read(path):
        return files.get(path)

    monkeypatch.setattr(cpu_budget, "_read", fake_read)
    monkeypatch.setattr(cpu_budget.os, "sched_getaffinity",
                        lambda pid: set(range(affinity)), raising=False)
    cpu_budget._budget.cache_clear()


@pytest.fixture(autouse=True)
def _clear_cache():
    cpu_budget._budget.cache_clear()
    yield
    cpu_budget._budget.cache_clear()


V2 = cpu_budget._CGROUP_V2_CPU_MAX
V1Q = cpu_budget._CGROUP_V1_QUOTA
V1P = cpu_budget._CGROUP_V1_PERIOD


@pytest.mark.parametrize("cpu_max,expected", [
    ("max 100000", 8),         # unlimited -> affinity
    ("200000 100000", 2),
    ("150000 100000", 2),      # ceil(1.5)
    ("50000 100000", 1),
    ("1600000 100000", 8),     # quota above affinity -> affinity
    ("garbage", 8),
])
def test_cgroup_v2_quota(monkeypatch, cpu_max, expected):
    _fake_files(monkeypatch, {V2: cpu_max})
    assert cpu_budget.effective_cpu_count() == expected


def test_cgroup_v1_quota(monkeypatch):
    _fake_files(monkeypatch, {V1Q: "300000", V1P: "100000"})
    assert cpu_budget.effective_cpu_count() == 3


def test_cgroup_v1_unlimited(monkeypatch):
    _fake_files(monkeypatch, {V1Q: "-1", V1P: "100000"})
    assert cpu_budget.effective_cpu_count() == 8


def test_no_cgroup_files_uses_affinity(monkeypatch):
    _fake_files(monkeypatch, {}, affinity=5)
    assert cpu_budget.effective_cpu_count() == 5


def test_affinity_unavailable_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.setattr(cpu_budget, "_read", lambda p: None)

    def boom(pid):
        raise OSError("nope")

    monkeypatch.setattr(cpu_budget.os, "sched_getaffinity", boom, raising=False)
    monkeypatch.setattr(cpu_budget.os, "cpu_count", lambda: None)
    assert cpu_budget.effective_cpu_count() == 1


def test_real_host_sane():
    assert cpu_budget.effective_cpu_count() >= 1


class _FakeTorch:
    def __init__(self, n):
        self.n = n
        self.set_calls = []

    def get_num_threads(self):
        return self.n

    def set_num_threads(self, n):
        self.set_calls.append(n)
        self.n = n


def test_clamp_lowers_when_over_budget(monkeypatch):
    _fake_files(monkeypatch, {V2: "200000 100000"})
    fake = _FakeTorch(16)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    msg = cpu_budget.clamp_torch_threads()
    assert fake.set_calls == [2]
    assert "clamped" in msg and "quota=2" in msg and "effective=2" in msg


def test_clamp_never_raises(monkeypatch):
    _fake_files(monkeypatch, {V2: "max 100000"}, affinity=12)
    fake = _FakeTorch(6)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    msg = cpu_budget.clamp_torch_threads()
    assert fake.set_calls == []
    assert msg == "cpus: affinity=12 quota=none effective=12 torch_threads=6 (unchanged)"
