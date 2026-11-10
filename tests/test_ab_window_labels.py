"""ab_qm8 window-labels profile: arms, and waiting for the replay that the
finalize stage writes minutes after the job reports "completed"."""
import json

import scripts.ab_qm8 as ab


def test_profile_arms():
    arms = dict(ab.PROFILES["window-labels"])
    assert arms["off"] == {}
    assert arms["wl_08"] == {"window_labels": True}
    assert arms["wl_065"] == {"window_labels": True, "window_labels_lineup_min_p": 0.65}


def test_wait_returns_once_the_replay_has_the_stage_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "REPO", tmp_path)
    meta = tmp_path / "web" / "replays" / "job1" / "meta.json"
    meta.parent.mkdir(parents=True)
    polls = {"n": 0}

    def saved(job_id, tries=1, delay=0.0):
        polls["n"] += 1
        if polls["n"] == 2:
            meta.write_text(json.dumps({"summary": {}}))               # replay, no stage yet
        if polls["n"] == 3:
            meta.write_text(json.dumps({"summary": {"window_labels": {"aborted": None}}}))
        return polls["n"] >= 2

    monkeypatch.setattr(ab, "_replay_saved", saved)
    assert ab._wait_window_labels("job1", timeout_s=5.0, delay=0.0) is True
    assert polls["n"] == 3


def test_wait_gives_up_after_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "REPO", tmp_path)
    monkeypatch.setattr(ab, "_replay_saved", lambda job_id, tries=1, delay=0.0: False)
    assert ab._wait_window_labels("job2", timeout_s=0.05, delay=0.01) is False
