"""--meta overrides on the GT scorer and the replay auditor, and the
auditor's score --json. No network, no video."""
import json
from types import SimpleNamespace

import pytest

import scripts.audit_replay as ar
import scripts.score_ground_truth as sg


def _replay_dir(tmp_path):
    own = tmp_path / "replays" / "rid" / "meta.json"
    own.parent.mkdir(parents=True)
    own.write_text(json.dumps({"tag": "own", "frames": []}))
    other = tmp_path / "arm.meta.json"
    other.write_text(json.dumps({"tag": "arm", "frames": []}))
    return other


ROW = {"id": "rid", "match_key": "2026cur_qm8", "created_at": "2026-11-10T09:00:00",
       "source_video_path": None, "meta_path": None}


def test_the_gt_scorer_can_score_an_override_meta(tmp_path, monkeypatch):
    other = _replay_dir(tmp_path)
    monkeypatch.setattr(sg, "REPLAYS_DIR", tmp_path / "replays")
    monkeypatch.setattr(sg, "replay_from_meta", lambda meta, **kw: meta)
    assert sg.load_replay(dict(ROW))["tag"] == "own"
    assert sg.load_replay(dict(ROW), meta_path=str(other))["tag"] == "arm"
    with pytest.raises(FileNotFoundError):
        sg.load_replay(dict(ROW), meta_path=str(tmp_path / "missing.json"))


def test_the_auditor_can_take_labels_from_an_override_meta(tmp_path, monkeypatch):
    import scripts.build_track_truth as btt
    import scripts.relabel_replay as rr
    other = _replay_dir(tmp_path)
    monkeypatch.setattr(sg, "REPLAYS_DIR", tmp_path / "replays")
    monkeypatch.setattr(sg, "replay_row", lambda rid: dict(ROW))
    monkeypatch.setattr(btt, "job_views", lambda *a: [])
    monkeypatch.setattr(rr, "job_roster", lambda mk: (["1"], ["2"]))
    assert ar.load_replay_inputs("rid")["meta"]["tag"] == "own"
    assert ar.load_replay_inputs("rid", str(other))["meta"]["tag"] == "arm"


def test_score_json_carries_the_estimates(tmp_path):
    d = tmp_path / "audit"
    d.mkdir()
    (d / "audit.json").write_text(json.dumps({
        "replay": "rid", "match_key": "2026cur_qm22", "meta": "arm.meta.json",
        "roster": {"red": ["1", "2", "3"], "blue": ["4", "5", "6"]},
        "strata": {"lab": {"N": 100}, "unl": {"N": 50}}, "frame_period_s": 0.2,
        "labelled_slot_s": 20.0, "unlabelled_slot_s": 10.0}))
    samples = [{"sample_id": f"s{i}", "views": ["main"], "stratum": "lab",
                "pipeline_label": "1", "slot": "R1", "period": "teleop"} for i in range(4)]
    samples.append({"sample_id": "s9", "views": ["main"], "stratum": "unl",
                    "pipeline_label": None, "slot": "R2", "period": "teleop"})
    ar.write_jsonl(d / "samples.jsonl", samples)
    ar.write_jsonl(d / "ans.jsonl", [
        {"sample_id": "s0", "team": "1", "confidence": 0.9},
        {"sample_id": "s1", "team": "2", "confidence": 0.9},
        {"sample_id": "s2", "team": "1", "confidence": 0.9},
        {"sample_id": "s3", "team": "1", "confidence": 0.9},
        {"sample_id": "s9", "team": "unsure", "confidence": 0.9}])
    out = tmp_path / "score.json"
    ar.cmd_score(SimpleNamespace(audit=str(d), answers=str(d / "ans.jsonl"), min_conf=0.7,
                                 no_gt=True, dump=None, json=str(out)))
    s = json.loads(out.read_text())
    assert s["match_key"] == "2026cur_qm22" and s["meta"] == "arm.meta.json"
    assert s["misid"] == pytest.approx(0.25) and s["labelled_slot_s"] == 20.0
    assert s["label_coverage"] == pytest.approx(20 / 30)
