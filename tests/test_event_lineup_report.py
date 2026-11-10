"""scripts/event_lineup_report.py: arm loading, the paired interval, the
spec section 8 criteria and the borrowed-row label count. No network."""
import json
from pathlib import Path

import pytest

import scripts.event_lineup_report as rep


def test_paired_ci_is_deterministic_and_brackets_the_mean():
    diffs = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]
    m, lo, hi = rep.paired_ci(diffs)
    assert lo <= m <= hi and rep.paired_ci(diffs) == (m, lo, hi)
    assert rep.paired_ci([]) is None


def _arm(tmp_path, name, rows, audits=None):
    d = tmp_path / name
    d.mkdir()
    for r in rows:
        r.setdefault("meta_out", str(d / f"{r['match_key']}.meta.json"))
        (d / f"{r['match_key']}.meta.json").write_text(json.dumps({"frames": []}))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for mk, misid in (audits or {}).items():
        a = d / "audit" / f"{mk}_x"
        a.mkdir(parents=True)
        mo = next(r["meta_out"] for r in rows if r["match_key"] == mk)
        (a / "score.json").write_text(json.dumps(
            {"match_key": mk, "misid": misid, "misid_s": misid * 100, "meta": mo}))
    return d


def _row(mk, unl, misid_s=None, wrong=None):
    score = None if misid_s is None else {"full": {"misid_s": misid_s, "wrong": wrong}}
    return {"match_key": mk, "replay_id": "r", "unlabelled_share": unl, "score": score,
            "row_sources": {}}


def test_criteria_pass_fail_and_not_run(tmp_path):
    base = rep.load_arm([_arm(tmp_path, "own",
                              [_row("2026cur_qm8", 0.40, 20.0, 0.03), _row("2026cur_qm11", 0.30)],
                              {"2026cur_qm8": 0.05, "2026cur_qm11": 0.04})])
    good = rep.load_arm([_arm(tmp_path, "full",
                              [_row("2026cur_qm8", 0.33, 24.0, 0.035), _row("2026cur_qm11", 0.24)],
                              {"2026cur_qm8": 0.05, "2026cur_qm11": 0.03})])
    assert all(c["pass"] for c in rep.criteria(base, good, fable_worst=0.05))
    bad = rep.load_arm([_arm(tmp_path, "bad",
                             [_row("2026cur_qm8", 0.39, 30.0, 0.06), _row("2026cur_qm11", 0.29)],
                             {"2026cur_qm8": 0.09, "2026cur_qm11": 0.08})])
    assert not any(c["pass"] for c in rep.criteria(base, bad, fable_worst=0.2))
    assert rep.criteria(base, good, fable_worst=None)[-1]["pass"] is None


def test_qm8_numbers_average_over_repeat_dirs(tmp_path):
    a = _arm(tmp_path, "a", [_row("2026cur_qm8", 0.4, 20.0, 0.02)])
    b = _arm(tmp_path, "b", [_row("2026cur_qm8", 0.4, 30.0, 0.04)])
    arm = rep.load_arm([a, b])
    assert [q["misid_s"] for q in arm["qm8"]] == [20.0, 30.0]
    assert list(arm["unlabelled"]) == ["2026cur_qm8"]            # first dir only


def test_borrowed_row_labels_are_lineup_labels_of_teams_whose_row_borrowed(tmp_path):
    d = tmp_path / "audit"
    d.mkdir()
    (d / "audit.json").write_text(json.dumps({"roster": {"red": ["2637", "2724", "6497"],
                                                         "blue": ["3128", "6773", "9066"]}}))
    samples = [{"sample_id": "s1", "frame_idx": 0, "slot": "B2", "pipeline_label": "6773"},
               {"sample_id": "s2", "frame_idx": 1, "slot": "B2", "pipeline_label": "6773"},
               {"sample_id": "s3", "frame_idx": 0, "slot": "B1", "pipeline_label": "3128"},
               {"sample_id": "s4", "frame_idx": 2, "slot": "B2", "pipeline_label": "6773"}]
    (d / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    answers = d / "fable.jsonl"
    answers.write_text("".join(json.dumps(a) + "\n" for a in [
        {"sample_id": "s1", "team": "6773", "confidence": 0.9},
        {"sample_id": "s2", "team": "9066", "confidence": 0.9},
        {"sample_id": "s3", "team": "9066", "confidence": 0.9},
        {"sample_id": "s4", "team": "6773", "confidence": 0.9}]))
    meta = {"frames": [
        {"robots": [{"slot": "B1", "team_number": "3128", "confirmed": False},
                    {"slot": "B2", "team_number": "6773", "confirmed": False}]},
        {"robots": [{"slot": "B2", "team_number": "6773", "confirmed": False}]},
        {"robots": [{"slot": "B2", "team_number": "6773", "confirmed": True}]}]}
    sources = {"6773": {"own": 0, "borrowed": 2, "elim": 0},
               "3128": {"own": 4, "borrowed": 0, "elim": 0}}
    # s3: 3128's row is all own; s4: a digit label (confirmed), not a lineup one
    assert rep.borrowed_label_counts(d, answers, meta, sources) == {"6773": (2, 1)}


QM8 = "2026cur_qm8"


def _qm8(tmp_path, name, misid_s, wrong):
    return rep.load_arm([_arm(tmp_path, name, [_row(QM8, 0.4, misid_s, wrong)])])


def _by(cs, name):
    return next(c for c in cs if c["name"] == name)


def test_qm8_thresholds_are_pinned(tmp_path):
    base = _qm8(tmp_path, "b", 20.0, 0.03)
    for s, ok in ((25.0, True), (25.1, False)):
        cs = rep.criteria(base, _qm8(tmp_path, f"m{s}", s, 0.03), None)
        assert _by(cs, "qm8 misID")["pass"] is ok
    for w, ok in ((0.04, True), (0.041, False)):
        cs = rep.criteria(base, _qm8(tmp_path, f"w{w}", 20.0, w), None)
        assert _by(cs, "qm8 wrong")["pass"] is ok


def _unl(tmp_path, name, shares):
    return rep.load_arm([_arm(tmp_path, name, [_row(m, v) for m, v in shares.items()])])


def test_unlabelled_drop_threshold_survives_float_noise(tmp_path):
    base = _unl(tmp_path, "b", {"qm1": 0.40, "qm2": 0.30})
    for t, ok in ((0.35, True), (0.351, False)):
        arm = _unl(tmp_path, f"t{t}", {"qm1": t, "qm2": t - 0.10})
        assert _by(rep.criteria(base, arm, None), "unlabelled drop")["pass"] is ok
    b3 = _unl(tmp_path, "b3", {"qm1": 0.3})
    assert _by(rep.criteria(b3, _unl(tmp_path, "t3", {"qm1": 0.25}), None),
               "unlabelled drop")["pass"] is True


def test_fable_threshold_and_inconclusive_teams():
    base = {"qm8": [], "unlabelled": {}, "audit": {}, "audit_s": {}}
    assert rep.criteria(base, base, 0.10)[-1]["pass"] is True
    assert rep.criteria(base, base, 0.11)[-1]["pass"] is False
    worst, inc = rep.fable_summary({"1": (4, 4), "2": (10, 1), "3": (5, 0)})
    assert worst == 0.1 and inc == ["1"]
    assert rep.fable_summary({"1": (4, 4)}) == (None, ["1"])


def _audited(tmp_path, name, audits):
    return rep.load_arm([_arm(tmp_path, name, [_row(m, 0.3) for m in audits], audits)])


def test_audit_bootstrap_bound_and_pairing(tmp_path):
    base = _audited(tmp_path, "b", {"qm1": 0.05, "qm2": 0.05})
    same = _audited(tmp_path, "s", {"qm1": 0.05, "qm2": 0.05})
    c = _by(rep.criteria(base, same, None), "audit misID paired")
    assert c["pass"] is True and "0.0" in c["value"]            # lower bound exactly 0
    better = _audited(tmp_path, "g", {"qm1": 0.02, "qm2": 0.03})
    assert _by(rep.criteria(base, better, None), "audit misID paired")["pass"] is True
    worse = _audited(tmp_path, "w", {"qm1": 0.09, "qm2": 0.08})
    assert _by(rep.criteria(base, worse, None), "audit misID paired")["pass"] is False
    only = _audited(tmp_path, "o", {"qm1": 0.05})                # qm2 in one arm only
    c = _by(rep.criteria(base, only, None), "audit misID paired")
    assert c["pass"] is None and "1/2" in c["value"] and c["info"]


def test_load_arm_rejects_bad_audit_inputs(tmp_path):
    d = _arm(tmp_path, "a", [_row("qm1", 0.3)], {"qm1": 0.05})
    dup = d / "audit" / "qm1_y"
    dup.mkdir()
    (dup / "score.json").write_text((d / "audit" / "qm1_x" / "score.json").read_text())
    with pytest.raises(SystemExit, match="duplicate"):
        rep.load_arm([d])
    d2 = _arm(tmp_path, "b", [_row("qm1", 0.3)], {"qm1": 0.05})
    sc = d2 / "audit" / "qm1_x" / "score.json"
    sc.write_text(json.dumps({**json.loads(sc.read_text()), "meta": "elsewhere.json"}))
    with pytest.raises(SystemExit, match="not this arm"):
        rep.load_arm([d2])


def _fable_fixture(tmp_path, audit_meta=None):
    meta = {"frames": [{"robots": [{"slot": "B2", "team_number": "6773", "confirmed": False}]}]}
    row = {**_row("2026cur_qm11", 0.2), "row_sources": {"6773": {"own": 0, "borrowed": 2, "elim": 0}}}
    d = _arm(tmp_path, "full", [row], {"2026cur_qm11": 0.03})
    mo = Path(row["meta_out"])
    mo.write_text(json.dumps(meta))
    a = d / "audit" / "2026cur_qm11_x"
    (a / "audit.json").write_text(json.dumps({
        "match_key": "2026cur_qm11", "meta": audit_meta or str(mo),
        "roster": {"red": ["1", "2", "3"], "blue": ["3128", "6773", "9066"]}}))
    (a / "samples.jsonl").write_text(json.dumps(
        {"sample_id": "s1", "frame_idx": 0, "slot": "B2", "pipeline_label": "6773"}) + "\n")
    (a / "ans.jsonl").write_text(json.dumps(
        {"sample_id": "s1", "team": "6773", "confidence": 0.9}) + "\n")
    base = _arm(tmp_path, "own", [_row("2026cur_qm11", 0.3)], {"2026cur_qm11": 0.05})
    return base, d, a


def test_main_smoke_with_a_fable_pair(tmp_path, capsys):
    base, d, a = _fable_fixture(tmp_path)
    rep.main(["--arm", f"own={base}", "--arm", f"event-full={d}",
              "--fable", f"{a}={a / 'ans.jsonl'}"])
    out = capsys.readouterr().out
    assert "verdict:" in out and "inconclusive" in out and "6773" in out


def test_main_fable_guards(tmp_path):
    base, d, a = _fable_fixture(tmp_path, audit_meta="elsewhere.json")
    with pytest.raises(SystemExit, match="not the event-full"):
        rep.main(["--arm", f"own={base}", "--arm", f"event-full={d}",
                  "--fable", f"{a}={a / 'ans.jsonl'}"])
    (a / "audit.json").write_text(json.dumps({"match_key": "2026cur_qm99", "meta": None}))
    with pytest.raises(SystemExit, match="not in the event-full"):
        rep.main(["--arm", f"own={base}", "--arm", f"event-full={d}",
                  "--fable", f"{a}={a / 'ans.jsonl'}"])
