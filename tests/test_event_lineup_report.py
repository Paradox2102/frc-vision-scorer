"""scripts/event_lineup_report.py: arm loading, the paired interval, the
spec section 8 criteria and the borrowed-row label count. No network."""
import json

import scripts.event_lineup_report as rep


def test_paired_ci_is_deterministic_and_brackets_the_mean():
    diffs = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]
    m, lo, hi = rep.paired_ci(diffs)
    assert lo <= m <= hi and rep.paired_ci(diffs) == (m, lo, hi)
    assert rep.paired_ci([]) is None


def _arm(tmp_path, name, rows, audits=None):
    d = tmp_path / name
    d.mkdir()
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for mk, misid in (audits or {}).items():
        a = d / "audit" / f"{mk}_x"
        a.mkdir(parents=True)
        (a / "score.json").write_text(json.dumps({"match_key": mk, "misid": misid}))
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
