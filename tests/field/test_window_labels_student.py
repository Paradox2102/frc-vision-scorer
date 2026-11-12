"""window_labels_reader: the S1 student digit reader in the window-label stage
(teacher-distillation spec section 14). Fakes only: a fake bumper detector and
S1, a fake Jev session, blank frames. No GPU, no network, no video."""
import json

import numpy as np
import pytest

from web.field import window_crops as wc
from web.field import window_labels as wl
from web.field import window_student as ws
from tests.field.test_window_labels import FakeSession, _Cal, _frames, _params, _source, _still

B1_XY, B2_XY = (3.0, 2.0), (9.0, 6.0)
# bumper box size per slot: the fake S1 tells the robots apart by crop width
WIDTH = {"B1": 40, "B2": 80}
TEAM = {40: "3128", 80: "6773"}


def _target(xy):
    u, v, _side = wc.target(_Cal(), xy)
    return u, v


class FakeStudent:
    """Bumpers exactly at each listed slot's target; S1 names the team the
    crop width maps to (``teams``: width -> team or None to abstain)."""

    def __init__(self, slots=("B1", "B2"), teams=None, colour="blue"):
        self.slots, self.teams, self.colour = slots, dict(TEAM if teams is None else teams), colour
        self.detected, self.read_n, self.closed = 0, 0, False
        self.model_key = "s1:fake"

    def detect(self, images):
        self.detected += len(images)
        out = []
        for _img in images:
            dets = []
            for slot in self.slots:
                u, v = _target(B1_XY if slot == "B1" else B2_XY)
                w = WIDTH[slot]
                dets.append(((u - w / 2, v - 12, u + w / 2, v + 12), 0.9, self.colour))
            out.append(dets)
        return out

    def read(self, crops, rosters, colours):
        self.read_n += len(crops)
        out = []
        for c in crops:
            w = min(WIDTH.values(), key=lambda x: abs(x - c.shape[1] / 1.5))   # padded 25%/side
            team = self.teams.get(w)
            probs = {t: 0.0 for t in rosters[0]["red"] + rosters[0]["blue"]}
            probs["other"] = 0.0
            if team:
                probs[team] = 0.99
            out.append({"probs": probs, "team": team, "p": 0.99, "abstain": team is None})
        return out

    def close(self):
        self.closed = True


def _run(frames, session, student=None, **kw):
    return wl.run_window_labels(frames, "video.mp4", _Cal(), session, _params(**kw),
                                frame_source=_source, student=student)


def _two_slots():
    return _frames({"B1": _still(range(20), B1_XY), "B2": _still(range(20), B2_XY)}, 20)


class Counting(FakeSession):
    def __init__(self, **kw):
        super().__init__(team_p={"9066": 0.9}, **kw)


def _team(res, slot):
    return res["labels"][(0, slot)][0]


# --------------------------------------------------------------------------- #
# pure helpers


def test_near_bumpers_keeps_the_nearest_and_those_close_to_it():
    side = 200.0
    dets = [((90, 90, 110, 110), 0.9, "red"),           # at the target, 20 px
            ((95, 80, 135, 110), 0.9, "red"),           # d 0.08: within the band
            ((150, 90, 190, 110), 0.9, "blue"),         # d 0.35: outside the band
            ((0, 0, 40, 40), 0.9, "blue"),              # too far
            ((96, 96, 104, 104), 0.9, "blue"),          # too small
            ((92, 92, 120, 120), 0.3, "blue")]          # below the crop conf
    got = ws.near_bumpers(dets, 100.0, 100.0, side)
    assert [g[1] for g in got] == [(90.0, 90.0, 110.0, 110.0), (95.0, 80.0, 135.0, 110.0)]
    assert ws.near_bumpers([((300, 300, 340, 340), 0.9, "red")], 100, 100, side) == []


def test_pad_box_matches_the_harvester():
    from scripts.teacher.harvest import pad_box
    for box in [(10.2, 20.7, 50.1, 41.9), (0.0, 0.0, 30.0, 30.0), (1900.0, 700.0, 1919.5, 722.0)]:
        assert ws.pad_box(box, 0.25, 1920, 723) == pad_box(box, 0.25, 1920, 723)


def test_sample_call_names_only_an_unopposed_s1_read():
    roster = ["3128", "6773"]
    named = {"team": "3128", "probs": {"3128": 0.97, "6773": 0.03, "other": 0.0}}
    other = {"team": "6773", "probs": {"3128": 0.01, "6773": 0.99, "other": 0.0}}
    abstain = {"team": None, "probs": {"3128": 0.9, "6773": 0.1, "other": 0.0}}
    assert ws.sample_call([], roster) is None
    assert ws.sample_call([named, abstain], roster) == {"3128": 0.97, "6773": 0.03, "other": 0.0}
    assert ws.sample_call([abstain, named], roster)["3128"] == 0.97
    assert ws.sample_call([abstain], roster) == ws.ABSTAIN       # S1's own abstain is kept
    assert ws.sample_call([named, other], roster) == ws.ABSTAIN  # two robots disagree
    assert wl.digit_decision([ws.ABSTAIN] * 4, roster) is None


# --------------------------------------------------------------------------- #
# the stage


def test_default_reader_is_jev_and_never_touches_the_student():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("student used in jev mode")

    assert wl.Params(fps=1.0, red=[], blue=[]).reader == "jev"
    s = Counting()
    res = _run(_two_slots(), s, lineup_min_p=1.01)
    res2 = _run(_two_slots(), Counting(), lineup_min_p=1.01, student=Boom())
    assert "student" not in res["stats"] and "reader" not in res["digit_calls"]
    assert res["stats"] == res2["stats"] and s.asked == [("team", 18)]


def test_student_mode_asks_jev_only_for_windows_s1_leaves_undecided():
    stu = FakeStudent(teams={40: "3128", 80: None})          # S1 reads B1, abstains on B2
    s = Counting()
    res = _run(_two_slots(), s, lineup_min_p=1.01, reader="student", student=stu)
    assert res["aborted"] is None
    assert _team(res, "B1") == "3128" and _team(res, "B2") == "9066"
    assert s.asked == [("team", 9)]                          # B2's window only
    st = res["stats"]["student"]
    assert st["windows_s1"] == 1 and st["windows_jev"] == 1 and st["jev_digit_calls"] == 9
    assert st["samples"] == 8 and st["bumper_found"] == 8 and st["named_samples"] == 4
    dc = res["digit_calls"]
    assert dc["reader"] == "student" and dc["sources"] == {0: "s1", 1: "jev"}
    assert list(dc["fingerprints"]) == [1]                   # S1 calls are never cached as Jev
    assert dc["calls"][0][0]["3128"] == 0.99 and dc["wide_p"][0][1]["3128"] == 0.99
    assert stu.read_n == 8 and not stu.closed               # a passed-in reader is the caller's


def test_student_only_never_asks_jev_in_the_digit_pass():
    stu = FakeStudent(teams={40: "3128", 80: None})
    s = Counting()
    res = _run(_two_slots(), s, lineup_min_p=1.01, reader="student_only", student=stu)
    assert s.asked == [] and res["stats"]["requests"] == 0
    assert _team(res, "B1") == "3128" and (0, "B2") not in res["labels"]
    assert res["stats"]["student"]["windows_jev"] == 0
    assert res["digit_calls"]["sources"] == {0: "s1", 1: "s1"}


def test_no_bumper_near_the_target_is_no_call():
    stu = FakeStudent(slots=("B1",))                         # nothing detected at B2
    res = _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student_only", student=stu)
    st = res["stats"]["student"]
    assert st["bumper_found"] == 4 and st["samples"] == 8
    assert res["digit_calls"]["calls"][1] == [None] * 4


def test_digit_cache_is_keyed_by_reader_mode():
    jev = _run(_two_slots(), Counting(), lineup_min_p=1.01)
    cache = json.loads(json.dumps(jev["digit_calls"]))
    # a jev cache is not used by student mode: B2's window is asked again
    s = Counting()
    stu_res = _run(_two_slots(), s, lineup_min_p=1.01, reader="student", digit_calls=cache,
                   student=FakeStudent(teams={40: "3128", 80: None}))
    assert s.asked == [("team", 9)]
    # a student cache is reused by student mode (its Jev windows) ...
    scache = json.loads(json.dumps(stu_res["digit_calls"]))
    s2 = Counting()
    again = _run(_two_slots(), s2, lineup_min_p=1.01, reader="student", digit_calls=scache,
                 student=FakeStudent(teams={40: "3128", 80: None}))
    assert s2.asked == [] and _team(again, "B2") == "9066"
    # ... but neither by jev mode nor by a student run whose S1 now abstains on B1
    s3 = Counting()
    _run(_two_slots(), s3, lineup_min_p=1.01, digit_calls=scache)
    assert s3.asked == [("team", 18)]
    s4 = Counting()
    _run(_two_slots(), s4, lineup_min_p=1.01, reader="student", digit_calls=scache,
         student=FakeStudent(teams={40: None, 80: None}))
    assert s4.asked == [("team", 9)]                         # B1 fresh, B2 from the cache


@pytest.mark.parametrize("student", [lambda: (_ for _ in ()).throw(FileNotFoundError("no ckpt")),
                                     "missing"])
def test_missing_models_fall_back_to_jev(student, monkeypatch, capsys):
    if student == "missing":                     # the default loader fails (no CUDA / files)
        class NoGpu:
            def __init__(self, *a, **k):
                raise RuntimeError("CUDA unavailable")
        monkeypatch.setattr(ws, "StudentReader", NoGpu)
        student = None
    s = Counting()
    res = _run(_two_slots(), s, lineup_min_p=1.01, reader="student", student=student)
    base = _run(_two_slots(), Counting(), lineup_min_p=1.01)
    assert res["aborted"] is None and s.asked == [("team", 18)]
    assert res["labels"] == base["labels"]
    assert res["stats"]["student"]["fallback"] and "reader" not in res["digit_calls"]
    assert "student reader unavailable" in capsys.readouterr().out


def test_an_owned_reader_is_closed_even_when_the_stage_aborts():
    made = []

    def factory():
        made.append(FakeStudent())
        return made[-1]

    ok = _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student_only", student=factory)
    assert ok["aborted"] is None and made[-1].closed
    bad = _run(_two_slots(), Counting(available=False), reader="student", student=factory)
    assert bad["aborted"] and made[-1].closed


def test_detector_failure_aborts_the_stage_cleanly():
    class Broken(FakeStudent):
        def detect(self, images):
            return []

    res = _run(_two_slots(), Counting(), reader="student", student=Broken())
    assert res["aborted"].startswith("bumper detector returned")
    assert res["labels"] is None


def test_detection_runs_on_one_field_when_deinterlacing(monkeypatch):
    seen = []
    real = ws.field_frame
    monkeypatch.setattr(ws, "field_frame", lambda img: seen.append(img.shape) or real(img))
    _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student_only",
         student=FakeStudent(), deinterlace=True)
    assert seen and all(sh == (723, 1920, 3) for sh in seen)
    seen.clear()
    _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student_only",
         student=FakeStudent())
    assert seen == []


def test_field_frame_keeps_the_size_and_drops_the_odd_rows():
    img = np.zeros((6, 4, 3), np.uint8)
    img[1::2] = 255
    out = ws.field_frame(img)
    assert out.shape == img.shape and out.max() == 0


# --------------------------------------------------------------------------- #
# S1 sample colours (s1_colours) and the colour-gated elimination


class NearestFirst(FakeStudent):
    """B1: a red bumper 6 px right of the target listed before a blue one
    on it (nearest = blue); B2: nothing detected."""

    def detect(self, images):
        self.detected += len(images)
        u, v = _target(B1_XY)
        return [[((u + 6 - 20, v - 12, u + 6 + 20, v + 12), 0.9, "red"),
                 ((u - 20, v - 12, u + 20, v + 12), 0.9, "blue")] for _img in images]


def test_s1_colours_keep_the_nearest_bumper_per_tile_aligned_with_s1_calls():
    res = _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student_only",
               student=NearestFirst())
    dc = res["digit_calls"]
    assert dc["s1_colours"] == {0: ["blue"] * 4, 1: [None] * 4}
    assert {w: len(c) for w, c in dc["s1_colours"].items()} == {
        w: len(c) for w, c in dc["s1_calls"].items()}
    assert dc["s1_calls"][1] == [None] * 4                   # no bumper: no call, no colour


def test_s1_colours_survive_json_and_a_cache_with_or_without_them_is_reused():
    stu = lambda: FakeStudent(teams={40: "3128", 80: None})  # noqa: E731
    first = _run(_two_slots(), Counting(), lineup_min_p=1.01, reader="student", student=stu())
    assert first["digit_calls"]["s1_colours"] == {0: ["blue"] * 4, 1: ["blue"] * 4}
    cache = json.loads(json.dumps(first["digit_calls"]))
    assert cache["s1_colours"] == {"0": ["blue"] * 4, "1": ["blue"] * 4}
    assert set(wl.load_digit_calls(cache)) == {"calls", "wide_p", "fingerprints"}
    old = {k: v for k, v in cache.items() if k != "s1_colours"}     # a pre-colour cache
    for c in (cache, old):
        s = Counting()
        again = _run(_two_slots(), s, lineup_min_p=1.01, reader="student", digit_calls=c,
                     student=stu())
        assert s.asked == [] and again["labels"] == first["labels"]
        assert again["digit_calls"]["s1_colours"] == first["digit_calls"]["s1_colours"]


B3_XY = (6.0, 4.0)


class ThreeBlue(FakeStudent):
    """Bumpers at B1 / B2 / B3 with per-slot widths and colours; S1 names by
    crop width (B1 abstains)."""
    W = {"B1": 40, "B2": 60, "B3": 80}
    XY = {"B1": B1_XY, "B2": B3_XY, "B3": B2_XY}

    def __init__(self, colours=None):
        super().__init__(teams={40: None, 60: "6773", 80: "3128"})
        self.cols = {"B1": "blue", "B2": "blue", "B3": "blue", **(colours or {})}

    def detect(self, images):
        self.detected += len(images)
        dets = []
        for slot, xy in self.XY.items():
            u, v = _target(xy)
            w = self.W[slot]
            dets.append(((u - w / 2, v - 12, u + w / 2, v + 12), 0.9, self.cols[slot]))
        return [list(dets) for _img in images]

    def read(self, crops, rosters, colours):
        self.read_n += len(crops)
        out = []
        for c in crops:
            w = min(self.W.values(), key=lambda x: abs(x - c.shape[1] / 1.5))
            team = self.teams.get(w)
            probs = {t: 0.0 for t in rosters[0]["red"] + rosters[0]["blue"]}
            probs["other"] = 0.0
            if team:
                probs[team] = 0.99
            out.append({"probs": probs, "team": team, "p": 0.99, "abstain": team is None})
        return out


def _three_slots():
    return _frames({s: _still(range(20), xy) for s, xy in ThreeBlue.XY.items()}, 20)


def test_stage_elimination_names_the_third_blue_robot_from_s1_reads():
    kw = dict(lineup_min_p=1.01, reader="student_only")
    off = _run(_three_slots(), Counting(), student=ThreeBlue(), **kw)
    assert (0, "B1") not in off["labels"] and "elim_frames" not in off["stats"]
    assert {_team(off, "B2"), _team(off, "B3")} == {"6773", "3128"}
    on = _run(_three_slots(), Counting(), student=ThreeBlue(), eliminate=True, **kw)
    assert all(on["labels"][(i, "B1")] == ("9066", wl.RANK_ELIM, 0.0, 20) for i in range(20))
    assert {k: on["stats"][k] for k in ("elim_frames", "elim_chains",
                                        "elim_chains_colour_skipped")} == {
        "elim_frames": 20, "elim_chains": 1, "elim_chains_colour_skipped": 0}
    b1 = wl.slot_label_to_id("B1")
    assert [r["source"] for r in on["segments"][b1]] == ["window_elim"]
    # a red bumper on B1's robot: a blue slot following a red robot is never eliminated
    red = _run(_three_slots(), Counting(), student=ThreeBlue({"B1": "red"}), eliminate=True, **kw)
    assert (0, "B1") not in red["labels"] and red["stats"]["elim_chains_colour_skipped"] == 1
