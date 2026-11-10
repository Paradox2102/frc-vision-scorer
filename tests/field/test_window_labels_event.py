"""Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md) in
the window-label stage: knob-off golden snapshot, store_anchors, borrowed
rows, elimination anchors, cards, anchors out, and the digit-call cache.
Fakes only: no network, no video, no GPU."""
import hashlib
import json
from pathlib import Path

from web.field import window_labels as wl
from tests.field.test_window_labels import (
    FakeSession, _frames, _lineup_frames, _lineup_session, _run, _still)

GOLDEN = Path(__file__).parent / "data" / "window_labels_golden.json"


def _snap(frames, session, **kw):
    """What the knob-off guarantee covers (labels, segments, stats, card
    JPEG) plus every lineup payload, as JSON-ready values."""
    sent = []
    real_ask = session.ask

    def ask(items, question):
        if question == "who":
            sent.extend(json.dumps(p, sort_keys=True) for p, _sizes in items)
        return real_ask(items, question)

    session.ask = ask
    res = _run(frames, session, **kw)
    labels = sorted([i, slot] + (list(e) if e else [None])
                    for (i, slot), e in res["labels"].items())
    return {"labels": labels,
            "segments": {str(k): v for k, v in res["segments"].items()},
            "stats": res["stats"],
            "card_sha256": (hashlib.sha256(res["lineup_jpeg"]).hexdigest()
                            if res["lineup_jpeg"] else None),
            "who_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in sent]}


def _golden(**kw):
    lineup = _lineup_frames(second_unread=True)
    digits = _frames({"B1": _still(range(20)), "B2": _still(range(20), (9.0, 6.0))}, 20)
    return json.loads(json.dumps({
        "lineup": _snap(lineup, _lineup_session(lineup, who_p={"A": 0.1, "B": 0.85,
                                                               "unsure": 0.05}),
                        seed=1, **kw),
        "digits_only": _snap(digits, FakeSession(team_p={"3128": 0.9, "unreadable": 0.1}),
                             **kw)}))


def test_knob_off_matches_the_golden_snapshot():
    assert _golden() == json.loads(GOLDEN.read_text())
