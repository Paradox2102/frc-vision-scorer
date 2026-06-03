import numpy as np
import pytest

from web.field.appearance_gallery import AppearanceGallery, classify_match


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_then_query_returns_team():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, legibility=0.9)
    ranked = g.query(e, "blue")
    assert ranked[0][0] == "1787"
    assert ranked[0][1] == pytest.approx(1.0, abs=1e-5)


def test_query_filters_by_color():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, legibility=0.9)
    assert g.query(e, "red") == []          # no red exemplars
    assert g.query(e, None)[0][0] == "1787"  # unknown color searches both


def test_dedup_rejects_near_duplicate_keeps_diverse():
    g = AppearanceGallery(dedup_sim=0.92, max_exemplars=8)
    base = _unit(1, 0, 0)
    near = _unit(1, 0.05, 0)        # cosine > 0.92 to base
    diverse = _unit(0, 1, 0)        # orthogonal -> kept
    g.add("254", "red", base, 0.9)
    g.add("254", "red", near, 0.9)      # rejected (too similar)
    g.add("254", "red", diverse, 0.9)   # kept (covers another viewpoint)
    assert g.exemplar_count("254", "red") == 2


def test_cap_evicts_lowest_legibility():
    g = AppearanceGallery(dedup_sim=0.999, max_exemplars=2)
    # all sufficiently distinct so dedup never triggers
    g.add("9", "red", _unit(1, 0, 0), legibility=0.3)
    g.add("9", "red", _unit(0, 1, 0), legibility=0.9)
    g.add("9", "red", _unit(0, 0, 1), legibility=0.8)  # full -> evict the 0.3
    assert g.exemplar_count("9", "red") == 2
    legs = sorted(g.legibilities("9", "red"))
    assert legs == [0.8, 0.9]


def test_query_returns_best_per_team_sorted():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    g.add("254", "blue", _unit(0, 1, 0), 0.9)
    q = _unit(0.9, 0.2, 0)  # closest to 1787
    ranked = g.query(q, "blue")
    assert [r[0] for r in ranked] == ["1787", "254"]
    assert ranked[0][1] > ranked[1][1]


def test_classify_match_confirm_supplement_none():
    # ranked = list[(team, sim)] sorted desc; second-best means next DISTINCT team
    assert classify_match([("1787", 0.90), ("254", 0.70)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("confirm", "1787", 0.90)
    # high sim but 2nd team too close -> supplement, not confirm
    assert classify_match([("1787", 0.90), ("254", 0.87)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80)[0] == "supplement"
    # moderate sim -> supplement
    assert classify_match([("1787", 0.82)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80)[0] == "supplement"
    # weak -> none
    assert classify_match([("1787", 0.50)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("none", None, 0.50)
    assert classify_match([], confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("none", None, 0.0)


def test_serialize_roundtrip():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    rows = g.serialize()
    assert rows and set(rows[0]) >= {"team", "alliance_color", "embedding", "legibility"}
    g2 = AppearanceGallery.from_rows(rows)
    assert g2.query(_unit(1, 0, 0), "blue")[0][0] == "1787"


def test_from_rows_recaps_oversized_bucket_keeping_highest_legibility():
    # The DB is append-only across matches, so a (team,color) bucket can arrive with
    # more rows than max_exemplars; from_rows must keep only the best `max_exemplars`.
    rows = [{"team": "1787", "alliance_color": "blue",
             "embedding": _unit(1, i * 0.3, 0), "legibility": leg}
            for i, leg in enumerate([0.1, 0.5, 0.9, 0.7, 0.3])]
    g = AppearanceGallery.from_rows(rows, max_exemplars=2)
    assert g.exemplar_count("1787", "blue") == 2
    assert sorted(g.legibilities("1787", "blue")) == [0.7, 0.9]


def test_query_returns_verified_flag_and_sorts_verified_first_on_tie():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("254", "red", e, 0.9, verified=False)
    g.add("1787", "red", e, 0.9, verified=True)  # identical sim -> verified wins tie
    ranked = g.query(e, "red")
    assert ranked[0][0] == "1787"      # verified team first
    assert ranked[0][2] is True        # 3-tuple: (team, sim, verified)
    assert len(ranked[0]) == 3


def test_verified_roundtrips_through_serialize_from_rows():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9, verified=True)
    rows = g.serialize()
    assert rows[0]["verified"] is True
    g2 = AppearanceGallery.from_rows(rows)
    assert g2.query(_unit(1, 0, 0), "blue")[0][2] is True
