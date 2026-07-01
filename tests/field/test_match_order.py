import pytest
from web.field.match_order import match_sort_key, select_epoch


def test_sort_key_orders_quals_numerically():
    k = match_sort_key
    assert k("2026hop_qm2") < k("2026hop_qm10") < k("2026hop_qm120")

def test_sort_key_quals_before_playoffs():
    assert match_sort_key("2026hop_qm120") < match_sort_key("2026hop_sf1m1")
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_f1m1")

def test_sort_key_playoff_set_then_match():
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_sf2m1")
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_sf1m2")

def test_select_epoch_picks_greatest_start_at_or_before():
    versions = [
        {"version_id": "a", "start_match": "2026hop_qm1"},
        {"version_id": "b", "start_match": "2026hop_qm50"},
    ]
    assert select_epoch(versions, "2026hop_qm72")["version_id"] == "b"
    assert select_epoch(versions, "2026hop_qm49")["version_id"] == "a"
    assert select_epoch(versions, "2026hop_qm50")["version_id"] == "b"

def test_select_epoch_legacy_none_covers_from_start():
    versions = [
        {"version_id": "legacy", "start_match": None},
        {"version_id": "b", "start_match": "2026hop_qm50"},
    ]
    assert select_epoch(versions, "2026hop_qm10")["version_id"] == "legacy"
    assert select_epoch(versions, "2026hop_qm80")["version_id"] == "b"

def test_select_epoch_earlier_only_version_still_covers_earlier_match():
    versions = [{"version_id": "b", "start_match": "2026hop_qm50"}]
    assert select_epoch(versions, "2026hop_qm10")["version_id"] == "b"

def test_select_epoch_empty_returns_none():
    assert select_epoch([], "2026hop_qm1") is None

def test_select_epoch_none_match_key_returns_latest_created():
    versions = [
        {"version_id": "a", "start_match": "2026hop_qm1", "created_at": "2026-06-01"},
        {"version_id": "b", "start_match": "2026hop_qm50", "created_at": "2026-06-08"},
    ]
    assert select_epoch(versions, None)["version_id"] == "b"

def test_select_epoch_breaks_start_match_ties_by_newest():
    # Recalibrating with the default start_match=None creates a second None-start
    # version; the NEWER one must win. Otherwise recalibration is silently ignored
    # and an older/broken stub (e.g. a K=None pinhole) gets selected and crashes.
    versions = [
        {"version_id": "20260630-104110", "start_match": None, "created_at": "2026-06-30T10:41:10"},
        {"version_id": "20260630-104112", "start_match": None, "created_at": "2026-06-30T10:41:12"},
    ]
    assert select_epoch(versions, "2026cur_qm1")["version_id"] == "20260630-104112"
    # order-independent: newest wins even if listed first
    assert select_epoch(list(reversed(versions)), "2026cur_qm1")["version_id"] == "20260630-104112"
