"""Scouting metric registry.

A metric is a ``Metric`` subclass decorated with ``@register``:

    @register
    class MyMetric(Metric):
        key = "my_metric"
        title = "My metric"
        version = 1
        years = None            # None = every season, else a tuple of years
        description = "..."
        default_params = {"threshold": 1.0}

        def compute(self, matches, params) -> dict:
            return {"teams": {"3128": {...}}, "params": params}

``compute`` takes the loader's ``list[MatchData]`` and returns plain JSON. The
``teams`` dict (keyed by team number as a string) is required — the export
envelope and the one-team API filter both read it. Any other top-level keys
(e.g. a ``legend``) pass through unchanged.

Adding a metric = one module under ``web/scouting/metrics/`` + one import line
in ``metrics/__init__.py``.
"""
from __future__ import annotations

import threading
from typing import Optional

_REGISTRY: dict = {}
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


class Metric:
    key: str = ""
    title: str = ""
    version: int = 1
    years: Optional[tuple] = None
    description: str = ""
    default_params: dict = {}

    def applies_to(self, year: int) -> bool:
        return self.years is None or int(year) in self.years

    def compute(self, matches: list, params: dict) -> dict:   # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:
        return {"key": self.key, "title": self.title, "version": self.version,
                "years": list(self.years) if self.years else None,
                "description": self.description, "default_params": dict(self.default_params)}


def register(cls):
    inst = cls()
    if not inst.key:
        raise ValueError(f"{cls.__name__} has no key")
    _REGISTRY[inst.key] = inst
    return cls


def _ensure_loaded() -> None:
    import web.scouting.metrics  # noqa: F401  (registers the built-in metrics)


def get_metric(key: str) -> Optional[Metric]:
    _ensure_loaded()
    return _REGISTRY.get(key)


def all_metrics() -> list:
    _ensure_loaded()
    return list(_REGISTRY.values())


def available_metrics(year: int) -> list:
    return [m for m in all_metrics() if m.applies_to(year)]


def merged_params(metric: Metric, params: Optional[dict]) -> dict:
    out = dict(metric.default_params)
    for k, v in (params or {}).items():
        if k not in out or v is None:
            continue
        if isinstance(out[k], bool):
            out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(out[k], (int, float)):
            out[k] = type(out[k])(v)
        else:
            out[k] = v
    return out


def run_metric(event_key: str, key: str, params: Optional[dict] = None, db_path=None,
               use_cache: bool = True) -> dict:
    """Load the event (cached on the replay set) and run one metric -> envelope."""
    from web.scouting.export import envelope
    from web.scouting.loader import event_season_year, load_event, source_fingerprint

    metric = get_metric(key)
    if metric is None:
        raise KeyError(f"unknown metric {key!r}")
    year = event_season_year(event_key)
    if not metric.applies_to(year):
        raise ValueError(f"metric {key!r} does not apply to {year}")
    p = merged_params(metric, params)
    fp = source_fingerprint(event_key, db_path)
    ck = (event_key, key, metric.version, tuple(sorted(p.items())), fp, str(db_path))
    with _CACHE_LOCK:
        if use_cache and ck in _CACHE:
            return _CACHE[ck]
    matches, skipped = _load_cached(event_key, fp, db_path, load_event)
    result = metric.compute(matches, p)
    env = envelope(metric, event_key, year, matches, skipped, result, p)
    with _CACHE_LOCK:
        _CACHE[ck] = env
        # keep the cache small: drop entries for stale fingerprints of this event
        for k in [k for k in _CACHE if k[0] == event_key and k[4] != fp]:
            _CACHE.pop(k, None)
    return env


_LOAD_CACHE: dict = {}


def _load_cached(event_key, fp, db_path, load_event) -> tuple:
    k = (event_key, fp, str(db_path))
    with _CACHE_LOCK:
        hit = _LOAD_CACHE.get(k)
    if hit is not None:
        return hit
    out = load_event(event_key, db_path)
    with _CACHE_LOCK:
        for old in [o for o in _LOAD_CACHE if o[0] == event_key]:
            _LOAD_CACHE.pop(old, None)
        _LOAD_CACHE[k] = out
    return out
