"""Environment lookups that honour pre-rename legacy variable names.

The mask tracker started life as a SAM2 wrapper, so its module-level tuning
constants were read from ``SAM2_*`` variables. SAM3 is the production backend
now and the canonical names are ``SAM3_*``; operator ``.env`` files and pod
profiles written before the rename keep working through this fallback. The
canonical name wins when both are set.
"""
from __future__ import annotations

import os
from typing import Optional

#: Prefix of the pre-rename variable names (``SAM2_TRACK_FPS`` etc.).
LEGACY_PREFIX = "SAM2_"


def env_get(name: str, default: Optional[str] = None,
            legacy: tuple = ()) -> Optional[str]:
    """``os.environ.get(name, default)``, falling back to each ``legacy`` name
    (in order) when ``name`` itself is not set."""
    for key in (name, *legacy):
        if key in os.environ:
            return os.environ[key]
    return default


def sam3_env(suffix: str, default: Optional[str] = None) -> Optional[str]:
    """``SAM3_<suffix>``, else the legacy pre-rename name, else ``default``."""
    return env_get(f"SAM3_{suffix}", default, legacy=(LEGACY_PREFIX + suffix,))
