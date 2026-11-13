"""Compat alias: this module was renamed to :mod:`web.field.mask_tracker`.

``Sam2FieldTracker`` is now ``MaskFieldTracker`` (the backend-neutral base
that ``Sam3FieldTracker`` subclasses). Old imports, scripts and pickles keep
working: importing this module yields the ``mask_tracker`` module object
itself, so every public and private name resolves (and monkeypatching a
constant here patches the one the tracker reads).
"""
import sys

from web.field import mask_tracker as _mask_tracker

sys.modules[__name__] = _mask_tracker
