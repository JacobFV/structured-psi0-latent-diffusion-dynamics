"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.features.legged`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.features.legged` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn("rrp.control.legged_latent is deprecated; import rrp.features.legged", DeprecationWarning, stacklevel=2)
_sys.modules[__name__] = _importlib.import_module("rrp.features.legged")
