"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.orchestration.discovery`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.orchestration.discovery` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn("rrp.ops.discovery is deprecated; import rrp.orchestration.discovery", DeprecationWarning, stacklevel=2)
_sys.modules[__name__] = _importlib.import_module("rrp.orchestration.discovery")
