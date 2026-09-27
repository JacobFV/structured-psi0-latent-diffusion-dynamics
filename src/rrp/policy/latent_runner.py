"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.controllers.latent_runner`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.controllers.latent_runner` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn("rrp.policy.latent_runner is deprecated; import rrp.controllers.latent_runner", DeprecationWarning, stacklevel=2)
_sys.modules[__name__] = _importlib.import_module("rrp.controllers.latent_runner")
