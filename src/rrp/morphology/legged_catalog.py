"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.evaluation.legged_catalog`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.evaluation.legged_catalog` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

if __name__ == "__main__":        # `python -m rrp.morphology.legged_catalog` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("rrp.morphology.legged_catalog is deprecated; use python -m rrp.evaluation.legged_catalog", DeprecationWarning, stacklevel=1)
    _runpy.run_module("rrp.evaluation.legged_catalog", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("rrp.morphology.legged_catalog is deprecated; import rrp.evaluation.legged_catalog", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module("rrp.evaluation.legged_catalog")
