"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.research.legged_t1_diag`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.research.legged_t1_diag` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

if __name__ == "__main__":        # `python -m rrp.learning.legged_t1_diag` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("rrp.learning.legged_t1_diag is deprecated; use python -m rrp.research.legged_t1_diag", DeprecationWarning, stacklevel=1)
    _runpy.run_module("rrp.research.legged_t1_diag", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("rrp.learning.legged_t1_diag is deprecated; import rrp.research.legged_t1_diag", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module("rrp.research.legged_t1_diag")
