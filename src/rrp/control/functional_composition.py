"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.teachers.functional_composition`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.teachers.functional_composition` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

if __name__ == "__main__":        # `python -m rrp.control.functional_composition` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("rrp.control.functional_composition is deprecated; use python -m rrp.teachers.functional_composition", DeprecationWarning, stacklevel=1)
    _runpy.run_module("rrp.teachers.functional_composition", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("rrp.control.functional_composition is deprecated; import rrp.teachers.functional_composition", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module("rrp.teachers.functional_composition")
