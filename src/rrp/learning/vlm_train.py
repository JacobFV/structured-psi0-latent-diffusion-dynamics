"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.training.vlm_train`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.training.vlm_train` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

if __name__ == "__main__":        # `python -m rrp.learning.vlm_train` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("rrp.learning.vlm_train is deprecated; use python -m rrp.training.vlm_train", DeprecationWarning, stacklevel=1)
    _runpy.run_module("rrp.training.vlm_train", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("rrp.learning.vlm_train is deprecated; import rrp.training.vlm_train", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module("rrp.training.vlm_train")
