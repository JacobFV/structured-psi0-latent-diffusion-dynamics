"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `rrp.research.bc_semantic_edits`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `rrp.research.bc_semantic_edits` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

if __name__ == "__main__":        # `python -m rrp.evaluation.bc_semantic_edits` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("rrp.evaluation.bc_semantic_edits is deprecated; use python -m rrp.research.bc_semantic_edits", DeprecationWarning, stacklevel=1)
    _runpy.run_module("rrp.research.bc_semantic_edits", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("rrp.evaluation.bc_semantic_edits is deprecated; import rrp.research.bc_semantic_edits", DeprecationWarning, stacklevel=2)
    _sys.modules[__name__] = _importlib.import_module("rrp.research.bc_semantic_edits")
