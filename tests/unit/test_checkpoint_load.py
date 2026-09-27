"""W4 gate: local checkpoints still load with the restructured code (host only; peer store: run the module there).

Loads a representative set (2 of each kind) of the `*.pt` files under the roots in RRP_CHECKPOINT_ROOTS
(os.pathsep-separated; default: this checkout's artifacts/) through the real loaders and W3's checkpoint_provenance
(rrp.evaluation.checkpoint_audit). Skipped when no checkpoint is present (fresh checkout: weights are not in git).
Full audit: `python -m rrp.evaluation.checkpoint_audit ROOT ...`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rrp.evaluation.checkpoint_audit import audit, find

REPO = Path(__file__).resolve().parents[2]
ROOTS = [Path(p).expanduser() for p in os.environ.get("RRP_CHECKPOINT_ROOTS", str(REPO / "artifacts")).split(os.pathsep) if p]


def test_representative_checkpoints_load():
    if not find(ROOTS):
        pytest.skip(f"no *.pt under {ROOTS} (weights are not in git; set RRP_CHECKPOINT_ROOTS)")
    rep = audit(ROOTS, per_kind=2)
    errors = [(r["path"], r["error"]) for r in rep["rows"] if "error" in r]
    assert not errors, errors
    assert rep["rrp_pickled_classes"] == [], rep["rrp_pickled_classes"]   # state dicts only: import paths irrelevant
