"""W4 restructure: every old module path keeps working (same module object) and on-disk pickle paths are unchanged."""
from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_layering import SHIMS

REPO = Path(__file__).resolve().parents[2]


def _import_or_skip(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:          # optional third-party dependency (e.g. transformers) absent
        if e.name and not e.name.startswith("rrp"):
            pytest.skip(f"{name}: optional dependency {e.name} missing")
        raise


@pytest.mark.parametrize("old", sorted(SHIMS))
def test_old_path_is_the_new_module(old):
    new = SHIMS[old]
    mod_new = _import_or_skip(new)
    sys.modules.pop(old, None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod_old = importlib.import_module(old)
    assert mod_old is mod_new
    assert any(issubclass(x.category, DeprecationWarning) and old in str(x.message) for x in w)
    parent, _, leaf = old.rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is mod_new


def test_policy_input_pickles_under_the_old_path():
    from rrp.features.featurizer import PolicyInput
    pi = PolicyInput(tokens={"morph": np.zeros((1, 2), np.float32)}, token_kind={"morph": np.zeros(1, int)},
                     act_node_feats=np.zeros((1, 3)), act_node_morph_index=np.zeros(1, int),
                     relations=np.zeros((0, 5), int), pointers=np.zeros((0, 4), int), pointer_text={},
                     q0=np.zeros(1), meta={"k": 1})
    b = pickle.dumps(pi, protocol=5)
    assert b"rrp.data.features" in b and b"rrp.features.featurizer" not in b   # unchanged dataset format
    back = pickle.loads(b)
    assert type(back) is PolicyInput and back.meta == {"k": 1}


def test_python_dash_m_old_path_forwards():
    """`python -m <old module>` still runs the moved module's __main__ (shim forwards via runpy)."""
    import os
    mains = [o for o in SHIMS if "run_module" in (REPO / "src" / Path(*o.split("."))).with_suffix(".py").read_text()]
    if not mains:
        pytest.skip("no moved __main__ modules yet")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for old in mains:
        r = subprocess.run([sys.executable, "-m", old, "--help"], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (old, r.stdout[-500:], r.stderr[-2000:])
