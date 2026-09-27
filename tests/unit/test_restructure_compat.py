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
    src = lambda m: (REPO / "src" / Path(*m.split("."))).with_suffix(".py").read_text()
    # only modules whose __main__ parses arguments (argparse): `--help` must not start real work
    mains = [o for o in SHIMS if "run_module" in src(o) and "argparse" in src(SHIMS[o])]
    if not mains:
        pytest.skip("no moved __main__ modules yet")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for old in mains:
        r = subprocess.run([sys.executable, "-m", old, "--help"], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (old, r.stdout[-500:], r.stderr[-2000:])


# ------------------------------------------------------------------ W4 deduplication: provably identical behaviour
def _old_ladder_wilson(k, n, z=1.96):          # verbatim copy of the pre-W4 rrp.evaluation.ladder.wilson
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def test_ladder_wilson_is_bitwise_unchanged():
    from rrp.evaluation.ladder import wilson
    for n in range(0, 61):
        for k in range(0, n + 1):
            assert wilson(k, n) == _old_ladder_wilson(k, n)
            assert wilson(k, n, 1.959964) == _old_ladder_wilson(k, n, 1.959964)
    assert type(wilson(3, 10)) is tuple


def test_seed_spec_single_implementation():
    from rrp.contracts.runs import parse_seed_spec
    import rrp.data.legged_collect as a, rrp.data.legged_latent_collect as b, rrp.evaluation.legged_latent_eval as c
    assert a._seeds is b._seeds is c._seeds is parse_seed_spec
    assert parse_seed_spec("3-6") == [3, 4, 5, 6] and parse_seed_spec("7") == [7] and parse_seed_spec("1,5") == [1, 5]


def test_cached_featurizer_single_implementation():
    from types import SimpleNamespace
    import rrp.features.featurizer as F
    from rrp.evaluation import ladder
    assert ladder._featurizer is F.cached_featurizer
    calls = []
    orig = F.featurizer_for
    F.featurizer_for = lambda s: calls.append(s) or object()
    try:
        s = SimpleNamespace()
        f1 = F.cached_featurizer(s)
        assert F.cached_featurizer(s) is f1 and s._rrp_featurizer is f1 and len(calls) == 1
        s2 = SimpleNamespace(_rrp_featurizer="installed")
        assert F.cached_featurizer(s2) == "installed" and len(calls) == 1
    finally:
        F.featurizer_for = orig


def test_peer_sync_revision_record():
    """scripts/peer_sync.sh push writes this JSON as .rrp_revision on the peer (read by W3 code_provenance)."""
    import json
    import shutil
    if not shutil.which("git") or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(["bash", str(REPO / "scripts" / "peer_sync.sh"), "revision"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert d["git_sha"] == head and isinstance(d["dirty"], bool)
