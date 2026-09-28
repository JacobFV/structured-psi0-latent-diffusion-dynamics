"""MJX legged FEASIBILITY PROTOTYPE (rrp.envs.mjx_legged, D-126 #15). The module must import without jax; with jax +
mujoco-mjx installed (isolated venv, not an rrp dependency), a short C-vs-MJX parity rollout on go2 stays close."""
import importlib

import numpy as np
import pytest


def test_imports_without_jax_and_labels_itself():
    mod = importlib.import_module("rrp.envs.mjx_legged")
    assert "prototype" in mod.SOURCE_LABEL and mod.PROTOTYPE_VERSION
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="isolated venv"):
            mod._jax()


def test_adaptations_are_explicit():
    mod = importlib.import_module("rrp.envs.mjx_legged")
    with pytest.raises(KeyError, match="unknown MJX adaptation"):
        mod.build_model("pquad4", "v2", adapt=["not_an_adaptation"])
    m, _, b, done = mod.build_model("pquad4", "v2", adapt=["pyramidal_cone"])
    assert done[0]["name"] == "pyramidal_cone" and int(m.opt.cone) == 0
    assert mod.c_rollout(m, b, 2).shape == (3, m.nq)


@pytest.mark.menagerie
def test_go2_parity_short():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")
    from rrp.envs.mjx_legged import parity_rollout
    r = parity_rollout("go2", ticks=5, contact="v2", adapt=["no_self_collision"])   # MJX lacks cylinder-box collisions
    assert np.isfinite(r["max_dqpos"])
    assert r["max_dqpos"] < 0.05, r            # loose: float32 + different solver implementation
    assert [a["name"] for a in r["adaptations"]] == ["no_self_collision"] and r["adaptation_effect_max_dqpos"] < 1e-9
