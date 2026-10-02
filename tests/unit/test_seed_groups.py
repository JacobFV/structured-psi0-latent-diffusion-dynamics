"""compute.seeds_per_job: n seeds of one stage in one process (harness/train/lockstep.py) are the independent runs.

Tiny CPU fixture (the relgen trainer tests' two-episode pack). Same seed -> the same init and the same per-step
draws whether a member trains alone or next to others; a failing member does not stop the rest."""
from __future__ import annotations

import pytest
import torch

import rrp.harness.train.latent_train as lt
from rrp.harness.train.lockstep import Outcome, run_members
from tests.unit.test_relgen_trainers import LATENT, POLICY, world  # noqa: F401  (the module-scoped fixture)

STEPS, B = 4, 4


@pytest.fixture(autouse=True)
def _no_eval(monkeypatch):
    monkeypatch.setattr(lt, "evaluate_representation", lambda *a, **k: {})
    monkeypatch.setattr(lt, "evaluate_generated", lambda *a, **k: {})


def _rep_cfg(world, seed):
    return dict(latent=dict(LATENT, encoder_factors=["preset:arm"]), packed_dir=str(world.pack), seed=seed, steps=STEPS,
                batch_size=B, zero_prev_action=True)


def _flow_cfg(rep, world, seed):
    return dict(representation=str(rep), packed_dir=str(world.pack), seed=seed, steps=STEPS, batch_size=B,
                zero_prev_action=True, snapshot_every=2, policy=dict(POLICY, factors=["preset:arm"], name="tinyflow"))


def _model(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _same_model(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _same_model(a[k], b[k])
        elif torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k


@pytest.fixture(scope="module")
def rep(world, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(lt, "evaluate_representation", lambda *a, **k: {})
    d = tmp_path_factory.mktemp("rep0")
    try:
        lt.train_representation(_rep_cfg(world, 1), d)
    finally:
        mp.undo()
    return d / "representation.pt"


@pytest.mark.parametrize("mode", ["lockstep", "threads"])
def test_flow_group_is_the_independent_runs(world, rep, tmp_path, mode):
    seeds = (5, 6, 7)
    for s in seeds:
        lt.train_latent_flow(_flow_cfg(rep, world, s), tmp_path / f"solo{s}")
    outs, info = lt.train_latent_flow_group([_flow_cfg(rep, world, s) for s in seeds], [tmp_path / f"grp{s}" for s in seeds], mode)
    assert info["mode"] == mode and info["rng_exact"], "the flow loop must not draw from the global RNG"
    assert [o.ok for o in outs] == [True] * 3
    for s, o in zip(seeds, outs):
        assert o.value["steps"] == STEPS
        _same_model(_model(tmp_path / f"solo{s}" / "policy.pt"), _model(tmp_path / f"grp{s}" / "policy.pt"))
        assert (tmp_path / f"grp{s}" / "policy_last.pt").exists()
    # seeds really differ (a group is not one run copied n times)
    a, b = _model(tmp_path / "grp5" / "policy.pt"), _model(tmp_path / "grp6" / "policy.pt")
    assert any(not torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


def test_representation_group_swaps_the_global_rng_per_member(world, tmp_path):
    """Stage A draws posterior noise from the global torch RNG: lockstep must keep each member's stream apart."""
    seeds = (1, 2)
    for s in seeds:
        lt.train_representation(_rep_cfg(world, s), tmp_path / f"solo{s}")
    outs, info = lt.train_representation_group([_rep_cfg(world, s) for s in seeds], [tmp_path / f"grp{s}" for s in seeds])
    assert all(o.ok for o in outs) and info["rng_exact"]
    for s in seeds:
        _same_model(_model(tmp_path / f"solo{s}" / "representation.pt"), _model(tmp_path / f"grp{s}" / "representation.pt"))
    # threads share the one global stream: Stage A is NOT exact there, and the run says so
    outs, info = lt.train_representation_group([_rep_cfg(world, s) for s in seeds], [tmp_path / f"thr{s}" for s in seeds], "threads")
    assert all(o.ok for o in outs) and info["rng_exact"] is False


def test_a_failing_member_stops_alone(world, rep, tmp_path):
    good, bad = _flow_cfg(rep, world, 5), dict(_flow_cfg(rep, world, 6), representation=str(tmp_path / "missing.pt"))
    outs, _ = lt.train_latent_flow_group([good, bad, _flow_cfg(rep, world, 7)], [tmp_path / f"m{i}" for i in range(3)])
    assert [o.ok for o in outs] == [True, False, True]
    assert isinstance(outs[1].value, Exception)
    with pytest.raises(Exception):
        outs[1].unwrap()
    assert (tmp_path / "m0" / "policy.pt").exists() and (tmp_path / "m2" / "policy.pt").exists()


def test_members_may_finish_at_different_steps():
    def member(n):
        for i in range(n):
            yield
        return n
    outs, _ = run_members([member(2), member(5), member(0)])
    assert [o.value for o in outs] == [2, 5, 0] and all(isinstance(o, Outcome) for o in outs)
