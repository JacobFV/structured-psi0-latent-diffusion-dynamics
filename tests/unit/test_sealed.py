"""H5 (audit D12): the sealed split guard (D-138 / D-146) and the sealed-body adapters. Red/green: every refusal has a
test that fails without the guard and passes with it."""
from __future__ import annotations

import json
import types

import pytest

from rrp.core.sealed import SEALED_LOG, SPLIT_PATH, SealedSplit, SealedSplitError
from rrp.core.paths import rrp_home

SEALED = ("g1_hands", "n1", "berkeley", "toddlerbot_2xc", "toddlerbot_2xm", "phum_9000123")
ADAPT = 1_000_005
EVAL0, DEV0 = 2_000_000, 3_000_000


@pytest.fixture(scope="module")
def split() -> SealedSplit:
    return SealedSplit.load()


def _cell(body="n1", method="m", seed=0, start=EVAL0):
    return dict(body=body, method=method, train_seed=seed, scenes=list(range(start, start + 100)))


def _code(exc) -> str:
    return exc.value.code


# ---------------------------------------------------------------- classification
def test_split_is_hash_pinned(tmp_path):
    p = tmp_path / "s.json"
    p.write_text((rrp_home() / SPLIT_PATH).read_text() + " ")
    with pytest.raises(SealedSplitError) as e:
        SealedSplit.load(p)
    assert _code(e) == "sealed_split_hash"
    with pytest.raises(SealedSplitError) as e:
        SealedSplit.load(tmp_path / "absent.json")
    assert _code(e) == "sealed_split_missing"


@pytest.mark.parametrize("b", SEALED)
def test_sealed_bodies(split, b):
    assert split.is_sealed_body(b)


@pytest.mark.parametrize("b", ("go2", "t1", "g1", "apollo", "adam_lite", "phum_17", "phum_8999999"))
def test_unsealed_bodies(split, b):
    assert not split.is_sealed_body(b)


# ---------------------------------------------------------------- refusal 1: training on sealed material
@pytest.mark.parametrize("b", SEALED)
def test_train_on_sealed_body_refused(split, b):
    with pytest.raises(SealedSplitError) as e:
        split.assert_train_allowed([b], range(0, 10))
    assert _code(e) == "sealed_body_in_training"
    with pytest.raises(SealedSplitError):
        split.assert_train_allowed([b], ())          # no seeds declared is not a pass
    split.assert_train_allowed([b], [ADAPT, ADAPT + 1])   # green: adaptation demos only


def test_eval_and_dev_seeds_never_train(split):
    for s in (EVAL0 + 3, DEV0 + 3):
        with pytest.raises(SealedSplitError) as e:
            split.assert_train_allowed(["go2"], [s])
        assert _code(e) == "sealed_eval_seed_in_training"
        with pytest.raises(SealedSplitError):
            split.assert_train_allowed(["n1"], [s])
    split.assert_train_allowed(["go2"], [5, 6, 7])   # legacy seeds of non-sealed bodies are untouched
    split.assert_train_allowed(["go2"], [])


def test_phum_gap_between_train_and_sealed_seeds_refused(split):
    with pytest.raises(SealedSplitError) as e:
        split.assert_train_allowed(["phum_5000000"], [1])
    assert _code(e) == "sealed_phum_seed_range"
    split.assert_train_allowed(["phum_999"], [1])


def test_dataset_guard_reads_shard_manifests(split, tmp_path):
    def shard(body, name, seeds):
        d = tmp_path / body
        d.mkdir(exist_ok=True)
        (d / name).write_text(json.dumps(dict(episodes=[dict(seed=s) for s in seeds])))
    shard("n1", "s0.json", [ADAPT, ADAPT + 1])
    split.assert_dataset_allowed(tmp_path, ["n1"])
    shard("n1", "s1.json", [5])                      # a source-range demo on a sealed body
    with pytest.raises(SealedSplitError):
        split.assert_dataset_allowed(tmp_path, ["n1"])
    shard("go2", "s0.json", [DEV0 + 1])              # a development seed in a non-sealed body's data
    with pytest.raises(SealedSplitError):
        split.assert_dataset_allowed(tmp_path, ["go2"])


def test_pipeline_stages_call_the_guard(tmp_path):
    from rrp.harness.pipelines import legged as pl
    with pytest.raises(SealedSplitError):
        pl.sealed_train_guard(["berkeley"], range(0, 4), "collect")
    pl.sealed_train_guard(["berkeley"], range(ADAPT, ADAPT + 4), "collect")
    d = tmp_path / "toddlerbot_2xc"
    d.mkdir()
    (d / "s0.json").write_text(json.dumps(dict(episodes=[dict(seed=EVAL0)])))
    with pytest.raises(SealedSplitError):
        pl.sealed_data_guard(tmp_path, ["toddlerbot_2xc"])


def test_tracker_training_bodies_include_recipe_groups_and_are_refused_when_sealed(tmp_path):
    from rrp.harness.pipelines import legged as pl
    assert "n1" in pl._tracker_bodies(dict(body="n1"))
    assert not any(SealedSplit.load().is_sealed_body(b) for b in pl._tracker_bodies(dict(engine="warp", recipe="shared_morph_v1")))
    r = tmp_path / "r.json"
    r.write_text(json.dumps(dict(groups=[[["go2", "berkeley"], 64]])))
    assert "berkeley" in pl._tracker_bodies(dict(recipe=str(r)))
    with pytest.raises(SealedSplitError):
        pl.sealed_train_guard(pl._tracker_bodies(dict(recipe=str(r))), (), "tracker training")


def test_trainers_call_the_guard():
    import inspect
    from rrp.harness.train import legged_bc, legged_dagger
    assert "SealedSplit" in inspect.getsource(legged_bc.train)
    assert "SealedSplit" in inspect.getsource(legged_dagger.collect)
    assert "SealedSplit" in inspect.getsource(legged_dagger.refit)


# ---------------------------------------------------------------- refusal 2: run once
def test_sealed_cell_runs_once(split, tmp_path):
    log = tmp_path / "log.jsonl"
    cell = _cell()
    with split.sealed_eval(cell, log) as cid:
        assert cid == "n1|m|s0"
        assert split.cell_state(cid, log) == "started"
    assert split.cell_state(cid, log) == "done"
    with pytest.raises(SealedSplitError) as e:
        with split.sealed_eval(cell, log):
            raise AssertionError("body must not run on a rerun")
    assert _code(e) == "sealed_cell_rerun"
    with split.sealed_eval(_cell(seed=1), log):      # another training seed is another cell
        pass
    assert [json.loads(x)["event"] for x in log.read_text().splitlines()] == ["start", "done", "start", "done"]


def test_exception_leaves_cell_open_and_only_infrastructure_failure_releases_it(split, tmp_path):
    log = tmp_path / "log.jsonl"
    cell = _cell(body="berkeley")
    with pytest.raises(RuntimeError):
        with split.sealed_eval(cell, log):
            raise RuntimeError("bad result")
    with pytest.raises(SealedSplitError) as e:       # a crash is not a free retry
        with split.sealed_eval(cell, log):
            pass
    assert _code(e) == "sealed_cell_rerun"
    with pytest.raises(SealedSplitError) as e:
        split.record_infrastructure_failure(cell, "  ", log)
    assert _code(e) == "sealed_cell_reason"
    split.record_infrastructure_failure(cell, "peer node lost", log)
    with split.sealed_eval(cell, log):               # released: exactly one more attempt
        pass
    assert split.cell_state(split.cell_id(cell), log) == "done"
    with pytest.raises(SealedSplitError):
        split.record_infrastructure_failure(cell, "after done", log)
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "infrastructure_failure", "start", "done"]
    assert rows[1]["reason"] == "peer node lost"


def test_check_cell(split, tmp_path):
    log = tmp_path / "log.jsonl"
    bad = [_cell(body="go2"),                                         # not sealed
           dict(_cell(), scenes=list(range(EVAL0, EVAL0 + 99))),      # 99 scenes
           dict(_cell(), scenes=[EVAL0] * 100),                       # duplicates
           _cell(start=DEV0)]                                         # dev seeds are not evaluation scenes
    for c in bad:
        with pytest.raises(SealedSplitError):
            with split.sealed_eval(c, log):
                pass
    assert not log.exists()


def test_sealed_body_evaluation_refused_outside_sealed_eval(split):
    with pytest.raises(SealedSplitError) as e:
        split.assert_eval_allowed("toddlerbot_2xm")
    assert _code(e) == "sealed_body_eval"
    split.assert_eval_allowed("go2")


def test_sealed_evaluation_decorator(tmp_path, monkeypatch):
    from rrp.harness.pipelines import legged as pl
    monkeypatch.setattr(pl.SealedSplit, "_log_path", staticmethod(lambda log: tmp_path / "log.jsonl"))
    calls = []

    @pl.sealed_evaluation(lambda ctx: (ctx.opts["body"], range(EVAL0, EVAL0 + 100)))
    def stage(ctx):
        calls.append(ctx.opts["body"])
        return "ran"

    ctx = lambda **o: types.SimpleNamespace(opts=o)
    assert stage(ctx(body="go2")) == "ran"                       # non-sealed: untouched, not logged
    assert not (tmp_path / "log.jsonl").exists()
    with pytest.raises(pl.StageError):                            # sealed without a declared cell
        stage(ctx(body="n1"))
    sc = dict(method="m", train_seed=0)
    assert stage(ctx(body="n1", sealed_cell=sc)) == "ran"
    with pytest.raises(SealedSplitError):                         # second run of the same cell
        stage(ctx(body="n1", sealed_cell=sc))
    assert calls == ["go2", "n1"]


# ---------------------------------------------------------------- sealed-body adapters: load and stand
@pytest.mark.menagerie
@pytest.mark.parametrize("key", ("berkeley", "toddlerbot_2xc", "toddlerbot_2xm", "n1", "g1_hands"))
def test_sealed_body_adapters_load_and_stand(key):
    """Load-and-stand only (no policy, no sealed evaluation): the adapter builds, maps to the morph slots, and the body
    does not fall in 1 s under the PD servo at its home pose."""
    import mujoco
    import numpy as np
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.morph_obs import MorphSpec
    model, _, meta = standalone_model(legged_body(key), contact="v2")
    b = LeggedBinding(model, meta)
    assert b.n >= 10 and b.nf == 2 and b.pitch_idx() is not None
    assert meta["limits_source"]
    MorphSpec(model, b, meta)
    d = mujoco.MjData(model)
    b.set_default(d)
    mujoco.mj_forward(model, d)
    z0 = float(d.qpos[b.qa + 2])
    for _ in range(int(1.0 / model.opt.timestep)):
        d.ctrl[b.pol_act] = b.q0
        if len(b.held_act):
            d.ctrl[b.held_act] = b.q0_held
        mujoco.mj_step(model, d)
    assert np.isfinite(d.qpos).all()
    assert b.tilt(d) < 0.6 and d.qpos[b.qa + 2] > 0.6 * z0, f"{key} fell: z {z0:.3f} -> {d.qpos[b.qa + 2]:.3f}, tilt {b.tilt(d):.2f}"


@pytest.mark.menagerie
def test_scaling_rule_torque_limits_are_declared_not_manufacturer():
    from rrp.bodies.legged import legged_body, standalone_model
    _, _, meta = standalone_model(legged_body("toddlerbot_2xc"), contact="v2")
    assert meta["limits_source"] == "declared_scaling"
