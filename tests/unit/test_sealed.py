"""H5 (audit D12) + round-2 SL: the sealed split guard (D-138 / D-146) for humanoid_v1, armdiv_v1 and cworld_pointer_v2, and
the sealed-body adapters. Red/green: every refusal has a test that fails without the guard and passes with it."""
from __future__ import annotations

import json
import types

import pytest

from rrp.core.sealed import SPLITS, SealedSplit, SealedSplitError
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
@pytest.mark.parametrize("name", sorted(SPLITS))
def test_split_is_hash_pinned(name, tmp_path):
    assert SealedSplit.load(name).name == name                        # the committed file matches its pin
    p = tmp_path / "s.json"
    p.write_text((rrp_home() / SPLITS[name]["path"]).read_text() + " ")
    with pytest.raises(SealedSplitError) as e:
        SealedSplit.load(name, path=p)
    assert _code(e) == "sealed_split_hash"
    with pytest.raises(SealedSplitError) as e:
        SealedSplit.load(name, path=tmp_path / "absent.json")
    assert _code(e) == "sealed_split_missing"


def test_unknown_split_refused():
    with pytest.raises(SealedSplitError) as e:
        SealedSplit.load("humanoid_v2")
    assert _code(e) == "sealed_split_unknown"


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
        assert cid == "n1|m|-|-|-|s0|evaluation"
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
    monkeypatch.setattr(pl.SealedSplit, "_log_path", lambda self, log: tmp_path / "log.jsonl")
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


# ---------------------------------------------------------------- round 2 (SL): native cell identity
def test_cell_identity_is_native_not_a_method_string(split, tmp_path):
    """Two cells that differ only in task, demo budget or adaptation are two run-once cells; nothing rides in `method`."""
    log = tmp_path / "log.jsonl"
    base = dict(_cell(), task="h_steps", budget=5, adaptation="adapt_bc")
    ids = {split.cell_id(dict(base, **kw)) for kw in ({}, dict(task="h_gap"), dict(budget=20), dict(adaptation="adapt_ppo"),
                                                    dict(budget=None), dict(seed_set="evaluation_b"))}
    assert len(ids) == 6 and split.cell_id(base) == "n1|m|h_steps|n5|adapt_bc|s0|evaluation"
    with split.sealed_eval(base, log):
        pass
    with split.sealed_eval(dict(base, budget=20), log):               # another budget is another cell
        pass
    with pytest.raises(SealedSplitError) as e:                        # the same native cell cannot run twice
        with split.sealed_eval(dict(base), log):
            pass
    assert _code(e) == "sealed_cell_rerun"
    with pytest.raises(SealedSplitError) as e:                        # a field outside the identity would silently merge cells
        split.cell_id(dict(base, note="x"))
    assert _code(e) == "sealed_cell_keys"


# ---------------------------------------------------------------- round 2 (SL): armdiv_v1
ARM_EVAL0 = 2_000_000


@pytest.fixture(scope="module")
def arm() -> SealedSplit:
    return SealedSplit.load("armdiv_v1")


@pytest.mark.parametrize("b", ("gen3_pg2", "rizon4_tf3", "iiwa14_tf3", "pa2s900002_pg2", "pa2s900011_tf3", "xarm7_pg2", "xarm7_tf3",
                               "panda_tf3", "lite6_pg2", "fr3_tf3"))
def test_armdiv_sealed_bodies(arm, b):
    assert arm.is_sealed_body(b)


@pytest.mark.parametrize("b", ("pa2s0_pg2", "pa2s9_tf3", "pa2s899999_pg2", "ur10e_tf3", "vx300s_pg2", "parm6_tf3", "parm5s_tf3", "panda_pg2"))
def test_armdiv_unsealed_bodies(arm, b):
    assert not arm.is_sealed_body(b)


def test_armdiv_source_pool_is_never_sealed(arm):
    pool = json.loads((rrp_home() / "research/splits/armdiv_pool_v1.json").read_text())
    assert not [k for k in pool["train_robots"] if arm.is_sealed_body(k)]
    split = json.loads((rrp_home() / SPLITS["armdiv_v1"]["path"]).read_text())
    assert all(arm.is_sealed_body(r) for t in split["targets"].values() for r in t["robots"])


def test_armdiv_training_guard(arm):
    with pytest.raises(SealedSplitError) as e:
        arm.assert_train_allowed(["gen3_pg2"], range(0, 5))
    assert _code(e) == "sealed_body_in_training"
    arm.assert_train_allowed(["gen3_pg2"], [1_000_003])                  # target-adaptation demos only
    for s in (ARM_EVAL0 + 1, 3_000_001):                                 # evaluation / development scenes never train
        with pytest.raises(SealedSplitError) as e:
            arm.assert_train_allowed(["ur10e_pg2"], [s])
        assert _code(e) == "sealed_eval_seed_in_training"
    arm.assert_train_allowed(["ur10e_pg2"], [0, 1, 2])
    with pytest.raises(SealedSplitError):
        arm.assert_eval_allowed("rizon4_tf3")
    arm.assert_eval_allowed("ur10e_tf3")


def test_armdiv_cells_run_once_in_their_own_log(arm, tmp_path):
    assert SPLITS["armdiv_v1"]["log"] != SPLITS["humanoid_v1"]["log"]
    log = tmp_path / "log.jsonl"
    cell = dict(body="gen3_pg2", method="latent_semfix", task="pick_place", budget=5, adaptation="joint_adapt", train_seed=1,
                scenes=list(range(ARM_EVAL0, ARM_EVAL0 + 100)))
    with arm.sealed_eval(cell, log) as cid:
        assert cid == "gen3_pg2|latent_semfix|pick_place|n5|joint_adapt|s1|evaluation"
    with pytest.raises(SealedSplitError):
        with arm.sealed_eval(cell, log):
            pass
    with pytest.raises(SealedSplitError) as e:                           # a non-sealed body is not a sealed cell
        with arm.sealed_eval(dict(cell, body="ur10e_pg2"), log):
            pass
    assert _code(e) == "sealed_cell_body"


def test_is_sealed_target_is_the_armdiv_split_rule():
    """One rule: the body-layer function every arm guard calls reads the pinned split (no second constant)."""
    from rrp.bodies.armdiv import is_sealed_target
    arm = SealedSplit.load("armdiv_v1")
    pool = json.loads((rrp_home() / "research/splits/armdiv_pool_v1.json").read_text())["train_robots"]
    keys = list(pool) + ["gen3_pg2", "gen3_tf3", "rizon4_pg2", "xarm7_pg2", "panda_tf3", "pa2s900002_pg2", "pa2s3_tf3", "panda_pg2"]
    assert [is_sealed_target(k) for k in keys] == [arm.is_sealed_body(k) for k in keys]
    assert is_sealed_target("xarm7_tf3") and not is_sealed_target("parm5s_tf3")


def test_arm_guards_use_is_sealed_target():
    """Every arm dev guard refuses an armdiv sealed body (before: GRPO anchors named only the legacy xarm7 / panda keys)."""
    from rrp.harness.train.grpo_anchor import AnchorConfig
    for b in ("gen3_pg2", "rizon4_tf3", "pa2s900002_pg2", "xarm7_tf3", "panda_tf3"):
        with pytest.raises(ValueError, match="sealed target bodies cannot be anchors"):
            AnchorConfig(robots=["ur10e_pg2", b])
    AnchorConfig(robots=["ur10e_pg2", "parm6_tf3"])
    from rrp.harness.train import latent_grpo
    assert not hasattr(latent_grpo, "TARGET_BODIES")
    with pytest.raises(ValueError, match="sealed target body"):
        latent_grpo.train_latent_grpo(latent_grpo.LatentGRPORunConfig(checkpoint="x", out_dir="y", robot="rizon4_tf3", allow_target=False))
    from rrp.cli import latent as cli_latent
    assert not hasattr(cli_latent, "TARGET_BODIES")
    for fn in (cli_latent._causal_common, cli_latent.cmd_semantic):
        a = types.SimpleNamespace(robots="gen3_pg2", seed_start=3_000_000)
        with pytest.raises(SystemExit, match="dev rule"):
            fn(a, None, None) if fn is cli_latent._causal_common else fn(a)


# ---------------------------------------------------------------- round 2 (SL): cworld_pointer_v2 (sealed seed lists)
def test_pointer_split_seals_seed_lists_not_bodies(tmp_path):
    ptr = SealedSplit.load("cworld_pointer_v2")
    assert not ptr.is_sealed_body("cw_pointer")
    sealed = ptr.spec["seeds"]["cw/calc_sum"]["sealed_id"]
    assert ptr.seed_kind(sealed[0]) == "evaluation" and ptr.seed_kind(ptr.spec["seeds"]["cw/calc_sum"]["dev"][0]) == "development"
    for s in (sealed[0], ptr.spec["seeds"]["cw/open_type"]["sealed_heldout"][0], ptr.spec["seeds"]["cw/calc_sum"]["dev"][0]):
        with pytest.raises(SealedSplitError) as e:
            ptr.assert_train_allowed(["cw_pointer"], [s])
        assert _code(e) == "sealed_eval_seed_in_training"
    ptr.assert_train_allowed(["cw_pointer"], [5, 399_999])
    log = tmp_path / "log.jsonl"
    cell = dict(body="cw_pointer", method="pointer_latent:abc", task="cw/calc_sum", seed_set="sealed_id", train_seed=0, scenes=sealed)
    with ptr.sealed_eval(cell, log) as cid:
        assert cid == "cw_pointer|pointer_latent:abc|cw/calc_sum|-|-|s0|sealed_id"
    with pytest.raises(SealedSplitError) as e:
        with ptr.sealed_eval(cell, log):
            pass
    assert _code(e) == "sealed_cell_rerun"
    for bad in (dict(cell, seed_set="dev"), dict(cell, task="cw/nope"), dict(cell, scenes=sealed[:-1]),
                dict(cell, seed_set="sealed_heldout", task="cw/drag_window")):     # drag_window has no held-out list
        with pytest.raises(SealedSplitError):
            with ptr.sealed_eval(bad, log):
                pass


# ---------------------------------------------------------------- round 2 (SL): rrp suite sealed-log
def test_sealed_log_cli_list_and_infra_failure(split, tmp_path, capsys):
    from rrp.cli.tools import TOOLS
    from rrp.core import sealed
    assert TOOLS[("suite", "sealed-log")][0] == "rrp.core.sealed:main"
    log = tmp_path / "log.jsonl"
    cell = _cell(body="n1", method="m", seed=3)
    with pytest.raises(RuntimeError):
        with split.sealed_eval(cell, log):
            raise RuntimeError("crash")
    cid = split.cell_id(cell)
    assert sealed.main(["list", "--split", "humanoid_v1", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert cid in out and "started" in out
    assert sealed.main(["list", "--split", "humanoid_v1", "--log", str(log), "--state", "done"]) == 0
    assert cid not in capsys.readouterr().out
    assert sealed.main(["infra-failure", "--split", "humanoid_v1", "--cell", cid, "--reason", " ", "--log", str(log)]) == 2
    assert split.cell_state(cid, log) == "started"                       # a blank reason releases nothing
    assert sealed.main(["infra-failure", "--split", "humanoid_v1", "--cell", cid, "--reason", "peer node lost", "--log", str(log)]) == 0
    assert split.cell_state(cid, log) == "infrastructure_failure"
    assert sealed.main(["infra-failure", "--split", "humanoid_v1", "--cell", "never|ran|-|-|-|s0|evaluation", "--reason", "x",
                        "--log", str(log)]) == 2                         # only an open cell can be released
    assert sealed.cell_table(log)[0]["reason"] == "peer node lost"
    with split.sealed_eval(cell, log):
        pass
    assert sealed.main(["list", "--split", "humanoid_v1", "--log", str(log), "--state", "done"]) == 0
    assert "starts=2" in capsys.readouterr().out


def test_tracker_trainer_guard_admits_a_sealed_body_only_on_an_adaptation_seed():
    """D-147 (2026-10-04): the Warp trainer's own guard passes its env seed; before, it passed none and refused even the pre-registered
    Level-1 adaptation (seed 1000000) of n1."""
    from types import SimpleNamespace as NS
    import pytest
    from rrp.core.sealed import SealedSplitError
    from rrp.harness.train.tracker_recipes import assert_trainable
    assert_trainable(NS(body=None, groups='[[["n1"], 1000]]', seed=1000000))
    with pytest.raises(SealedSplitError):
        assert_trainable(NS(body=None, groups='[[["n1"], 1000]]', seed=1))
    assert_trainable(NS(body="t1", groups=None, seed=1))
