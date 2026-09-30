"""H6: the humanoid transfer pipeline (collect -> pack -> train_* -> eval_transfer -> sealed_eval) and the `rrp eval humanoid-transfer`
driver. No simulation: packs are built from fake shards, the tools run against a stubbed evaluate()."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rrp.harness import dag
from rrp.harness.eval import humanoid_eval as HE
from rrp.harness.pipelines import humanoid as HP
from rrp.harness.pipelines.base import _REGISTRY, StageError, _load_families

ROOT = Path(__file__).resolve().parents[2]
HELD_OUT = ("h_steps_carry", "h_gap_cart")                   # evaluation-only recipes (tests/unit/test_humanoid_recipes.py)
RECIPES = sorted(p for p in (ROOT / "recipes/humanoid").glob("transfer_h_*.yaml")      # the `preset:legged` arm; the `_legged_none` control arm is checked in test_humanoid_recipes.py
                 if p.stem.removeprefix("transfer_") not in HELD_OUT and not p.stem.endswith("_legged_none"))
UPDATES = {5: 150, 20: 300, 100: 600}


def _shards(tmp, body, ticks_of, n):
    d = tmp / "data" / body
    d.mkdir(parents=True)
    for s in range(n):
        (d / f"s{s}-{s}.json").write_text(json.dumps(dict(episodes=[dict(seed=s, ticks=ticks_of(s), status="ok")])))
        (d / f"s{s}-{s}.npz").write_bytes(b"x")
    return tmp / "data"


def test_pack_budget_accounting_counts_teacher_ticks(tmp_path):
    data = _shards(tmp_path, "t1", lambda s: 10 + s, 6)
    pk = HP.make_pack(data, tmp_path / "pack", task="h_walk", bodies=["t1"], budgets=[2, 5], updates={2: 150, 5: 300})
    r2, r5 = pk["records"]["h_walk|t1|n2"], pk["records"]["h_walk|t1|n5"]
    assert (r2["demos"], r2["teacher_ticks"], r2["env_samples"], r2["updates"]) == (2, 10 + 11, 0, 150)
    assert r5["teacher_ticks"] == sum(10 + s for s in range(5)) and r5["seeds"] == [0, 1, 2, 3, 4]
    assert r2["seeds"] == r5["seeds"][:2]                                   # nested: the first N demos
    d = tmp_path / "pack/n2/t1"
    assert sorted(p.name for p in d.glob("*.npz")) == ["s0-0.npz", "s1-1.npz"] and (d / "s0-0.npz").is_symlink()
    with pytest.raises(StageError, match="only 6 collected"):
        HP.make_pack(data, tmp_path / "p2", task="h_walk", bodies=["t1"], budgets=[7], updates={7: 1})
    with pytest.raises(StageError, match="matched update count"):
        HP.make_pack(data, tmp_path / "p3", task="h_walk", bodies=["t1"], budgets=[2], updates={5: 1})


def test_shard_with_two_episodes_is_refused(tmp_path):
    d = tmp_path / "t1"
    d.mkdir()
    (d / "s0-1.json").write_text(json.dumps(dict(episodes=[dict(seed=0, ticks=1), dict(seed=1, ticks=1)])))
    with pytest.raises(StageError, match="exactly one"):
        HP.shard_index(tmp_path, "t1")


def test_trainer_acquisition_needs_the_matched_update_count():
    pack = dict(records={"h_walk|t1|n5": dict(task="h_walk", body="t1", budget=5, demos=5, teacher_ticks=50, env_samples=0, updates=150, seeds=[0])})
    assert HP.acquisition_record(pack, "h_walk", "t1", 5, 150)["teacher_ticks"] == 50
    with pytest.raises(StageError, match="matched count"):
        HP.acquisition_record(pack, "h_walk", "t1", 5, 149)
    with pytest.raises(StageError, match="no record"):
        HP.acquisition_record(pack, "h_walk", "t1", 20, 300)


def _row(method, acq, budget=5, level="2"):
    return dict(level=level, task="h_walk", body="n1", budget=budget, method=method, train_seed=0, acquisition=acq)


def test_equal_acquisition_green_and_red():
    a = dict(demos=5, teacher_ticks=50, env_samples=0, updates=150)
    assert HE.check_equal_acquisition([_row("bc_sft", a), _row("semfix_refit", dict(a)), _row("zs", dict(HE.ZERO_ACQ), budget=None)]) == []
    bad = HE.check_equal_acquisition([_row("bc_sft", a), _row("semfix_refit", dict(a, teacher_ticks=51))])
    assert len(bad) == 1 and "unequal acquisition" in bad[0]
    assert "zero-shot cell carries" in HE.check_equal_acquisition([_row("zs", a, budget=None)])[0]


def test_sealed_cell_id_carries_task_budget_and_adaptation():
    from rrp.core.sealed import SealedSplit
    cfg = HE.validate_config(dict(name="x", task="h_walk", bodies=["n1"], train_seeds=[0],
                                  levels={"2": dict(unit="demos", budgets=[5, 20], updates={"5": 150, "20": 300})},
                                  methods=[dict(name=n, level=2, kind="bc", source="bc", trained=True, budgeted=True, policy="legged_bc",
                                                producer="adapt_bc") for n in ("bc_sft", "bc_other")]))
    scenes = list(range(2_000_000, 2_000_100))
    base = HE.expand_cells(cfg)[0]
    ids = {SealedSplit.cell_id(HE.sealed_cell(cfg, dict(base, **kw), scenes))
           for kw in ({}, dict(budget=20), dict(task="h_turn"), dict(method="bc_other"), dict(train_seed=1))}
    assert len(ids) == 5
    assert SealedSplit.cell_id(HE.sealed_cell(cfg, base, scenes)) == "n1|bc_sft|h_walk|n5|adapt_bc|s0|evaluation"


def test_pipeline_family_stages_registered():
    _load_families()
    for st in ("collect", "pack", "train_rep", "train_flow", "train_bc", "adapt_refit", "adapt_flow", "adapt_bc", "adapt_ppo",
               "eval_transfer", "sealed_eval"):
        assert ("humanoid", st) in _REGISTRY, st


def _plan(recipe):
    _load_families()
    return dag.plan_dag(dag.load_dag(recipe))


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_every_humanoid_recipe_plans_and_its_config_paths_match_the_planned_outs(recipe, tmp_path):
    plan = _plan(recipe)
    task = recipe.stem.removeprefix("transfer_")
    assert {n.split(".")[0].split("@")[0] for n in plan.nodes} >= {"collect_src", "collect_tgt", "collect_sealed", "pack", "pack_sealed",
                                                                  "rep", "flow", "bc", "eval", "sealed", "eval_ref", "sealed_ref"}
    assert plan.nodes["sealed@semfix.s0"].rc.options["sealed"] is False          # sealed cells run once: off until flipped
    cfg = HE.validate_config(json.loads(json.dumps(plan.nodes["eval@semfix.s0"].rc.options["transfer"])))
    assert cfg["task"] == task
    for pk in cfg["pack"]:                                                       # the config's pack paths are the planned pack outs
        assert HE.fmt(pk, task=task).rsplit("/", 1)[0] in (plan.nodes["pack"].rc.out, plan.nodes["pack_sealed"].rc.out)
    for sysname, node, want in (("semfix", "flow", "policy.pt"), ("nosem", "flow", "policy.pt"), ("bc", "bc", "policy.pt")):
        m = HE.method_of(cfg, f"{sysname}_zeroshot" if sysname != "bc" else "bc_zeroshot")
        for s in (0, 1):
            p = HE.fmt(m["kw"].get("flow") or m["kw"]["checkpoint"], seed=s, task=task)
            assert p == plan.nodes[f"{node}@{sysname}.s{s}"].rc.out + "/" + want
    # cells whose run does not exist are reported, never skipped or faked
    pack = HE.load_packs(cfg, tmp_path)
    cells = HE.expand_cells(cfg)
    states = {HE.cell_key(c): HE.cell_state(cfg, c, tmp_path, pack)["status"] for c in cells}
    assert {k for k, v in states.items() if v == "ready"} == {k for k in states if "|teacher|" in k}   # only the scripted teacher needs no run
    # a method whose run a stage of the recipe produces is pending (zero-shot) or unaccounted (no pack yet), never a silent miss: no missing_run
    assert "pending" in states.values() and "missing_run" not in states.values()


def test_level1_only_where_existing_controllers_exist():
    for task, has in (("h_steps", True), ("h_gap", True), ("h_walk", False), ("h_place", False)):
        cfg = HE.validate_config(json.loads(json.dumps(_plan(ROOT / f"recipes/humanoid/transfer_{task}.yaml").nodes["eval@bc.s0"].rc.options["transfer"])))
        assert ("1" in cfg["levels"]) == has
        assert {m["source"] for m in cfg["methods"] if m["level"] == 2} <= {"learned", "bc"}
        assert all(m["source"] in HE.SOURCES for m in cfg["methods"])


def test_source_label_must_match_the_policy(tmp_path):
    cfg = HE.validate_config(dict(name="x", task="h_walk", bodies=["t1"], train_seeds=[0], levels={"2": dict(unit="demos", budgets=[5], updates={5: 1})},
                                  methods=[dict(name="t", level=0, kind="teacher", source="learned", policy="teacher:h_walk")]))
    cell = HE.expand_cells(cfg)[0]
    with pytest.raises(ValueError, match="declared source 'learned'"):
        HE.build_policy(cfg, cell, tmp_path)


def test_register_skips_a_leaf_eval_parser():
    from rrp.cli import tools
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("eval", help="leaf command")
    tools.register(sub)                                                          # must not raise / replace the leaf
    assert ("eval", "humanoid-transfer") in tools.TOOLS


def test_steps_tool_keeps_argv_and_json_shape(tmp_path, monkeypatch):
    ep = SimpleNamespace(outcome="success", failure_reason=None, seed=7200, time=3.0, wall_s=0.5,
                         metrics=dict(scene=dict(h_frac=0.1, x_end=2.0), x=1.9, sim_time=3.0))
    monkeypatch.setattr(HE, "resolve_actor", lambda body, actor: f"{body}:v")
    monkeypatch.setattr("rrp.harness.eval.evaluate.evaluate", lambda *a, **k: [ep])
    pol = SimpleNamespace(info=SimpleNamespace(source="privileged_teacher"), entry=SimpleNamespace(extra_obs="none"))
    monkeypatch.setattr("rrp.policies.teachers.humanoid.make_rl_expert", lambda arg: pol)
    out = tmp_path / "s.json"
    assert HE.steps_main(["h1", "a.pt", "--n", "1", "--out", str(out), "--h-frac", "0.1"]) == 0
    d = json.loads(out.read_text())
    assert d["success"] == 1 and d["fell"] == 0 and d["rows"][0]["status"] == "success" and d["rows"][0]["h_frac"] == 0.1
    assert {"body", "actor", "n", "source", "rows"} <= set(d)
