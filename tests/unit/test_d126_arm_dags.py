"""D-126 arm recipe templates (#6, #9, #10): they plan, keep the sealed protocol and matched budgets. (The #4 ablation
overlays moved to the legacy area with the closed armabl track.)"""
from __future__ import annotations

from pathlib import Path

import pytest

from rrp.harness.dag import DagError, load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "recipes"


def test_list_extends_refuses_cycles(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(f"extends: [{DAGS / 'templates/arm_lineage.yaml'}, a.yaml]\nname: a\n")
    with pytest.raises(DagError, match="circular"):
        load_dag(a)


def test_grpo_template_matched_budget_and_anchors():
    p = plan_dag(load_dag(DAGS / "templates/arm_grpo.yaml"), source="t")
    lat = [n for n in p.nodes.values() if n.name == "grpo"]
    bc = p.nodes["grpo_bc"]
    assert len(lat) == 4 and bc.rc.options["method"] == "bc"
    for n in lat:
        assert n.rc.options["method"] == "latent" and n.rc.inputs["representation"].endswith("representation.pt")
        assert n.rc.params["budget_env_steps"] == max(bc.rc.params["budgets"])        # matched budget
        assert n.rc.params["grpo"] == bc.rc.params["grpo"]                              # same objective / sampler
        assert n.rc.options["anchor"] == bc.rc.options["anchor"]                        # same anchors and rule
        assert n.rc.params["robot"] == bc.rc.params["robot"]
        assert n.rc.options["grasp_contact"] == "v2.1" and n.placement == "peer" and n.resources.gpu
        assert "@grpo" not in n.rc.inputs["flow"] and "SET_IN_CHILD_DAG" in n.rc.inputs["flow"]
    assert set(lat[0].rc.options["anchor"]["robots"]).isdisjoint({"xarm7_pg2", "xarm7_tf3", "panda_tf3",
                                                                   lat[0].rc.params["robot"]})
    assert p.nodes["final_bc"].rc.inputs["bc_policy"].endswith(f"policy_b{max(bc.rc.params['budgets'])}.pt")
    assert "grpo@semfix.s1" in p.nodes["final@semfix.s1"].deps


def test_target_templates_sealed_and_fair():
    lat = plan_dag(load_dag(DAGS / "templates/arm_targets_latent.yaml"), source="t")
    bc = plan_dag(load_dag(DAGS / "templates/arm_targets_bc.yaml"), source="t")
    for plan in (lat, bc):
        for nid, n in plan.nodes.items():
            assert n.placement == "peer", nid
            if n.rc.stage == "target_eval":
                assert n.rc.options["sealed_run"] is True and not n.rc.options.get("smoke"), nid
                assert n.rc.options["robot"] in ("xarm7_pg2", "xarm7_tf3", "panda_tf3", "parm5s_tf3", "parm5l_pg2")
            if n.rc.stage == "target_adapt":
                assert n.rc.options["budget"] in (5, 20, 100) and n.rc.options["adapt_seed"] in (1701, 1702, 1703), nid
                assert n.rc.inputs["packed_dir"].endswith(f"pack-{n.rc.options['target']}_s0"), nid
    # same budgets and targets for both methods; latent: flow and system-0 adaptation, BC: its SFT
    kinds = lambda plan: sorted({n.rc.options["kind"] for n in plan.nodes.values() if n.rc.stage == "target_eval"})
    assert kinds(lat) == ["adapted:flow_sft:b100", "adapted:flow_sft:b20", "adapted:flow_sft:b5",
                          "adapted:system0_refit:b100", "adapted:system0_refit:b20", "adapted:system0_refit:b5",
                          "source", "zero_shot"]
    assert kinds(bc) == ["adapted:bc_sft:b100", "adapted:bc_sft:b20", "adapted:bc_sft:b5", "source", "zero_shot"]
    assert {n.rc.seed for n in bc.nodes.values() if n.rc.stage == "train_bc"} == {1702, 1703}
    pk = lambda plan: {n.rc.params["train_robots"][0]: {k: v for k, v in n.rc.params.items() if k != "name"}
                       for n in plan.nodes.values() if n.rc.stage == "pack"}
    assert pk(lat) == pk(bc)                                     # identical target demo packs


def test_arm_lineage_template_stays_default_off():
    """Default-off: the arm lineage template plans without any D-126 option (the rendered nodes of every kept recipe are
    pinned by the recipe.* goldens, tests/unit/test_recipes.py)."""
    p = plan_dag(load_dag(DAGS / "templates/arm_lineage.yaml"), source="x")
    for n in p.nodes.values():
        assert "grasp_contact" not in n.rc.options and "chunk_blend" not in n.rc.options
        assert n.rc.stage not in ("grpo", "target_eval", "target_adapt")
