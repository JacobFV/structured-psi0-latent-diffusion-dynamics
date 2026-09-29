"""D-126 arm DAG templates and recipe overlays (#4, #6, #9, #10): they plan, change only what they declare, keep the
sealed protocol and matched budgets, and never collide with the parent's outputs."""
from __future__ import annotations

from pathlib import Path

import pytest

from rrp.harness.dag import DagError, load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "dags"
OVERLAYS = ("stageA_beta_kl", "bcdagger_schedule", "refit_lengths")
DECLARED = {   # node -> the only (section, key path) that may differ from the parent
    "stageA_beta_kl": {"stageA": [("params", ("latent", "beta_kl"))]},
    "bcdagger_schedule": {**{n: [("options", ("episodes",))] for n in ("bc1", "bc2", "bc3")},
                          **{n: [("params", ("dagger_frac",))] for n in ("rzbcdag1", "rzbcdag1long", "rzbcdag2")}},
    "refit_lengths": {n: [("params", ("steps",))] for n in ("rzbcdag1", "rzbcdag2", "rzgendag1", "rzgendag2", "rzgendag3")},
}


def _strip(d: dict, drop: list) -> dict:
    import copy
    d = copy.deepcopy(d)
    for sec, path in drop:
        cur = d[sec]
        for k in path[:-1]:
            cur = cur[k]
        cur.pop(path[-1], None)
    return d


def _names(d):
    """Names/labels embed the lineage prefix; the comparison is about recipe content."""
    return {k: v for k, v in d.items() if k != "name"} if isinstance(d, dict) else d


@pytest.mark.parametrize("ov", OVERLAYS)
def test_recipe_overlay_changes_only_declared_keys(tmp_path, ov):
    child = tmp_path / "c.yaml"
    child.write_text(f"extends: [{DAGS / 'arm_lineage_v2.yaml'}, {DAGS / 'overlays/arm_recipe' / (ov + '.yaml')}]\n"
                     f"name: c_{ov}\n")
    parent = plan_dag(load_dag(DAGS / "arm_lineage_v2.yaml"), source="p").select(points=[{"variant": "nosem"}])
    comp = plan_dag(load_dag(child), source="c")
    assert {n.point["variant"] for n in comp.nodes.values()} == {"nosem"}
    assert set(comp.nodes) == set(parent.nodes)
    assert not {n.rc.out for n in comp.nodes.values()} & {n.rc.out for n in parent.nodes.values()}
    changed = set()
    for nid, n in comp.nodes.items():
        a, b = parent.nodes[nid].rc.model_dump(), n.rc.model_dump()
        drop = DECLARED[ov].get(n.name, [])
        pa, pb = _names(_strip(a, drop)["params"]), _names(_strip(b, drop)["params"])
        if isinstance(pa.get("latent"), dict):
            pa["latent"], pb["latent"] = _names(pa["latent"]), _names(pb["latent"])
        assert pa == pb, nid
        assert _strip(a, drop)["options"] == _strip(b, drop)["options"], nid
        assert a["flags"] == b["flags"] and a["stage"] == b["stage"] and a["seed"] == b["seed"], nid
        fa, fb = _names(a["params"]), _names(b["params"])
        if isinstance(fa.get("latent"), dict):
            fa, fb = dict(fa, latent=_names(fa["latent"])), dict(fb, latent=_names(fb["latent"]))
        if fa != fb or a["options"] != b["options"]:
            changed.add(n.name)
    assert changed and changed <= set(DECLARED[ov]), (ov, changed)     # (refit_lengths: bcdag2 already had 16000)


def test_list_extends_refuses_cycles(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(f"extends: [{DAGS / 'arm_lineage.yaml'}, a.yaml]\nname: a\n")
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


def test_existing_arm_dags_still_plan_identically():
    """Default-off: the committed arm DAGs plan to the same node set and config hashes with the D-126 code (the hash
    values are pinned by tests/unit/test_dag.py's legacy-config equality; here: nothing new leaks into them)."""
    for f in ("arm_lineage.yaml", "arm_lineage_v2.yaml", "armexpert_v6dart.yaml"):
        p = plan_dag(load_dag(DAGS / f), source="x")
        for n in p.nodes.values():
            assert "grasp_contact" not in n.rc.options and "chunk_blend" not in n.rc.options
            assert n.rc.stage not in ("grpo", "target_eval", "target_adapt")
