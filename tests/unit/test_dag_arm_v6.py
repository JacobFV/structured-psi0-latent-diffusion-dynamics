"""The v6 arm lineage DAG (D-110 relaunch, D-127): grasp contact v2.1 in every simulated stage, v6 inputs only,
peer placement only, and the unchanged v1 training recipe."""
import json
import re
from pathlib import Path

from rrp.orchestration.dag import load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
TRAINING = {"train_rep", "train_flow", "flow_ft", "refit"}     # read packs only; simulate nothing


def _plan(name):
    return plan_dag(load_dag(ROOT / "dags" / name), source="t")


def test_every_simulated_stage_uses_grasp_v2_1():
    plan = _plan("arm_lineage_v6.yaml")
    assert len(plan.order) == 100
    sim = 0
    for nid in plan.order:
        rc = plan.nodes[nid].rc.model_dump()
        if rc["stage"] in TRAINING:
            continue
        sim += 1
        assert rc["options"].get("grasp_contact") == "v2.1", nid
    assert sim == 4 * 14                  # 8 collections + 5 evals + edit suite, per variant x seed


def test_v6_inputs_placement_and_flags():
    plan = _plan("arm_lineage_v6.yaml")
    for nid in plan.order:
        n = plan.nodes[nid]
        rc = n.rc.model_dump()
        blob = json.dumps(rc)
        assert not re.search(r"v3dart|v4dart|v5dart|bcv2_|baselines_bc_ckpts|(?<!bcv6_)direct1701_u12000", blob), nid
        assert n.placement == "peer", nid
        assert rc["flags"]["zero_prev_action"] is True, nid
        if "packed_dir" in rc["inputs"]:
            assert rc["inputs"]["packed_dir"] == "packed/latent_pp_v6dart_s1_H16", nid
        if "bc_policy" in rc["inputs"]:
            assert rc["inputs"]["bc_policy"] == "runs/armexpert_bcv6/baseline_direct_action/seed1701/source:policy.pt", nid
            assert rc["options"]["bc_label"] == "bcv6_direct1701_final", nid


def test_v6_training_recipe_equals_v1():
    v1, v6 = _plan("arm_lineage.yaml"), _plan("arm_lineage_v6.yaml")
    for nid in v6.order:
        a, b = v1.nodes[nid].rc.model_dump(), v6.nodes[nid].rc.model_dump()
        if a["stage"] not in TRAINING:
            continue
        pa, pb = dict(a["params"]), dict(b["params"])
        for p in (pa, pb):
            p.pop("name", None); p.pop("latent", None)
            if "policy" in p: p["policy"] = {k: v for k, v in p["policy"].items() if k != "name"}
        assert pa == pb, nid
        assert a["flags"] == b["flags"], nid


def test_all_arm_dags_respect_broker_time_cap():
    """The broker refuses max_seconds > 21600 at launch (a node that exceeds it fails before running)."""
    for f in ("arm_lineage.yaml", "arm_lineage_v2.yaml", "arm_lineage_v6.yaml"):
        plan = _plan(f)
        for nid in plan.order:
            assert 0 < plan.nodes[nid].resources.max_seconds <= 21600, (f, nid)
