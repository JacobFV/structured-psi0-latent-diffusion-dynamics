"""ARM V6 TARGET BODIES DAGs (roadmap #9/#10): sealed protocol, grasp_v2.1, v6 inputs, peer only, and matched
budgets / adapt seeds between the latent route and the BC baseline."""
import json
from pathlib import Path

from rrp.harness.dag import load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
PROTO = json.loads((ROOT / "configs/eval/latent_slice1.json").read_text())
TARGETS = set(PROTO["targets"])


def _nodes(name):
    plan = plan_dag(load_dag(ROOT / "dags" / name), source="t")
    return [(nid, plan.nodes[nid], plan.nodes[nid].rc.model_dump()) for nid in plan.order]


def _common(nodes):
    for nid, n, rc in nodes:
        assert n.placement == "peer", nid
        assert rc["options"].get("grasp_contact") == "v2.1", nid
        if rc["stage"] == "target_eval":            # deployment input; adaptation stages inherit the flag from the
            assert rc["flags"]["zero_prev_action"] is True, nid   # checkpoint they adapt (all trained with it true)
        assert n.resources.max_seconds <= 21600, nid
        o = rc["options"]
        if rc["stage"] == "target_eval":
            assert o.get("sealed_run") is True and not o.get("smoke"), nid
        if rc["stage"] == "target_adapt":
            assert o["target"] in TARGETS and o["budget"] in PROTO["sft_budgets"] and o["budget"] > 0, nid
            assert int(o["adapt_seed"]) in PROTO["seeds"], nid


def test_latent_targets_dag():
    nodes = _nodes("arm_targets_v6_latent.yaml")
    assert len(nodes) == 167
    _common(nodes)
    for nid, n, rc in nodes:
        blob = json.dumps(rc["inputs"])
        if "flow" in rc["inputs"] and not rc["inputs"]["flow"].startswith("runs/armtgt"):
            assert rc["inputs"]["flow"].startswith("runs/armv6/arm6-") and "flow_ft-gdag2h_s" in blob, nid
        if rc["inputs"].get("representation", "").startswith("runs/armv6"):
            assert "refit-gendag3_noqd_s" in rc["inputs"]["representation"], nid
        if rc["stage"] == "target_adapt":
            assert int(rc["options"]["adapt_seed"]) == 1700 + rc["seed"], nid


def test_bc_targets_dag_matches_latent_budgets_and_seeds():
    bc, lat = _nodes("arm_targets_v6_bc.yaml"), _nodes("arm_targets_v6_latent.yaml")
    assert len(bc) == 49
    _common(bc)
    assert not any(rc["stage"] == "train_bc" for _, _, rc in bc)          # experts are reused, not retrained
    for nid, n, rc in bc:
        if "bc_policy" in rc["inputs"] and rc["inputs"]["bc_policy"].startswith("runs/armexpert_bcv6"):
            assert rc["inputs"]["bc_policy"].endswith("/source:policy.pt"), nid
    key = lambda nodes, m: sorted((rc["options"]["target"], rc["options"]["budget"], int(rc["options"]["adapt_seed"]))
                                  for _, _, rc in nodes if rc["stage"] == "target_adapt" and rc["options"]["method"] == m)
    b = key(bc, "bc_sft")
    assert set(b) == set(key(lat, "flow_sft"))            # same (target, budget, adapt seed) cells
    assert set(b) == set(key(lat, "system0_refit"))
    ev = lambda nodes: sorted(rc["options"]["robot"] for _, _, rc in nodes if rc["stage"] == "target_eval")
    assert set(ev(bc)) == set(ev(lat)) == TARGETS | set(PROTO["source_heldout_eval_robots"])
