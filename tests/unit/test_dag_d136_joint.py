"""D-136 pre-registered joint adaptation DAG: sealed evals only on the new-arm targets, protocol budgets/seeds, update
counts matched to BC SFT (SFT_STEPS), the pre-registered variant (split, gen_frac 0.5), D-135 cells untouched."""
import json
from pathlib import Path

from rrp.orchestration.dag import load_dag, plan_dag
from rrp.pipelines.arm import SFT_STEPS
from rrp.training.joint_adapt import split_steps

ROOT = Path(__file__).resolve().parents[2]
PROTO = json.loads((ROOT / "configs/eval/latent_slice1.json").read_text())


def test_split_steps_total_matches_budget():
    for s in SFT_STEPS.values():
        f, r = split_steps(s, "split")
        assert f + r == s and abs(f - r) <= 1
    assert split_steps(600, "joint") == (600, 600)          # one optimizer: each update touches both modules


def test_d136_dag():
    plan = plan_dag(load_dag(ROOT / "dags/arm_targets_d136_joint.yaml"), source="t")
    nodes = [plan.nodes[n].rc.model_dump() for n in plan.order]
    assert len(nodes) == 72
    for rc in nodes:
        o = rc["options"]
        assert o["grasp_contact"] == "v2.1"
        assert "armja136-" in rc["lineage"]
        if rc["stage"] == "target_eval":
            assert o["sealed_run"] is True and o["robot"] in ("xarm7_pg2", "xarm7_tf3")
            assert rc["flags"]["zero_prev_action"] is True
        else:
            assert rc["stage"] == "target_adapt" and o["method"] == "joint_adapt"
            assert (o["joint_mode"], o["gen_frac"]) == ("split", 0.5)
            assert o["budget"] in PROTO["sft_budgets"] and int(o["adapt_seed"]) == 1700 + rc["seed"]
            assert int(o["adapt_seed"]) in PROTO["seeds"]
