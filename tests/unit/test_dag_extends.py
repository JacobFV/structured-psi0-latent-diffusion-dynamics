"""`extends:` DAG overlays (D-102): the v2-teacher arm DAG inherits the arm lineage DAG and changes only names, matrix,
data/expert inputs and resources; no v1-trained input may remain."""
import json
import re
from pathlib import Path

import pytest

from rrp.harness.dag import DagError, load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]


def test_v2_arm_dag_uses_only_v2_inputs():
    plan = plan_dag(load_dag(ROOT / "dags/arm_lineage_v2.yaml"), source="t")
    assert len(plan.order) == 100                       # 25 nodes x {semfix, nosem} x seeds {1, 2}
    for nid in plan.order:
        rc = plan.nodes[nid].rc.model_dump()
        blob = json.dumps(rc)
        assert not re.search(r"v3dart|(?<!bcv2_)direct1701_u12000|baselines_bc_ckpts", blob), nid
        assert rc["flags"]["zero_prev_action"] is True, nid
        assert rc["variant"] in ("semfix", "nosem")
        if "packed_dir" in rc["inputs"]:
            assert rc["inputs"]["packed_dir"] == "packed/latent_pp_v4dart_s1_H16"
        if "bc_policy" in rc["inputs"]:
            assert rc["inputs"]["bc_policy"].endswith("seed1701/source:policy_u12000.pt")


def test_v2_training_configs_match_v1_recipe_except_inputs():
    v1 = plan_dag(load_dag(ROOT / "dags/arm_lineage.yaml"), source="t")
    v2 = plan_dag(load_dag(ROOT / "dags/arm_lineage_v2.yaml"), source="t")
    strip = lambda d: {k: v for k, v in d.items() if k not in ("name",)}
    for nid in v2.order:
        a, b = v1.nodes[nid].rc.model_dump(), v2.nodes[nid].rc.model_dump()
        pa = strip(a["params"]); pb = strip(b["params"])
        pa.pop("latent", None); pb.pop("latent", None); pa.get("policy", {}).pop("name", None); pb.get("policy", {}).pop("name", None)
        assert pa == pb, nid                            # seeds, steps, lr, fractions, z-noise: unchanged
        assert a["flags"] == b["flags"], nid


def test_extends_cycle_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n")
    with pytest.raises(DagError):
        load_dag(tmp_path / "a.yaml")
