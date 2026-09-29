"""Legged DAG template for FUTURE runs (dags/templates/legged_v2_gated.yaml; D-114, D-123 backlog): the tracker is validated
and gated before any collection, the dataset gate is enforced, and the recipe equals the completed W8 go2 DAG."""
import hashlib
import json
from pathlib import Path

import pytest

from rrp.harness.dag import load_dag, plan_dag

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "dags/templates/legged_v2_gated.yaml"
GO2_SHA = "af3f06f4e029e9f92fafc50e6174ffdb171c5512fbab4cd6fac18e6ce23cf18b"


def _child(tmp_path: Path, body="go2", sha=GO2_SHA) -> Path:
    p = tmp_path / "legged_v2_x.yaml"
    p.write_text(f"extends: {TEMPLATE}\nname: legged_v2_x\nlabel_prefix: 'l8z_{{tg}}{{seed}}'\nglobal_label_prefix: 'l8z'\n"
                 f"vars:\n  body: {body}\n  tracker_sha256: {sha}\n")
    return p


def test_template_validates_tracker_before_collect_and_enforces_gates():
    plan = plan_dag(load_dag(TEMPLATE), source="t")
    val = plan.nodes["validate"]
    assert val.rc.stage == "validate_tracker" and val.placement == "peer"
    assert val.rc.options["robust"] is True and val.rc.options["tracker_sha256"] == "SET_IN_CHILD_DAG"
    assert "validate" in plan.nodes["collect"].deps
    assert plan.order.index("validate") < plan.order.index("collect")
    for nid, n in plan.nodes.items():
        assert n.rc.options.get("gate", "enforce") != "report", nid        # no report-only gates in the template
        assert n.placement == "peer", nid                                  # D-115


def test_child_dag_matches_go2_recipe_and_declares_one_tracker(tmp_path):
    child = plan_dag(load_dag(_child(tmp_path)), source="t")
    go2 = plan_dag(load_dag(ROOT / "dags/legged_v2_go2.yaml"), source="t")
    assert set(child.order) == set(go2.order) | {"validate"}
    strip = lambda d: {k: v for k, v in d.items() if k != "name"}
    for nid in go2.order:
        a, b = go2.nodes[nid].rc.model_dump(), child.nodes[nid].rc.model_dump()
        assert strip(a["params"]) == strip(b["params"]), nid
        assert a["flags"] == b["flags"], nid
        assert a["stage"] == b["stage"] and a["seed"] == b["seed"] and a["variant"] == b["variant"], nid
    shas = {n.rc.options["tracker_sha256"] for n in child.nodes.values() if "tracker_sha256" in n.rc.options}
    assert shas == {GO2_SHA}
    assert child.nodes["collect"].rc.options["actuator_limits"] == "sourced_v1"


def test_validate_tracker_checks_declared_sha_before_running(tmp_path):
    from rrp.core.runconfig import RunConfig, RunIndex
    from rrp.harness.pipelines.base import StageContext, StageError
    from rrp.harness.pipelines.legged import check_tracker_sha
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"tracker weights")
    good = hashlib.sha256(b"tracker weights").hexdigest()

    def ctx(**opts):
        rc = RunConfig.model_validate(dict(
            schema_version="runconfig-1", family="legged", stage="validate_tracker", variant="semfix", seed=0, lineage="l",
            track="t", flags=dict(zero_prev_action=None, realizer_anchor=None, realizer_drop_qd=None, probe_lv_min=None,
                                  qd_dropout=None, contact_version="contact_v2"), options=dict(body="go2", **opts)))
        return StageContext(rc=rc, index=RunIndex(), root=tmp_path)
    assert check_tracker_sha(ctx(actor=str(actor))) is None                    # nothing declared: no check
    assert check_tracker_sha(ctx(actor=str(actor), tracker_sha256=good)) == good
    with pytest.raises(StageError, match="tracker sha"):
        check_tracker_sha(ctx(actor=str(actor), tracker_sha256="SET_IN_CHILD_DAG"))
    with pytest.raises(StageError, match="not found"):
        check_tracker_sha(ctx(actor=str(tmp_path / "missing.pt"), tracker_sha256=good))
