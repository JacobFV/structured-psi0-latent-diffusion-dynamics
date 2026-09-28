"""W6 gates (D-112): synthetic inputs per criterion, the pipeline gate hook and run-dag's handling of a gated failure."""
from __future__ import annotations

import json

import pytest

from rrp.evaluation.gates import check_arm_dataset, check_dataset, check_legged_dataset, check_tracker, policy_flags


def _val(**kw):
    fw = dict(slip_ratio=0.05, cot=0.4, peak_force_bw=2.5, joint_limit_margin_min=0.2)
    fw.update(kw.pop("fw", {}))
    v = dict(body="x", family="quadruped", gate=dict(no_fall_rate=1.0, contact_gate=dict(slip_ratio=fw["slip_ratio"])),
             summary=dict(forward=fw, stand=dict(peak_force_bw=1.2, joint_limit_margin_min=0.3)),
             robustness=dict(nominal_forward_ratio=1.0, training_range={},
                             conditions={c: dict(no_fall_rate=1.0, forward_ratio=0.95) for c in ("mu_lo", "push")}))
    v.update(kw)
    return v


def _status(r, name):
    return next(c["status"] for c in r["criteria"] if c["name"] == name)


def test_tracker_pass_and_each_failure():
    assert check_tracker(_val())["verdict"] == "pass"
    assert _status(check_tracker(_val(fw=dict(slip_ratio=0.2))), "slip_ratio_forward") == "fail"
    assert _status(check_tracker(_val(fw=dict(cot=1.2))), "cot_forward") == "fail"            # quadruped <= 1.0
    assert _status(check_tracker(_val(fw=dict(cot=1.2), family="humanoid")), "cot_forward") == "pass"   # biped <= 2.0
    assert _status(check_tracker(_val(fw=dict(peak_force_bw=3.6))), "peak_foot_force_bw") == "fail"
    assert _status(check_tracker(_val(fw=dict(peak_force_bw=3.2))), "peak_foot_force_bw") == "pass"      # quadruped <= 3.5
    assert _status(check_tracker(_val(fw=dict(peak_force_bw=3.2), family="humanoid")), "peak_foot_force_bw") == "fail"  # biped <= 3.0
    assert _status(check_tracker(_val(fw=dict(joint_limit_margin_min=0.01))), "joint_limit_margin") == "fail"
    v = _val()
    v["gate"]["no_fall_rate"] = 0.8
    assert check_tracker(v)["verdict"] == "fail"
    v = _val()
    v["robustness"]["conditions"]["push"] = dict(no_fall_rate=1.0, forward_ratio=0.75)       # drop 0.25 > 0.20
    r = check_tracker(v)
    assert r["verdict"] == "fail" and _status(r, "robust_in_training_range") == "fail" and "push" in r["failed"][0]


def test_tracker_missing_measurements_are_incomplete_not_fail():
    v = _val()
    v.pop("robustness")
    for s in v["summary"].values():
        s.pop("peak_force_bw")
    r = check_tracker(v)
    assert r["verdict"] == "incomplete" and set(r["not_evaluated"]) == {"peak_foot_force_bw", "robust_in_training_range"}


def _leg(n, slip, sigma=0.0, status="success"):
    return [dict(body="b", sigma=sigma, status=status, tracker_version="t", motion=dict(family="legged", slip_ratio=slip))
            for _ in range(n)]


def test_legged_dataset():
    assert check_legged_dataset(_leg(96, 0.05) + _leg(4, 0.3))["verdict"] == "pass"          # 96% < 0.15
    r = check_legged_dataset(_leg(94, 0.05) + _leg(6, 0.3))
    assert r["verdict"] == "fail" and _status(r, "slip_ok_fraction") == "fail"
    r = check_legged_dataset(_leg(50, 0.05) + _leg(1, 0.05, status="fell") + _leg(1, 0.05, sigma=0.2, status="fell"))
    assert _status(r, "falls_at_sigma0") == "fail"                                            # the sigma-0 fall counts
    assert check_legged_dataset(_leg(50, 0.05) + _leg(2, 0.05, sigma=0.3, status="fell"))["verdict"] == "pass"
    assert check_dataset(None, _leg(10, 0.05))["gate"] == "legged_dataset"


def _arm(n, step=0.2, jerk=7.0, margin=0.1, pen=0.0005, grasp="grasp_v2", body="panda_pg2"):
    return [dict(robot_key=body, status="success", exec_noise=0.0,
                 motion=dict(family="arm", phase_switch_vel_step_max=step, cmd_jerk_rms=jerk, joint_limit_margin_min=margin,
                             penetration_max_m=pen, grasp_contact_version=grasp)) for _ in range(n)]


def test_arm_dataset():
    ref = dict(source="test", bodies=dict(panda_pg2=5.0))
    assert check_arm_dataset(_arm(20), reference=ref)["verdict"] == "pass"
    assert _status(check_arm_dataset(_arm(20, step=0.9), reference=ref), "phase_switch_vel_step") == "fail"
    assert _status(check_arm_dataset(_arm(20, jerk=11.0), reference=ref), "cmd_jerk_rms_vs_v2_teacher") == "fail"   # 2.2x
    assert _status(check_arm_dataset(_arm(20, margin=0.01), reference=ref), "joint_limit_margin") == "fail"
    # procedural parm* arms: the margin is reported (labelled), never gated; menagerie arms in the same dataset stay enforced
    ref2 = dict(source="test", bodies=dict(panda_pg2=5.0, parm6_tf3=5.0))
    r = check_arm_dataset(_arm(20, margin=0.0, body="parm6_tf3"), reference=ref2)
    assert _status(r, "joint_limit_margin_procedural") == "labelled" and r["verdict"] == "pass"
    assert all(c["name"] != "joint_limit_margin" for c in r["criteria"])
    r = check_arm_dataset(_arm(20, margin=0.0, body="parm6_tf3") + _arm(20, margin=0.01), reference=ref2)
    assert _status(r, "joint_limit_margin") == "fail" and _status(r, "joint_limit_margin_procedural") == "labelled"
    assert _status(check_arm_dataset(_arm(20, pen=0.004), reference=ref), "penetration") == "fail"          # grasp_v2: gated
    r = check_arm_dataset(_arm(20, pen=0.02, grasp="grasp_v1"), reference=ref)                              # grasp_v1: labelled
    assert _status(r, "penetration") == "labelled" and r["verdict"] == "pass"
    r = check_arm_dataset(_arm(20, body="other"), reference=ref)
    assert _status(r, "cmd_jerk_rms_vs_v2_teacher") == "not_evaluated" and r["verdict"] == "incomplete"
    # teacher_quality rows are accepted too
    tq = [dict(robot="panda_pg2", outcome="success", vel_jump_switch_max=0.2, joint_cmd_jerk_rms=6.0,
               joint_limit_margin_min=0.1, pen_max_m=0.001, grasp_contact="v2", feasible=True)] * 10
    assert check_arm_dataset(tq, reference=ref)["verdict"] == "pass"


def test_arm_joint_margin_reported_for_dart_episodes():
    """D-118 (2): the joint-limit margin is enforced on clean episodes, reported (not gated) on DART episodes."""
    ref = dict(source="test", bodies=dict(panda_pg2=5.0))
    dart = _arm(20, margin=-0.01)
    for e in dart:
        e["exec_noise"] = 0.08
    r = check_arm_dataset(_arm(20) + dart, reference=ref)
    assert _status(r, "joint_limit_margin") == "pass" and _status(r, "joint_limit_margin_dart") == "labelled"
    assert r["verdict"] == "pass"
    r = check_arm_dataset(_arm(20, margin=-0.01) + dart, reference=ref)
    assert _status(r, "joint_limit_margin") == "fail"


def test_policy_flags_are_reported_only():
    r = policy_flags([dict(motion=dict(chunk_vel_step_max=x)) for x in (1.0, 2.0, 3.0)])
    assert r["verdict"] == "reported" and r["flagged"] == 2


def test_pipeline_apply_gate_writes_report_and_raises(tmp_path):
    from types import SimpleNamespace
    from rrp.pipelines.base import GateFailed, apply_gate
    bad = check_legged_dataset(_leg(10, 0.5))
    ctx = SimpleNamespace(out=tmp_path, opts={}, log=lambda m: None)
    with pytest.raises(GateFailed, match="slip_ok_fraction"):
        apply_gate(ctx, bad)
    assert json.loads((tmp_path / "gate_report.json").read_text())["verdict"] == "fail"
    ctx.opts = {"gate": "report"}
    assert apply_gate(ctx, bad)["verdict"] == "fail"                     # report-only mode records, does not raise


def test_run_dag_fails_a_gated_node_without_retry(tmp_path):
    from tests.unit.test_dag import FakeRunner, _ex

    class GateRunner(FakeRunner):
        def gate_report(self, node):
            p = self.root / node.rc.out / "gate_report.json"
            return json.loads(p.read_text()) if p.exists() else None

    r = GateRunner(tmp_path, fail={"b@sem.s1": 5})              # b has retries: 1
    ex = _ex(tmp_path, r)
    ob = ex.plan.nodes["b@sem.s1"].rc.out
    (tmp_path / ob).mkdir(parents=True)
    (tmp_path / ob / "gate_report.json").write_text(json.dumps(dict(gate="legged_dataset", verdict="fail",
                                                                     failed=["slip_ok_fraction=0.5 (threshold x)"])))
    ex.run()
    e = ex.ledger.node("b@sem.s1")
    assert e["state"] == "failed" and len(e["attempts"]) == 1 and "gate legged_dataset failed" in e["last_error"]
    assert ex.ledger.node("c@sem.s1")["state"] == "blocked"
