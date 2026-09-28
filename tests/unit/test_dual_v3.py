"""D-126 W12 items: dual teacher v3 (#18), dual dataset gate, dual pipeline stages + DAG template (#33), coordination
tasks (#22). Defaults must be unchanged (v2 teacher, v2 collection jobs and episode keys, existing gates)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------------------- teacher v3
def test_minjerk_superposition_is_c2_and_reaches_goals():
    from rrp.teachers.dual_smooth import SmoothArmMover, minjerk
    m = object.__new__(SmoothArmMover)
    m.base, m.t = np.zeros(3), 0.0
    m.subs = [(0.0, 1.0, np.array([0.1, 0.0, 0.0])), (0.6, 0.8, np.array([0.0, 0.05, 0.0]))]   # blended corner
    ts = np.arange(0, 1.6, 0.002)
    P = []
    for t in ts:
        m.t = t
        P.append(m._pos())
    P = np.array(P)
    v = np.diff(P, axis=0) / 0.002
    a = np.diff(v, axis=0) / 0.002
    assert np.allclose(P[-1], [0.1, 0.05, 0.0], atol=1e-9) and np.allclose(v[0], 0, atol=1e-3)
    assert np.abs(np.diff(a, axis=0)).max() < 0.1          # no acceleration jumps (C2), incl. at the blend start 0.6 s
    assert minjerk(0.5) == pytest.approx(0.5) and minjerk(-1) == 0 and minjerk(2) == 1


def test_v3_options_dart_gate_and_factory():
    from rrp.teachers.dual_smooth import DART_PHASES, V3Options, dart_phase_allowed
    assert V3Options.from_dict({"minjerk": False}).minjerk is False and V3Options().confirm_grasp
    with pytest.raises(ValueError):
        V3Options.from_dict({"nonsense": True})
    for contact_phase in ("l_hold", "r_descend", "r_close", "r_lift", "r_align", "r_insert"):
        assert not dart_phase_allowed("support_insert", contact_phase)
    for p in ("l_close", "l_present", "l_hold", "l_open", "r_close", "r_descend"):
        assert not dart_phase_allowed("handover", p)
    assert dart_phase_allowed("support_insert", "r_transit") and dart_phase_allowed("handover", "r_transport")
    assert not dart_phase_allowed("unknown_task", "r_transit") and set(DART_PHASES) == {"support_insert", "handover"}


def test_make_dual_teacher_default_is_v2():
    from rrp.teachers.dual import HandoverTeacher, SupportInsertTeacher
    from rrp.teachers.dual_coord import CarryTrayTeacherStub, PivotTeacherStub
    from rrp.teachers.dual_smooth import (DEFAULT_DUAL_TEACHER, HandoverTeacherV3, SupportInsertTeacherV3,
                                          make_dual_teacher)
    from rrp.teachers.dual_validate import make_session
    assert DEFAULT_DUAL_TEACHER == "v2"
    s = make_session("support_insert", "parm5_pg2__parm5_pg2", 0)
    t = make_dual_teacher("support_insert", s)
    assert type(t) is SupportInsertTeacher and not hasattr(t, "limits")
    with pytest.raises(ValueError):
        make_dual_teacher("support_insert", s, "v2", {"minjerk": False})
    t3 = make_dual_teacher("support_insert", s, "v3", {"deep_grasp": False})
    assert isinstance(t3, SupportInsertTeacherV3) and t3.grasp_depth == t.grasp_depth and t3.o.minjerk
    assert make_dual_teacher("support_insert", s, "v3").grasp_depth == 0.03
    h = make_session("handover", "parm5_pg2__parm5_pg2", 0)
    assert type(make_dual_teacher("handover", h)) is HandoverTeacher
    assert isinstance(make_dual_teacher("handover", h, "v3"), HandoverTeacherV3)
    p = make_session("pivot_against_surface", "parm5_pg2__parm5_pg2", 0)
    assert isinstance(make_dual_teacher("pivot_against_surface", p), PivotTeacherStub)
    c = make_session("carry_tray_level", "parm5_pg2__parm5_pg2", 0)
    stub = make_dual_teacher("carry_tray_level", c)
    assert isinstance(stub, CarryTrayTeacherStub) and stub.stub is True
    for sess, te in ((s, t3), (p, make_dual_teacher("pivot_against_surface", p)), (c, stub)):
        for _ in range(5):                                     # tiny: the teachers act without errors
            sess.step(te.act())


# ----------------------------------------------------------------------------------------------- collection defaults
def test_collect_jobs_and_episode_keys_unchanged_by_default():
    from rrp.data.collect_dual import EXTRA_KEYS, build_jobs, collect_dual_episode
    cfg = dict(out_dir="x", task="handover", items=[dict(pair="a__b", split="s", seed_start=3, episodes=2)],
               noise_levels=[0.0, 0.04], noise_burst=[25, 5], max_steps=800, stop_after_success=10)
    jobs, ex = build_jobs(cfg)
    assert ex == {} and len(jobs) == 4 and all(len(j) == 9 for j in jobs)
    assert jobs[1] == ("handover", "a__b", 3, "x", "s", 800, 0.04, 10, (25, 5))
    jobs2, ex2 = build_jobs(dict(cfg, teacher_version="v2", record_quality=False))
    assert jobs2 == jobs and ex2 == {}                          # explicit defaults = absent
    jobs3, ex3 = build_jobs(dict(cfg, teacher_version="v3", noise_phase_gate=True))
    assert ex3 == {"teacher_version": "v3", "noise_phase_gate": True} and jobs3[0][9] == ex3
    assert set(EXTRA_KEYS) >= {"contact_labels", "teacher_version", "record_quality", "noise_phase_gate"}
    from rrp.teachers.dual import TEACHERS
    from rrp.teachers.dual_validate import make_session
    s = make_session("handover", "parm5_pg2__parm5_pg2", 0)
    rec = collect_dual_episode(s, TEACHERS["handover"](s), max_steps=3, episode_id="e")
    new_keys = {"motion", "teacher_version", "teacher_options", "teacher_limits", "dart_variant"}
    assert not new_keys & set(rec.public["meta"]) and "contact_frames" not in rec.private
    s = make_session("handover", "parm5_pg2__parm5_pg2", 0)
    from rrp.teachers.dual_smooth import make_dual_teacher
    rec = collect_dual_episode(s, make_dual_teacher("handover", s, "v3"), max_steps=3, episode_id="e",
                               record_quality=True, noise_phase_gate=True, exec_noise=0.04)
    m = rec.public["meta"]
    assert m["teacher_version"] == "dual_v3_minjerk" and m["dart_variant"] == "phase_gated_v1"
    assert m["motion"]["family"] == "dual" and set(m["motion"]["per_arm"]) == {"left", "right"}


# ----------------------------------------------------------------------------------------------- dual gate
def _row(status="success", step=0.3, margin=0.1, jerk=10.0, pen=0.001, slip=0.002, dpos=0.001, drot=0.02, order=0.0,
         noise=0.0, pair="panda_pg2__ur5e_pg2"):
    arm = dict(phase_switch_vel_step_max=step, joint_limit_margin_min=margin, cmd_jerk_rms=jerk)
    return dict(status=status, robot_key=pair, exec_noise=noise,
                motion=dict(family="dual", per_arm=dict(left=arm, right=dict(arm)), penetration_max_m=pen,
                            grasp_contact_version="grasp_v2.1",
                            contact=dict(cf_support_anchor_slip_max_m=slip, cf_held_pos_drift_grip_max_m=dpos,
                                         cf_held_rot_drift_grip_max_rad=drot, cf_contact_order_error=order)))


def test_dual_dataset_gate():
    from rrp.evaluation.gates import check_dataset, check_dual_dataset
    ok = check_dual_dataset([_row() for _ in range(20)])
    assert ok["verdict"] == "pass", ok["failed"]
    assert check_dataset(None, [_row() for _ in range(20)])["gate"] == "dual_dataset"
    for kw, name in ((dict(step=0.9), "phase_switch_vel_step"), (dict(jerk=40.0), "cmd_jerk_rms_vs_arm_v2_teacher"),
                     (dict(pen=0.01), "penetration"), (dict(slip=0.02), "support_slip"),
                     (dict(dpos=0.02), "grip_drift_pos"), (dict(drot=0.3), "grip_drift_rot"),
                     (dict(order=0.5), "contact_order"), (dict(margin=0.0), "joint_limit_margin")):
        r = check_dual_dataset([_row(**kw) for _ in range(20)])
        assert r["verdict"] == "fail" and any(f.startswith(name) for f in r["failed"]), (kw, r["failed"])
    # DART episodes: maintained-contact values reported, not gated
    r = check_dual_dataset([_row() for _ in range(20)] + [_row(noise=0.04, slip=0.05) for _ in range(5)])
    assert r["verdict"] == "pass" and any(c["name"] == "support_slip_dart" and c["status"] == "labelled"
                                          for c in r["criteria"])
    # procedural arms: joint margin not gated; no quality records -> incomplete, never pass
    assert check_dual_dataset([_row(margin=0.0, pair="parm5_pg2__parm5_pg2") for _ in range(20)])["verdict"] == "pass"
    assert check_dual_dataset([dict(status="success", robot_key="a__b")])["verdict"] == "incomplete"


def test_arm_gate_family_detection_unchanged():
    from rrp.evaluation.gates import _family
    assert _family([dict(robot_key="panda_pg2", control_dt=0.05)]) == "arm"
    assert _family([dict(motion=dict(family="dual"))]) == "dual"


# ----------------------------------------------------------------------------------------------- pipeline + DAG
def test_dual_pipeline_guards_and_template():
    from rrp.contracts.runconfig import RunConfig
    from rrp.orchestration.dag import load_dag, plan_dag
    from rrp.pipelines.base import Pipeline, StageContext, StageError
    from rrp.pipelines import dual
    p = Pipeline("dual")
    with pytest.raises(StageError, match="label source"):
        p.spec("dagger_collect").fn(None)
    plan = plan_dag(load_dag(ROOT / "dags/templates/dual_lineage.yaml"), source="t")
    stages = {n.rc.stage for n in plan.nodes.values()}
    assert {"collect", "pack", "train_rep", "probes", "train_flow", "eval_r2", "heldout", "edits"} <= stages
    for nid, n in plan.nodes.items():
        assert n.placement == "peer", nid
        if n.rc.stage in ("train_rep", "train_flow", "refit", "flow_ft"):
            assert n.rc.flags.zero_prev_action is True, nid           # B-1 correct
        if n.rc.stage == "collect":
            nat = n.rc.params
            assert nat["teacher_version"] == "v3" and nat["record_quality"] and nat["noise_phase_gate"] \
                and nat["contact_labels"], nid
            assert n.rc.options.get("gate", "enforce") == "enforce"
    st = next(n for n in plan.nodes.values() if n.rc.stage == "train_rep" and n.rc.variant == "semfix")
    assert st.rc.to_native()["latent"]["probe_lv_min"] == -4.0
    # the B-1 guard refuses a new dual training config with zero_prev_action false
    rc = st.rc.model_copy(update=dict(flags=st.rc.flags.model_copy(update=dict(zero_prev_action=False))))
    ctx = StageContext(rc=rc, index=None, root=ROOT)
    with pytest.raises(StageError, match="zero_prev_action"):
        dual._b1(ctx)


# ----------------------------------------------------------------------------------------------- coordination tasks
def test_coordination_tasks_compile_and_estimators():
    from rrp.contracts.task import TaskDefinition
    from rrp.envs.dual import pivot_angle_from_height
    from rrp.tasks.compiler import compile_task
    for name, edges in (("pivot_against_surface", {("brace", "pivot", "maintained_during"),
                                                   ("contact", "pivot", "maintained_during")}),
                        ("carry_tray_level", {("carry", "release", "enables")})):
        c = compile_task(TaskDefinition.model_validate(json.loads((ROOT / "tasks" / f"{name}.json").read_text())))
        assert edges <= {(e.src, e.dst, e.type) for e in c.edges}
    L, h = 0.05, 0.015
    for th in (0.0, 0.4, 1.0, 1.2):
        z = L * math.sin(th) + h * math.cos(th)
        assert pivot_angle_from_height(z, L, h) == pytest.approx(th, abs=1e-9)
    from rrp.teachers.dual_validate import make_session
    s = make_session("carry_tray_level", "parm5_pg2__parm5_pg2", 1)
    assert s.truth_predicate("level_error_rad", ["tray"]) == pytest.approx(0.0, abs=1e-6)
    assert s.estimate("level_error_rad", ["tray"])[0] == 0.0                     # not carried by two hands
    p = make_session("pivot_against_surface", "parm5_pg2__parm5_pg2", 1)
    assert p.truth_predicate("pivot_angle_rad", ["box"]) == pytest.approx(0.0, abs=1e-6)
    assert not p.privileged_success() and not s.privileged_success()
