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
    from rrp.policies.teachers.dual_smooth import SmoothArmMover, minjerk
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
    from rrp.policies.teachers.dual_smooth import DART_PHASES, V3Options, dart_phase_allowed
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
    from rrp.policies.teachers.dual import HandoverTeacher, SupportInsertTeacher
    from rrp.policies.teachers.dual_smooth import (DEFAULT_DUAL_TEACHER, HandoverTeacherV3, SupportInsertTeacherV3,
                                          make_dual_teacher)
    from rrp.policies.teachers.dual_validate import make_session
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
    with pytest.raises(ValueError, match="parked"):                     # D-146 item 5: the W12 coordination stub is parked
        make_dual_teacher("pivot_against_surface", make_session("pivot_against_surface", "parm5_pg2__parm5_pg2", 0))
    for _ in range(5):                                         # tiny: the v3 teacher acts without errors
        s.step(t3.act())


def test_dual_training_parked():
    """D-146 item 5 / unit A3: no dual dagger_collect, no dual BC stage, no W12 coordination stubs; the collect and
    evaluation stages the dual_lineage template keeps stay registered."""
    import importlib.util
    from rrp.harness.pipelines.base import Pipeline
    st = Pipeline("dual").stages()
    assert "dagger_collect" not in st and "train_bc" not in st
    assert {"collect", "eval_r2", "heldout", "edits"} <= set(st)
    assert importlib.util.find_spec("rrp.policies.teachers.dual_coord") is None


# ----------------------------------------------------------------------------------------------- collection defaults
def test_collect_jobs_and_episode_keys_unchanged_by_default():
    from rrp.harness.data.collect_dual import EXTRA_KEYS, build_jobs, collect_dual_episode
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
    from rrp.policies.teachers.dual import TEACHERS
    from rrp.policies.teachers.dual_validate import make_session
    s = make_session("handover", "parm5_pg2__parm5_pg2", 0)
    rec = collect_dual_episode(s, TEACHERS["handover"](s), max_steps=3, episode_id="e")
    new_keys = {"motion", "teacher_version", "teacher_options", "teacher_limits", "dart_variant"}
    assert not new_keys & set(rec.public["meta"]) and "contact_frames" not in rec.private
    s = make_session("handover", "parm5_pg2__parm5_pg2", 0)
    from rrp.policies.teachers.dual_smooth import make_dual_teacher
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
    from rrp.harness.eval.gates import check_dataset, check_dual_dataset
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
    from rrp.harness.eval.gates import _family
    assert _family([dict(robot_key="panda_pg2", control_dt=0.05)]) == "arm"
    assert _family([dict(motion=dict(family="dual"))]) == "dual"


# ----------------------------------------------------------------------------------------------- pipeline + DAG
def test_dual_pipeline_guards_and_template():
    from rrp.core.runconfig import RunConfig
    from rrp.harness.dag import load_dag, plan_dag
    from rrp.harness.pipelines.base import Pipeline, StageContext, StageError
    from rrp.harness.pipelines import dual
    p = Pipeline("dual")
    with pytest.raises(StageError, match="not implemented"):          # parked (D-146 item 5): not registered at all
        p.spec("dagger_collect")
    plan = plan_dag(load_dag(ROOT / "recipes/templates/dual_lineage.yaml"), source="t")
    stages = {n.rc.stage for n in plan.nodes.values()}
    # R2 DP: parked means no training node (no Stage A / probes / flow / refit); the evals take an external checkpoint
    assert stages == {"collect", "pack", "eval_r2", "heldout", "edits"}
    assert "refit" not in p.stages()
    for nid, n in plan.nodes.items():
        assert n.placement == "peer", nid
        if n.rc.stage == "collect":
            nat = n.rc.params
            assert nat["teacher_version"] == "v3" and nat["record_quality"] and nat["noise_phase_gate"] \
                and nat["contact_labels"], nid
            assert n.rc.options.get("gate", "enforce") == "enforce"
        if n.rc.stage in ("eval_r2", "heldout", "edits"):
            paths = n.rc.input_paths(n.index if hasattr(n, "index") else None)
            assert set(paths) == {"checkpoint"} and "SET_IN_CHILD_DAG" in paths["checkpoint"], nid    # refuses until set
    # the B-1 guard of the (still registered) dual training stages refuses zero_prev_action false
    ev = next(n for n in plan.nodes.values() if n.rc.stage == "eval_r2")
    rc = ev.rc.model_copy(update=dict(flags=ev.rc.flags.model_copy(update=dict(zero_prev_action=False))))
    with pytest.raises(StageError, match="zero_prev_action"):
        dual._b1(StageContext(rc=rc, index=None, root=ROOT))


def test_dual_eval_stages_take_an_external_checkpoint_and_the_registry_tasks():
    from rrp.harness.pipelines import dual
    from rrp.tasks.spec import tasks_in
    assert dual.DUAL_TASKS == tasks_in("mujoco/dual") and {"support_insert", "handover"} <= set(dual.DUAL_TASKS)
    import inspect
    assert 'ctx.inp("checkpoint")' in inspect.getsource(dual._evaluate) and "ctx.inp(\"flow\")" not in inspect.getsource(dual)


def test_dual_teacher_episodes_run_on_rollout():
    """R2 DP: dual_validate's episode code lives in harness.eval.dual_teacher_quality on harness.rollout; the private
    episode loop (run_dual_teacher_episode) is deleted and functional_composition runs on the same rollout."""
    from rrp.harness.eval import dual_teacher_quality as Q
    from rrp.policies.teachers import dual, dual_validate, functional_composition as FC
    assert not hasattr(dual, "run_dual_teacher_episode") and not hasattr(dual, "DualTeacherResult")
    assert [n for n in ("run_one", "main", "summarize") if hasattr(dual_validate, n)] == []
    assert callable(Q.run_one) and callable(Q.validate_main) and callable(Q.summarize_validation)
    row = FC.run("parm5l_pg2__parm6_pg2", 3, "stale_receipt", max_steps=6)
    assert row["condition"] == "stale_receipt" and row["statuses"] and not row["privileged_success"]
    assert row["failure_reason"].startswith("ended_in_phase:")
    rows = [Q.run_one("support_insert", "parm5l_pg2__parm6_pg2", sd, max_steps=6) for sd in (3, 2)]
    assert [r["status"] for r in rows] == ["failure", "infeasible"] and rows[1]["failure_reason"].startswith("infeasible:")
    assert rows[0]["steps"] == 6 and rows[1]["steps"] == 0 and rows[1]["statuses"] == {}
    summ = Q.summarize_validation(rows)
    assert summ["support_insert|parm5l_pg2__parm6_pg2"]["n"] == 2


def test_dual_swap_slots_is_the_registry_edit():
    """PACKET_EDITS["swap_slots"] is packets.chunk_hook("swap_assembly", a=0, b=1) (P3's locked-equal edit)."""
    from rrp.harness.eval.dual_latent_eval import PACKET_EDITS
    from rrp.policies.packets import apply_edit

    class _P:
        def __init__(self, z, source="learned", sampling=None):
            self.z, self.source, self.sampling = z, source, sampling

        def model_copy(self, update):
            q = _P(self.z, self.source, self.sampling)
            q.__dict__.update(update)
            return q

    z = np.random.default_rng(0).normal(size=(4, 2, 5)).astype(np.float32)
    out = PACKET_EDITS["swap_slots"](0, _P(z))
    np.testing.assert_array_equal(out.z, apply_edit("swap_assembly", z, a=0, b=1))
    np.testing.assert_array_equal(out.z, z[:, ::-1])
    assert out.source == "debug" and out.sampling["intervention"] == "swap_assembly"


# ----------------------------------------------------------------------------------------------- coordination tasks
def test_coordination_tasks_compile_and_estimators():
    from rrp.core.task import TaskDefinition
    from rrp.envs.mujoco.dual import pivot_angle_from_height
    from rrp.tasks.compiler import compile_task
    for name, edges in (("pivot_against_surface", {("brace", "pivot", "maintained_during"),
                                                   ("contact", "pivot", "maintained_during")}),):
        c = compile_task(TaskDefinition.model_validate(json.loads((ROOT / "src" / "rrp" / "tasks" / "graphs" / f"{name}.json").read_text())))
        assert edges <= {(e.src, e.dst, e.type) for e in c.edges}
    L, h = 0.05, 0.015
    for th in (0.0, 0.4, 1.0, 1.2):
        z = L * math.sin(th) + h * math.cos(th)
        assert pivot_angle_from_height(z, L, h) == pytest.approx(th, abs=1e-9)
    from rrp.policies.teachers.dual_validate import make_session
    p = make_session("pivot_against_surface", "parm5_pg2__parm5_pg2", 1)
    assert p.truth_predicate("pivot_angle_rad", ["box"]) == pytest.approx(0.0, abs=1e-6)
    assert not p.privileged_success()
