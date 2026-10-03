"""D-147 (hum-teachers): the scripted humanoid manipulation teachers derive their stance, balance gains, squat plan and palm geometry
from each body's model (`policies/teachers/humanoid.py`: BodyStance, SquatPlanner, UpperIK) instead of t1-tuned constants.

Kinematic checks on the Menagerie humanoids t1 / g1 / h1 (no tracker weights: the static-support legs are planned, the tracker is a
stub), one tiny physics check (g1 holds a planned squat), and two pure-math checks. Red on the t1-only code: g1's straight-knee IK
diverged, g1's CoM reference was a heel-corner touch site, h1 had no squat plan within 0.25 m, the ankle gains were t1's numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_wholebody import _binding, _put, registry  # noqa: E402,F401  (registry is a fixture)

from rrp.policies.teachers import humanoid as TH  # noqa: E402


def _model(body: str):
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedBinding
    r = legged_body(body)
    m = r.spec.copy().compile()
    return m, LeggedBinding(m, r.meta, "")


def _session(tmp_path, registry, body: str, task: str, seed: int = 0):
    from rrp.envs.mujoco.humanoid_scenes import make_humanoid_session
    from rrp.envs.mujoco.legged import build_waypoint_contact
    _put(tmp_path, body, _binding(build_waypoint_contact(body, 0, contact="v2")), upper=False, seed=5)
    registry()
    return make_humanoid_session(task=task, body=body, seed=seed, tracker=f"{body}:stub", tracker_kind="learned")


# ---------------------------------------------------------------- pure math
def test_geom_world_box_is_exact_for_a_rotated_box():
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(pos=[1.0, 2.0, 3.0], quat=[np.cos(0.25), 0.0, 0.0, np.sin(0.25)])     # 0.5 rad about z
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.1, 0.2])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = TH._geom_world_box(m, d, 0)
    c, s = np.cos(0.5), np.sin(0.5)
    ex = np.array([0.3 * c + 0.1 * s, 0.3 * s + 0.1 * c, 0.2])
    assert np.allclose(lo, [1.0, 2.0, 3.0] - ex) and np.allclose(hi, [1.0, 2.0, 3.0] + ex)


def test_balance_gain_rule_closes_the_loop_at_the_t1_ratios():
    """K (1 + KP h) = ALPHA m g h and the damping ratio ZETA, whatever the body: the rule, not the numbers, is what transfers."""
    st = TH.BodyStance.__new__(TH.BodyStance)
    for mass, h, K, kd in ((31.6, 0.54, 100.0, 4.0), (51.4, 0.93, 80.0, 6.0)):
        mgh, inertia = mass * st.G * h, mass * h * h
        KP = (st.ALPHA * mgh / K - 1.0) / h
        k_net = K * (1.0 + KP * h) - mgh
        KD = (2.0 * st.ZETA * np.sqrt(k_net * inertia) - kd) / (K * h)
        assert K * (1.0 + KP * h) == pytest.approx(st.ALPHA * mgh)
        assert (kd + K * KD * h) / (2.0 * np.sqrt(k_net * inertia)) == pytest.approx(st.ZETA)


# ---------------------------------------------------------------- morphology (Menagerie t1 / g1 / h1, kinematics only)
@pytest.mark.menagerie
def test_t1_keeps_its_validated_ankle_gains_and_support_centre():
    st = TH.BodyStance(*_model("t1"))
    assert (st.KP, st.KI, st.KD) == pytest.approx((8.0, 2.0, 0.5), rel=0.01)       # U2: 20/20 per task with these
    assert st.support_x0 == pytest.approx(0.05, abs=1e-3) and st.sign == -1.0


@pytest.mark.menagerie
def test_sole_centre_is_the_middle_of_the_contact_geoms_not_the_touch_site():
    m, b = _model("g1")
    st = TH.BodyStance(m, b)
    d = mujoco.MjData(m)
    b.set_default(d)
    mujoco.mj_forward(m, d)
    site_x = float(np.mean([d.site_xpos[s][0] for s in b.foot_sids]))
    assert site_x == pytest.approx(-0.05, abs=2e-3)                    # g1's touch site: one heel-corner sphere of four
    assert st.support_x0 == pytest.approx(0.035, abs=2e-3)             # heel -0.05 .. toe 0.12
    assert st.KP > 8.0                                                  # weaker ankles per m g h than t1: a larger gain


@pytest.mark.menagerie
@pytest.mark.parametrize("body,ankles", [("t1", ("Left_Ankle_Pitch", "Right_Ankle_Pitch")),
                                         ("g1", ("left_ankle_pitch_joint", "right_ankle_pitch_joint")),
                                         ("h1", ("left_ankle", "right_ankle"))])
def test_ankle_pitch_joints_come_from_the_leg_chain_and_axis(body, ankles):
    m, b = _model(body)
    st = TH.BodyStance(m, b)
    names = tuple(m.joint(int(m.actuator_trnid[b.pol_act[i], 0])).name for i in st.ankle_ix)
    assert names == ankles


@pytest.mark.menagerie
def test_squat_ik_starts_off_the_straight_knee_singularity():
    m, b = _model("g1")
    pl = TH.SquatPlanner(m, b)
    for dz in (0.025, 0.1, 0.2):
        p = pl.pose(dz, dz / 0.79)
        assert p["ok"] and p["foot_err"] < 2e-3, (dz, p["foot_err"])     # the q0 seed diverged to 1.2 m foot error


@pytest.mark.menagerie
@pytest.mark.parametrize("body,kind,r", [("t1", "geom_long_axis", 0.03), ("g1", "forearm", 0.0271), ("h1", "geom_long_axis", 0.025)])
def test_palm_bar_axis_and_radius_are_derived(tmp_path, registry, body, kind, r):
    s = _session(tmp_path, registry, body, "h_reach")
    ik = TH.UpperIK(s.model, s.binding, s.palm_ids())
    rec = ik.record()
    assert {rec[sd]["bar"] for sd in ("left", "right")} == {kind}
    assert ik.palm_r["left"] == pytest.approx(r, abs=1e-3)
    assert s.model.body(ik.chest_bid).name.endswith(("Trunk", "torso_link"))   # the frame that twists with the waist


@pytest.mark.menagerie
@pytest.mark.parametrize("body", ["t1", "g1"])
def test_squat_plan_exists_and_t1_keeps_its_u2_plan(tmp_path, registry, body):
    TH._PLANS.clear()
    s = _session(tmp_path, registry, body, "h_squat_pick")
    t = TH.SquatPickTeacher(s)
    sq = t.derived()["squat"]
    assert sq["ik_err"] < t.IK_TOL and all(t.plan["ok"])
    if body == "t1":                                                    # U2: dz 0.225 m, pitch 1.5 dz, level bars
        assert (sq["dz"], sq["bar_z"]) == (0.225, 0.0) and sq["pitch"] == pytest.approx(0.3375, abs=1e-3)


@pytest.mark.slow
@pytest.mark.menagerie
def test_h1_squat_plan_tilts_the_bar_and_pitches_past_its_ankle_stop(tmp_path, registry):
    TH._PLANS.clear()
    s = _session(tmp_path, registry, "h1", "h_squat_pick")
    sq = TH.SquatPickTeacher(s).derived()["squat"]
    assert sq["dz_L"] > 0.25 and sq["bar_z"] != 0.0 and sq["pitch"] > sq["dz_L"]     # no level-bar plan exists for h1's forearm palms


@pytest.mark.menagerie
def test_at_feet_with_the_default_feet_is_the_default_plan():
    m, b = _model("t1")
    pl = TH.SquatPlanner(m, b)
    d = mujoco.MjData(m)
    b.set_default(d)
    mujoco.mj_forward(m, d)
    p0, p1 = pl.pose(0.1, 0.15), pl.at_feet(d, (0.0, 0.0), 0.0).pose(0.1, 0.15)
    assert np.allclose(p0["legs"], p1["legs"], atol=1e-4)


# ---------------------------------------------------------------- tiny physics: the planned squat is held (gravity feed-forward)
@pytest.mark.menagerie
def test_g1_holds_its_planned_squat(tmp_path, registry):
    TH._PLANS.clear()
    s = _session(tmp_path, registry, "g1", "h_squat_pick")
    t = TH.SquatPickTeacher(s)
    b = s.binding
    for _ in range(int((t.tl["squat"] + 0.4) / t.dt)):                 # down to the full squat (the arms still hang)
        s.step(t.act())
        assert not s.fell
    planned = float(t.plan["root_pos"][-1][2])
    assert abs(float(s.data.qpos[b.qa + 2]) - planned) < 0.03          # without tau / kp it sank 5-10 cm and sat back
    e = float(s.data.subtree_com[b.root_bid][0] - t.stance.support_centre(s.data)[0])
    assert abs(e) < 0.03
