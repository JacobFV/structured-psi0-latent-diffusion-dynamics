"""W6: motion-quality metric math (known values) and the physics perturbations (tiny rollouts)."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from rrp.envs.motion_quality import (chunk_boundary_steps, cost_of_transport, finite_diff, jerk_stats,
                                           joint_limit_margin, slip_ratio)
from rrp.envs.perturb import CtrlDelay, PhysicsPerturbation, PushHook, apply_model


def test_jerk_of_cubic_is_exact():
    dt = 0.02
    t = np.arange(50) * dt
    q = np.stack([2.0 * t ** 3, -0.5 * t ** 3], 1)          # d3/dt3 = 12 and -3
    js = jerk_stats(q, dt)
    assert js["joint_jerk_peak"] == pytest.approx(12.0, rel=1e-6)
    assert js["joint_jerk_rms"] == pytest.approx(math.sqrt((144 + 9) / 2), rel=1e-6)
    assert jerk_stats(q[:3], dt)["joint_jerk_rms"] is None           # too short for a third difference
    assert finite_diff(t, dt, 1) == pytest.approx(np.ones(49))


def test_chunk_boundary_velocity_step():
    dt = 0.1
    # velocity 1 rad/s until tick 10, then 3 rad/s: the only velocity step (2 rad/s) is at tick 10
    c = np.array([[k * dt * 1.0] if k <= 10 else [1.0 + (k - 10) * dt * 3.0] for k in range(20)])
    r = chunk_boundary_steps(c, dt, boundary_ticks=[10])
    assert r["chunk_vel_step_max"] == pytest.approx(2.0)
    assert r["vel_step_any_max"] == pytest.approx(2.0)
    r2 = chunk_boundary_steps(c, dt, boundary_ticks=[5, 15])         # boundaries elsewhere see no step
    assert r2["chunk_vel_step_max"] == pytest.approx(0.0, abs=1e-9)
    assert r2["vel_step_any_max"] == pytest.approx(2.0)


def test_joint_limit_margin():
    lo, hi = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0])    # third joint unlimited (hi <= lo): skipped
    q = np.array([[0.0, 1.0, 5.0], [0.5, 1.8, -3.0]])
    # joint 0: min(1.5, 0.5)/2 = 0.25 ; joint 1: min(1.8, 0.2)/2 = 0.1
    assert joint_limit_margin(q, lo, hi) == pytest.approx(0.1)
    assert joint_limit_margin(np.array([[0.0, 1.0, 0.0]]), lo, hi) == pytest.approx(0.5)


def test_slip_ratio_and_cot():
    assert slip_ratio([0.1, 0.3], [1.0, 1.0]) == pytest.approx(0.2)
    assert slip_ratio([0.1], [0.0]) == pytest.approx(0.1 / 0.02)       # speed floor
    assert slip_ratio([], [1.0]) is None
    assert cost_of_transport(98.1, 10.0, 1.0) == pytest.approx(1.0)
    assert cost_of_transport(98.1, 10.0, 0.1) is None


BOX = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
<worldbody><geom name="floor" type="plane" size="5 5 .1" friction="0.8 0.01 0.001"/>
<body name="box" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="2" friction="1 0.02 0.002"/></body>
<body name="arm" pos="2 0 1"><joint name="j" type="hinge" axis="0 0 1"/><geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass="1"/></body>
</worldbody><actuator><position name="a" joint="j" kp="100" kv="10"/></actuator></mujoco>"""


def test_push_impulse_gives_expected_velocity_change():
    m = mujoco.MjModel.from_xml_string(BOX)
    d = mujoco.MjData(m)
    bid = m.body("box").id
    pert = PhysicsPerturbation(push_impulse_Ns=3.0, push_time_s=0.05, push_duration_s=0.1, push_dir_rad=0.0)
    h = PushHook(bid, pert, seed=0)
    while d.time < 0.3:
        h(d, m.opt.timestep)
        mujoco.mj_step(m, d)
    assert d.qvel[0] == pytest.approx(3.0 / 2.0, rel=0.03)          # dv = J / m along +x
    assert abs(d.qvel[1]) < 1e-9
    assert h.record()["push_impulse_applied_Ns"] == pytest.approx(3.0, rel=0.03)


def test_apply_model_scales_and_nominal_is_noop():
    m = mujoco.MjModel.from_xml_string(BOX)
    ref = mujoco.MjModel.from_xml_string(BOX)
    assert apply_model(m, PhysicsPerturbation(), robot_bodies=[2], com_body=2, act_ids=[0]) == {}
    assert np.array_equal(m.geom_friction, ref.geom_friction) and np.array_equal(m.body_mass, ref.body_mass)
    rec = apply_model(m, PhysicsPerturbation(friction_scale=0.5, mass_scale=1.2, com_offset_m=(0.02, 0, 0), kp_scale=0.7),
                      robot_bodies=[2], com_body=2, act_ids=[0])
    assert m.geom_friction == pytest.approx(0.5 * ref.geom_friction)
    assert m.body_mass[2] == pytest.approx(1.2 * ref.body_mass[2]) and m.body_mass[1] == ref.body_mass[1]
    assert m.body_ipos[2][0] == pytest.approx(ref.body_ipos[2][0] + 0.02)
    assert m.actuator_gainprm[0, 0] == pytest.approx(70.0) and m.actuator_biasprm[0, 1] == pytest.approx(-70.0)
    assert m.actuator_biasprm[0, 2] == pytest.approx(-7.0)
    assert rec["n_servos"] == 1
    m2 = mujoco.MjModel.from_xml_string(BOX)
    apply_model(m2, PhysicsPerturbation(object_friction_scale=0.25, object_mass_scale=3.0), robot_bodies=[], com_body=2,
                act_ids=[], object_bodies=[1])
    assert m2.geom_friction[1] == pytest.approx(0.25 * ref.geom_friction[1]) and m2.geom_friction[0] == pytest.approx(ref.geom_friction[0])
    assert m2.body_mass[1] == pytest.approx(3 * ref.body_mass[1])


def test_ctrl_delay_shifts_by_n_substeps():
    m = mujoco.MjModel.from_xml_string(BOX)
    d = mujoco.MjData(m)
    dl = CtrlDelay([0], 3, np.zeros(1))
    seen = []
    for k in range(1, 8):
        d.ctrl[0] = k                         # the controller writes k at substep k
        dl(d)
        seen.append(float(d.ctrl[0]))
    assert seen == [0, 0, 0, 1, 2, 3, 4]


def test_terrain_heightfield_is_named_floor_and_bounded():
    from rrp.bodies.legged import bump_heights, legged_world
    h = bump_heights(0.03, seed=5, half_m=2.0)
    assert h.max() == pytest.approx(0.03, rel=0.05) and h.min() >= 0
    n = h.shape[0]
    assert h[n // 2, n // 2] == 0.0                                # flat spawn area
    s = legged_world("t", None, contact="v2", terrain=dict(amp_m=0.03, seed=5, half_m=2.0))
    m = s.compile()
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD and m.hfield_size[0][2] == pytest.approx(0.03)
    flat = legged_world("t", None, contact="v2").compile()
    assert flat.geom_type[mujoco.mj_name2id(flat, mujoco.mjtObj.mjOBJ_GEOM, "floor")] == mujoco.mjtGeom.mjGEOM_PLANE
