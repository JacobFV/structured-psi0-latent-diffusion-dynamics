"""D-126 #27 base-state estimator: synthetic known answers for the filter math, real-body kinematics, and the
LeggedSession option (default byte-identical; estimator mode changes sensing only, never physics)."""
import hashlib
import math

import mujoco
import numpy as np
import pytest

from rrp.envs.state_estimator import (BaseStateEstimator, EstimatorConfig, LegKinematics, leg_odometry,
                                      quat_to_rot)

G = 9.81


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def test_quat_to_rot_yaw90():
    R = quat_to_rot(_yaw_quat(math.pi / 2))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_static_stays_zero():
    e = BaseStateEstimator()
    feet = np.array([[0.2, 0.1, -0.3], [0.2, -0.1, -0.3], [-0.2, 0.1, -0.3], [-0.2, -0.1, -0.3]])
    for _ in range(500):
        e.update(quat=[1, 0, 0, 0], gyro=np.zeros(3), acc=[0, 0, G], foot_pos_b=feet, foot_vel_b=np.zeros((4, 3)),
                 touch=np.full(4, 10.0), dt=0.02)
    assert np.allclose(e.v_w, 0, atol=1e-12) and np.allclose(e.p_w, 0, atol=1e-12)


def test_imu_only_integrates_exactly():
    """No stance: v = a t, p = sum of Euler steps (known closed form)."""
    e = BaseStateEstimator()
    a = np.array([0.5, -0.2, 0.0])
    dt, n = 0.02, 100
    for _ in range(n):
        e.update(quat=[1, 0, 0, 0], gyro=np.zeros(3), acc=a + [0, 0, G], foot_pos_b=np.zeros((2, 3)),
                 foot_vel_b=np.zeros((2, 3)), touch=np.zeros(2), dt=dt)
    assert np.allclose(e.v_w, a * n * dt, atol=1e-12)
    assert np.allclose(e.p_w, a * dt * dt * n * (n + 1) / 2, atol=1e-12)
    assert e.n_stance == 0


def test_leg_odometry_recovers_translation_and_rotation():
    """Stance feet fixed in the world: base moves with v and spins with w; foot velocity relative to the base in
    the body frame is -(v + w x r). The odometry must return v exactly."""
    rng = np.random.default_rng(0)
    v, w = np.array([0.4, -0.1, 0.02]), np.array([0.1, -0.05, 0.6])
    r = rng.normal(0, 0.2, (4, 3))
    vrel = -(v[None] + np.cross(w[None], r))
    vb, n = leg_odometry(r, vrel, w, [True, True, False, True])
    assert n == 3 and np.allclose(vb, v, atol=1e-12)
    vb, n = leg_odometry(r, vrel, w, [False] * 4)
    assert vb is None and n == 0


def test_leg_velocity_rotated_to_world():
    """Body frame yawed 90 deg; body-frame forward velocity 1 m/s appears as +y in the world after convergence."""
    e = BaseStateEstimator(EstimatorConfig(tau_s=0.05))
    q = _yaw_quat(math.pi / 2)
    r = np.array([[0.2, 0.1, -0.3], [-0.2, -0.1, -0.3]])
    v_b = np.array([1.0, 0, 0])
    for _ in range(200):
        e.update(quat=q, gyro=np.zeros(3), acc=[0, 0, G], foot_pos_b=r, foot_vel_b=-np.tile(v_b, (2, 1)),
                 touch=[5, 5], dt=0.02)
    assert np.allclose(e.v_w, [0, 1, 0], atol=1e-9)
    assert np.allclose(e.v_b, v_b, atol=1e-9)


def test_complementary_bounds_accelerometer_bias():
    """A constant accelerometer bias b with a true zero velocity: the steady-state error of the filter is
    b * tau (continuous limit; exactly b*dt*(1-k)/k for the discrete update), not unbounded as with pure IMU."""
    tau, dt, b = 0.1, 0.01, np.array([0.3, 0, 0])
    e = BaseStateEstimator(EstimatorConfig(tau_s=tau))
    for _ in range(3000):
        e.update(quat=[1, 0, 0, 0], gyro=np.zeros(3), acc=b + [0, 0, G], foot_pos_b=np.zeros((1, 3)),
                 foot_vel_b=np.zeros((1, 3)), touch=[5], dt=dt)
    k = dt / (tau + dt)
    assert np.allclose(e.v_w, b * dt * (1 - k) / k, atol=1e-9)
    assert abs(e.v_w[0] - b[0] * tau) < 0.01


def test_state_roundtrip():
    e = BaseStateEstimator()
    e.update(quat=_yaw_quat(0.3), gyro=[0, 0, 0.1], acc=[0.1, 0, G], foot_pos_b=np.zeros((1, 3)),
             foot_vel_b=np.zeros((1, 3)), touch=[5], dt=0.02)
    e2 = BaseStateEstimator()
    e2.load(e.state())
    assert e2.state() == e.state()


# ------------------------------------------------------------------ real-body kinematics (procedural hexapod)
def _hexapod():
    from rrp.bodies.legged import hexapod, standalone_model
    from rrp.envs.legged_core import LeggedBinding
    model, _, meta = standalone_model(hexapod())
    b = LeggedBinding(model, meta)
    jq = np.array(b.pol_qadr)
    jd = np.array(b.pol_dadr)
    return model, b, jq, jd


def test_leg_kinematics_matches_true_feet_in_root_frame():
    model, b, jq, jd = _hexapod()
    d = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    b.set_default(d, xy=(1.3, -0.4), yaw=0.9, noise=0.2, rng=rng)
    d.qpos[b.qa + 3:b.qa + 7] = quat = np.array([0.9, 0.1, -0.05, 0.4]) / np.linalg.norm([0.9, 0.1, -0.05, 0.4])
    d.qvel[jd] = rng.normal(0, 1.0, len(jd))
    d.qvel[b.da:b.da + 6] = [0.5, -0.2, 0.1, 0.3, -0.4, 0.7]       # base motion must not enter the adapter
    mujoco.mj_forward(model, d)
    kin = LegKinematics.from_binding(b, jq, jd)
    pos, vel = kin.feet(d.qpos[jq], d.qvel[jd])
    R = quat_to_rot(quat)
    p0 = d.qpos[b.qa:b.qa + 3]
    truth = np.array([R.T @ (d.site_xpos[s] - p0) for s in b.foot_sids])
    assert np.allclose(pos, truth, atol=1e-9)
    # velocities: finite difference of the adapter's own foot positions under joint motion only
    h = 1e-6
    pos2, _ = kin.feet(d.qpos[jq] + h * d.qvel[jd], d.qvel[jd])
    assert np.allclose(vel, (pos2 - pos) / h, atol=1e-4)


def test_leg_odometry_on_real_kinematics_known_base_velocity():
    """Real hexapod: pick base (v, w) and joint velocities such that stance feet are world-static; odometry returns v."""
    model, b, jq, jd = _hexapod()
    d = mujoco.MjData(model)
    b.set_default(d, yaw=0.0)
    mujoco.mj_forward(model, d)
    kin = LegKinematics.from_binding(b, jq, jd)
    v, w = np.array([0.3, 0.05, 0.0]), np.array([0.0, 0.0, 0.2])
    pos, _ = kin.feet(d.qpos[jq], np.zeros(len(jd)))
    # target foot velocities relative to the base for world-static feet, solve J qd = -(v + w x r) per foot
    kin.feet(d.qpos[jq], np.zeros(len(jd)))
    qd = np.zeros(model.nv)
    J = np.zeros((3, model.nv))
    for i, s in enumerate(b.foot_sids):
        mujoco.mj_jacSite(model, kin.d, J, None, s)
        cols = [c for c in jd if np.abs(J[:, c]).sum() > 1e-9]
        sol, *_ = np.linalg.lstsq(J[:, cols], -(v + np.cross(w, pos[i])), rcond=None)
        qd[cols] = sol
    _, vel = kin.feet(d.qpos[jq], qd[jd])
    vb, n = leg_odometry(pos, vel, w, np.ones(len(pos), bool))
    assert n == len(pos) and np.allclose(vb, v, atol=1e-6)


# ------------------------------------------------------------------ session option
def _session_traj(source, steps=12):
    from rrp.contracts.action import NativeCommand
    from rrp.envs.legged import LeggedSession, build_waypoint_contact
    sc = build_waypoint_contact("hexapod6", 3)
    kw = {} if source is None else dict(base_state_source=source)
    s = LeggedSession(sc, tracker_kind="cpg", seed=3, **kw)
    s.reset()
    h = hashlib.sha256()
    obs = []
    for k in range(steps):
        cmd = NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.3, 0.0, 0.2]},
                            source="scripted_teacher")
        r = s.step(cmd)
        h.update(np.asarray(s.data.qpos).tobytes())
        obs.append(r.observation)
    return s, h.hexdigest(), obs


def test_session_default_is_truth_noise_and_estimator_changes_sensing_only():
    s0, h0, o0 = _session_traj(None)
    s1, h1, o1 = _session_traj("truth_noise")
    s2, h2, o2 = _session_traj("estimator")
    assert h0 == h1 == h2                              # physics identical: the option is sensing-only
    assert s0.base_estimator is None and o0[-1].measured_node_state.base_vel_estimate is None
    assert o0[-1].model_dump_json(exclude={"observation_id"}) == o1[-1].model_dump_json(exclude={"observation_id"})
    ve = o2[-1].measured_node_state.base_vel_estimate
    assert ve is not None and ve.shape == (3,) and np.isfinite(ve).all()
    assert np.isfinite(s2.speed_est) and s2.speed_est != s0.speed_est
    assert s0.loc.tolist() == s2.loc.tolist()          # localization noise stream unchanged
    snap = s2.snapshot()
    st = s2.base_estimator.state()
    s2.base_estimator.reset()
    s2.restore(snap)
    assert s2.base_estimator.state() == st


def test_session_rejects_unknown_source():
    from rrp.envs.legged import LeggedSession, build_waypoint_contact
    with pytest.raises(ValueError):
        LeggedSession(build_waypoint_contact("hexapod6", 3), tracker_kind="cpg", seed=3, base_state_source="truth")


# golden: computed on origin/main 733b02a (before the option existed) with the same script; default path unchanged
GOLDEN_DEFAULT_HEXAPOD6_S3 = "b209b12be5536a17b12c8863a835933d900c8d3ae419dafea12982aa375fe203"


def test_default_session_byte_identical_to_pre_d126():
    from rrp.contracts.action import NativeCommand
    from rrp.envs.legged import LeggedSession, build_waypoint_contact
    s = LeggedSession(build_waypoint_contact("hexapod6", 3), tracker_kind="cpg", seed=3)
    s.reset()
    h = hashlib.sha256()
    for _ in range(15):
        r = s.step(NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.3, 0.0, 0.2]},
                                 source="scripted_teacher"))
        h.update(np.asarray(s.data.qpos).tobytes())
        h.update(r.observation.model_dump_json(exclude={"observation_id"}).encode())
        h.update(repr(s.speed_est).encode())
    assert h.hexdigest() == GOLDEN_DEFAULT_HEXAPOD6_S3
