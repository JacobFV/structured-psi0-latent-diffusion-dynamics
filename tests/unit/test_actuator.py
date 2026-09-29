"""actuator_v2: latency applies targets exactly L substeps late; PD torque never exceeds the torque-speed envelope."""
import warnings

import mujoco
import numpy as np

from rrp.bodies.legged import legged_body, standalone_model
from rrp.envs.mujoco.legged_core import LeggedBinding
from rrp.bodies.actuator import KNEE, ActuatorModel


def _setup(lat_ms):
    warnings.filterwarnings("ignore")
    m, _, meta = standalone_model(legged_body("hexapod6"), contact="v2")
    b = LeggedBinding(m, meta)
    act = ActuatorModel(m, b, 1, None, name=meta["name"], randomize=False, latency_ms=lat_ms)
    d = mujoco.MjData(m)
    b.set_default(d)
    mujoco.mj_forward(m, d)
    act.reset(0, d.ctrl[b.pol_act].copy())
    return m, b, act, d


def test_latency_substeps():
    m, b, act, d = _setup(6.0)              # 3 substeps at dt 0.002
    t0 = act.current[0].copy()
    act.command(0, t0 + 0.1)
    seen = [act.substep_ctrl(0, d).copy() for _ in range(5)]
    moved = [not np.allclose(u, np.clip(t0, b.lo, b.hi), atol=1e-6) for u in seen]
    assert moved == [False, False, False, True, True]


def test_torque_speed_clamp():
    m, b, act, d = _setup(0.0)
    act.command(0, b.hi.copy())
    d.qvel[b.pol_dadr] = 0.9 * act.vmax      # deep in the derated region
    u = act.substep_ctrl(0, d)
    q, qd = d.qpos[b.pol_qadr], d.qvel[b.pol_dadr]
    tau = act.kp * (u - q) - act.kd * qd
    lim = act.eff * np.clip((act.vmax - np.abs(qd)) / ((1 - KNEE) * act.vmax), 0, 1)
    assert np.all(np.abs(tau) <= lim + 1e-6)
