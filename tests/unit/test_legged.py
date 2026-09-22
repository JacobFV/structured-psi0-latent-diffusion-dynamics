import mujoco
import numpy as np

from rrp.control.legged_core import LeggedBinding
from rrp.morphology.legged import hexapod, standalone_model


def _setup():
    model, _, meta = standalone_model(hexapod())
    b = LeggedBinding(model, meta)
    d = mujoco.MjData(model)
    b.set_default(d, yaw=0.7)
    d.qvel[b.da + 3:b.da + 6] = [0.3, -0.2, 0.5]
    mujoco.mj_forward(model, d)
    return model, b, d


def test_imu_equals_root_state():
    model, b, d = _setup()
    quat, gyro = b.imu(d)
    sq = d.sensordata[b.imu_quat:b.imu_quat + 4]
    sg = d.sensordata[b.imu_gyro:b.imu_gyro + 3]
    assert np.allclose(sq, quat, atol=1e-6) or np.allclose(sq, -quat, atol=1e-6)
    assert np.allclose(sg, gyro, atol=1e-6)


def test_public_obs_has_no_privileged_base_state():
    model, b, d = _setup()
    cmd, a = np.array([0.1, 0, 0.2]), np.zeros(b.n)
    o1 = b.public_obs(d, cmd, a, 0.3)
    d.qpos[b.qa:b.qa + 3] += [5.0, -3.0, 0.2]          # base position (incl. height)
    d.qvel[b.da:b.da + 3] = [1.0, 0.5, -0.3]           # base linear velocity
    d.qpos[b.qa + 3:b.qa + 7] = [np.cos(1.2), 0, 0, np.sin(1.2)]   # yaw only (gravity unchanged)
    o2 = b.public_obs(d, cmd, a, 0.3)
    assert o1.shape == (b.obs_dim,)
    assert np.allclose(o1, o2, atol=1e-6)
