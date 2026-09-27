"""contact_v1 vs contact_v2: tiny physics checks (stick below atan(mu), slide above, creep, penetration)."""
import itertools
import math

import mujoco
import numpy as np
import pytest

from rrp.physics.contact import CONTACT_MODELS, ContactRandomizer, resolve
from rrp.bodies.legged import legged_world


def _slope_model(contact, foot, mu=None):
    s = legged_world("slope", None, contact=contact, size=5.0)
    if mu is not None:  # same Coulomb coefficient for both versions, so only the contact model differs
        fl = s.geom("floor")
        fl.friction = [mu, fl.friction[1], fl.friction[2]]
    b = s.worldbody.add_body(name="foot", pos=[0, 0, 0.1])
    b.add_freejoint()
    if foot == "box":        # flat sole, 30 kg (humanoid foot under body weight)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.11, 0.05, 0.02], mass=30.0)
    else:                    # stool on 4 sphere feet (point-foot quadruped / hexapod)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.02], mass=15.0, contype=0, conaffinity=0)
        for x, y in itertools.product([-0.1, 0.1], [-0.1, 0.1]):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], pos=[x, y, -0.03], mass=0.01)
    return s.compile()


def slope_run(contact, foot, tan_ratio, mu=1.0, T=2.5):
    """Tilt gravity so tan(slope) = tan_ratio * mu; return (mean downhill speed after 0.5 s, max penetration mm)."""
    m = _slope_model(contact, foot, mu)
    ang = math.atan(tan_ratio * mu)
    m.opt.gravity[:] = [9.81 * math.sin(ang), 0, -9.81 * math.cos(ang)]
    d = mujoco.MjData(m)
    v, pen = [], 0.0
    for _ in range(int(T / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.time > 0.5:
            v.append(d.qvel[0])
            pen = max([pen] + [-d.contact[i].dist for i in range(d.ncon)])
    return float(np.mean(v)), pen * 1000


@pytest.mark.parametrize("foot", ["box", "stool"])
def test_stick_below_slide_above_and_less_creep(foot):
    v1_stick, _ = slope_run("v1", foot, 0.9)
    v2_stick, pen2 = slope_run("v2", foot, 0.9)
    v2_slide, _ = slope_run("v2", foot, 1.2)
    assert v2_stick < 2e-3                      # sticks (< 2 mm/s) at 90% of the friction limit
    assert v2_slide > 0.5                       # slides clearly above atan(mu)
    assert v2_stick < 0.3 * v1_stick            # v2 creeps much less than v1 at the same mu
    assert pen2 < 3.0                           # penetration in the mm range


def test_landing_penetration_mm():
    m = _slope_model("v2", "box")
    d = mujoco.MjData(m)
    d.qpos[2] = 0.07                            # 5 cm drop of a 30 kg sole
    pen = 0.0
    for _ in range(500):
        mujoco.mj_step(m, d)
        pen = max([pen] + [-d.contact[i].dist for i in range(d.ncon)])
    assert 0.0 < pen * 1000 < 5.0


def test_v1_unchanged_and_resolve():
    s = legged_world("w", None, contact="v1")
    m = s.compile()
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and m.opt.impratio == 1
    assert np.allclose(m.geom_friction[g], [1.0, 0.02, 0.001]) and m.geom_priority[g] == 0
    assert resolve("contact_v2") == "v2" and resolve("V1") == "v1"
    m2 = legged_world("w", None, contact="v2").compile()
    assert m2.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m2.opt.impratio == CONTACT_MODELS["v2"]["impratio"]
    assert m2.opt.timestep <= 0.002
    from rrp.physics.contact import model_contact_version
    assert model_contact_version(m) == "contact_v1" and model_contact_version(m2) == "contact_v2"


def test_randomizer_ranges():
    m = _slope_model("v2", "box")
    R = CONTACT_MODELS["v2"]["randomization"]
    cr = ContactRandomizer(m, 1, np.random.default_rng(0))
    for _ in range(20):
        p = cr.resample()
        assert R["mu"][0] <= p["mu"] <= R["mu"][1]
        assert R["timeconst"][0] <= p["timeconst"] <= R["timeconst"][1]
        assert 0 <= cr.latency() <= R["latency_substeps"][1]
