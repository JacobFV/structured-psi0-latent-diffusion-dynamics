"""W7 grasp contact v2 (D-108): no interpenetration holding, Coulomb-like slip under load, versioning + provenance."""
import math

import pytest

mujoco = pytest.importorskip("mujoco")

from rrp.core.provenance import physics_provenance
from rrp.harness.eval.grasp_rig import build, run
from rrp.bodies import grasp_contact as GC


def test_versioning_and_provenance():
    m1, i1 = build("v1", "pg2")
    m2, i2 = build("v2", "tf3")
    assert GC.model_grasp_version(m1) is None and i1["changed"] == 0
    assert GC.model_grasp_version(m2) == "grasp_v2" and i2["pads"] == 3 and i2["objects"] == 1
    assert "grasp_contact_version" not in physics_provenance(m1).to_dict()        # legacy records unchanged
    assert physics_provenance(m2).to_dict()["grasp_contact_version"] == "grasp_v2"
    pad = next(g for g in range(m2.ngeom) if GC.PAD_RE.search(m2.geom(g).name))
    assert m2.geom_priority[pad] == 1 and 0.8 <= m2.geom_friction[pad][0] <= 1.2 and m2.geom_condim[pad] >= 4
    grip = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, "r0_act_grip")
    assert abs(m2.actuator_forcerange[grip][1] - 2.2) < 1e-9          # ~40 N at the 55 mm fingertip
    with pytest.raises(ValueError):
        GC.resolve("v3")


@pytest.mark.parametrize("kind", ["pg2", "tf3"])
def test_held_lift_penetration_below_3mm(kind):
    r = run("v2", kind, yaw=math.radians(15))
    assert r["lift_held"] and r["pen_max_during_lift_mm"] < 3.0 and r["lift_slip_mm"] < 3.0


@pytest.mark.parametrize("fscale", [1.0, 0.05])
def test_slip_under_load_is_coulomb(fscale):
    """Parallel jaw: slip onset at m*g ~= mu * sum(N) (within 25 %), also at x0.05 friction."""
    r = run("v2", "pg2", fscale)
    assert r["slip_ratio"] is not None and 0.8 <= r["slip_ratio"] <= 1.25, r
