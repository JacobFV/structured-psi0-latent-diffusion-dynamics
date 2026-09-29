"""Body-model torque limits: sourced_v1 is the default, versioned in meta, model text and physics provenance."""
import re
import warnings

import mujoco
import pytest

from rrp.bodies.legged import legged_body, standalone_model
from rrp.core.provenance import physics_provenance
from rrp.bodies.actuator import SOURCED

pytestmark = pytest.mark.menagerie      # every test builds Menagerie legged bodies


@pytest.mark.parametrize("body", ["t1", "g1", "h1", "go2", "anymal_c"])
def test_default_is_sourced(body):
    warnings.filterwarnings("ignore")
    mod = legged_body(body)
    assert mod.meta["actuator_limits"] == "sourced_v1"
    m, _, meta = standalone_model(mod, contact="v2")
    for a in meta["actuator_adapter"]["actuators"]:
        src = next((e for pat, e, _ in SOURCED[body] if re.search(pat, a["joint"])), None)
        if src is None:
            continue
        uid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "r0_" + a["actuator"])
        assert m.actuator_forcerange[uid, 1] == pytest.approx(src)
    assert physics_provenance(m).actuator_limits == "sourced_v1"


def test_legacy_option_and_t1_change():
    warnings.filterwarnings("ignore")
    leg = {a["joint"]: a["effort"] for a in legged_body("t1", limits="legacy_gains_v0").meta["actuator_adapter"]["actuators"]}
    new = {a["joint"]: a["effort"] for a in legged_body("t1").meta["actuator_adapter"]["actuators"]}
    assert leg["Left_Knee_Pitch"] == 130.0 and new["Left_Knee_Pitch"] == 60.0
    m, _, _ = standalone_model(legged_body("t1", limits="legacy_gains_v0"), contact="v2")
    assert physics_provenance(m).actuator_limits == "legacy_gains_v0"
