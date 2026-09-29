"""armdiv (D-137) workbench keys: procedural v2 arms `pa2s<seed>_<pg2|tf3>` and the additional menagerie arms
`<gen3|iiwa14|...>_<pg2|tf3>` listed in research/splits/armdiv_candidates_v1.json. Additive: no existing key changes.
The home pose of a v2 arm is solved once per seed per process (IK, deterministic) and cached."""
from __future__ import annotations

import json
from functools import lru_cache, partial

from rrp.core.paths import rrp_home

CANDIDATES = "research/splits/armdiv_candidates_v1.json"


@lru_cache(maxsize=None)
def _v2_params(seed: int):
    from rrp.bodies.generators_v2 import sample_arm_v2, solve_home
    p = sample_arm_v2(seed)
    p.home = tuple(solve_home(p))
    return p


def _gripper(g: str):
    from rrp.bodies.fixtures import gripper_module, three_finger_module
    return gripper_module() if g == "pg2" else three_finger_module()


def build_v2(seed: int, gripper: str):
    import copy
    from rrp.bodies.generators_v2 import procedural_arm_v2
    from rrp.bodies.surgery import attach
    return attach(procedural_arm_v2(copy.deepcopy(_v2_params(int(seed)))), _gripper(gripper), port_id="wrist")


def build_menagerie_v2(key: str, gripper: str):
    from rrp.bodies.importers import menagerie_arm
    from rrp.bodies.surgery import attach
    return attach(menagerie_arm(key), _gripper(gripper), port_id="wrist")


def candidates() -> dict:
    p = rrp_home() / CANDIDATES
    return json.loads(p.read_text()) if p.exists() else {}


def armdiv_robots() -> dict:
    c = candidates()
    if not c:
        return {}
    out = {}
    for g in c.get("grippers", ["pg2", "tf3"]):
        for s in list(c.get("procedural_train_seeds", [])) + list(c.get("procedural_sealed_seeds", [])):
            out[f"pa2s{s}_{g}"] = partial(build_v2, int(s), g)
        from rrp.bodies.importers import MENAGERIE
        if MENAGERIE.exists():
            for k in c.get("menagerie_v2", []):
                out[f"{k}_{g}"] = partial(build_menagerie_v2, k, g)
    return out


# D-137 sealed targets (research/splits/armdiv_v1.json). Whole target FAMILIES are protected: every key of the kinova /
# flexiv / kuka arms and every sealed-range procedural seed (>= 900000) is refused wherever the D-025 target guard applies
# (ladder dev evaluations, DAgger collections, causal-edit and GRPO development), like xarm7_* / panda_tf3.
ARMDIV_TARGETS = ("gen3_pg2", "rizon4_tf3", "pa2s900002_pg2", "pa2s900003_tf3")
_TARGET_FAMILY_PREFIXES = ("gen3_", "rizon4_", "iiwa14_")


def is_armdiv_sealed(key: str) -> bool:
    if key in ARMDIV_TARGETS or key.startswith(_TARGET_FAMILY_PREFIXES):
        return True
    if key.startswith("pa2s"):
        try:
            return int(key[4:].split("_")[0]) >= 900000
        except ValueError:
            return False
    return False
