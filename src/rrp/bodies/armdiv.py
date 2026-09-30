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


# D-137 sealed targets (research/splits/armdiv_v1.json). The ONE rule lives in `rrp.core.sealed.SealedSplit` ("armdiv_v1", hash
# pinned): every target robot, every key of a target / excluded family (kinova gen3, flexiv rizon4, kuka iiwa14, xarm7, lite6,
# fr3, panda_tf3) and every sealed-range procedural seed (pa2s<seed>, seed >= 900000). `is_sealed_target` is what every guard
# calls (ladder dev evaluations, DAgger collections, causal edits, GRPO development and its anchors, target_eval smoke).
def is_sealed_target(key: str) -> bool:
    from rrp.core.sealed import SealedSplit
    return SealedSplit.load("armdiv_v1").is_sealed_body(key)
