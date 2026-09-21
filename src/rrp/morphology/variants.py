"""Registered procedural arm variants (all share the synthetic procedural_arm_family lineage)."""
from __future__ import annotations

from functools import partial

from .generators import ArmParams, procedural_arm
from .fixtures import gripper_module, three_finger_module
from .surgery import attach

VARIANTS = {
    "parm5s": dict(axes=("z", "y", "y", "y", "z"), lengths=(0.12, 0.30, 0.27, 0.10, 0.06)),
    "parm5l": dict(axes=("z", "y", "y", "y", "z"), lengths=(0.14, 0.38, 0.33, 0.10, 0.06)),
    "parm6": dict(axes=("z", "y", "y", "z", "y", "z"), lengths=(0.12, 0.34, 0.30, 0.06, 0.08, 0.06)),
    "parm7": dict(axes=("z", "y", "z", "y", "z", "y", "z"), lengths=(0.12, 0.20, 0.16, 0.18, 0.14, 0.08, 0.06)),
}


def _build(name, gripper):
    p = ArmParams(name=name, **VARIANTS[name])
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(procedural_arm(p), g, port_id="wrist")


def registered_variants() -> dict:
    out = {}
    for n in VARIANTS:
        out[f"{n}_pg2"] = partial(_build, n, "parallel")
        out[f"{n}_tf3"] = partial(_build, n, "three_finger")
    return out
