"""Workbench/experiment robot catalogue: fixed keys -> constructors (no user-supplied paths)."""
from __future__ import annotations

from functools import partial

from .fixtures import arm_with_port, gripper_module, three_finger_module
from .surgery import attach


def _fixture(gripper: str, **arm):
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(arm_with_port(**arm), g, port_id="wrist")


def _menagerie(key: str, gripper: str):
    from .importers import menagerie_arm
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(menagerie_arm(key), g, port_id="wrist")


def workbench_robots() -> dict:
    robots = {
        "parm5_pg2": partial(_fixture, "parallel"),
        "parm5_tf3": partial(_fixture, "three_finger"),
    }
    from .importers import ARMS, MENAGERIE
    if MENAGERIE.exists():
        for key in ARMS:
            for g, gname in (("parallel", "pg2"), ("three_finger", "tf3")):
                robots[f"{key}_{gname}"] = partial(_menagerie, key, g)
    try:
        from .variants import registered_variants
        robots.update(registered_variants())
    except ImportError:
        pass
    return robots
