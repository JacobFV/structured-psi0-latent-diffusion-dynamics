"""Workbench/experiment robot catalogue: fixed keys -> constructors (no user-supplied paths)."""
from __future__ import annotations

from functools import partial

from .fixtures import arm_with_port, gripper_module, three_finger_module
from .surgery import attach


def _fixture(gripper: str, **arm):
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(arm_with_port(**arm), g, port_id="wrist")


def workbench_robots() -> dict:
    robots = {
        "parm5_pg2": partial(_fixture, "parallel"),
        "parm5_tf3": partial(_fixture, "three_finger"),
    }
    try:
        from .variants import registered_variants
        robots.update(registered_variants())
    except ImportError:
        pass
    return robots
