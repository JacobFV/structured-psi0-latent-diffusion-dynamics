"""Workbench/experiment robot catalogue: fixed keys -> constructors (no user-supplied paths)."""
from __future__ import annotations

from functools import partial

from rrp.bodies.fixtures import arm_with_port, gripper_module, three_finger_module
from rrp.bodies.surgery import attach


def _fixture(gripper: str, **arm):
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(arm_with_port(**arm), g, port_id="wrist")


def _menagerie(key: str, gripper: str):
    from rrp.bodies.importers import menagerie_arm
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(menagerie_arm(key), g, port_id="wrist")


# W11 extension hook: robots from outside rrp (register_robot, or entry points in the group "rrp.robots" whose target
# is a module that calls register_robot on import, or a zero-argument callable returning {key: factory}).
ROBOT_ENTRY_POINT_GROUP = "rrp.robots"
_EXTERNAL_ROBOTS: dict = {}
_ROBOT_PLUGINS_LOADED = False


def register_robot(key: str, factory) -> None:
    """Add a workbench robot `key -> factory()` (factory returns an assembled robot, like the built-ins).
    Built-in keys cannot be replaced; re-registering the same factory is a no-op."""
    if key in _EXTERNAL_ROBOTS and _EXTERNAL_ROBOTS[key] is not factory:
        raise ValueError(f"robot {key!r} is already registered")
    _EXTERNAL_ROBOTS[key] = factory


def unregister_robot(key: str) -> None:
    _EXTERNAL_ROBOTS.pop(key, None)


def _load_robot_plugins() -> None:
    global _ROBOT_PLUGINS_LOADED
    if _ROBOT_PLUGINS_LOADED:
        return
    _ROBOT_PLUGINS_LOADED = True
    from importlib.metadata import entry_points
    for ep in entry_points(group=ROBOT_ENTRY_POINT_GROUP):
        obj = ep.load()                                  # a broken plugin raises (no silent skip)
        if callable(obj):
            for k, f in (obj() or {}).items():
                register_robot(k, f)


def workbench_robots() -> dict:
    robots = {
        "parm5_pg2": partial(_fixture, "parallel"),
        "parm5_tf3": partial(_fixture, "three_finger"),
    }
    from rrp.bodies.importers import ARMS, MENAGERIE
    if MENAGERIE.exists():
        for key in ARMS:
            for g, gname in (("parallel", "pg2"), ("three_finger", "tf3")):
                robots[f"{key}_{gname}"] = partial(_menagerie, key, g)
    try:
        from rrp.bodies.variants import registered_variants
        robots.update(registered_variants())
    except ImportError:
        pass
    from rrp.bodies.armdiv import armdiv_robots          # D-137: additive keys (pa2s*, gen3_*, ...)
    for k, f in armdiv_robots().items():
        robots.setdefault(k, f)
    _load_robot_plugins()
    clash = sorted(set(_EXTERNAL_ROBOTS) & set(robots))
    if clash:
        raise ValueError(f"external robots shadow built-in keys: {clash}")
    robots.update(_EXTERNAL_ROBOTS)
    return robots
