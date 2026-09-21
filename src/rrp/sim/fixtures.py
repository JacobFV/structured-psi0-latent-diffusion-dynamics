"""Fixture sessions built from real procedural models (no mocks)."""
from __future__ import annotations

from rrp.morphology.fixtures import arm_with_port, gripper_module, three_finger_module
from rrp.morphology.surgery import attach
from .native import Session
from .scenario import build_pick_place, build_reach


def fixture_robot(gripper: str = "parallel", **arm_kw):
    g = gripper_module() if gripper == "parallel" else three_finger_module()
    return attach(arm_with_port(**arm_kw), g, port_id="wrist")


def make_pick_place_session(seed: int = 0, gripper: str = "parallel", n_distractors: int = 0, **kw) -> Session:
    sc = build_pick_place(fixture_robot(gripper), seed, n_distractors=n_distractors)
    return Session(sc, seed=seed, **kw)


def make_arm_session(seed: int = 7, **kw) -> Session:
    return make_pick_place_session(seed=seed, **kw)


def make_reach_session(seed: int = 0, gripper: str = "parallel", **kw) -> Session:
    return Session(build_reach(fixture_robot(gripper), seed), seed=seed, **kw)
