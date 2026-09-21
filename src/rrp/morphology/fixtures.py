"""Small real model records for tests and the first vertical slice."""
from __future__ import annotations

from .generators import ArmParams, GripperParams, procedural_arm, gripper_module as _gripper


def arm_with_port(**kw):
    return procedural_arm(ArmParams(**kw))


def gripper_module(**kw):
    return _gripper(GripperParams(**kw))


def three_finger_module(**kw):
    kw.setdefault("name", "tf3")
    return _gripper(GripperParams(kind="three_finger", **kw))
