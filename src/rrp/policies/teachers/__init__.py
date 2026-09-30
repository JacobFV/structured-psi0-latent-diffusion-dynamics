"""Scripted (privileged) teachers for arm, dual-arm, legged, humanoid and ComputerWorld tasks, and the Policy adapter
over them.

Every teacher reads privileged session state, so its adapter declares `requires.privileged` and source
`scripted_teacher` (docs/architecture.md section 3). Which task has which teacher is data: `TaskSpec.teacher` names a
POLICIES key ("teacher:<task>", rrp.policies.base) and each key names one of the factories below (or a factory of a
later unit's own module). A factory takes `arg=<task>` plus optional `version` / `options` and returns a TeacherPolicy.
"""
from __future__ import annotations

from typing import Callable

from rrp.policies.base import Act, PolicyInfo, Requirements


class TeacherPolicy:
    """One scripted teacher per episode; act() = teacher.act() (NativeCommand, or {robot: NativeCommand} for dual).
    `make(env)` builds the teacher of one episode; `kinds` are the action kinds it emits."""

    def __init__(self, task: str, make: Callable, version: str, kinds: tuple[str, ...]):
        self.make = make
        self.info = PolicyInfo(f"teacher:{task}", "scripted_teacher", version,
                               Requirements(frozenset(kinds), observations=frozenset(), tasks=frozenset({task}),
                                            privileged=True))
        self.teachers = []

    def reset(self, spec, task, seeds, *, envs=None):
        self.teachers = [self.make(e) for e in envs]

    def act(self, obs):
        return {i: Act(self.teachers[i].act()) for i in obs}


def make_arm_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    from rrp.policies.teachers.arm_smooth import DEFAULT_ARM_TEACHER, make_arm_teacher
    v = version or DEFAULT_ARM_TEACHER
    return TeacherPolicy(arg, lambda s: make_arm_teacher(s, v, **(options or {})), f"arm:{v}",
                         ("joint_position", "gripper"))


def make_dual_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    from rrp.policies.teachers.dual_smooth import DEFAULT_DUAL_TEACHER, make_dual_teacher
    v = version or DEFAULT_DUAL_TEACHER
    return TeacherPolicy(arg, lambda s: make_dual_teacher(arg, s, v, options), f"dual:{v}",
                         ("joint_position", "gripper"))


def make_waypoint_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    from rrp.policies.teachers.legged import WaypointTeacher
    return TeacherPolicy(arg, lambda s: WaypointTeacher(s, **(options or {})), "waypoint", ("base_velocity",))


def make_loco_pick_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None
                                  ) -> TeacherPolicy:
    from rrp.policies.teachers.legged_loco import LOCO_PICK_TEACHER_VERSION, LocoPickTeacherStub
    return TeacherPolicy(arg, lambda s: LocoPickTeacherStub(s, **(options or {})), LOCO_PICK_TEACHER_VERSION,
                         ("base_velocity",))


def make_steps_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    from rrp.policies.teachers.humanoid import StepsHeadingTeacher
    return TeacherPolicy(arg, lambda s: StepsHeadingTeacher(s), StepsHeadingTeacher.__name__, ("base_velocity",))


def make_gap_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    from rrp.policies.teachers.humanoid import GapTeacher
    return TeacherPolicy(arg, lambda s: GapTeacher(s), GapTeacher.__name__, ("base_velocity",))


def make_cw_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    """ComputerWorld teachers: rrp.policies.teachers.computerworld.teacher_factory."""
    from rrp.policies.teachers.computerworld import teacher_factory
    make, v, kinds = teacher_factory(arg, version, options)
    return TeacherPolicy(arg, make, v, kinds)
