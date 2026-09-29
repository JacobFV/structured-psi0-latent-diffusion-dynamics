"""Scripted (privileged) teachers for arm, dual-arm, legged and humanoid tasks, and the Policy adapter over them.

Every teacher reads privileged session state, so its adapter declares `requires.privileged` and source
`scripted_teacher` (docs/architecture.md section 3). make_teacher(arg=<task>) is registered as "teacher:*".
"""
from __future__ import annotations

from rrp.policies.base import Act, PolicyInfo, Requirements

ARM_TASKS = ("pick_place",)
DUAL_TASKS = ("support_insert", "handover", "assign_left", "assign_right", "pivot_against_surface", "carry_tray_level")
LEGGED_TASKS = ("waypoint_contact", "loco_pick", "h_steps", "h_gap")


def _factory(task: str, version: str | None, options: dict | None):
    """-> (make(session) -> teacher, teacher version string, action kinds it emits)."""
    if task.startswith("cw/"):         # ComputerWorld teachers: rrp.policies.teachers.computerworld.teacher_factory
        from rrp.policies.teachers.computerworld import teacher_factory
        return teacher_factory(task, version, options)
    if task in ARM_TASKS:
        from rrp.policies.teachers.arm_smooth import DEFAULT_ARM_TEACHER, make_arm_teacher
        v = version or DEFAULT_ARM_TEACHER
        return (lambda s: make_arm_teacher(s, v, **(options or {}))), f"arm:{v}", ("joint_position", "gripper")
    if task in DUAL_TASKS:
        from rrp.policies.teachers.dual_smooth import DEFAULT_DUAL_TEACHER, make_dual_teacher
        v = version or DEFAULT_DUAL_TEACHER
        return (lambda s: make_dual_teacher(task, s, v, options)), f"dual:{v}", ("joint_position", "gripper")
    if task == "waypoint_contact":
        from rrp.policies.teachers.legged import WaypointTeacher
        return (lambda s: WaypointTeacher(s, **(options or {}))), "waypoint", ("base_velocity",)
    if task == "loco_pick":
        from rrp.policies.teachers.legged_loco import LOCO_PICK_TEACHER_VERSION, LocoPickTeacherStub
        return (lambda s: LocoPickTeacherStub(s, **(options or {}))), LOCO_PICK_TEACHER_VERSION, ("base_velocity",)
    if task in ("h_steps", "h_gap"):
        from rrp.policies.teachers.humanoid import GapTeacher, StepsHeadingTeacher
        cls = StepsHeadingTeacher if task == "h_steps" else GapTeacher
        return (lambda s: cls(s)), cls.__name__, ("base_velocity",)
    raise KeyError(f"no scripted teacher for task {task!r}")


class TeacherPolicy:
    """One scripted teacher per episode; act() = teacher.act() (NativeCommand, or {robot: NativeCommand} for dual)."""

    def __init__(self, task: str, version: str | None = None, options: dict | None = None):
        self.make, tv, kinds = _factory(task, version, options)
        self.info = PolicyInfo(f"teacher:{task}", "scripted_teacher", tv,
                               Requirements(frozenset(kinds), observations=frozenset(), tasks=frozenset({task}),
                                            privileged=True))
        self.teachers = []

    def reset(self, spec, task, seeds, *, envs=None):
        self.teachers = [self.make(e) for e in envs]

    def act(self, obs):
        return {i: Act(self.teachers[i].act()) for i in obs}


def make_teacher(*, arg: str, version: str | None = None, options: dict | None = None) -> TeacherPolicy:
    return TeacherPolicy(arg, version, options)
