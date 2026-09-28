"""SCRIPTED TEACHER STUB for the loco_pick task (D-126 roadmap #34; rrp.envs.legged_scenes). DEFAULT OFF.

Label: source = "scripted_teacher", teacher_variant = "loco_pick_stub_v0", stub = True. It is NOT a working demonstrator:
* phase `walk_to_standoff` IS implemented: base_velocity commands (same law as rrp.teachers.legged.WaypointTeacher)
  to the standoff point in front of the table, then an in-place turn to face the table;
* phases `reach` / `grasp` / `lift` are NOT implemented. The legged session accepts only the `base_velocity` group
  (the arm actuators are held at their default pose by the locomotion tracker), so there is no arm command interface
  yet. When the walk phase ends the stub returns a zero base command, sets `done = True` and
  `outcome = "stub_manipulation_not_implemented"`; `arm_command()` raises NotImplementedError. A run with this stub can
  therefore never succeed at `pick`; its episodes must be reported as STUB, never as teacher successes.

PRIVILEGED inputs (teacher only, never in a deployable observation): the true base pose
(`session.base_pose_truth()`) and the scenario's standoff pose / table yaw (`scenario.meta["standoff"]`, computed from
the true table pose at build time).
"""
from __future__ import annotations

import math

import numpy as np

from rrp.contracts.action import NativeCommand

LOCO_PICK_TEACHER_VERSION = "loco_pick_stub_v0"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class LocoPickTeacherStub:
    source = "scripted_teacher"
    privileged = True
    stub = True
    version = LOCO_PICK_TEACHER_VERSION

    def __init__(self, session, speed_frac: float = 0.5, turn_gain: float = 1.5, pos_tol: float = 0.12,
                 yaw_tol: float = 0.15):
        self.s = session
        L = session.robots[0].meta["legged"]
        self.r = L["command_ranges"]
        self.vmax = speed_frac * self.r["vx"][1]
        self.wmax = 0.8 * self.r["wz"][1]
        self.k, self.pos_tol, self.yaw_tol = turn_gain, pos_tol, yaw_tol
        so = session.scenario.meta["standoff"]           # PRIVILEGED (from the true table pose)
        self.goal = np.array(so["xy"], float)
        self.goal_yaw = float(so["yaw"])
        self.phase = "walk_to_standoff"
        self.done = False
        self.outcome = None

    def meta(self) -> dict:
        return dict(source=self.source, teacher_variant=self.version, stub=True, privileged_teacher=True,
                    phase=self.phase, outcome=self.outcome)

    def command_values(self) -> np.ndarray:
        if self.done:
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()          # PRIVILEGED
        d = self.goal - np.array([x, y])
        dist = float(np.linalg.norm(d))
        if self.phase == "walk_to_standoff":
            if dist <= self.pos_tol:
                self.phase = "face_table"
            else:
                err = _wrap(math.atan2(d[1], d[0]) - yaw)
                wz = float(np.clip(self.k * err, -self.wmax, self.wmax))
                vx = 0.0 if abs(err) > 1.0 else self.vmax * max(0.0, math.cos(err)) ** 2 * min(1.0, dist / 0.6 + 0.3)
                return np.array([vx, 0.0, wz])
        if self.phase == "face_table":
            err = _wrap(self.goal_yaw - yaw)
            if abs(err) > self.yaw_tol:
                return np.array([0.0, 0.0, float(np.clip(self.k * err, -self.wmax, self.wmax))])
            self.phase = "reach"
        # reach / grasp / lift: NOT IMPLEMENTED (no arm command interface on the legged session yet)
        self.done = True
        self.outcome = "stub_manipulation_not_implemented"
        return np.zeros(3)

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(),
                             groups={"base_velocity": self.command_values().tolist()}, source="scripted_teacher")

    def arm_command(self, *_a, **_k):
        raise NotImplementedError("loco_pick arm reach/grasp/lift is a STUB (D-126 #34): the legged session exposes only "
                                  "base_velocity; an arm command group + a grasp teacher are future work")
