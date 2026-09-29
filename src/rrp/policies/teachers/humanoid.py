"""W13 humanoid task command layers (label scripted_command, PRIVILEGED truth) and expert wiring.

StepsHeadingTeacher (h_steps): walk +x at 0.6 vx_max, steering yaw and lateral offset to 0 -- the same law as the GPU expert
env (rrp.envs.warp_task_env.WarpStepsEnv._command). The joint-level expert is a LearnedTracker with the privileged height
scan attached (attach_steps_scan), label privileged_teacher:rl_expert:<sha>.
"""
from __future__ import annotations

import math

import numpy as np

from rrp.core.action import NativeCommand


class StepsHeadingTeacher:
    source = "scripted_teacher"          # the scripted command layer (contract literal)
    privileged = True

    def __init__(self, session):
        self.s = session
        L = session.robots[0].meta["legged"]
        self.vx = 0.6 * L["command_ranges"]["vx"][1]
        self.L = float(session.scenario.meta["L"])
        self.done = False

    def command_values(self) -> np.ndarray:
        rt = self.s.runtime
        if rt.succeeded() or any(i.status == "failed" for i in rt.instances.values()):
            self.done = True
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()            # PRIVILEGED
        tgt = 0.5 * math.atan(-y / self.L)
        head = math.atan2(math.sin(tgt - yaw), math.cos(tgt - yaw))
        return np.array([self.vx, 0.0, float(np.clip(1.5 * head, -0.5, 0.5))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")


def attach_steps_scan(tracker, session):
    """Give a steps expert its PRIVILEGED height scan (true base pose + the scenario's staircase)."""
    from rrp.envs.mujoco.humanoid_scenes import steps_scan_np
    b = session.binding
    L, h = float(session.scenario.meta["L"]), float(session.scenario.meta["staircase"]["h"])
    tracker.extra_fn = lambda d: steps_scan_np(d.qpos[b.qa:b.qa + 7], L, h)
    return tracker


class GapTeacher:
    """h_gap_sidestep command layer (scripted_teacher, PRIVILEGED truth): the WarpGapEnv._command law. Phase 1: keep yaw 0,
    walk +x at 0.5 vx_max and sidestep onto the gap centre; phase 2 (0.5 L past the wall): stop and turn to psi_f."""
    source = "scripted_teacher"
    privileged = True

    def __init__(self, session):
        self.s = session
        self.r = session.robots[0].meta["legged"]["command_ranges"]
        self.m = session.scenario.meta
        self.phase2 = 0.0

    def command_values(self) -> np.ndarray:
        x, y, yaw = self.s.base_pose_truth()
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        if x > self.m["wall_x"] + 0.5 * self.m["L"]:
            self.phase2 = 1.0
        if self.phase2:
            e = wrap(self.m["psi_f"] - yaw)
            return np.array([0.0, 0.0, 0.0 if abs(e) < 0.1 else float(np.clip(1.5 * e, -0.5, 0.5))])
        vy = float(np.clip(1.5 * math.cos(yaw) * (self.m["y_c"] - y), self.r["vy"][0], self.r["vy"][1]))
        return np.array([0.5 * self.r["vx"][1], vy, float(np.clip(1.5 * wrap(-yaw), -0.5, 0.5))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")
