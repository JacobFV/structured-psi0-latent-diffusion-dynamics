"""W13 humanoid task command layers (label scripted_command, PRIVILEGED truth) and expert wiring.

StepsHeadingTeacher (h_steps): walk +x at 0.6 vx_max, steering yaw and lateral offset to 0 -- the same law as the GPU expert
env (rrp.envs.warp_task_env.WarpStepsEnv._command). The joint-level expert is a LearnedTracker with the privileged height
scan attached (attach_steps_scan), label privileged_teacher:rl_expert:<sha>.
"""
from __future__ import annotations

import math

import numpy as np

from rrp.contracts.action import NativeCommand


class StepsHeadingTeacher:
    source = "scripted_command"
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
                             source="scripted_command")


def attach_steps_scan(tracker, session):
    """Give a steps expert its PRIVILEGED height scan (true base pose + the scenario's staircase)."""
    from rrp.envs.humanoid_scenes import steps_scan_np
    b = session.binding
    L, h = float(session.scenario.meta["L"]), float(session.scenario.meta["staircase"]["h"])
    tracker.extra_fn = lambda d: steps_scan_np(d.qpos[b.qa:b.qa + 7], L, h)
    return tracker
