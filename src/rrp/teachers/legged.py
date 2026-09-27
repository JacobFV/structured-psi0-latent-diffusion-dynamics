"""Scripted waypoint teacher for `waypoint_contact` (label: scripted_teacher, PRIVILEGED).

Uses the TRUE base pose and TRUE waypoint positions (privileged truth bus) to emit
base-velocity commands to the body's tracker: turn toward the active waypoint, walk forward
with speed shaped by heading error and distance, then halt (zero command) for the stance
event. It demonstrates the high-level command stream; it is not a model result.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from rrp.contracts.action import NativeCommand


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointTeacher:
    source = "scripted_teacher"
    privileged = True

    def __init__(self, session, speed_frac: float = 0.6, turn_gain: float = 1.5, arc_only: bool = False):
        """arc_only: for trackers validated only for walking/arc turns (no in-place turning), keep a
        minimum forward speed while turning (declared teacher variant, recorded in episode meta)."""
        self.arc_only = arc_only
        self.s = session
        L = session.robots[0].meta["legged"]
        self.r = L["command_ranges"]
        self.vmax = speed_frac * self.r["vx"][1]
        self.wmax = 0.8 * self.r["wz"][1]
        self.k = turn_gain
        self.phase = "init"
        self.done = False
        self.targets = {o.task_entity: o.sim_body for o in session.scenario.objects if o.task_entity}

    def feasibility(self) -> dict:
        return {"feasible": True, "reason": None}

    def _target_xy(self, entity):
        bid = mujoco.mj_name2id(self.s.model, mujoco.mjtObj.mjOBJ_BODY, self.targets[entity])
        return self.s.data.xpos[bid][:2].copy()     # PRIVILEGED

    def command_values(self) -> np.ndarray:
        rt = self.s.runtime
        st = {e: i.status for e, i in rt.instances.items()}
        if rt.succeeded() or any(v == "failed" for v in st.values()):
            self.done = True
            self.phase = "done"
            return np.zeros(3)
        if st.get("walk_to_a") in ("active", "pending", "ready"):
            ent = "waypoint_a"
            self.phase = "walk_to_a"
        elif st.get("walk_to_b") in ("active", "pending", "ready"):
            ent = "waypoint_b"
            self.phase = "walk_to_b"
        else:
            self.phase = "halt"
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()            # PRIVILEGED
        tx, ty = self._target_xy(ent)
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        err = wrap(math.atan2(dy, dx) - yaw)
        wz = float(np.clip(self.k * err, -self.wmax, self.wmax))
        vx = self.vmax * max(0.0, math.cos(err)) ** 2 * min(1.0, dist / 0.6 + 0.3)
        if self.arc_only:
            vx = max(vx, 0.5 * self.vmax)
        elif abs(err) > 1.0:
            vx = 0.0
        return np.array([vx, 0.0, wz])

    def act(self) -> NativeCommand:
        v = self.command_values()
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": v.tolist()},
                             source="scripted_teacher")
