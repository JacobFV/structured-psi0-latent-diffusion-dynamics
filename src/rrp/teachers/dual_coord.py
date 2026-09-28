"""Scripted teacher STUBS for the D-126 #22 coordination tasks (label: scripted_teacher, PRIVILEGED object poses).

STUB = True: these run the task's phase structure end to end but are NOT validated (no success-rate gate, no
feasibility sweep, no data). They exist so the scenes, task graphs, public estimators and quality recorder can be
exercised; a task becomes data-worthy only after a validation sweep like rrp.teachers.dual_validate plus the dual
dataset gate. Both use the v3 min-jerk mover (rrp.teachers.dual_smooth.SmoothArmMover).

pivot_against_surface: left braces the stop block (press on its top, as the support_insert support hand); right
    descends behind the box's free end at low height and pushes it toward the stop along an arc, so the box pivots
    about its bottom edge against the stop's face; holds at the target angle until the runtime reports success.
carry_tray_level: both hands descend onto the tray handles, close (contact-confirmed), lift TOGETHER on one shared
    min-jerk tray path (same waypoints for both TCPs, offset by the handle spacing, so the tray stays level), carry to
    the target, lower, open when `release` is active, retreat.
"""
from __future__ import annotations

import numpy as np

from rrp.teachers.dual import DualTeacherBase
from rrp.teachers.dual_smooth import SmoothArmMover, V3Options

STUB = True
SOURCE_DETAIL = "scripted_teacher:stub_unvalidated"


class _StubBase(DualTeacherBase):
    stub = STUB
    version = "stub"
    teacher_version = "dual_coord_stub_v0"

    def __init__(self, session, speed: float = 0.25):
        super().__init__(session, speed)
        self.o = V3Options()
        self.arms = {e: SmoothArmMover(session, e, speed, self.o) for e in ("left", "right")}
        self.limits = {}

    def feasibility(self, tol: float = 0.015) -> dict:
        errs, bad = {}, []
        for ent, pts in self._key_points().items():
            for k, p in pts.items():
                e = self.arms[ent].ik_error(p)
                errs[f"{ent}.{k}"] = e
                if e > tol:
                    bad.append(f"{ent}.{k}")
        return {"feasible": not bad, "ik_errors": errs, "unreachable": bad}

    def _touching(self, ent) -> bool:
        tv = self.s.touch_values(ent)
        return bool(len(tv) and (tv >= 0.2).sum() >= min(2, len(tv)))


class PivotTeacherStub(_StubBase):
    TARGET = 1.15            # rad, beyond the task's 1.0 completion threshold

    def _geom(self):
        g = self.s.scenario.meta["declared_geometry"]
        sp, sR = self._body("stop")
        bp, bR = self._body("box")
        return g, sp, bp

    def _key_points(self):
        g, sp, bp = self._geom()
        return {"left": {"brace": sp + [0, 0, g["stop"]["half_extents"][2] + 0.005]},
                "right": {"behind": np.r_[bp[0] - g["box"]["half_extents"][0] - 0.02, bp[1], 0.03]}}

    def _plan(self):
        L, R = self.arms["left"], self.arms["right"]
        pl, pr = self.phase["left"], self.phase["right"]
        g, sp, bp = self._geom()
        sh, bh = g["stop"]["half_extents"], g["box"]["half_extents"]
        L.grip = L.h.closed_value if L.h.gripper_kind != "aloha" else 0.012
        top = sp + np.array([0, 0, sh[2]])
        if pl == "start":
            self._next("left", "l_stage")
        elif pl == "l_stage":
            self._staging("left", "l_pre")
        elif pl == "l_pre":
            L.set_goal(top + [0, 0, 0.08], 0.25)
            if L.reached(0.008):
                self._next("left", "l_descend")
        elif pl == "l_descend":
            L.set_goal(top + [0, 0, -0.02], 0.04)
            if self._touching("left") or L.reached(0.004):
                L.hold_here()
                self._press = L.tcp()[0].copy()
                self._next("left", "l_hold")
        elif pl == "l_hold":
            L.integral = False
            L.set_goal(self._press - [0, 0, 0.006], 0.02)
            if self.s.runtime.succeeded() or pr == "r_done":
                self._next("left", "l_retreat")
        elif pl == "l_retreat":
            L.integral = True
            L.set_goal(self._press + [0, 0, 0.08], 0.15)
            if L.reached(0.02):
                self._next("left", "l_done")
        R.grip = R.h.closed_value if R.h.gripper_kind != "aloha" else 0.012
        # the pivot edge: the box's bottom edge at the stop face (declared geometry + initial layout)
        if not hasattr(self, "_edge"):
            self._edge = np.array([sp[0] - sh[0], bp[1], 0.0])
        Lb = 2 * bh[0]
        if pr == "start":
            self._next("right", "r_stage")
        elif pr == "r_stage":
            if pl in ("l_hold",):
                self._staging("right", "r_pre")
        elif pr == "r_pre":
            R.set_goal(np.r_[self._edge[0] - Lb - 0.025, bp[1], 0.10], 0.25)
            if R.reached(0.01):
                self._next("right", "r_descend")
        elif pr == "r_descend":
            R.set_goal(np.r_[self._edge[0] - Lb - 0.025, bp[1], bh[2] + 0.005], 0.08)
            if R.reached(0.006):
                self._th = 0.0
                self._next("right", "r_push")
        elif pr == "r_push":
            # the free end of the box follows a circle of radius Lb about the pivot edge; the fingertips push it
            self._th = min(self.TARGET, self._th + 0.25 * self.s.dt)
            th = self._th
            p = self._edge + np.array([-np.cos(th) * (Lb + 0.01), 0.0, np.sin(th) * (Lb + 0.01) + bh[2] * np.cos(th)])
            R.set_goal(p, 0.05)
            if self.s.runtime.succeeded() or (th >= self.TARGET and self.t_phase["right"] > 8.0):
                self._next("right", "r_retreat")
        elif pr == "r_retreat":
            tcp, _ = R.tcp()
            R.set_goal(np.r_[tcp[0] - 0.05, tcp[1], 0.15], 0.15)
            if R.reached(0.02):
                self._next("right", "r_done")

    @property
    def done(self):
        return self.phase["right"] == "r_done" and self.phase["left"] == "l_done"


class CarryTrayTeacherStub(_StubBase):
    LIFT_Z = 0.12

    def _handles(self):
        g = self.s.scenario.meta["declared_geometry"]["tray"]
        tp, tR = self._body("tray")
        yoff, hh = g["handle_offset_y"], g["handle_half"]
        top = g["half_extents"][2] + 2 * hh[2]
        return tp, {"left": tp + tR @ np.array([0, yoff, top - 0.012]), "right": tp + tR @ np.array([0, -yoff, top - 0.012])}

    def _key_points(self):
        tp, h = self._handles()
        tgt = np.array(self.s.scenario.meta["privileged_layout"]["target_xy"])
        return {e: {"grasp": h[e], "carry": np.r_[tgt + (h[e] - tp)[:2], self.LIFT_Z]} for e in ("left", "right")}

    def _plan(self):
        tp, h = self._handles()
        both = lambda p: all(self.phase[e] == p for e in ("left", "right"))
        for e in ("left", "right"):
            A = self.arms[e]
            ph = self.phase[e]
            if ph == "start":
                A.grip = A.open_value
                self._next(e, "stage")
            elif ph == "stage":
                self._staging(e, "pre")
            elif ph == "pre":
                A.set_goal(h[e] + [0, 0, 0.08], 0.25, 0.0)
                if A.reached(0.01):
                    self._next(e, "descend")
            elif ph == "descend":
                A.set_goal(h[e], 0.08)
                if A.reached(0.005):
                    self._g = getattr(self, "_g", {})
                    self._g[e] = A.goal.copy()
                    self._next(e, "close")
            elif ph == "close":
                A.grip = A.closed_for(0.012)
                if self._touching(e) and self.t_phase[e] > 0.6:
                    self._next(e, "ready")
            elif ph == "ready" and both("ready"):
                # one shared tray path: both TCPs get the same displacement (level carry)
                self._off = {x: self._g[x] - tp for x in ("left", "right")}
                self._base = tp.copy()
                for x in ("left", "right"):
                    self._next(x, "lift")
            elif ph == "lift":
                A.set_goal(np.r_[(self._base + self._off[e])[:2], self.LIFT_Z + self._off[e][2]], 0.08)
                if both("lift") and all(self.arms[x].reached(0.01) for x in ("left", "right")):
                    for x in ("left", "right"):
                        self._next(x, "carry")
            elif ph == "carry":
                tgt = np.array(self.s.scenario.meta["privileged_layout"]["target_xy"])
                A.set_goal(np.r_[tgt + self._off[e][:2], self.LIFT_Z + self._off[e][2]], 0.08)
                if both("carry") and all(self.arms[x].reached(0.01) for x in ("left", "right")):
                    for x in ("left", "right"):
                        self._next(x, "lower")
            elif ph == "lower":
                tgt = np.array(self.s.scenario.meta["privileged_layout"]["target_xy"])
                A.set_goal(np.r_[tgt + self._off[e][:2], self._g[e][2] + 0.004], 0.05)
                if both("lower") and all(self.arms[x].reached(0.006) for x in ("left", "right")) \
                        and self.status("release") == "active":
                    for x in ("left", "right"):
                        self._next(x, "open")
            elif ph == "open":
                A.grip = A.open_value
                if self.t_phase[e] > 0.6:
                    self._next(e, "retreat")
            elif ph == "retreat":
                A.set_goal(np.r_[A.goal[:2], 0.25], 0.15)
                if A.reached(0.02):
                    self._next(e, "done")

    @property
    def done(self):
        return all(self.phase[e] == "done" for e in ("left", "right"))


COORD_TEACHERS = {"pivot_against_surface": PivotTeacherStub, "carry_tray_level": CarryTrayTeacherStub}
