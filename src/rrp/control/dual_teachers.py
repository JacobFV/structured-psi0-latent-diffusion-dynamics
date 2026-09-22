"""Scripted PRIVILEGED dual-arm teachers (label: scripted_teacher) for support_insert and handover.

Privileged inputs: simulator object poses (peg/bar/fixture) and the true in-hand offset.
Public inputs deliberately consumed: task-runtime statuses and RECEIPTS. In particular the
support_insert teacher places the peg using the `locate#k.hole_frame` receipt bound to the
align/insert events, not the true hole pose -- so changing the locate output changes the
downstream align/insert commands (data-dependent functional composition; see
research/reports/functional_composition.md). Demonstrations only, never model results.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import NativeCommand
from rrp.control.ik import down_rotation

SOURCE = "scripted_teacher"
GRASP_YAW = math.pi / 2          # palm long axis along world x: keeps the two hands apart along y


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ArmMover:
    """TCP-waypoint follower for one manipulator handle: rate-limited TCP command, integral
    correction of steady-state error (FK of measured joints), DLS IK with azimuth seeds."""

    def __init__(self, session, ent: str, speed: float = 0.3):
        self.s, self.ent = session, ent
        self.h = session.handles[ent]
        self.r = session.robots[self.h.robot]
        self.q_arm = self.r.controller.current_targets(session.data)[self.h.arm_group].copy()
        self.grip = self.h.open_value
        self.speed = speed
        self.goal = None
        self.yaw = GRASP_YAW
        self.tcp_cmd = None
        self.ioff = np.zeros(3)
        self.last_tcp = None
        self.t_goal = 0.0
        self.err = 1.0
        self.n = 1.0
        self.stalled = False
        self.max_ik_err = 0.0
        self.integral = True

    def closed_for(self, obj_half: float) -> float:
        h = self.h
        if h.gripper_kind == "three_finger":
            gp = h.gripper_params
            reach = (gp.get("finger_base_radius", 0.052) - obj_half) / gp.get("finger_length", 0.055)
            theta = float(np.arcsin(np.clip(reach, 0.0, 1.0)))
            return min(h.closed_value, theta + 0.1)
        return h.closed_value

    def set_goal(self, goal, speed: float | None = None, yaw: float | None = None):
        goal = np.asarray(goal, float)
        if self.goal is None or np.linalg.norm(goal - self.goal) > 1e-4:
            self.goal = goal
            self.t_goal = 0.0
        if speed is not None:
            self.speed = speed
        if yaw is not None:
            self.yaw = yaw

    def tcp(self):
        return self.s._fk_site(self.r, self.h.tcp_site)

    def step(self) -> dict:
        dt = self.s.dt
        self.t_goal += dt
        tcp_now, _ = self.tcp()
        if self.tcp_cmd is None:
            self.tcp_cmd = tcp_now.copy()
        goal_true = self.goal if self.goal is not None else tcp_now
        err_vec = goal_true - tcp_now
        if self.integral and np.linalg.norm(self.tcp_cmd - (goal_true + self.ioff)) < 1e-6 and \
                np.linalg.norm(err_vec) < 0.04:
            self.ioff = np.clip(self.ioff + 0.3 * err_vec, -0.04, 0.04)
        goal = goal_true + self.ioff
        d = goal - self.tcp_cmd
        n = float(np.linalg.norm(d))
        step = self.speed * dt
        self.tcp_cmd = goal.copy() if n <= step else self.tcp_cmd + d / n * step
        self.n = n
        self.err = float(np.linalg.norm(err_vec))
        v = np.linalg.norm(tcp_now - self.last_tcp) / dt if self.last_tcp is not None else 1.0
        self.last_tcp = tcp_now
        self.stalled = n < step and v < 0.01 and self.t_goal > 1.0
        q, e = self.h.ik.solve(self.s.data.qpos.copy(), self.q_arm, self.tcp_cmd, down_rotation(self.yaw),
                               seeds=self.seeds(self.tcp_cmd))
        self.max_ik_err = max(self.max_ik_err, float(e))
        self.q_arm = q
        return {self.h.arm_group: q.tolist(), self.h.grip_group: [float(self.grip)]}

    def reached(self, tol: float) -> bool:
        return self.n <= self.speed * self.s.dt + 1e-9 and (self.err < tol or (self.stalled and self.err < 2 * tol))

    def seeds(self, target):
        h = self.h
        home = np.array(h.home) if h.home else self.q_arm.copy()
        az = math.atan2(target[1] - h.base_pos[1], target[0] - h.base_pos[0])
        dyaw = _wrap_pi(az - h.home_azimuth)
        out = []
        for flip in (0.0, 0.3, -0.3):
            s = home.copy()
            s[0] = home[0] + dyaw + flip
            out.append(s)
        return out

    def ik_error(self, target, yaw=GRASP_YAW) -> float:
        q, e = self.h.ik.solve(self.s.data.qpos.copy(), self.q_arm, np.asarray(target, float), down_rotation(yaw),
                               iters=120, seeds=self.seeds(target))
        return float(e)

    _STATE = ("q_arm", "grip", "speed", "goal", "yaw", "tcp_cmd", "ioff", "last_tcp", "t_goal", "err", "n", "stalled")

    def state(self):
        import copy
        return {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}

    def load(self, st):
        import copy
        for k in self._STATE:
            setattr(self, k, copy.deepcopy(st[k]))


class DualTeacherBase:
    phase_names: tuple = ()

    def __init__(self, session, speed: float = 0.3):
        self.s = session
        self.arms = {e: ArmMover(session, e, speed) for e in ("left", "right")}
        self.phase = {"left": "start", "right": "start"}
        self.t_phase = {"left": 0.0, "right": 0.0}
        self.log: list[dict] = []

    def _body(self, name):
        m, d = self.s.model, self.s.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()

    def _next(self, ent, p):
        self.log.append(dict(t=float(self.s.data.time), arm=ent, frm=self.phase[ent], to=p))
        self.phase[ent], self.t_phase[ent] = p, 0.0

    def status(self, ev):
        return self.s.runtime.status(ev)

    def act(self) -> dict:
        for e in self.t_phase:
            self.t_phase[e] += self.s.dt
        self._plan()
        per: dict[int, dict] = {}
        for e, a in self.arms.items():
            per.setdefault(a.h.robot, {}).update(a.step())
        return {i: NativeCommand(controller_version=self.s.robots[i].controller.version, groups=g, source=SOURCE)
                for i, g in per.items()}

    @property
    def phase_label(self) -> str:
        return f"L:{self.phase['left']}|R:{self.phase['right']}"


class SupportInsertTeacher(DualTeacherBase):
    """left: establish + maintain support on the fixture; right: acquire peg, align with the
    LOCATE RECEIPT hole frame, insert, release. Waits on public runtime statuses."""

    HOVER = 0.10
    ALIGN_HOVER = 0.012         # peg bottom above the hole top during align
    INSERT_DEPTH = 0.032

    def __init__(self, session, speed: float = 0.3, grasp_depth: float = 0.02, frame_override=None):
        super().__init__(session, speed)
        self.grasp_depth = grasp_depth
        self.frame_override = frame_override    # diagnostic hook: replace the bound hole frame
        g = session.scenario.meta["declared_geometry"]
        self.peg_half = g["peg"]["half_length"]
        self.peg_r = g["peg"]["radius"]
        self.grasp_xyz = None
        self.contact_z = None
        self.contact_xy = None
        self.hole_frame_used = None

    # privileged helpers
    def _support_point(self):
        fp, fR = self._body("fixture")
        lay = self.s.scenario.meta["privileged_layout"]
        so = lay["support_offset"]
        return fp + fR @ np.array([so[0], so[1], self.s.scenario.meta["declared_geometry"]["fixture"]["height"]])

    def _peg_bottom_in_tcp(self):
        """privileged: true peg-bottom offset from the right TCP (world frame vector)."""
        p, R = self._body("peg")
        tcp, _ = self.arms["right"].tcp()
        return p - R[:, 2] * self.peg_half - tcp

    def _touchdown(self, L):
        """Freeze the support contact where it happened: command = measured TCP (no integral)."""
        tcp, _ = L.tcp()
        self.contact_z, self.contact_xy = float(tcp[2]), tcp[:2].copy()
        L.integral = False
        L.ioff = np.zeros(3)
        L.tcp_cmd = tcp.copy()
        self._next("left", "l_hold")

    def bound_hole_frame(self):
        if self.frame_override is not None:
            return self.frame_override
        fr = self.s._bound_frame("hole")
        if fr is None:
            return None
        return fr[0], fr[1][:, 2], dict(event=fr[2].event_id, attempt=fr[2].attempt, version=fr[2].version)

    def _plan(self):
        L, R = self.arms["left"], self.arms["right"]
        pl, pr = self.phase["left"], self.phase["right"]
        # ---------------- left arm: support
        sp = self._support_point()
        L.yaw = GRASP_YAW
        L.grip = L.h.closed_value
        if pl == "start":
            self._next("left", "l_pre")
        elif pl == "l_pre":
            L.set_goal(sp + [0, 0, self.HOVER], 0.3)
            if L.reached(0.006):
                self._next("left", "l_descend")
        elif pl == "l_descend":
            tcp_z = float(L.tcp()[0][2])
            L.set_goal(sp + [0, 0, -0.03], 0.12 if tcp_z > sp[2] + 0.04 else 0.03)   # slow near contact; stops on touch
            t = self.s.touch_values("left")
            if len(t) and t.max() > 0.5:
                self._touchdown(L)
            elif L.reached(0.005):
                self._touchdown(L)
        elif pl == "l_hold":
            # contact point frozen at touch-down; no integral action while pressing (the commanded
            # depth is intentionally unreachable -> bounded pressing force, no lateral drift)
            L.integral = False
            L.set_goal(np.r_[self.contact_xy, self.contact_z - 0.006], 0.02)
            if self.status("insert") == "succeeded" or (self.phase["right"] == "r_done"):
                self._next("left", "l_retreat")
        elif pl == "l_retreat":
            L.integral = True
            L.set_goal(np.r_[self.contact_xy, sp[2] + self.HOVER], 0.2)
            if L.reached(0.02):
                self._next("left", "l_done")
        # ---------------- right arm: acquire, align, insert
        peg, pR = self._body("peg")
        top = peg + pR[:, 2] * self.peg_half
        close_v = R.closed_for(self.peg_r)
        if pr == "start":
            self._next("right", "r_pre")
        elif pr == "r_pre":
            R.grip = R.h.open_value
            R.set_goal(np.r_[top[:2], top[2] + self.HOVER], 0.3)
            if R.reached(0.012):
                self._next("right", "r_descend")
        elif pr == "r_descend":
            R.set_goal(np.r_[top[:2], top[2] - self.grasp_depth], 0.15)
            if R.reached(0.006):
                self.grasp_xyz = R.goal.copy()
                self._next("right", "r_close")
        elif pr == "r_close":
            R.set_goal(self.grasp_xyz, 0.1)
            R.grip = close_v
            if self.t_phase["right"] > 0.8:
                self._next("right", "r_lift")
        elif pr == "r_lift":
            R.set_goal(np.r_[self.grasp_xyz[:2], 0.25], 0.25)
            if R.reached(0.02):
                self._next("right", "r_wait")
        elif pr in ("r_wait", "r_transit", "r_align"):
            fr = self.bound_hole_frame()
            if fr is None or self.status("align") not in ("active", "succeeded") and pr == "r_wait":
                R.set_goal(np.r_[self.grasp_xyz[:2] if pr == "r_wait" else R.goal[:2], 0.25], 0.2)
                return
            hp, n_up, info = fr
            self.hole_frame_used = dict(pos=[float(x) for x in hp], **info)
            off = self._peg_bottom_in_tcp()
            if pr == "r_wait":
                self._next("right", "r_transit")
            elif pr == "r_transit":
                R.set_goal(hp + n_up * 0.06 - off, 0.25)
                if R.reached(0.01):
                    self._next("right", "r_align")
            else:
                R.set_goal(hp + n_up * self.ALIGN_HOVER - off, 0.05)
                if self.status("insert") in ("active", "succeeded"):
                    self._ins_off = off           # freeze the in-hand offset for the insertion stroke
                    self._next("right", "r_insert")
        elif pr == "r_insert":
            fr = self.bound_hole_frame()
            if fr is not None:
                hp, n_up, info = fr
                R.set_goal(hp - n_up * self.INSERT_DEPTH - self._ins_off, 0.03)
            if self.status("insert") == "succeeded":
                self._next("right", "r_open")
            elif self.status("insert") == "failed":
                self._next("right", "r_abort")
        elif pr == "r_open":
            R.grip = R.h.open_value
            if self.t_phase["right"] > 0.6:
                self._next("right", "r_retreat")
        elif pr == "r_retreat":
            tcp, _ = R.tcp()
            R.set_goal(np.r_[tcp[:2] if R.goal is None else R.goal[:2], 0.22], 0.2)
            if R.reached(0.02):
                self._next("right", "r_done")
        elif pr == "r_abort":
            R.set_goal(np.r_[R.goal[:2], 0.25], 0.1)

    def feasibility(self, tol: float = 0.012) -> dict:
        """Analytic IK check of the key waypoints BEFORE execution (true layout, tool down)."""
        sp = self._support_point()
        peg, pR = self._body("peg")
        top = peg + pR[:, 2] * self.peg_half
        hole = np.array(self.s.scenario.meta["privileged_layout"]["hole_world"])
        dz = self.grasp_depth + self.peg_half * 2 - self.grasp_depth   # tcp above peg bottom when held
        wps = {"left": {"support_hover": sp + [0, 0, self.HOVER], "support": sp + [0, 0, 0.005]},
               "right": {"pregrasp": np.r_[top[:2], top[2] + self.HOVER], "grasp": np.r_[top[:2], top[2] - self.grasp_depth],
                         "transit": hole + [0, 0, 0.06 + dz - self.grasp_depth],
                         "inserted": hole + [0, 0, -self.INSERT_DEPTH + 2 * self.peg_half - self.grasp_depth]}}
        errs, bad = {}, []
        for ent, w in wps.items():
            for k, p in w.items():
                e = self.arms[ent].ik_error(p)
                errs[f"{ent}.{k}"] = e
                if e > tol:
                    bad.append(f"{ent}.{k}")
        return {"feasible": not bad, "ik_errors": errs, "unreachable": bad}

    @property
    def done(self):
        return self.phase["right"] in ("r_done", "r_abort") and self.phase["left"] in ("l_done", "l_hold") and \
            (self.phase["right"] == "r_abort" or self.phase["left"] == "l_done")


class HandoverTeacher(DualTeacherBase):
    """left (giver) grasps the bar at one end and presents it; right (receiver) grasps the
    other end while the giver still holds (overlapping held_by states); giver releases only
    when the runtime's `release` event is active; receiver places on the target zone."""

    HANDOVER_POINT = np.array([0.40, 0.0, 0.20])
    GRASP_DZ = 0.006

    def __init__(self, session, speed: float = 0.3):
        super().__init__(session, speed)
        g = session.scenario.meta["declared_geometry"]["bar"]
        self.half = np.array(g["half_extents"])
        self.d = g["grasp_offset"]
        self.gl = None
        self.gr = None
        self.off_l = None
        self.off_r = None

    def _bar(self):
        c, R = self._body("bar")
        a = R[:, 0]
        if a[1] < 0:      # axis pointing toward the giver (+y)
            a = -a
        return c, R, a

    def _yaw_along(self, a):
        return float(math.atan2(a[1], a[0]))

    def _plan(self):
        L, R = self.arms["left"], self.arms["right"]
        pl, pr = self.phase["left"], self.phase["right"]
        c, Rb, a = self._bar()
        yaw_bar = _wrap_pi(self._yaw_along(a))
        if yaw_bar > math.pi / 2:
            yaw_bar -= math.pi
        elif yaw_bar < -math.pi / 2:
            yaw_bar += math.pi
        h = self.half[2]
        # ---------------- giver
        if pl == "start":
            L.grip = L.h.open_value
            self._next("left", "l_pre")
        elif pl == "l_pre":
            gpt = c + a * self.d
            L.set_goal(np.r_[gpt[:2], 0.12], 0.3, yaw_bar)
            if L.reached(0.012):
                self._next("left", "l_descend")
        elif pl == "l_descend":
            gpt = c + a * self.d
            L.set_goal(np.r_[gpt[:2], c[2] + self.GRASP_DZ], 0.12, yaw_bar)
            if L.reached(0.006):
                self.gl = L.goal.copy()
                self._next("left", "l_close")
        elif pl == "l_close":
            L.set_goal(self.gl, 0.1)
            L.grip = L.closed_for(self.half[1])
            if self.t_phase["left"] > 0.8:
                self._next("left", "l_lift")
        elif pl == "l_lift":
            L.set_goal(np.r_[self.gl[:2], 0.14], 0.2)
            if L.reached(0.02):
                tcp, Rt = L.tcp()
                self.off_l = Rt.T @ (c - tcp)            # privileged in-hand offset (bar centre in TCP frame)
                self._next("left", "l_present")
        elif pl == "l_present":
            from rrp.control.ik import down_rotation as _dr
            L.set_goal(self.HANDOVER_POINT - _dr(GRASP_YAW) @ self.off_l, 0.2, GRASP_YAW)
            if L.reached(0.012) and self.t_phase["left"] > 0.5:
                self._next("left", "l_hold")
        elif pl == "l_hold":
            if self.status("release") == "active":
                self._next("left", "l_open")
        elif pl == "l_open":
            L.grip = L.h.open_value
            if self.t_phase["left"] > 0.6:
                self._next("left", "l_retreat")
        elif pl == "l_retreat":
            L.set_goal(np.r_[L.goal[:2] + np.array([0.0, 0.08]), 0.30], 0.2)
            if L.reached(0.02):
                self._next("left", "l_done")
        # ---------------- receiver
        if pr == "start":
            R.grip = R.h.open_value
            self._next("right", "r_wait")
        elif pr == "r_wait":
            if self.status("receive") == "active" and pl == "l_hold":
                self._next("right", "r_pre")
        elif pr == "r_pre":
            gpt = c - a * self.d
            R.set_goal(np.r_[gpt[:2], c[2] + 0.08], 0.3, yaw_bar)
            if R.reached(0.012):
                self._next("right", "r_descend")
        elif pr == "r_descend":
            gpt = c - a * self.d
            R.set_goal(np.r_[gpt[:2], c[2] + self.GRASP_DZ], 0.1, yaw_bar)
            if R.reached(0.006):
                self.gr = R.goal.copy()
                self._next("right", "r_close")
        elif pr == "r_close":
            R.set_goal(self.gr, 0.1)
            R.grip = R.closed_for(self.half[1])
            # keep holding in place until the giver's release is confirmed by the runtime
            if self.t_phase["right"] > 0.8 and self.status("release") == "succeeded":
                tcp, Rt = R.tcp()
                self.off_r = c - tcp
                self._next("right", "r_transport")
        elif pr == "r_transport":
            zone = self._body("target_zone")[0]
            R.set_goal(np.r_[zone[:2] - self.off_r[:2], 0.16], 0.2)
            if R.reached(0.015):
                self._next("right", "r_lower")
        elif pr == "r_lower":
            zone = self._body("target_zone")[0]
            R.set_goal(np.r_[zone[:2] - self.off_r[:2], h + 0.012 - self.off_r[2]], 0.1)
            if R.reached(0.01):
                self._next("right", "r_open")
        elif pr == "r_open":
            R.grip = R.h.open_value
            if self.t_phase["right"] > 0.6:
                self._next("right", "r_retreat")
        elif pr == "r_retreat":
            R.set_goal(np.r_[R.goal[:2], 0.25], 0.2)
            if R.reached(0.02):
                self._next("right", "r_done")

    def feasibility(self, tol: float = 0.012) -> dict:
        c, Rb, a = self._bar()
        zone = self._body("target_zone")[0]
        wps = {"left": {"grasp": np.r_[(c + a * self.d)[:2], c[2]],
                        "present": self.HANDOVER_POINT + np.array([0, self.d, 0])},
               "right": {"receive": self.HANDOVER_POINT - np.array([0, self.d, 0]),
                         "place": np.r_[zone[:2] + np.array([0, -self.d]), self.half[2] * 2 + 0.012]}}
        errs, bad = {}, []
        for ent, w in wps.items():
            for k, p in w.items():
                e = self.arms[ent].ik_error(p)
                errs[f"{ent}.{k}"] = e
                if e > tol:
                    bad.append(f"{ent}.{k}")
        return {"feasible": not bad, "ik_errors": errs, "unreachable": bad}

    @property
    def done(self):
        return self.phase["right"] == "r_done" and self.phase["left"] == "l_done"


TEACHERS = {"support_insert": SupportInsertTeacher, "handover": HandoverTeacher}


@dataclass
class DualTeacherResult:
    success: bool = False                  # public runtime success
    privileged_evaluator_success: bool = False
    controller_source: str = SOURCE
    privileged_inputs: bool = True
    steps: int = 0
    failure_reason: str | None = None
    rejected_commands: int = 0
    feasibility: dict | None = None
    statuses: dict = field(default_factory=dict)
    transitions: list = field(default_factory=list)
    phase_log: list = field(default_factory=list)
    truth: dict = field(default_factory=dict)
    wall_s: float = 0.0


def run_dual_teacher_episode(session, teacher, max_control_steps: int = 1200, check_feasibility: bool = True,
                             on_step=None) -> DualTeacherResult:
    res = DualTeacherResult()
    t0 = time.time()
    if check_feasibility:
        f = teacher.feasibility()
        res.feasibility = f
        if not f["feasible"]:
            res.failure_reason = f"infeasible:{','.join(f['unreachable'])}"
            res.wall_s = time.time() - t0
            return res
    for k in range(max_control_steps):
        cmd = teacher.act()
        out = session.step(cmd)
        if on_step is not None:
            on_step(k, cmd, out)
        if out.rejected:
            res.rejected_commands += 1
        res.steps = k + 1
        if teacher.done:
            break
    for _ in range(5):
        session.step(None)
    res.success = bool(session.runtime.succeeded())
    res.privileged_evaluator_success = bool(session.privileged_success())
    res.statuses = {e: (v.status, v.attempt, v.reason) for e, v in session.runtime.instances.items()}
    res.transitions = list(session.runtime.transitions)
    res.phase_log = list(teacher.log)
    if hasattr(session, "insertion_truth") and session.scenario.name == "support_insert":
        res.truth = session.insertion_truth()
    if not res.privileged_evaluator_success:
        res.failure_reason = f"ended_in_phase:{teacher.phase_label}"
    res.wall_s = time.time() - t0
    return res
