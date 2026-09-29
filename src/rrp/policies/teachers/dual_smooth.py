"""Dual-arm scripted teacher v3 (W12 / D-126 #18): NEW version, default OFF (v2 = rrp.teachers.dual stays the default).
Label: scripted_teacher (PRIVILEGED: object poses and in-hand offsets, as v2). Diagnosis: research/tracks/w12.md §7.

Changes vs v2, each ablatable through `V3Options` (all True = full v3; all False = v2's task logic on the v3 mover):
  minjerk        TCP motion = superposition of minimum-jerk Cartesian submovements (C2 blended corners, as arm_smooth v2);
                 residual steady-state error corrected by short min-jerk correction moves (replaces v2's integral steps);
                 tool orientation and gripper opening follow min-jerk profiles (replaces constant-rate slews).
  limit_aware    after each DLS IK solve, a null-space step moves joints away from their limits (keeps the TCP pose).
  confirm_grasp  close -> wait for touch on >= 2 sensors (public) -> settle -> lift slowly; no contact -> regrasp.
  deep_grasp     support_insert peg grasp depth GRASP_DEPTH_V3 (parallel 3 cm under the top; v2: 2 cm).
  grip_limits    in-grip drift of the held object (privileged, TCP frame) > GRIP_DRIFT_M during lift/carry -> regrasp
                 (support_insert; <= MAX_REGRASP) and counted (handover).
  support_limits support-hand slip from its frozen contact point (public FK) > SUPPORT_SLIP_M -> re-press, counted.
Phase-gated DART (D-121): `dart_phase_allowed(task, arm_phase)` = the free-space phases in which collect_dual may perturb
an arm (`noise_phase_gate: true`); never during descend/close/lift, the support press, align/insert, offer/receive/release.

W7 owns the grasp contact model (rrp.physics.grasp_contact): the pad-cylinder friction-cone saturation and the pg2
gripper force shortfall (~16-20 N of 40 N) found by scripts/w12/grip_diag.py are NOT fixed here. `GRIP_CLOSE_OVERRIDE`
is the hook: a per-gripper-kind closed command (None = the body's declared closed value) to use once W7 decides how
the pg2 servo should reach its design force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import mujoco
import numpy as np

from rrp.bodies.ik import down_rotation, rot_error
from rrp.policies.teachers.dual import (GRASP_YAW, ROT_RATE, JOINT_RATE, ArmMover, HandoverTeacher, SupportInsertTeacher,
                               TEACHERS, _rotvec_mat)

DEFAULT_DUAL_TEACHER = "v2"
DUAL_TEACHER_VERSIONS = {"v2": "dual_v2_rate_limited", "v3": "dual_v3_minjerk"}
VMAX_ACC = 1.0                   # m/s^2 Cartesian acceleration bound for submovement durations
MIN_SUB_T = 0.25                 # s, shortest submovement
CORR_T = 0.35                    # s, shortest correction submovement
CORR_V = 0.05                    # m/s, peak speed of a correction submovement (errors < 2 cm only)
MAX_CORR = 3
GRIP_T = 0.5                     # s, gripper open/close ramp
GRASP_DEPTH_V3 = {"parallel": 0.03, "three_finger": 0.04, "aloha": 0.03}
GRIP_DRIFT_M = 0.003
SUPPORT_SLIP_M = 0.005
MAX_REGRASP = 2
TOUCH_ON = 0.2
GRIP_CLOSE_OVERRIDE: dict = {}   # hook for W7 (gripper kind -> closed command); empty = declared closed values
LIMIT_GAIN = 0.02
LIMIT_BAND = 0.1                 # fraction of the joint range near each limit where the null-space push acts

# D-121-style phase gating: arm phases in which executed-command noise is allowed (free space only)
DART_PHASES = {
    "support_insert": {"l_stage", "l_pre", "l_retreat", "l_done", "r_stage", "r_pre", "r_wait", "r_transit",
                       "r_retreat", "r_done"},
    "handover": {"l_stage", "l_pre", "l_retreat", "l_done", "r_stage", "r_wait", "r_pre", "r_transport",
                 "r_retreat", "r_done"},
}


def dart_phase_allowed(task: str, arm_phase: str) -> bool:
    return arm_phase in DART_PHASES.get(task, set())


def minjerk(tau):
    t = np.clip(tau, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


def minjerk_duration(dist: float, vmax: float, amax: float = VMAX_ACC, tmin: float = MIN_SUB_T) -> float:
    if dist <= 1e-9:
        return tmin
    return max(tmin, 1.875 * dist / max(vmax, 1e-6), math.sqrt(5.7735 * dist / amax))


@dataclass
class V3Options:
    minjerk: bool = True
    limit_aware: bool = True
    confirm_grasp: bool = True
    deep_grasp: bool = True
    grip_limits: bool = True
    support_limits: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "V3Options":
        d = dict(d or {})
        bad = set(d) - {f.name for f in fields(cls)}
        if bad:
            raise ValueError(f"unknown teacher v3 options {sorted(bad)}")
        return cls(**d)


class SmoothArmMover(ArmMover):
    """v3 mover: same interface as ArmMover (set_goal / step / reached / grip), min-jerk internals."""

    def __init__(self, session, ent, speed=0.3, opts: V3Options | None = None):
        super().__init__(session, ent, speed)
        self.o = opts or V3Options()
        self.t = 0.0
        self.base = None            # TCP command origin of the plan
        self.subs: list = []        # (t0, T, delta)
        self.coff = np.zeros(3)     # accumulated correction offset (steady-state sag), kept across goals
        self.ncorr = 0
        self.rot_plan = None        # (t0, T, R0, rv, R_target)
        self.grip_cmd = None
        self.grip_plan = None       # (t0, T, g0, g1)

    # ---- plan helpers
    def _final(self):
        return self.base + sum((d for _, _, d in self.subs), np.zeros(3))

    def _pos(self):
        return self.base + sum((d * minjerk((self.t - t0) / T) for t0, T, d in self.subs), np.zeros(3))

    def plan_done(self) -> bool:
        return all(self.t >= t0 + T - 1e-9 for t0, T, _ in self.subs)

    def hold_here(self):
        """Freeze the plan at the measured TCP (touch-down)."""
        tcp, _ = self.tcp()
        self.base, self.subs, self.coff, self.ncorr = tcp.copy(), [], np.zeros(3), 0
        self.goal = tcp.copy()

    def set_goal(self, goal, speed=None, yaw=None):
        goal = np.asarray(goal, float)
        if speed is not None:
            self.speed = speed
        if yaw is not None:
            self.yaw = yaw
        if self.base is None:
            self.base = self.tcp()[0].copy()
        if self.goal is not None and np.linalg.norm(goal - self.goal) <= 1e-4:
            return
        if self.goal is None or np.linalg.norm(goal - self.goal) > 2e-3:
            self.n, self.err, self.stalled, self.t_goal, self.ncorr = float("inf"), float("inf"), False, 0.0, 0
        self.goal = goal
        delta = goal + self.coff - self._final()
        self.subs.append((self.t, minjerk_duration(float(np.linalg.norm(delta)), self.speed), delta))

    def _grip_now(self):
        if self.grip_cmd is None:
            self.grip_cmd = float(self.grip)
        tgt = float(GRIP_CLOSE_OVERRIDE.get(self.h.gripper_kind, self.grip)) if self.grip == self.h.closed_value \
            else float(self.grip)
        if self.grip_plan is None or abs(self.grip_plan[3] - tgt) > 1e-9:
            self.grip_plan = (self.t, GRIP_T, self.grip_cmd, tgt)
        t0, T, g0, g1 = self.grip_plan
        self.grip_cmd = float(g0 + (g1 - g0) * minjerk((self.t - t0) / T))
        return self.grip_cmd

    def _orient(self):
        if self.R_cmd is None:
            _, Rn = self.tcp()
            self.R_cmd = Rn.copy()
        cur_yaw = float(math.atan2(self.R_cmd[1, 0], self.R_cmd[0, 0]))
        tgt_yaw = self.yaw + round((cur_yaw - self.yaw) / self.yaw_sym) * self.yaw_sym
        Rt = down_rotation(tgt_yaw)
        if self.rot_plan is None or np.linalg.norm(rot_error(self.rot_plan[4], Rt)) > 1e-3:
            rv = rot_error(self.R_cmd, Rt)
            ang = float(np.linalg.norm(rv))
            self.rot_plan = (self.t, max(MIN_SUB_T, 1.875 * ang / ROT_RATE), self.R_cmd.copy(), rv, Rt)
        t0, T, R0, rv, _ = self.rot_plan
        self.R_cmd = _rotvec_mat(rv * minjerk((self.t - t0) / T)) @ R0
        return self.R_cmd

    def _limit_refine(self, q, pos, R):
        """Null-space step away from joint limits (keeps the 6-D TCP task to first order)."""
        ik = self.h.ik
        lo, hi = ik.lo, ik.hi
        rng = np.maximum(hi - lo, 1e-6)
        x = (q - lo) / rng
        g = np.where(x < LIMIT_BAND, (LIMIT_BAND - x), 0.0) - np.where(x > 1 - LIMIT_BAND, (x - 1 + LIMIT_BAND), 0.0)
        if not np.any(g):
            return q
        ik.fk(self.s.data.qpos.copy(), q)
        jacp = np.zeros((3, ik.model.nv))
        jacr = np.zeros((3, ik.model.nv))
        mujoco.mj_comPos(ik.model, ik.data)
        mujoco.mj_jacSite(ik.model, ik.data, jacp, jacr, ik.sid)
        J = np.vstack([jacp[:, ik.dadr], jacr[:, ik.dadr]])
        N = np.eye(len(q)) - np.linalg.pinv(J, rcond=1e-3) @ J
        return np.clip(q + N @ (LIMIT_GAIN * g * rng), lo, hi)

    # ---- per tick
    def step(self) -> dict:
        dt = self.s.dt
        self.t += dt
        self.t_goal += dt
        tcp_now, _ = self.tcp()
        if self.base is None:
            self.base = tcp_now.copy()
        goal_true = self.goal if self.goal is not None else tcp_now
        err_vec = goal_true - tcp_now
        if self.integral and self.plan_done() and self.ncorr < MAX_CORR and 0.0015 < np.linalg.norm(err_vec) < 0.02 \
                and self.t_goal > 0.2:
            corr = err_vec
            self.coff = np.clip(self.coff + corr, -0.04, 0.04)
            self.subs.append((self.t, minjerk_duration(float(np.linalg.norm(corr)), CORR_V, tmin=CORR_T), corr))
            self.ncorr += 1
        # fold finished submovements into the base (bounded list)
        done = [s for s in self.subs if self.t >= s[0] + s[1]]
        if done:
            self.base = self.base + sum((d for _, _, d in done), np.zeros(3))
            self.subs = [s for s in self.subs if self.t < s[0] + s[1]]
        self.tcp_cmd = self._pos()
        self.n = float(np.linalg.norm(self._final() - self.tcp_cmd))
        self.err = float(np.linalg.norm(err_vec))
        v = np.linalg.norm(tcp_now - self.last_tcp) / dt if self.last_tcp is not None else 1.0
        self.last_tcp = tcp_now
        self.stalled = self.plan_done() and v < 0.01 and self.t_goal > 1.0
        R_cmd = self._orient()
        q, e = self.h.ik.solve(self.s.data.qpos.copy(), self.q_arm, self.tcp_cmd, R_cmd, seeds=self.seeds(self.tcp_cmd))
        if self.o.limit_aware:
            q = self._limit_refine(q, self.tcp_cmd, R_cmd)
        self.max_ik_err = max(self.max_ik_err, float(e))
        self.q_arm = self.q_arm + np.clip(q - self.q_arm, -JOINT_RATE * dt, JOINT_RATE * dt)   # safety clip (v2's)
        return {self.h.arm_group: self.q_arm.tolist(), self.h.grip_group: [self._grip_now()]}

    def reached(self, tol: float) -> bool:
        return self.plan_done() and (self.err < tol or (self.stalled and self.err < 2 * tol))

    _STATE = ArmMover._STATE + ("t", "base", "subs", "coff", "ncorr", "rot_plan", "grip_cmd", "grip_plan")


class _V3Mixin:
    """Common v3 machinery: movers, option handling, limit counters, per-arm phase interception."""
    version = "v3"
    teacher_version = DUAL_TEACHER_VERSIONS["v3"]

    def _v3_init(self, session, speed, options):
        self.o = options if isinstance(options, V3Options) else V3Options.from_dict(options)
        self.arms = {e: SmoothArmMover(session, e, speed, self.o) for e in ("left", "right")}
        self.limits = dict(regrasps=0, grip_drift_events=0, support_slip_events=0, grip_drift_max_m=0.0,
                           support_slip_max_m=0.0, no_contact_close=0)
        self._grip_ref = {}

    def _intercept(self, owned: dict):
        """Run v2's _plan with the phases v3 owns hidden (sentinel), then restore them."""
        saved = dict(self.phase)
        for e, p in owned.items():
            if self.phase[e] in p:
                self.phase[e] = "_v3"
        try:
            super()._plan()
        finally:
            for e in owned:
                if self.phase[e] == "_v3":
                    self.phase[e] = saved[e]

    def _touching(self, ent) -> bool:
        tv = self.s.touch_values(ent)
        return bool(len(tv) and (tv >= TOUCH_ON).sum() >= min(2, len(tv)))

    def _in_grip(self, ent, body):
        """PRIVILEGED: object position in the TCP frame."""
        p, _ = self._body(body)
        tcp, R = self.arms[ent].tcp()
        return R.T @ (p - tcp)

    def _drift(self, ent, body) -> float:
        if ent not in self._grip_ref:
            self._grip_ref[ent] = self._in_grip(ent, body)
            return 0.0
        d = float(np.linalg.norm(self._in_grip(ent, body) - self._grip_ref[ent]))
        self.limits["grip_drift_max_m"] = max(self.limits["grip_drift_max_m"], d)
        return d

    def _confirm_close(self, ent, arm, closed, nxt, fail):
        """close ramp -> touch confirmed -> settle 0.3 s -> nxt; no contact within 2 s -> fail."""
        arm.grip = closed
        key = f"_touch_t_{ent}"
        if self._touching(ent):
            t0 = getattr(self, key, None)
            if t0 is None:
                setattr(self, key, self.t_phase[ent])
            elif self.t_phase[ent] - t0 >= 0.3:
                setattr(self, key, None)
                self._grip_ref.pop(ent, None)
                self._next(ent, nxt)
        elif self.t_phase[ent] > 2.0:
            setattr(self, key, None)
            self.limits["no_contact_close"] += 1
            self._next(ent, fail)


class SupportInsertTeacherV3(_V3Mixin, SupportInsertTeacher):
    """support_insert v3 (see module doc). Phases added: r_regrasp (open, back off, re-approach), l_repress."""

    def __init__(self, session, speed: float = 0.3, grasp_depth: float | None = None, frame_override=None,
                 options: V3Options | dict | None = None):
        super().__init__(session, speed, grasp_depth, frame_override)
        self._v3_init(session, speed, options)
        self._slow = False
        self._slip_over = False
        if grasp_depth is None and self.o.deep_grasp:
            self.grasp_depth = GRASP_DEPTH_V3.get(self.arms["right"].h.gripper_kind, 0.03)

    def _touchdown(self, L):
        tcp, _ = L.tcp()
        self.contact_z, self.contact_xy = float(tcp[2]), tcp[:2].copy()
        L.integral = False
        L.hold_here()
        self._next("left", "l_hold")

    def _plan(self):
        o = self.o
        own_r = set()
        if o.confirm_grasp:
            own_r |= {"r_close"}
        if o.grip_limits:
            own_r |= {"r_lift", "r_regrasp"}
        own_l = ({"l_hold"} if o.support_limits else set()) | ({"l_descend"} if o.minjerk else set())
        self._intercept({"right": own_r, "left": own_l})
        L, R = self.arms["left"], self.arms["right"]
        pr, pl = self.phase["right"], self.phase["left"]
        if pr in own_r:
            close_v = R.closed_for(self.peg_r)
            if pr == "r_close":
                R.set_goal(self.grasp_xyz, 0.1)
                self._confirm_close("right", R, close_v, "r_lift", "r_regrasp")
            elif pr == "r_lift":
                R.set_goal(np.r_[self.grasp_xyz[:2], 0.25], 0.08 if self.t_phase["right"] < 1.0 else 0.2)
                if self._drift("right", "peg") > GRIP_DRIFT_M:
                    self.limits["grip_drift_events"] += 1
                    self._next("right", "r_regrasp")
                elif R.reached(0.02):
                    self._next("right", "r_wait")
            elif pr == "r_regrasp":
                R.grip = R.open_value
                tcp, _ = R.tcp()
                R.set_goal(np.r_[tcp[:2], tcp[2] + 0.06], 0.1)
                if self.t_phase["right"] > 0.8:
                    self.limits["regrasps"] += 1
                    self._next("right", "r_pre" if self.limits["regrasps"] <= MAX_REGRASP else "r_abort")
        if pl in own_l and pl == "l_descend":
            # min-jerk approach to 1.2 cm above the support point, then a slow (2 cm/s) press-in that stops on touch:
            # v2's single fast descend stopped mid-motion at touch-down (a velocity step)
            sp = self._support_point()
            if not self._slow:
                L.set_goal(sp + [0, 0, 0.012], 0.12)
                if L.reached(0.004):
                    self._slow = True
            else:
                L.set_goal(sp + [0, 0, -0.03], 0.02)
            t = self.s.touch_values("left")
            if (len(t) and t.max() > 0.5) or (self._slow and L.reached(0.005)):
                self._slow = False
                self._touchdown(L)
        if pl in own_l and pl == "l_hold":
            tcp, _ = L.tcp()
            slip = float(np.linalg.norm(tcp[:2] - self.contact_xy))
            self.limits["support_slip_max_m"] = max(self.limits["support_slip_max_m"], slip)
            L.integral = False
            over = slip > SUPPORT_SLIP_M
            if over and not self._slip_over:
                self.limits["support_slip_events"] += 1          # counted once per excursion (rising edge)
            self._slip_over = over
            L.set_goal(np.r_[self.contact_xy, self.contact_z - 0.006], 0.02)
            if self.status("insert") == "succeeded" or (self.phase["right"] == "r_done"):
                self._next("left", "l_retreat")


class HandoverTeacherV3(_V3Mixin, HandoverTeacher):
    """handover v3: min-jerk mover, contact-confirmed closes for both hands, giver in-grip drift counted."""

    def __init__(self, session, speed: float = 0.3, options: V3Options | dict | None = None):
        super().__init__(session, speed)
        self._v3_init(session, speed, options)
        self._lret = None
        self._drift_flag = False

    def _plan(self):
        o = self.o
        own_l = ({"l_close"} if o.confirm_grasp else set()) | ({"l_retreat"} if o.minjerk else set())
        own_r = {"r_close"} if o.confirm_grasp else set()
        self._intercept({"left": own_l, "right": own_r})
        L, R = self.arms["left"], self.arms["right"]
        if self.phase["left"] == "l_retreat" and "l_retreat" in own_l:
            # v2 recomputes its retreat target from its own previous goal every tick (goal[:2] + 8 cm in y), so the target
            # runs away until the arm saturates its joint-rate clip; v3 fixes the target once on entry
            if self._lret is None:
                self._lret = np.r_[L.goal[:2] + np.array([0.0, 0.08]), 0.30]
            L.set_goal(self._lret, 0.2)
            if L.reached(0.02):
                self._next("left", "l_done")
        if self.phase["left"] == "l_close" and "l_close" in own_l:
            L.set_goal(self.gl, 0.1)
            self._confirm_close("left", L, L.closed_for(self.half[1]), "l_lift", "l_pre")
        if self.phase["right"] == "r_close" and "r_close" in own_r:
            R.set_goal(self.gr, 0.1)
            R.grip = R.closed_for(self.half[1])
            if self._touching("right") and self.t_phase["right"] > 0.8 and self.status("release") == "succeeded":
                tcp, _ = R.tcp()
                self.off_r = self._body("bar")[0] - tcp
                self._next("right", "r_transport")
            elif self.t_phase["right"] > 6.0 and not self._touching("right"):
                self.limits["no_contact_close"] += 1
                self._next("right", "r_pre")
        if o.grip_limits and self.phase["left"] in ("l_lift", "l_present", "l_hold"):
            if self._drift("left", "bar") > GRIP_DRIFT_M and not self._drift_flag:
                self.limits["grip_drift_events"] += 1
                self._drift_flag = True


TEACHERS_V3 = {"support_insert": SupportInsertTeacherV3, "handover": HandoverTeacherV3}


def make_dual_teacher(task: str, session, version: str | None = None, options: dict | None = None):
    """version None/'v2' -> the v2 teacher (default, unchanged); 'v3' -> the v3 teacher with `options`."""
    v = version or DEFAULT_DUAL_TEACHER
    from rrp.policies.teachers.dual_coord import COORD_TEACHERS
    if task in COORD_TEACHERS:                       # D-126 #22 coordination tasks: labelled stubs only
        return COORD_TEACHERS[task](session)
    if v == "v2":
        if options:
            raise ValueError("teacher options apply to v3 only")
        return TEACHERS[task](session)
    if v == "v3":
        if task not in TEACHERS_V3:
            raise ValueError(f"no v3 teacher for {task} (v3: {sorted(TEACHERS_V3)})")
        return TEACHERS_V3[task](session, options=options)
    raise ValueError(f"unknown dual teacher version {v!r} ({sorted(DUAL_TEACHER_VERSIONS)})")
