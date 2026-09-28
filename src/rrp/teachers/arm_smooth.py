"""Arm pick-place teacher versions (W7).

v1 = the original waypoint FSM (`rrp.teachers.arm.PickPlaceTeacher`), still the DEFAULT everywhere.
v2 = `SmoothPickPlaceTeacher`: the same task logic with time-parameterized minimum-jerk motion and a force-checked grasp.
Versions are selected by name (`make_arm_teacher(session, "v2")`) and recorded in provenance as
`scripted_teacher:<teacher version id>` (`teacher_source`).

Both versions read PRIVILEGED simulator state (object poses) and are labelled scripted_teacher.

v2 design (diagnosis in research/tracks/armexpert.md):
- Motion = superposition of minimum-jerk (quintic, zero boundary velocity/acceleration) Cartesian submovements; the
  next submovement starts before the previous one ends (blended corners, C2 path). Plans: approach (hover over the
  cube -> vertical descent along the tool axis), carry (vertical lift -> transport -> lower), retreat (vertical).
- Each plan is converted to joint commands at the control rate by warm-started IK before execution; the plan is
  stretched in time until the joint velocities/accelerations respect limits derived from the body's actuators
  (a_max = 0.5 * tau_max / M_jj at the current configuration, capped by nominal limits). If Cartesian IK tracking
  fails (joint limit / local minimum), the plan falls back to a joint-space quintic to a seeded IK solution.
- Grasp yaw is chosen among the cube x gripper symmetric yaws (pg2: 90 deg, three-finger: 30 deg) by IK
  reachability of hover + grasp, clearance of the fingertips from other objects, then closeness to the current yaw.
- Three-finger grippers are pre-shaped (opened only as far as needed) instead of splayed fully open.
- Gripper close = min-jerk ramp; on contact (public touch sensors on >= 2 fingers) the target becomes the measured
  finger position plus a small squeeze; settle, then lift only when the grasp is confirmed (touch + blocked width).
  No contact -> regrasp. Opening is a ramp as well.
- End-of-plan residuals are corrected by short min-jerk correction moves (smooth replacement of v1's integral step).
- Every phase has a timeout with recovery (retreat and re-approach with the next yaw candidate) instead of stalling.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from rrp.bodies.ik import down_rotation
from rrp.contracts.action import NativeCommand
from rrp.contracts.provenance import Source, source_label
from rrp.teachers.arm import SOURCE, PickPlaceTeacher, _yaw_of_quat

DEFAULT_ARM_TEACHER = "v1"
TEACHER_VERSIONS = {"v1": "pick_place_v1_waypoint", "v2": "pick_place_v2_minjerk",
                    "v2lim": "pick_place_v2_minjerk_lim"}     # D-126 #8: v2 + limit-aware IK (bodies/ik.py limit_margin)
V2LIM_DEFAULT_MARGIN = 0.05     # fraction of each joint's range (the D-112 gate asks >= 0.02; the backlog: keep >= 0.05)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def minjerk(tau):
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 on [0, 1], clamped outside."""
    t = np.clip(tau, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


# peak |ds/dtau| and |d2s/dtau2| of the min-jerk profile
MJ_VPEAK = 1.875
MJ_APEAK = 5.7735


def minjerk_duration(dist: float, vmax: float, amax: float, tmin: float = 0.0) -> float:
    if dist <= 1e-9:
        return tmin
    return max(tmin, MJ_VPEAK * dist / vmax, math.sqrt(MJ_APEAK * dist / amax))


class SmoothPickPlaceTeacher(PickPlaceTeacher):
    """v2: minimum-jerk, limit-respecting, force-checked pick-place teacher (privileged, scripted).

    A plan is a list of segments, each a joint-space delta trajectory D_i (D_i(0) = 0) started at tick k_i:
    q(k) = q0 + sum_i D_i(k - k_i). Joint segments are quintics (free-space moves: pregrasp, transport); Cartesian
    segments are straight tool-axis lines tracked by warm-started IK (descend, lift, lower, retreat, corrections).
    Starting a segment before the previous one ends blends the corner."""

    # nominal limits (declared, recorded in diag); the actuator-derived acceleration limit is min'ed in
    V_JOINT = 1.5          # rad/s
    A_JOINT = 6.0          # rad/s^2
    V_FREE = 0.5           # m/s peak TCP speed, free-space moves
    A_FREE = 2.0           # m/s^2
    V_AXIAL = 0.25         # m/s peak along the tool axis (lift / retreat)
    V_APPROACH = 0.15      # m/s peak for the final approach to grasp / place
    HOVER = 0.13
    LIFT_Z = 0.18
    RETREAT_DZ = 0.09
    BLEND = 0.3            # fraction of a segment overlapped by the next one
    TOUCH_THR = 0.2        # N, the same threshold as the public held_by estimate
    MAX_ATTEMPTS = 3
    IK_LIMIT_MARGIN = 0.0  # D-126 #8: 0 = historical IK (v2); > 0 only in the version-bumped v2lim teacher
    VERSION_KEY = "v2"

    def __init__(self, session, robot: int = 0, obj: str = "cube", zone: str = "target_zone", rng=None,
                 ik_limit_margin: float | None = None, **kw):
        self.ik_limit_margin = float(self.IK_LIMIT_MARGIN if ik_limit_margin is None else ik_limit_margin)
        if self.ik_limit_margin > 0 and self.VERSION_KEY == "v2":
            raise ValueError("ik_limit_margin > 0 changes the teacher: use version 'v2lim' (pick_place_v2_minjerk_lim)")
        # extra IK keyword for the limit-aware solve; empty for v2 so its calls are exactly the historical ones
        self._ikkw = dict(limit_margin=self.ik_limit_margin) if self.ik_limit_margin > 0 else {}
        super().__init__(session, robot=robot, obj=obj, zone=zone, rng=rng)
        m, d = session.model, session.data
        self.three = (self.r.meta.get("gripper_params") or {}).get("kind") == "three_finger"
        g = self.r.controller.groups["gripper"]
        self.g_lo, self.g_hi = float(g.lower[0]), float(g.upper[0])
        if self.three:
            # pre-shape: open only ~0.55 rad short of the contact angle (v1 splays to the joint limit, whose 9 cm
            # fingertip reach lands on neighbouring objects)
            self.open_v = float(np.clip(self.closed_v - 0.1 - 0.55, self.g_lo, self.g_hi))
            self.squeeze = 0.03                    # rad past the measured contact angle
            self.yaw_period = math.pi / 6          # cube 90 deg x fingers 120 deg
        else:
            self.squeeze = -0.003                  # m (opening per finger; closing = decreasing)
            self.yaw_period = math.pi / 2          # cube 90 deg x parallel jaw 180 deg
        self.grip = float(self.r.controller.current_targets(d)["gripper"][0])
        # actuator-derived acceleration limits: 0.5 * tau_max / M_jj at the current configuration
        self.v_joint = np.full(len(self.q_arm), self.V_JOINT)
        self.a_joint = np.full(len(self.q_arm), self.A_JOINT)
        a_act, err = None, None
        try:
            ids = self.r.controller.act_ids["arm"]
            tau = np.array([np.abs(m.actuator_forcerange[u]).max() * abs(m.actuator_gear[u, 0])
                            if m.actuator_forcelimited[u] else np.inf for u in ids])
            mjj = []
            e = np.zeros(m.nv)
            out = np.zeros(m.nv)
            for dof in self.r.ik.dadr:
                e[:] = 0.0
                e[dof] = 1.0
                mujoco.mj_mulM(m, d, out, e)
                mjj.append(out[dof])
            a_act = 0.5 * tau / np.maximum(np.array(mjj), 1e-6)
            self.a_joint = np.minimum(self.a_joint, a_act)
        except Exception as ex:  # noqa: BLE001 - limits stay nominal; recorded
            err = repr(ex)[:200]
        self.diag = dict(version=TEACHER_VERSIONS[self.VERSION_KEY], v_joint=self.v_joint.tolist(), a_joint=self.a_joint.tolist(),
                         a_actuator=None if a_act is None else np.asarray(a_act).tolist(), limits_error=err,
                         attempts=0, plans=[], yaw_choice=None, contact_grip=None, events=[])
        if self._ikkw:
            self.diag["ik_limit_margin"] = self.ik_limit_margin
        self.phase = "pregrasp"
        self.t_phase = 0.0
        self.q_plan: list = []          # precomputed arm commands of the active plan
        self.plan_phases: list = []
        self.k_plan = 0
        self.grip_plan: list = []
        self.attempt = 0
        self.yaw_rank = 0
        self.stage = "approach"
        self.corrections = 0
        self.ioff = np.zeros(3)
        self.contact_ticks = 0
        self.lost_ticks = 0
        self.done_flag = False
        self.rel_grasp = None
        self._start_approach()

    # ------------------------------------------------------------------ geometry helpers
    def _tcp(self):
        return self.s._fk_site(self.r, self.tcp_site)

    def _tool_yaw(self):
        _, R = self._tcp()
        return float(math.atan2(R[1, 0], R[0, 0]))

    def _fk(self, q):
        return self.r.ik.fk(self.s.data.qpos.copy(), q)

    def _others(self):
        out = []
        for o in self.s.scenario.objects:
            if o.kind == "object" and o.sim_body != self.obj:
                p, _ = self._body(o.sim_body)
                out.append((p, float(np.max(o.size)) * math.sqrt(2)))
        return out

    def _finger_xy(self, center, yaw):
        """Fingertip xy positions (pre-shaped) for a grasp yaw; used only for clearance scoring."""
        if self.three:
            gp = self.r.meta.get("gripper_params") or {}
            r0, L = gp.get("finger_base_radius", 0.052), gp.get("finger_length", 0.055)
            rad = r0 - L * math.sin(self.open_v) + 0.009
            angs = [yaw + 2 * math.pi * k / 3 for k in range(3)]
        else:
            rad = 0.012 + float(self.open_v) + 0.012
            angs = [yaw + math.pi / 2, yaw - math.pi / 2]
        return [center[:2] + rad * np.array([math.cos(a), math.sin(a)]) for a in angs]

    def _ik_point(self, p, yaw, q0):
        q, e = self.r.ik.solve(self.s.data.qpos.copy(), q0, p, down_rotation(yaw), iters=150, tol=2e-4,
                               seeds=self._ik_seeds(p), **self._ikkw)
        return q, float(e)

    def _margin(self, q):
        lo, hi = self.r.ik.lo, self.r.ik.hi
        return float(np.min(np.minimum(q - lo, hi - q) / np.maximum(hi - lo, 1e-6)))

    # ------------------------------------------------------------------ segments
    def _joint_seg(self, q_start, q_end, phase, tmin=0.4, v_cart=None):
        dt = self.s.dt
        dq = np.abs(q_end - q_start)
        T = max(tmin, *[minjerk_duration(x, v, a) for x, v, a in zip(dq, self.v_joint, self.a_joint)])
        if v_cart:                                  # TCP speed cap along the joint-space path
            p = [self._fk(q_start + (q_end - q_start) * float(minjerk(u)))[0] for u in np.linspace(0, 1, 21)]
            L = float(sum(np.linalg.norm(b - a) for a, b in zip(p, p[1:])))
            T = max(T, minjerk_duration(L, v_cart, self.A_FREE))
        n = max(1, int(math.ceil(T / dt)))
        path = [q_start + (q_end - q_start) * float(minjerk((k + 1) / n)) for k in range(n)]
        return dict(kind="joint", phase=phase, q_start=q_start, q_end=q_end, path=path, T=n * dt)

    def _cart_seg(self, q_start, p_end, yaw, phase, vmax, tmin=0.3, reach_retry=False):
        """Straight TCP line from FK(q_start) to p_end (tool down, fixed yaw), min-jerk timed, IK-tracked.
        Stretched until the joint path meets the joint limits; falls back to a joint quintic if IK tracking fails."""
        dt = self.s.dt
        p0, _ = self._fk(q_start)
        dist = float(np.linalg.norm(p_end - p0))
        T = minjerk_duration(dist, vmax, self.A_FREE, tmin)
        R = down_rotation(yaw)
        qfull = self.s.data.qpos.copy()
        info = dict(stretch=1.0)
        for _ in range(3):
            n = max(1, int(math.ceil(T / dt)))
            q = q_start.copy()
            path, worst = [], 0.0
            for k in range(n):
                p = p0 + (p_end - p0) * float(minjerk((k + 1) / n))
                q, e = self.r.ik.solve(qfull, q, p, R, iters=80, tol=2e-4, **self._ikkw)
                worst = max(worst, float(e))
                path.append(q.copy())
            Q = np.vstack([q_start, *path])
            v = np.abs(np.diff(Q, axis=0)) / dt
            a = np.abs(np.diff(Q, 2, axis=0)) / dt ** 2 if len(Q) > 2 else np.zeros((1, len(q)))
            f = max(1.0, float(np.max(v / self.v_joint)), math.sqrt(float(np.max(a / self.a_joint))))
            info["ik_worst"] = round(worst, 4)
            if worst > 5e-3 or f > 3.0:
                break
            if f <= 1.02:
                return dict(kind="cart", phase=phase, q_start=q_start, q_end=path[-1], path=path, T=n * dt, **info)
            T *= f * 1.03
            info["stretch"] = round(info["stretch"] * f * 1.03, 3)
        # IK tracking failed. If the END point itself is out of reach (joint limit), track the straight line to the
        # closest reachable end point instead (a joint-space arc near the cube knocks it away); otherwise (local
        # minimum / singular path) fall back to a joint quintic to a seeded IK solution.
        q_end, e = self._ik_point(p_end, yaw, q_start)
        if e > 2e-3 and not reach_retry:
            p_reach = self._fk(q_end)[0]
            seg = self._cart_seg(q_start, p_reach, yaw, phase, vmax, tmin, reach_retry=True)
            if not seg.get("fallback"):
                seg.update(reach_limited=True, ik_worst=round(max(seg.get("ik_worst", 0.0), e), 4))
                return seg
        seg = self._joint_seg(q_start, q_end, phase, tmin=tmin)
        seg.update(fallback=True, fallback_err=round(e, 4), **info)
        return seg

    def _compose(self, q0, segs, name):
        """q(k) = q0 + sum_i D_i(k - k_i), segment i+1 starting BLEND of segment i before its end (in ticks).
        The blend is reduced if the overlapping velocities would exceed the joint limits."""
        dt = self.s.dt
        blend = self.BLEND
        for _ in range(3):
            starts, k = [], 0
            for i, sg in enumerate(segs):
                starts.append(k)
                n = len(sg["path"])
                ov = blend * n if (i + 1 < len(segs) and sg.get("blend_ok", True) and
                                   segs[i + 1].get("blend_ok", True)) else 0.0
                k += max(1, int(round(n - ov)))
            total = max(s_ + len(sg["path"]) for s_, sg in zip(starts, segs))
            qs, phases = [], []
            for kk in range(total):
                q = q0.copy()
                ph = segs[0]["phase"]
                for s_, sg in zip(starts, segs):
                    j = kk - s_
                    if j < 0:
                        continue
                    ph = sg["phase"]
                    q = q + ((sg["path"][j] if j < len(sg["path"]) else sg["path"][-1]) - sg["q_start"])
                qs.append(q)
                phases.append(ph)
            Q = np.vstack([q0, *qs])
            v = np.abs(np.diff(Q, axis=0)) / dt
            f = float(np.max(v / self.v_joint))
            if f <= 1.1 or blend == 0.0:
                break
            blend = 0.0 if blend < 0.1 else blend / 2
        self.q_plan, self.plan_phases, self.k_plan = qs, phases, 0
        last = segs[-1]
        self.plan_reach_err = float(last.get("fallback_err", last.get("ik_worst", 0.0)) or 0.0)
        self.diag["plans"].append(dict(name=name, ticks=len(qs), T=round(len(qs) * dt, 3), blend=blend,
                                       segs=[dict(phase=sg["phase"], kind=sg["kind"], T=round(sg["T"], 3),
                                                  **{k: sg[k] for k in ("stretch", "ik_worst", "fallback", "reach_limited",
                                                                        "fallback_err") if k in sg})
                                             for sg in segs]))

    # ------------------------------------------------------------------ plans
    def _grasp_targets(self):
        cube, cq = self._body(self.obj)
        hover = np.array([cube[0], cube[1], cube[2] + self.HOVER])
        grasp = np.array([cube[0], cube[1], cube[2] + 0.004])
        return cube, cq, hover, grasp

    def _yaw_candidates(self):
        cube, cq, hover, grasp = self._grasp_targets()
        cur = self._fk(self.q_arm)[1]
        cur = float(math.atan2(cur[1, 0], cur[0, 0]))
        base = _yaw_of_quat(cq)
        n = int(round(2 * math.pi / self.yaw_period))
        cands = []
        for k in range(n):
            y = cur + _wrap(base + k * self.yaw_period - cur)
            q_h, e_h = self._ik_point(hover, y, self.q_arm)
            q_g, e_g = self._ik_point(grasp, y, q_h)
            clear = min((min(np.linalg.norm(f - p[:2]) - r for f in self._finger_xy(cube, y))
                         for p, r in self._others()), default=1.0)
            cands.append(dict(yaw=y, ok=bool(e_h < 5e-3 and e_g < 5e-3), err=round(max(e_h, e_g), 4),
                              clear=round(clear, 4), margin=round(min(self._margin(q_h), self._margin(q_g)), 4),
                              dyaw=round(abs(y - cur), 3), q_h=q_h))
        # reachable first, then fingertip clearance >= 1.5 cm, then joint margin >= 2 %, then the smallest wrist roll
        cands.sort(key=lambda c: (not c["ok"], 0.0 if c["ok"] else c["err"], c["clear"] < 0.015, c["margin"] < 0.02,
                                  c["dyaw"]))
        return cands

    def _start_approach(self):
        cands = self._yaw_candidates()
        ok = [c for c in cands if c["ok"]] or [c for c in cands if c["err"] < 0.016] or cands[:1]
        # retries cycle through the reachable yaws (or the near-reachable ones, within the 1.6 cm grasp acceptance),
        # repeating the best one rather than trying a yaw that cannot reach the cube
        c = ok[self.yaw_rank % len(ok)]
        self.yaw = c["yaw"]
        self.diag["yaw_choice"] = dict({k: v for k, v in c.items() if k != "q_h"}, rank=self.yaw_rank, n=len(cands))
        _, _, hover, grasp = self._grasp_targets()
        q0 = self.q_arm.copy()
        s1 = self._joint_seg(q0, c["q_h"], "pregrasp", tmin=0.6, v_cart=self.V_FREE)
        s2 = self._cart_seg(s1["q_end"], grasp, self.yaw, "descend", self.V_APPROACH, tmin=0.5)
        s1["blend_ok"] = s2["blend_ok"] = True
        self._compose(q0, [s1, s2], f"approach{self.attempt}")
        self.grasp_goal = grasp
        self.stage, self.corrections, self.ioff = "approach", 0, np.zeros(3)
        self._grip_ramp(self.open_v, 0.4)

    def _start_carry(self):
        zone, _ = self._body(self.zone)
        h = self.s.scenario.object(self.obj).size[2]
        q0 = self.q_arm.copy()
        p0, R0 = self._fk(q0)
        y0 = float(math.atan2(R0[1, 0], R0[0, 0]))
        lift = np.array([p0[0], p0[1], self.LIFT_Z])
        above = np.array([zone[0], zone[1], self.LIFT_Z])
        place = np.array([zone[0], zone[1], h + 0.012])
        s1 = self._cart_seg(q0, lift, y0, "lift", self.V_AXIAL, tmin=0.6)
        q_above, e = self._ik_point(above, y0, s1["q_end"])
        s2 = self._joint_seg(s1["q_end"], q_above, "transport", tmin=0.6, v_cart=self.V_FREE)
        # keep the carried object high: if the joint-space arc dips > 3 cm, track the straight line instead
        zmin = min(self._fk(s1["q_end"] + (q_above - s1["q_end"]) * float(minjerk(u)))[0][2]
                   for u in np.linspace(0, 1, 11))
        if zmin < self.LIFT_Z - 0.03:
            s2 = self._cart_seg(s1["q_end"], above, y0, "transport", self.V_FREE, tmin=0.6)
        s3 = self._cart_seg(s2["q_end"], place, y0, "lower", self.V_APPROACH * 1.5, tmin=0.5)
        self._compose(q0, [s1, s2, s3], "carry")
        self.place_goal = place
        self.stage, self.corrections, self.ioff = "carry", 0, np.zeros(3)

    def _start_retreat(self, phase="retreat", then="done"):
        q0 = self.q_arm.copy()
        p0, R0 = self._fk(q0)
        up = np.array([p0[0], p0[1], min(p0[2] + self.RETREAT_DZ, 0.25)])
        s1 = self._cart_seg(q0, up, float(math.atan2(R0[1, 0], R0[0, 0])), phase, self.V_AXIAL, tmin=0.5)
        self._compose(q0, [s1], phase)
        self.stage = "retreat" if then == "done" else "reapproach_up"

    def _correct(self, goal, phase):
        """Short min-jerk correction move of the command by the measured residual (smooth integral action)."""
        tcp, _ = self._tcp()
        self.ioff = np.clip(self.ioff + (goal - tcp), -0.02, 0.02)
        q0 = self.q_arm.copy()
        _, R0 = self._fk(q0)
        yaw = self.yaw if phase == "descend" else float(math.atan2(R0[1, 0], R0[0, 0]))
        s1 = self._cart_seg(q0, goal + self.ioff, yaw, phase, self.V_APPROACH, tmin=0.25)
        self._compose(q0, [s1], f"correct_{phase}")
        self.corrections += 1

    def _grip_ramp(self, target, T):
        n = max(1, int(math.ceil(T / self.s.dt)))
        g0 = self.grip
        self.grip_plan = [g0 + (target - g0) * float(minjerk((k + 1) / n)) for k in range(n)]

    # ------------------------------------------------------------------ public sensing (touch / width)
    def _touch_count(self):
        return int((self.s._touch_values(self.r) > self.TOUCH_THR).sum())

    def _width(self):
        return self.s._grip_width(self.r)

    # ------------------------------------------------------------------ control
    def act(self) -> NativeCommand:
        dt = self.s.dt
        self.t_phase += dt
        if self.q_plan and self.k_plan < len(self.q_plan):
            self.q_arm = self.q_plan[self.k_plan].copy()
            ph = self.plan_phases[self.k_plan]
            self.k_plan += 1
            if ph != self.phase and self.stage in ("approach", "carry", "retreat", "reapproach_up"):
                self._next(ph)
        plan_done = self.k_plan >= len(self.q_plan)
        if self.grip_plan:
            self.grip = self.grip_plan.pop(0)
        self._fsm(plan_done)
        self.grip = float(np.clip(self.grip, self.g_lo, self.g_hi))
        return NativeCommand(controller_version=self.r.controller.version,
                             groups={"arm": self.q_arm.tolist(), "gripper": [self.grip]}, source=SOURCE)

    def _event(self, what):
        self.diag["events"].append(dict(t=round(float(self.s.data.time), 3), phase=self.phase, what=what))

    def _fsm(self, plan_done):
        tcp, _ = self._tcp()
        if self.stage == "approach":
            if not plan_done:
                return
            err = float(np.linalg.norm(self.grasp_goal - tcp))
            # reach-limited (the IK itself misses the goal): corrections cannot help; grasp from the closest
            # reachable pose when it is within 1.6 cm (v1 accepts a stalled 1.5 cm as well)
            reach = self.plan_reach_err > 4e-3
            d = self.grasp_goal - tcp
            # reach-limited and only too HIGH (elbow at its limit): the fingers can still take the cube's upper half
            high_only = reach and d[2] < 0 and float(np.linalg.norm(d[:2])) < 0.008 and err < 0.022
            if err < 0.006 or ((self.corrections >= 3 or reach) and err < 0.016) or high_only:
                self._begin_close()
            elif self.corrections < 3 and not reach:
                self._correct(self.grasp_goal, "descend")
            else:                                   # blocked (collision / reach): retreat and try another yaw
                self._event(f"approach_blocked err={err:.3f}")
                self._retry()
        elif self.stage == "close":
            nt = self._touch_count()
            if self.contact_grip is None:
                if nt >= 2:
                    w = self._width()
                    self.contact_grip = w
                    self.diag["contact_grip"] = w
                    tgt = (w + self.squeeze) if w is not None else self.closed_v
                    lo, hi = sorted((self.closed_v, self.open_v))
                    self._grip_ramp(float(np.clip(tgt, lo, hi)), 0.1)
                    self.settle_t = 0.0
                elif not self.grip_plan:
                    self.no_contact_t += self.s.dt
                    if self.no_contact_t > 0.3:
                        self._event("close_no_contact")
                        self._retry()
            else:
                self.settle_t += self.s.dt
                self.contact_ticks = self.contact_ticks + 1 if nt >= 2 else 0
                if self.settle_t >= 0.25 and self.contact_ticks >= 3 and self._blocked():
                    self._next("lift")
                    self.rel_grasp = None
                    self._start_carry()
                elif self.settle_t > 1.0:
                    self._event("grasp_not_confirmed")
                    self._retry()
        elif self.stage == "carry":
            # privileged (the teacher already reads object poses): object left the hand = moved > 4 cm relative to
            # the TCP since the grasp was confirmed
            cube, _ = self._body(self.obj)
            rel = cube - tcp
            if self.rel_grasp is None:
                self.rel_grasp = rel
            self.lost_ticks = self.lost_ticks + 1 if np.linalg.norm(rel - self.rel_grasp) > 0.04 else 0
            if self.phase in ("lift", "transport", "lower") and self.lost_ticks >= 3:
                self._event("object_lost")
                self._retry()
                return
            if not plan_done:
                return
            err = float(np.linalg.norm(self.place_goal - tcp))
            if err < 0.012 or self.corrections >= 3 or self.plan_reach_err > 4e-3:
                self._next("open")
                self.stage = "open"
                self._grip_ramp(self.open_v, 0.35)
            else:
                self._correct(self.place_goal, "lower")
        elif self.stage == "open":
            if not self.grip_plan and self.t_phase >= 0.5:
                self._next("retreat")
                self._start_retreat()
        elif self.stage == "retreat":
            if plan_done and self.t_phase > 0.2:
                self.done_flag = True
        elif self.stage == "reapproach_up":
            if plan_done:
                self._next("pregrasp")
                self._start_approach()

    def _blocked(self):
        """Public: the gripper is held short of its command by the object (same notion as the held_by estimate)."""
        w = self._width()
        if w is None:
            return True
        return abs(self.grip - w) > (0.004 if self.three else 0.0005) or self._touch_count() >= 2

    def _begin_close(self):
        self._next("close")
        self.stage = "close"
        self.grasp_xyz = self._tcp()[0]
        self.contact_grip = None
        self.no_contact_t = 0.0
        self.contact_ticks = 0
        span = abs(self.closed_v - self.open_v)
        self._grip_ramp(self.closed_v, max(0.35, 0.6 * span / (0.65 if self.three else 0.045)))

    def _retry(self):
        self.attempt += 1
        self.diag["attempts"] = self.attempt
        if self.attempt >= self.MAX_ATTEMPTS:
            self._event("give_up")
            self.stage = "failed"
            self.q_plan = []
            return
        self.yaw_rank += 1
        self._grip_ramp(self.open_v, 0.35)
        self._next("regrasp")
        self._start_retreat(phase="regrasp", then="approach")

    def _next(self, p):
        self.phase, self.t_phase = p, 0.0

    _STATE = ("phase", "t_phase", "grip", "q_arm", "yaw", "grasp_xyz", "ioff", "q_plan", "plan_phases", "k_plan",
              "grip_plan", "attempt", "yaw_rank", "stage", "corrections", "contact_ticks", "lost_ticks", "done_flag",
              "diag", "open_v", "rel_grasp", "plan_reach_err")
    _OPT = ("grasp_goal", "place_goal", "contact_grip", "no_contact_t", "settle_t")

    def state(self) -> dict:
        st = {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}
        st.update({k: copy.deepcopy(getattr(self, k)) for k in self._OPT if hasattr(self, k)})
        return st

    def load(self, st: dict):
        for k, v in st.items():
            setattr(self, k, copy.deepcopy(v))

    @property
    def done(self):
        return self.done_flag or self.stage == "failed"


class LimitAwarePickPlaceTeacher(SmoothPickPlaceTeacher):
    """v2lim (D-126 #8, D-114 (3)): v2 with limit-aware IK (bodies/ik.py `limit_margin`) in every hover / grasp / line
    solve, so the procedural arms keep a joint-limit margin. A different teacher version (never mixed with v2 data);
    `ik_limit_margin` may override the default 0.05 and is recorded in diag and the dataset flags."""
    IK_LIMIT_MARGIN = V2LIM_DEFAULT_MARGIN
    VERSION_KEY = "v2lim"


def ARM_TEACHERS():
    return {"v1": PickPlaceTeacher, "v2": SmoothPickPlaceTeacher, "v2lim": LimitAwarePickPlaceTeacher}


def teacher_version_id(version: str) -> str:
    return TEACHER_VERSIONS[version]


def teacher_source(version: str) -> str:
    """Canonical source label for a teacher version, e.g. scripted_teacher:pick_place_v2_minjerk."""
    return source_label(Source.SCRIPTED_TEACHER, teacher_version_id(version))


def make_arm_teacher(session, version: str = DEFAULT_ARM_TEACHER, **kw):
    cls = ARM_TEACHERS()[version]
    t = cls(session, **kw)
    t.teacher_version = teacher_version_id(version)
    return t
