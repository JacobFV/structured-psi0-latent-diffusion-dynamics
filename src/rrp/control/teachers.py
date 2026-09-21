"""Scripted teachers. They read PRIVILEGED simulator state (object poses) and are labelled
`scripted_teacher` everywhere. Their traces are demonstrations, never learned-policy results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import NativeCommand
from rrp.control.ik import down_rotation
from rrp.contracts.errors import ControllerRejection

SOURCE = "scripted_teacher"


def _yaw_of_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclass
class TeacherResult:
    actions: list = field(default_factory=list)          # native commands per control step
    observations: list = field(default_factory=list)     # public observation ids
    phases: list = field(default_factory=list)
    success: bool = False                                  # public runtime success
    privileged_evaluator_success: bool = False
    controller_source: str = SOURCE
    privileged_inputs: bool = True
    steps: int = 0
    failure_reason: str | None = None
    rejected_commands: int = 0
    feasibility: dict | None = None


class PickPlaceTeacher:
    """FSM over TCP waypoints solved by DLS IK; uses privileged cube/target poses."""

    def __init__(self, session, robot: int = 0, speed: float = 0.35, obj: str = "cube", zone: str = "target_zone",
                 rng=None):
        self.s = session
        self.r = session.robots[robot]
        self.obj, self.zone = obj, zone
        self.speed = speed
        self.phase = "pregrasp"
        self.t_phase = 0.0
        self.tcp_cmd = None
        self.rng = rng or np.random.default_rng(0)
        g = self.r.controller.groups["gripper"]
        self.open_v, self.closed_v = g.upper[0], g.lower[0]
        gp = self.r.meta.get("gripper_params") or {}
        if gp.get("kind") == "three_finger":
            # module-specific strategy: close to the angle where fingertips meet the object
            # side plus a squeeze margin (closing fully would curl the object into the palm)
            half = self.s.scenario.object(obj).size[0]
            reach = (gp.get("finger_base_radius", 0.052) - half) / gp.get("finger_length", 0.055)
            theta = float(np.arcsin(np.clip(reach, 0.0, 1.0)))
            self.open_v, self.closed_v = g.lower[0], min(g.upper[0], theta + 0.1)
        self.grip = self.open_v
        self.q_arm = self.r.controller.current_targets(session.data)["arm"].copy()
        self.yaw = 0.0
        self.tcp_site = self.r.tcp_sites[next(a.id for a in self.r.spec.assemblies if a.kind in ("gripper", "hand"))]
        self.base = np.array(session.scenario.robots[robot].base_pos)
        self.grasp_xyz = None
        self.ioff = np.zeros(3)
        self.last_tcp = None

    def _body(self, name):
        m, d = self.s.model, self.s.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return d.xpos[bid].copy(), d.xquat[bid].copy()

    def _goal(self):
        cube, cq = self._body(self.obj)
        zone, _ = self._body(self.zone)
        h = self.s.scenario.object(self.obj).size[2]
        hover = 0.13
        if self.phase in ("pregrasp",):
            return np.array([cube[0], cube[1], cube[2] + hover])
        if self.phase in ("descend",):
            return np.array([cube[0], cube[1], cube[2] + 0.004])
        if self.phase == "close":
            return self.grasp_xyz.copy()
        if self.phase == "lift":
            return np.array([self.grasp_xyz[0], self.grasp_xyz[1], 0.18])
        if self.phase == "transport":
            return np.array([zone[0], zone[1], 0.18])
        if self.phase in ("lower", "open"):
            return np.array([zone[0], zone[1], h + 0.012])
        return np.array([zone[0], zone[1], 0.2])

    def act(self) -> NativeCommand:
        dt = self.s.dt
        self.t_phase += dt
        tcp_now, _ = self.s._fk_site(self.r, self.tcp_site)
        if self.tcp_cmd is None:
            self.tcp_cmd = tcp_now.copy()
        cube, cq = self._body(self.obj)
        if self.phase in ("pregrasp", "descend"):
            self.yaw = _yaw_of_quat(cq)
            self.yaw = (self.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4   # cube symmetry
        goal_true = self._goal()
        # integral correction of steady-state tracking error (gravity/finite servo gains)
        err_vec = goal_true - tcp_now
        if np.linalg.norm(self.tcp_cmd - goal_true) < 1e-6 and np.linalg.norm(err_vec) < 0.04:
            self.ioff = np.clip(self.ioff + 0.25 * err_vec, -0.04, 0.04)
        goal = goal_true + self.ioff
        d = goal - self.tcp_cmd
        n = np.linalg.norm(d)
        step = self.speed * dt
        self.tcp_cmd = goal.copy() if n <= step else self.tcp_cmd + d / n * step
        err = np.linalg.norm(goal_true - tcp_now)
        speed = np.linalg.norm(tcp_now - self.last_tcp) / dt if self.last_tcp is not None else 1.0
        self.last_tcp = tcp_now
        stalled = n < 1e-6 and speed < 0.01 and self.t_phase > 1.0
        if self.phase == "pregrasp" and (err < 0.015 or (stalled and err < 0.03)) and n < 1e-6:
            self._next("descend")
        elif self.phase == "descend" and (err < 0.008 or (stalled and err < 0.015)) and n < 1e-6:
            self._next("close")
        elif self.phase == "close":
            self.grip = self.closed_v
            if self.t_phase > 0.7:
                self._next("lift")
        elif self.phase == "lift" and n < 1e-6 and (err < 0.02 or (stalled and err < 0.04)):
            self._next("transport")
        elif self.phase == "transport" and n < 1e-6 and (err < 0.015 or (stalled and err < 0.03)):
            self._next("lower")
        elif self.phase == "lower" and n < 1e-6 and (err < 0.01 or (stalled and err < 0.02)):
            self._next("open")
        elif self.phase == "open":
            self.grip = self.open_v
            if self.t_phase > 0.6:
                self._next("retreat")
        q, _ = self.r.ik.solve(self.s.data.qpos.copy(), self.q_arm, self.tcp_cmd, down_rotation(self.yaw),
                               seeds=self._ik_seeds(self.tcp_cmd))
        self.q_arm = q
        return NativeCommand(controller_version=self.r.controller.version,
                             groups={"arm": q.tolist(), "gripper": [float(self.grip)]}, source=SOURCE)

    _STATE = ("phase", "t_phase", "tcp_cmd", "grip", "q_arm", "yaw", "grasp_xyz", "ioff", "last_tcp")

    def state(self) -> dict:
        import copy
        return {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}

    def load(self, st: dict):
        import copy
        for k in self._STATE:
            setattr(self, k, copy.deepcopy(st[k]))

    def _next(self, p):
        self.ioff = np.zeros(3) if p in ("transport", "retreat") else self.ioff
        if p == "close":
            self.grasp_xyz = self.tcp_cmd.copy()   # freeze: later goals must not chase the held object
        self.phase, self.t_phase = p, 0.0

    def _ik_seeds(self, target):
        home = np.array(self.r.meta.get("home") or np.zeros(len(self.q_arm)))
        yaw = float(np.arctan2(target[1] - self.base[1], target[0] - self.base[0]))
        # the asset's zero-yaw convention may differ: rotate the base joint by the azimuth
        # difference between the target and the home TCP
        dyaw = yaw - float(self.r.meta.get("home_tcp_azimuth", 0.0))
        seeds = []
        for flip in (0.0, 0.3, -0.3):
            s = home.copy()
            s[0] = home[0] + dyaw + flip
            seeds.append(s)
        return seeds

    def feasibility(self, tol: float = 0.012) -> dict:
        """Analytic check BEFORE execution: IK must reach every waypoint (tool down).
        Uses the declared kinematic model; infeasible episodes are recorded, not attempted."""
        cube, cq = self._body(self.obj)
        zone, _ = self._body(self.zone)
        h = self.s.scenario.object(self.obj).size[2]
        wps = {"pregrasp": [cube[0], cube[1], cube[2] + 0.13], "grasp": [cube[0], cube[1], cube[2] + 0.004],
               "lift": [cube[0], cube[1], 0.18], "transport": [zone[0], zone[1], 0.18],
               "place": [zone[0], zone[1], h + 0.012]}
        errs = {}
        q = self.q_arm.copy()
        for k, wp in wps.items():
            wp = np.array(wp)
            q, e = self.r.ik.solve(self.s.data.qpos.copy(), q, wp, down_rotation(0.0), iters=120,
                                   seeds=self._ik_seeds(wp))
            errs[k] = float(e)
        bad = {k: v for k, v in errs.items() if v > tol}
        return {"feasible": not bad, "ik_errors": errs, "unreachable": sorted(bad)}

    @property
    def done(self):
        return self.phase == "retreat" and self.t_phase > 0.8


def run_teacher_episode(session, teacher, max_control_steps: int = 600, record_obs: bool = False,
                        check_feasibility: bool = True) -> TeacherResult:
    res = TeacherResult()
    if check_feasibility and hasattr(teacher, "feasibility"):
        f = teacher.feasibility()
        res.feasibility = f
        if not f["feasible"]:
            res.failure_reason = f"infeasible:{','.join(f['unreachable'])}"
            return res
    for k in range(max_control_steps):
        cmd = teacher.act()
        res.actions.append(cmd.groups)
        res.phases.append(teacher.phase)
        out = session.step(cmd)
        if out.rejected:
            res.rejected_commands += 1
        if record_obs:
            res.observations.append(out.observation.observation_id)
        res.steps = k + 1
        if teacher.done:
            break
    session.step(None)
    res.success = session.runtime.succeeded()
    res.privileged_evaluator_success = session.privileged_success()
    if not res.privileged_evaluator_success:
        res.failure_reason = f"ended_in_phase:{teacher.phase}"
    return res


def run_fixture_pick_place(seed: int = 0, max_control_steps: int = 600, gripper: str = "parallel",
                           n_distractors: int = 0) -> TeacherResult:
    from rrp.sim.fixtures import make_pick_place_session
    s = make_pick_place_session(seed=seed, gripper=gripper, n_distractors=n_distractors)
    return run_teacher_episode(s, PickPlaceTeacher(s), max_control_steps)
