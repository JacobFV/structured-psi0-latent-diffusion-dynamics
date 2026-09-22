"""Multi-robot / dual-arm session.

`DualSession` extends the native `Session` without changing its single-robot behaviour:
  * manipulator handles: one per task manipulator entity (`left`, `right`), each resolving to
    (robot index, gripper assembly, arm/gripper command groups, arm joints, IK solver, touch
    sensors, width sensor, public base frame). Works for two separately mounted arms AND for
    one dual-arm body with two gripper assemblies (e.g. menagerie ALOHA).
  * PUBLIC estimators for the support_insert / handover predicates, computed only from
    detector measurements, FK of measured joints, touch/width sensors and declared geometry:
      frame_estimate_valid, held_by, reachable, lateral_error_m, axis_error_rad,
      insertion_depth_m, supported, inside.
    Hole-relative predicates are evaluated in the frame of the RECEIPT bound to the consuming
    event (`frame_binding`), never in "the latest frame of that type".
  * privileged evaluator for physical insertion (truth bus only).
  * multi-robot command path: `step({robot_idx: NativeCommand})` (native) plus a flat
    `{"r<i>:<group>": values}` form used by the multi-robot featurizer/action space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import NativeCommand
from rrp.contracts.observation import SensorChannel, PredicateEstimate
from rrp.contracts.task import EntityBinding, OutputBinding
from rrp.control.ik import IKSolver
from .native import Session
from .scenario import Scenario
from .sensors import camera_visibility, read_sensor

EST_VERSION = "rrp.sim.dual.public_estimators/v1"
STATIC_FEATURE_MIN_N = 12          # detections averaged before a static feature frame is valid
STATIC_FEATURE_RESET_M = 0.015     # a jump larger than this restarts the average (feature moved)
ANCHOR_SLIP_M = 0.012              # maintained contact anchor invalid when the TCP slides further
ANCHOR_LOSS_S = 0.3                # contact must be lost this long before the anchor is invalid
EXTRA_PUBLIC_PREDICATES = ("supported", "inside")


@dataclass
class ManipHandle:
    entity: str
    robot: int
    prefix: str
    assembly: str
    tcp_site: str
    arm_group: str
    grip_group: str
    arm_joints: list
    touch: list
    width_sensor: str | None
    home: list | None
    base_pos: np.ndarray            # public: declared arm base frame (FK of the fixed base)
    reach_m: float
    gripper_kind: str               # parallel | three_finger | aloha
    open_value: float
    closed_value: float
    ik: IKSolver | None = None
    home_azimuth: float = 0.0
    gripper_params: dict = field(default_factory=dict)


def _full(prefix, n):
    return n if n.startswith(prefix) else prefix + n


def build_handles(scenario: Scenario) -> dict[str, ManipHandle]:
    m = scenario.model
    out = {}
    for ri, mr in enumerate(scenario.robots):
        rs, meta, prefix = mr.robot_spec, mr.meta, mr.prefix
        contract = rs.controller_contracts[0]
        addr2j = {a.address: a.joint for a in rs.actuators}
        jaddr2name = {j.address: j.name for j in rs.joints}
        declared = {d["assembly"]: d for d in meta.get("manipulators", [])}
        for ent, asm_id in mr.manipulator_bindings.items():
            asm = next(a for a in rs.assemblies if a.id == asm_id)
            d = declared.get(asm_id)
            if d is None:     # single-arm robot: conventional group names
                d = dict(arm_group="arm", grip_group="gripper", touch_sensors=meta.get("touch_sensors", []),
                         width_sensor=meta.get("width_sensor"), home=meta.get("home"))
            groups = {g.name: g for g in contract.command_groups}
            ag, gg = groups[d["arm_group"]], groups[d["grip_group"]]
            arm_joints = [jaddr2name[addr2j[a]] for a in ag.actuators]
            gp = meta.get("gripper_params") or {}
            kind = d.get("gripper_kind") or gp.get("kind", "parallel")
            if kind == "three_finger":
                ov, cv = gg.lower[0], gg.upper[0]
            else:
                ov, cv = gg.upper[0], gg.lower[0]
            j0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, arm_joints[0])
            dd = mujoco.MjData(m)
            mujoco.mj_kinematics(m, dd)
            base = dd.xpos[m.jnt_bodyid[j0]].copy()
            reach = float(d.get("reach_m") or meta.get("reach_m") or sum(meta.get("params", {}).get("lengths", (0.8,))))
            out[ent] = ManipHandle(ent, ri, prefix, asm_id, asm.frame.site, d["arm_group"], d["grip_group"], arm_joints,
                                   [_full(prefix, t) for t in d.get("touch_sensors", [])],
                                   _full(prefix, d["width_sensor"]) if d.get("width_sensor") else None,
                                   d.get("home"), base, reach, kind, float(ov), float(cv),
                                   gripper_params=dict(gp) if kind != "aloha" else dict(d.get("gripper_params", {})))
    for h in out.values():
        h.ik = IKSolver(m, h.tcp_site, h.arm_joints)
    return out


def _home_azimuths(model: mujoco.MjModel, handles: dict[str, ManipHandle]):
    """Azimuth of each TCP at its declared home pose, around its own arm base (IK seeding).
    NOTE: never write homes into model.qpos0 -- MuJoCo hinge kinematics are relative to qpos0."""
    d = mujoco.MjData(model)
    for h in handles.values():
        if h.home:
            for jn, v in zip(h.arm_joints, h.home):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                d.qpos[model.jnt_qposadr[jid]] = v
    mujoco.mj_kinematics(model, d)
    for h in handles.values():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, h.tcp_site)
        p = d.site_xpos[sid] - h.base_pos
        h.home_azimuth = float(math.atan2(p[1], p[0]))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class DualSession(Session):
    """Session over several manipulators; single-robot behaviour of the base class is untouched."""

    def __init__(self, scenario: Scenario, **kw):
        self.handles = build_handles(scenario)
        _home_azimuths(scenario.model, self.handles)
        self._static: dict[int, list] = {}
        self._inhand: dict[tuple, dict] = {}
        self._anchors: dict[tuple, dict] = {}
        self.fixture_init = None
        super().__init__(scenario, **kw)
        # the base class built one IK per robot from the first gripper; handles carry per-arm IK

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed=None):
        self._static, self._inhand, self._anchors = {}, {}, {}
        obs = super().reset(seed)
        if self._apply_declared_home():
            obs = self._last_obs
        self.fixture_init = self._body_pose("fixture") if self._has_body("fixture") else None
        return obs

    def _apply_declared_home(self) -> bool:
        """Homes for manipulators the base Session does not know (multi-arm bodies)."""
        todo = [h for h in self.handles.values() if h.home and not self.robots[h.robot].arm_joints]
        if not todo:
            return False
        m, d = self.model, self.data
        for h in todo:
            r = self.robots[h.robot]
            for jn, v in zip(h.arm_joints, h.home):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                d.qpos[m.jnt_qposadr[jid]] = v
                d.qvel[m.jnt_dofadr[jid]] = 0
            d.ctrl[r.controller.act_ids[h.arm_group]] = h.home
        mujoco.mj_forward(m, d)
        for r in self.robots:
            r.controller.prev = r.controller.current_targets(d)
            r.controller.target = dict(r.controller.prev)
        for _ in range(50):
            mujoco.mj_step(m, d)
        self._sense()
        self._last_obs = self.observe()
        return True

    def _has_body(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

    def _body_pose(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy()

    # ------------------------------------------------------------------ public sensing
    def _sense(self):
        meas = {}
        for i, o in enumerate(self.detectables):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)
            p = self.data.xpos[bid].copy()
            vis = camera_visibility(self.model, self.data, self.det_cfg.camera, o.sim_body, p, self.det_cfg.max_range)
            if vis and self.env_rng.random() > self.det_cfg.dropout:
                meas[i] = p + self.env_rng.normal(0, self.det_cfg.pos_sigma, 3)
            else:
                meas[i] = None
        self.tracker.update(float(self.data.time), meas)
        # static-feature averaging (public estimator for fixed features such as a hole)
        for i, o in enumerate(self.detectables):
            if o.kind != "feature" or meas.get(i) is None:
                continue
            acc = self._static.setdefault(i, [])
            if acc and np.linalg.norm(meas[i] - np.mean(acc, 0)) > STATIC_FEATURE_RESET_M + 3 * self.det_cfg.pos_sigma:
                acc.clear()
            acc.append(meas[i])
            del acc[:-60]
        if hasattr(self, "runtime"):
            self._update_inhand()

    def tcp_pose(self, ent: str):
        h = self.handles[ent]
        return self._fk_site(self.robots[h.robot], h.tcp_site)

    def touch_values(self, ent: str) -> np.ndarray:
        h = self.handles[ent]
        vals = [read_sensor(self.model, self.data, n) for n in h.touch]
        return np.array([float(v[0]) if v is not None else 0.0 for v in vals])

    def grip_width(self, ent: str) -> float | None:
        h = self.handles[ent]
        if not h.width_sensor:
            return None
        v = read_sensor(self.model, self.data, h.width_sensor)
        return float(v[0]) if v is not None else None

    def _closed_cmd(self, ent: str) -> bool:
        h = self.handles[ent]
        r = self.robots[h.robot]
        cmd = float(self.data.ctrl[r.controller.act_ids[h.grip_group][0]])
        g = r.controller.groups[h.grip_group]
        lo, hi = g.lower[0], g.upper[0]
        if h.gripper_kind == "three_finger":
            return bool(cmd > lo + 0.25 * (hi - lo))
        w = self.grip_width(ent)
        return bool(cmd < lo + 0.5 * (hi - lo) and (w is None or w > lo + 0.002))

    def held_estimate(self, obj: str, ent: str) -> bool:
        touch = self.touch_values(ent)
        touch_ok = int((touch > 0.2).sum()) >= min(2, len(touch)) if len(touch) else False
        pos, _ = self._track(obj)
        tcp, _ = self.tcp_pose(ent)
        near = True if pos is None else float(np.linalg.norm(pos - tcp)) < 0.10
        return bool(touch_ok and self._closed_cmd(ent) and near)

    def _update_inhand(self):
        """Public in-hand pose belief: running mean of (detected object - FK tcp) in the TCP
        frame while the object is estimated held; cleared when it is released."""
        for ent in self.handles:
            for i, o in enumerate(self.detectables):
                if o.kind != "object":
                    continue
                obj = next((e for e, s in self.entity_slots.items() if s == i), None)
                if obj is None:
                    continue
                key = (obj, ent)
                if not self.held_estimate(obj, ent):
                    self._inhand.pop(key, None)
                    continue
                tr = self.tracker.tracks[i]
                if not tr.visible or tr.mean is None:
                    continue
                tcp, R = self.tcp_pose(ent)
                rel = R.T @ (np.array(tr.mean) - tcp)
                b = self._inhand.setdefault(key, dict(n=0, mean=np.zeros(3)))
                b["n"] += 1
                b["mean"] = b["mean"] + (rel - b["mean"]) / min(b["n"], 50)

    def inhand_offset(self, obj: str, ent: str) -> np.ndarray | None:
        b = self._inhand.get((obj, ent))
        return None if b is None or b["n"] < 3 else b["mean"].copy()

    # ------------------------------------------------------------------ bound frames
    def _bound_frame(self, target_entity: str):
        """Hole frame from the receipt bound (frame_binding) to an event targeting this entity.
        Prefers active events. Returns (pos, R, receipt) or None."""
        evs = self.runtime.compiled.definition.events
        cands = []
        for e in evs:
            fb = e.frame_binding
            if fb is None:
                continue
            if not any(s.role == "target" and isinstance(s.binding, EntityBinding) and s.binding.entity.id == target_entity
                       for s in e.roles):
                continue
            st = self.runtime.instances[e.id].status if e.id in self.runtime.instances else "pending"
            cands.append((0 if st == "active" else 1, e.id, fb))
        for _, _, fb in sorted(cands):
            r = self.runtime.receipts.latest(fb.event_id, fb.attempt, fb.output_name)
            if r is not None and r.valid:
                return np.array(r.value["pos"]), _quat_to_mat(r.value["quat_wxyz"]), r
        return None

    def peg_estimate(self, obj: str):
        """(bottom point, axis pointing into the insertion direction) from public quantities."""
        holder = next((e for e in self.handles if self.held_estimate(obj, e)), None)
        if holder is None:
            return None
        off = self.inhand_offset(obj, holder)
        if off is None:
            return None
        tcp, R = self.tcp_pose(holder)
        g = self.scenario.meta.get("declared_geometry", {}).get(obj, {})
        axis = R[:, 2]                    # grasped along the approach axis (declared grasp convention)
        centre = tcp + R @ off
        return centre + axis * g.get("half_length", 0.0), axis

    # ------------------------------------------------------------------ public estimators
    def estimate(self, predicate: str, args: list[str]):
        if predicate == "held_by" and len(args) == 2 and args[1] in self.handles:
            return self.held_estimate(args[0], args[1]), True, 0.9
        if predicate == "reachable" and len(args) == 2 and args[0] in self.handles:
            p, _ = self._track(args[1])
            if p is None:
                return None, False, 0.0
            h = self.handles[args[0]]
            d = float(np.linalg.norm(p[:2] - h.base_pos[:2]))
            return bool(0.1 < d < 0.9 * h.reach_m), True, 0.8
        if predicate == "frame_estimate_valid" and len(args) == 1:
            s = self.entity_slots.get(args[0])
            if s is None:
                return None, False, 0.0
            acc = self._static.get(s, [])
            ok = len(acc) >= STATIC_FEATURE_MIN_N and self.tracker.tracks[s].visible
            return bool(ok), True, 0.9
        if predicate in ("lateral_error_m", "axis_error_rad", "insertion_depth_m", "inside") and len(args) == 2:
            fr = self._bound_frame(args[1])
            pe = self.peg_estimate(args[0])
            if fr is None or pe is None:
                return None, False, 0.0
            hp, hR, _ = fr
            bottom, axis = pe
            return self._hole_metric(predicate, hp, hR[:, 2], bottom, axis, args[1]), True, 0.8
        if predicate == "supported" and len(args) == 1:
            p, _ = self._track(args[0])
            if p is None:
                return None, False, 0.0
            half = self.scenario.meta.get("declared_geometry", {}).get(args[0], {}).get("half_extent", 0.1)
            for ent in self.handles:
                t = self.touch_values(ent)
                tcp, _ = self.tcp_pose(ent)
                if len(t) and t.max() > 0.2 and np.all(np.abs(tcp[:2] - p[:2]) < half * 1.5) and \
                        tcp[2] < p[2] + 0.06 and not any(
                            self.held_estimate(o, ent) for o, sl in self.entity_slots.items()
                            if o != args[0] and self.detectables[sl].kind == "object"):
                    return True, True, 0.8
            return False, True, 0.8
        return super().estimate(predicate, args)

    def _hole_metric(self, predicate, hp, n_up, bottom, axis, hole_ent):
        d = bottom - hp
        depth = float(-(d @ n_up))
        lat = float(np.linalg.norm(d - (d @ n_up) * n_up))
        ang = float(math.acos(np.clip(axis @ -n_up, -1, 1)))
        if predicate == "lateral_error_m":
            return lat
        if predicate == "axis_error_rad":
            return ang
        if predicate == "insertion_depth_m":
            return depth
        hw = self.scenario.meta.get("declared_geometry", {}).get(hole_ent, {}).get("half_width", 0.014)
        return bool(depth >= 0.02 and lat < hw)

    def _task_predicates(self):
        out = super()._task_predicates()
        seen = {(p, tuple(a)) for p, a in out}
        for e in self.runtime.compiled.definition.events:
            for c in e.desired_effects:
                if c.predicate in EXTRA_PUBLIC_PREDICATES:
                    args = [a.entity.id for a in c.arguments if isinstance(a, EntityBinding)]
                    if (c.predicate, tuple(args)) not in seen:
                        seen.add((c.predicate, tuple(args)))
                        out.append((c.predicate, args))
        return out

    def observe(self):
        obs = super().observe()
        # per-assembly sensor channels for bodies carrying several grippers (e.g. ALOHA)
        t = float(self.data.time)
        for ent, h in self.handles.items():
            if self.scenario.robots[h.robot].meta.get("manipulators"):
                tv = self.touch_values(ent)
                obs.declared_sensor_channels.append(SensorChannel(
                    name=f"{h.robot}:{h.assembly}:touch", kind="touch", values=tv,
                    mask=np.ones_like(tv, dtype=bool), timestamp=t))
                w = self.grip_width(ent)
                if w is not None:
                    obs.declared_sensor_channels.append(SensorChannel(
                        name=f"{h.robot}:{h.assembly}:grip_width", kind="gripper_width", values=np.array([w]),
                        mask=np.ones(1, dtype=bool), timestamp=t))
        return obs

    # ------------------------------------------------------------------ receipts
    def _output_provider(self, event_id, output_name, output_type, obs):
        ev = self.runtime.compiled.event(event_id)
        if output_type == "frame_estimate":
            ent = next((s.binding.entity.id for s in ev.roles if s.role == "patient"
                        and isinstance(s.binding, EntityBinding)), None)
            slot = self.entity_slots.get(ent)
            acc = self._static.get(slot, [])
            if slot is None or len(acc) < STATIC_FEATURE_MIN_N:
                return None
            A = np.array(acc)
            var = (self.det_cfg.pos_sigma ** 2) / len(A)
            return {"pos": A.mean(0).tolist(), "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "frame": "world",
                    "estimator": "static_feature_average/v1", "n_detections": int(len(A)),
                    "axis_source": "declared_fixture_up", "covariance_diag": [var] * 3 + [0.02 ** 2] * 3}
        if output_type == "contact_anchor":
            actor = next((s.binding.entity.id for s in ev.roles if s.role == "actor"
                          and isinstance(s.binding, EntityBinding)), None)
            if actor not in self.handles:
                return None
            key = (event_id, actor)
            tv = self.touch_values(actor)
            tcp, R = self.tcp_pose(actor)
            prev = self._anchors.get(key)
            in_contact = bool(len(tv) and tv.max() >= 0.2)
            if not in_contact:
                # contact lost -> anchor invalid, debounced over ANCHOR_LOSS_S (contact chatter)
                if prev is None:
                    return None
                prev.setdefault("_lost_since", float(self.data.time))
                if float(self.data.time) - prev["_lost_since"] >= ANCHOR_LOSS_S:
                    self._anchors.pop(key, None)
                    return None
                return {k: v for k, v in prev.items() if not k.startswith("_")}
            if prev is not None:
                prev.pop("_lost_since", None)
            if prev is not None:
                if np.linalg.norm(tcp - np.array(prev["pos"])) <= ANCHOR_SLIP_M:
                    return {k: v for k, v in prev.items() if not k.startswith("_")}   # unchanged: no churn
                self._anchors.pop(key, None)
                return None                       # anchor slipped -> invalid; re-established next tick
            a = {"pos": [round(float(x), 4) for x in tcp], "normal": [round(float(x), 4) for x in -R[:, 2]],
                 "tangent": None, "tangent_yaw_uncertain": True, "frame": "world",
                 "covariance_diag": [1e-4, 1e-4, 1e-4]}
            self._anchors[key] = dict(a)
            return dict(a)
        return super()._output_provider(event_id, output_name, output_type, obs)

    # ------------------------------------------------------------------ privileged truth
    def truth_predicate(self, pred, args, held=None):
        if pred in ("lateral_error_m", "axis_error_rad", "insertion_depth_m", "inside") and len(args) == 2:
            p, R = self._body_pose(args[0])
            hp, hR = self._body_pose(args[1])
            g = self.scenario.meta["declared_geometry"][args[0]]
            axis = -R[:, 2] if (R[:, 2] @ hR[:, 2]) > 0 else R[:, 2]     # peg axis pointing into the hole
            return self._hole_metric(pred, hp, hR[:, 2], p + axis * g["half_length"], axis, args[1])
        if pred == "held_by" and len(args) == 2 and args[1] in self.handles:
            held = held if held is not None else self._held_truth(self.truth().contacts)
            o = next((x for x in self.scenario.objects if x.task_entity == args[0]), None)
            return bool(o and o.sim_body in held.get(args[1], []))
        return super().truth_predicate(pred, args, held)

    def insertion_truth(self) -> dict:
        p, R = self._body_pose("peg")
        hp, hR = self._body_pose("hole")
        axis = -R[:, 2] if (R[:, 2] @ hR[:, 2]) > 0 else R[:, 2]
        bottom = p + axis * self.scenario.meta["declared_geometry"]["peg"]["half_length"]
        d = bottom - hp
        depth = float(-(d @ hR[:, 2]))
        lat = float(np.linalg.norm(d - (d @ hR[:, 2]) * hR[:, 2]))
        fp, fR = self._body_pose("fixture")
        f0 = self.fixture_init[0] if self.fixture_init else fp
        return dict(depth=depth, lateral=lat, axis_err=float(math.acos(np.clip(axis @ -hR[:, 2], -1, 1))),
                    fixture_shift=float(np.linalg.norm(fp[:2] - f0[:2])), fixture_tilt=float(math.acos(np.clip(fR[2, 2], -1, 1))))

    def privileged_success(self) -> bool:
        if self.scenario.name == "support_insert":
            t = self.insertion_truth()
            hw = self.scenario.meta["declared_geometry"]["hole"]["half_width"]
            return bool(t["depth"] >= 0.02 and t["lateral"] < hw and t["fixture_tilt"] < 0.1)
        return super().privileged_success()

    # ------------------------------------------------------------------ multi-robot commands
    def command_from_flat(self, flat: dict, source: str) -> dict[int, NativeCommand]:
        """{"r<i>:<group>": values} -> {robot_idx: NativeCommand} (multi-robot action space)."""
        per: dict[int, dict] = {}
        for k, v in flat.items():
            ri, g = k.split(":", 1)
            per.setdefault(int(ri[1:]), {})[g] = list(v)
        return {i: NativeCommand(controller_version=self.robots[i].controller.version, groups=g, source=source)
                for i, g in per.items()}
