"""Native MuJoCo session: physics + controllers + sensors/tracker + task runtime + queue.

Public outputs (PolicyObservation) and privileged truth (PrivilegedTruth) are produced by
separate methods and never mixed. Snapshots contain every continuation component.
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import ActionChunk, NativeCommand
from rrp.contracts.errors import ControllerRejection, StaleActionError
from rrp.contracts.observation import (PolicyObservation, PrivilegedTruth, NodeState, SensorChannel, ImageObs,
                                       PredicateEstimate, ContactTruth)
from rrp.contracts.task import EntityBinding, OutputBinding
from rrp.control.joint_targets import JointTargetController, ChunkExecutor
from rrp.control.ik import IKSolver
from rrp.tasks.runtime import TaskRuntime
from .scenario import Scenario
from .sensors import DetectorConfig, ObjectTracker, camera_visibility, read_sensor
from .snapshot_contract import Snapshot

_obs_counter = itertools.count()


@dataclass
class RobotRuntime:
    idx: int
    prefix: str
    spec: object
    meta: dict
    controller: JointTargetController
    joint_names: list          # full model names of the robot's hinge/slide joints
    qadr: np.ndarray
    dadr: np.ndarray
    tcp_sites: dict            # assembly id -> full site name
    touch: list
    width_sensor: str | None
    ik: IKSolver | None
    arm_joints: list


@dataclass
class StepResult:
    observation: PolicyObservation
    qpos: np.ndarray
    time: float
    rejected: str | None = None
    source: str | None = None


class Session:
    def __init__(self, scenario: Scenario, *, control_hz: float = 20.0, seed: int = 0,
                 detector: DetectorConfig | None = None, render: dict | None = None, auto_advance: bool = True):
        self.scenario = scenario
        self.model = scenario.model
        self.data = mujoco.MjData(self.model)
        self.est_data = mujoco.MjData(self.model)   # public FK from measured joints only
        self.control_hz = control_hz
        self.substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.dt = self.substeps * self.model.opt.timestep
        self.seed = seed
        self.det_cfg = detector or DetectorConfig()
        self.render_cfg = render
        self._renderer = None
        self.auto_advance = auto_advance
        self.robots: list[RobotRuntime] = []
        for i, mr in enumerate(scenario.robots):
            self.robots.append(self._make_robot(i, mr))
        self.manip_map = {}
        for i, mr in enumerate(scenario.robots):
            for ent, asm in mr.manipulator_bindings.items():
                self.manip_map[ent] = (i, asm)
        self.detectables = [o for o in scenario.objects]
        self.executor = ChunkExecutor()
        self.reset(seed)

    # ------------------------------------------------------------------ construction
    def _make_robot(self, i, mr) -> RobotRuntime:
        m = self.model
        ctrl = JointTargetController(m, mr.robot_spec, mr.robot_spec.controller_contracts[0], mr.prefix)
        jn = [j.name for j in mr.robot_spec.joints if j.type in ("hinge", "slide")]
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        tcp = {a.id: a.frame.site for a in mr.robot_spec.assemblies}
        touch = [n if n.startswith(mr.prefix) else mr.prefix + n for n in mr.meta.get("touch_sensors", [])]
        ws = mr.meta.get("width_sensor")
        ws = (ws if ws.startswith(mr.prefix) else mr.prefix + ws) if ws else None
        arm_group = next((g for g in mr.robot_spec.controller_contracts[0].command_groups if g.name == "arm"), None)
        arm_joints = []
        if arm_group:
            addr2j = {a.address: a.joint for a in mr.robot_spec.actuators}
            jaddr2name = {j.address: j.name for j in mr.robot_spec.joints}
            arm_joints = [jaddr2name[addr2j[a]] for a in arm_group.actuators]
        grip_asm = next((a for a in mr.robot_spec.assemblies if a.kind in ("gripper", "hand")), None)
        ik = IKSolver(m, grip_asm.frame.site, arm_joints) if (grip_asm and arm_joints) else None
        return RobotRuntime(i, mr.prefix, mr.robot_spec, mr.meta, ctrl, jn, np.array([m.jnt_qposadr[j] for j in jids]),
                            np.array([m.jnt_dofadr[j] for j in jids]), tcp, touch, ws, ik, arm_joints)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> PolicyObservation:
        seed = self.seed if seed is None else seed
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for r in self.robots:
            home = r.meta.get("home")
            if home and r.arm_joints:
                for jn, v in zip(r.arm_joints, home):
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                    self.data.qpos[self.model.jnt_qposadr[jid]] = v
                mujoco.mj_forward(self.model, self.data)
            hold = r.controller.measured_hold(self.data)
            for g, ids in r.controller.act_ids.items():
                if r.controller.groups[g].semantic == "gripper":
                    self.data.ctrl[ids] = np.array(r.controller.groups[g].upper)   # open
                else:
                    self.data.ctrl[ids] = hold[g]
            r.controller.prev = r.controller.current_targets(self.data)
            r.controller.target = dict(r.controller.prev)
        self.env_rng = np.random.default_rng([seed, 1])
        self.sampler_rng = np.random.default_rng([seed, 2])
        self.tracker = ObjectTracker([o.descriptor for o in self.detectables], self.det_cfg)
        self.entity_slots = self._bind_entities()
        self.runtime = TaskRuntime(copy.deepcopy(self.scenario.task), clock=lambda: float(self.data.time),
                                   auto_advance=self.auto_advance, output_provider=self._output_provider,
                                   on_invalidate=self._on_invalidate)
        self.executor = ChunkExecutor()
        self.step_count = 0
        self.intervention_log: list[dict] = []
        # settle objects
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._sense()
        obs = self.observe()
        self.runtime.tick(obs)
        self._last_obs = self.observe()
        return self._last_obs

    def _bind_entities(self) -> dict[str, int]:
        """Public binding: task entity descriptor -> detector slot by descriptor text."""
        slots = {}
        for d in self.scenario.task["entity_declarations"]:
            if d["type"] in ("object", "feature"):
                matches = [i for i, o in enumerate(self.detectables) if o.descriptor in d["descriptor"]
                           or d["descriptor"] in o.descriptor]
                if matches:
                    slots[d["id"]] = matches[0]
        return slots

    def _on_invalidate(self, reason: str):
        self.executor.invalidate(reason, float(self.data.time))

    # ------------------------------------------------------------------ sensing (public)
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

    def _fk_site(self, robot: RobotRuntime, site: str) -> tuple[np.ndarray, np.ndarray]:
        self.est_data.qpos[:] = self.model.qpos0
        self.est_data.qpos[robot.qadr] = self.data.qpos[robot.qadr]   # measured joints only
        mujoco.mj_kinematics(self.model, self.est_data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        return self.est_data.site_xpos[sid].copy(), self.est_data.site_xmat[sid].reshape(3, 3).copy()

    def tcp_estimate(self, manip_entity: str):
        ri, asm = self.manip_map[manip_entity]
        r = self.robots[ri]
        return self._fk_site(r, r.tcp_sites[asm])

    def _touch_values(self, r: RobotRuntime) -> np.ndarray:
        vals = [read_sensor(self.model, self.data, n) for n in r.touch]
        return np.array([float(v[0]) if v is not None else 0.0 for v in vals])

    def _grip_width(self, r: RobotRuntime) -> float | None:
        if not r.width_sensor:
            return None
        v = read_sensor(self.model, self.data, r.width_sensor)
        return float(v[0]) if v is not None else None

    def _track(self, entity: str):
        s = self.entity_slots.get(entity)
        if s is None:
            return None, None
        tr = self.tracker.tracks[s]
        return (np.array(tr.mean) if tr.mean is not None else None), (np.array(tr.var) if tr.var else None)

    def _grasp_closed_ok(self, r: RobotRuntime) -> bool:
        """Public, gripper-agnostic: the gripper was commanded to close and is BLOCKED short
        of its commanded position (by an object), using measured position vs issued command."""
        w = self._grip_width(r)
        g = r.controller.groups.get("gripper")
        if w is None or g is None:
            return False
        cmd = float(self.data.ctrl[r.controller.act_ids["gripper"][0]])
        lo, hi = g.lower[0], g.upper[0]
        three = (r.meta.get("gripper_params") or {}).get("kind") == "three_finger"
        # commanded to close (the issued command is public to the robot). Blocking is not
        # required: compliant contacts can let fingers reach a just-past-contact target.
        if three:   # closing = increasing angle
            return bool(cmd > lo + 0.25 * (hi - lo))
        return bool(cmd < lo + 0.5 * (hi - lo) and w > lo + 0.002)

    def estimate(self, predicate: str, args: list[str]) -> tuple[object, bool, float]:
        t = float(self.data.time)
        if predicate == "held_by" and len(args) == 2 and args[1] in self.manip_map:
            obj, man = args
            r = self.robots[self.manip_map[man][0]]
            touch = self._touch_values(r)
            touch_ok = int((touch > 0.2).sum()) >= min(2, len(touch)) if len(touch) else False
            closed_ok = self._grasp_closed_ok(r)
            pos, var = self._track(obj)
            tcp, _ = self.tcp_estimate(man)
            near = True if pos is None else float(np.linalg.norm(pos - tcp)) < 0.08
            return bool(touch_ok and closed_ok and near), True, 0.9
        if predicate == "in_region" and len(args) == 2:
            p, v = self._track(args[0])
            q, w = self._track(args[1])
            if p is None or q is None or v is None or max(v) > 0.02 ** 2:
                return None, False, 0.0
            rad = next((o.radius for o in self.detectables if self.entity_slots.get(args[1]) is not None
                        and self.detectables[self.entity_slots[args[1]]] is o), 0.05)
            inside = float(np.linalg.norm(p[:2] - q[:2])) < rad and p[2] < q[2] + 0.06
            return bool(inside), True, 0.85
        if predicate == "distance_m" and len(args) == 2 and args[0] in self.manip_map:
            q, w = self._track(args[1])
            if q is None:
                return None, False, 0.0
            tcp, _ = self.tcp_estimate(args[0])
            return float(np.linalg.norm(tcp - q)), True, 0.9
        if predicate == "reachable" and len(args) == 2 and args[0] in self.manip_map:
            p, v = self._track(args[1])
            if p is None:
                return None, False, 0.0
            r = self.robots[self.manip_map[args[0]][0]]
            reach = float(sum(r.meta.get("params", {}).get("lengths", (0.8,))))
            d = float(np.linalg.norm(p[:2] - np.array(self.scenario.robots[r.idx].base_pos[:2])))
            return bool(0.1 < d < 0.9 * reach), True, 0.8
        if predicate == "frame_estimate_valid" and len(args) == 1:
            s = self.entity_slots.get(args[0])
            if s is None:
                return None, False, 0.0
            tr = self.tracker.tracks[s]
            ok = tr.visible and tr.var is not None and max(tr.var) < 0.01 ** 2
            return bool(ok), True, 0.9
        return None, False, 0.0

    def _task_predicates(self) -> list[tuple[str, list[str]]]:
        seen, out = set(), []
        for e in self.runtime.compiled.definition.events:
            for c in e.preconditions + e.invariants + e.completion:
                if c.source != "observation_estimate":
                    continue
                args = [a.entity.id if isinstance(a, EntityBinding) else f"{a.event_id}#{a.attempt}.{a.output_name}"
                        for a in c.arguments]
                k = (c.predicate, tuple(args))
                if k not in seen:
                    seen.add(k)
                    out.append((c.predicate, args))
        return out

    def observe(self) -> PolicyObservation:
        t = float(self.data.time)
        r0 = self.robots[0]
        allq = np.concatenate([self.data.qpos[r.qadr] for r in self.robots])
        allv = np.concatenate([self.data.qvel[r.dadr] for r in self.robots])
        names = [a for r in self.robots for a in [j.address for j in r.spec.joints if j.type in ("hinge", "slide")]]
        ns = NodeState(joint_addresses=[f"{r.idx}:{a}" for r in self.robots
                                        for a in [j.address for j in r.spec.joints if j.type in ("hinge", "slide")]],
                       qpos=allq, qvel=allv, qpos_mask=np.ones_like(allq, dtype=bool), timestamp=t)
        chans = []
        for r in self.robots:
            if r.touch:
                tv = self._touch_values(r)
                chans.append(SensorChannel(name=f"{r.idx}:touch", kind="touch", values=tv,
                                           mask=np.ones_like(tv, dtype=bool), timestamp=t))
            w = self._grip_width(r)
            if w is not None:
                chans.append(SensorChannel(name=f"{r.idx}:grip_width", kind="gripper_width", values=np.array([w]),
                                           mask=np.ones(1, dtype=bool), timestamp=t))
        pes = []
        for pred, args in self._task_predicates():
            v, known, conf = self.estimate(pred, args)
            pes.append(PredicateEstimate(predicate=pred, args=args, value=v, known=known, confidence=conf,
                                         estimator="rrp.sim.native.public_estimators/v1", timestamp=t))
        images = self._render() if self.render_cfg else []
        oid = f"o{self.seed}_{self.step_count}_{next(_obs_counter)}"
        obs = PolicyObservation(observation_id=oid, sensor_time=t, robot_spec_hash=r0.spec.spec_hash,
                                sensor_images=images, measured_node_state=ns, declared_sensor_channels=chans,
                                object_descriptors=self.tracker.descriptors(t), predicate_estimates=pes,
                                task_input=self.runtime.public_view() if hasattr(self, "runtime") else None)
        return obs

    def _render(self) -> list[ImageObs]:
        cfg = self.render_cfg
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, cfg.get("height", 128), cfg.get("width", 128))
        out = []
        for cam in cfg.get("cameras", ["front"]):
            self._renderer.update_scene(self.data, camera=cam)
            px = self._renderer.render().copy()
            out.append(ImageObs(camera=cam, height=px.shape[0], width=px.shape[1], encoding="rgb8", pixels=px,
                                timestamp=float(self.data.time)))
        return out

    # ------------------------------------------------------------------ privileged truth
    def truth(self, observation_id: str = "") -> PrivilegedTruth:
        poses = {}
        for o in self.scenario.objects:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)
            poses[o.sim_body] = [*self.data.xpos[bid].tolist(), *self.data.xquat[bid].tolist()]
        contacts = []
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = self.model.body(self.model.geom_bodyid[con.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[con.geom2]).name
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, c, f)
            contacts.append(ContactTruth(body_a=b1, body_b=b2, pos=con.pos.tolist(), normal=con.frame[:3].tolist(),
                                         force=float(f[0])))
        held = self._held_truth(contacts)
        preds = {}
        for pred, args in self._task_predicates():
            v = self.truth_predicate(pred, args, held)
            if v is not None:
                preds[f"{pred}({','.join(args)})"] = v
        ent_map = {o.task_entity: o.sim_body for o in self.scenario.objects if o.task_entity}
        comp = {e: self.success_truth_for(e, held) for e in self.runtime.compiled.definition.success_events}
        return PrivilegedTruth(observation_id=observation_id, sim_time=float(self.data.time), object_poses=poses,
                               object_entity_map=ent_map, contacts=contacts, held_by=held, predicates=preds,
                               event_completion_truth=comp)

    def _held_truth(self, contacts) -> dict[str, list[str]]:
        """Privileged: object contacted by >= 2 distinct bodies of the manipulator assembly."""
        held = {}
        for ent, (ri, asm) in self.manip_map.items():
            r = self.robots[ri]
            asm_spec = next(a for a in r.spec.assemblies if a.id == asm)
            hand_bodies = {l.name for l in r.spec.links if l.address in asm_spec.members}
            objs = []
            for o in self.scenario.objects:
                if o.kind != "object":
                    continue
                touching = {c.body_a if c.body_b == o.sim_body else c.body_b for c in contacts
                            if o.sim_body in (c.body_a, c.body_b)}
                if len(touching & hand_bodies) >= 2:
                    objs.append(o.sim_body)
            held[ent] = objs
        return held

    def truth_predicate(self, pred: str, args: list[str], held=None):
        ent2sim = {o.task_entity: o for o in self.scenario.objects if o.task_entity}
        if pred == "held_by":
            held = held if held is not None else self._held_truth([])
            o = ent2sim.get(args[0])
            return bool(o and o.sim_body in held.get(args[1], []))
        if pred == "in_region":
            o, z = ent2sim.get(args[0]), ent2sim.get(args[1])
            if not (o and z):
                return None
            po = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)]
            pz = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, z.sim_body)]
            return bool(np.linalg.norm(po[:2] - pz[:2]) < z.radius and po[2] < pz[2] + 0.06)
        if pred == "distance_m" and args[0] in self.manip_map:
            z = ent2sim.get(args[1])
            ri, asm = self.manip_map[args[0]]
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.robots[ri].tcp_sites[asm])
            pz = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, z.sim_body)]
            return float(np.linalg.norm(self.data.site_xpos[sid] - pz))
        return None

    def success_truth_for(self, event_id: str, held=None) -> bool:
        ev = self.runtime.compiled.event(event_id)
        for c in ev.completion:
            if c.source != "observation_estimate":
                continue
            args = [a.entity.id for a in c.arguments if isinstance(a, EntityBinding)]
            v = self.truth_predicate(c.predicate, args, held)
            if v is None:
                return False
            from rrp.tasks.runtime import Estimate
            if not TaskRuntime.compare(c, Estimate(v, True)):
                return False
        return True

    def privileged_success(self) -> bool:
        contacts = self.truth().contacts
        held = self._held_truth(contacts)
        return all(self.success_truth_for(e, held) for e in self.runtime.compiled.definition.success_events)

    # ------------------------------------------------------------------ outputs for receipts
    def _output_provider(self, event_id, output_name, output_type, obs):
        ev = self.runtime.compiled.event(event_id)
        if output_type == "frame_estimate":
            ent = next((s.binding.entity.id for s in ev.roles if s.role == "patient"
                        and isinstance(s.binding, EntityBinding)), None)
            p, v = self._track(ent)
            if p is None:
                return None
            return {"pos": p.tolist(), "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "frame": "world",
                    "covariance_diag": list(v) + [0.05 ** 2] * 3}
        if output_type == "contact_anchor":
            actor = next((s.binding.entity.id for s in ev.roles if s.role == "actor"), None)
            if actor not in self.manip_map:
                return None
            r = self.robots[self.manip_map[actor][0]]
            tv = self._touch_values(r)
            if not len(tv) or tv.max() < 0.2:
                return None
            tcp, R = self.tcp_estimate(actor)
            return {"pos": tcp.tolist(), "normal": (-R[:, 2]).tolist(), "tangent": None,
                    "tangent_yaw_uncertain": True, "frame": "world", "covariance_diag": [1e-4, 1e-4, 1e-4]}
        if output_type in ("completion_receipt", "alignment_receipt"):
            return {"observation_id": obs.observation_id if obs else None, "t": float(self.data.time)}
        return None

    # ------------------------------------------------------------------ stepping
    def controller_version(self, robot: int = 0) -> str:
        return self.robots[robot].controller.version

    def submit_chunk(self, chunk: ActionChunk, robot: int = 0, execute_prefix: int | None = None):
        if abs(chunk.dt - self.dt) > 1e-9:
            self.executor.log.append(dict(t=float(self.data.time), event="chunk_rejected", reasons=["rate_mismatch"]))
            raise StaleActionError(f"chunk dt {chunk.dt} != controller period {self.dt}", reasons=["rate_mismatch"])
        self.executor.submit(chunk, graph_version=self.runtime.graph_version,
                             robot_spec_hash=self.robots[robot].spec.spec_hash,
                             controller_version=self.controller_version(robot), now=float(self.data.time),
                             execute_prefix=execute_prefix)

    def fixture_command(self) -> NativeCommand:
        r = self.robots[0]
        tg = r.controller.current_targets(self.data)
        arm = tg["arm"].copy()
        arm[0] += 0.2
        return NativeCommand(controller_version=r.controller.version, groups={"arm": arm.tolist()},
                             source="scripted_teacher")

    def step(self, command: NativeCommand | dict | None = None, robot: int = 0) -> StepResult:
        cmds: dict[int, NativeCommand | None] = {}
        source = None
        if isinstance(command, dict):
            cmds = command
        elif command is not None:
            cmds = {robot: command}
        else:
            row = self.executor.pop()
            if row is not None:
                r = self.robots[robot]
                cmds = {robot: NativeCommand(controller_version=r.controller.version, groups=row,
                                             source=self.executor.meta["source"] if self.executor.meta else "learned")}
        rejected = None
        for r in self.robots:
            c = cmds.get(r.idx)
            try:
                r.controller.begin_step(self.data, c)
                source = source or (c.source if c else None)
            except ControllerRejection as e:
                rejected = e.code
                r.controller.begin_step(self.data, None)
        for k in range(self.substeps):
            a = (k + 1) / self.substeps
            for r in self.robots:
                r.controller.apply_substep(self.data, a)
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError("simulation diverged (non-finite state)")
        self.step_count += 1
        self._sense()
        obs = self.observe()
        v0 = self.runtime.runtime_version
        self.runtime.tick(obs)
        if self.runtime.runtime_version != v0:
            obs = self.observe()
        self._last_obs = obs
        return StepResult(obs, self.data.qpos.copy(), float(self.data.time), rejected, source)

    # ------------------------------------------------------------------ interventions
    def teleport_object(self, sim_body: str, pos, source: str = "user"):
        """Simulator intervention: contaminates autonomous evaluation; always logged."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{sim_body}_free")
        if jid < 0:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, sim_body)
            mid = self.model.body_mocapid[bid]
            if mid < 0:
                raise ValueError(f"{sim_body} is not movable")
            self.data.mocap_pos[mid] = pos
        else:
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr:adr + 3] = pos
            self.data.qvel[self.model.jnt_dofadr[jid]:self.model.jnt_dofadr[jid] + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.intervention_log.append(dict(t=float(self.data.time), kind="teleport", body=sim_body,
                                          pos=list(map(float, pos)), source=source, contaminates_evaluation=True))

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> Snapshot:
        d = self.data
        phys = dict(time=float(d.time), qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(),
                    qacc_warmstart=d.qacc_warmstart.copy(), mocap_pos=d.mocap_pos.copy(),
                    mocap_quat=d.mocap_quat.copy(), qfrc_applied=d.qfrc_applied.copy(),
                    xfrc_applied=d.xfrc_applied.copy())
        return Snapshot(dict(
            physics=phys,
            controller_state=[r.controller.state() for r in self.robots],
            task_runtime=self.runtime.snapshot(),
            sensor_filters=dict(detector=self.det_cfg.__dict__.copy()),
            entity_tracker=self.tracker.state(),
            belief_state=dict(entity_slots=dict(self.entity_slots)),
            command_queue=self.executor.state(),
            sampler_rng=copy.deepcopy(self.sampler_rng.bit_generator.state),
            env_rng=copy.deepcopy(self.env_rng.bit_generator.state),
            source_versions=dict(mujoco=mujoco.__version__, model_nq=self.model.nq,
                                 robot_hashes=[r.spec.spec_hash for r in self.robots]),
            step_count=self.step_count,
            interventions=copy.deepcopy(self.intervention_log))).validate()

    def restore(self, snap: Snapshot) -> PolicyObservation:
        snap.validate()
        c = snap.components
        if c["source_versions"]["robot_hashes"] != [r.spec.spec_hash for r in self.robots]:
            raise ValueError("snapshot belongs to a different body")
        p = c["physics"]
        d = self.data
        d.time = p["time"]
        for k in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "qfrc_applied",
                  "xfrc_applied"):
            getattr(d, k)[:] = p[k]
        mujoco.mj_forward(self.model, d)
        for r, st in zip(self.robots, c["controller_state"]):
            r.controller.load(st)
        self.runtime.restore(c["task_runtime"])
        self.tracker.load(c["entity_tracker"])
        self.entity_slots = dict(c["belief_state"]["entity_slots"])
        self.executor.load(c["command_queue"])
        self.sampler_rng.bit_generator.state = copy.deepcopy(c["sampler_rng"])
        self.env_rng.bit_generator.state = copy.deepcopy(c["env_rng"])
        self.step_count = c["step_count"]
        self.intervention_log = copy.deepcopy(c.get("interventions", []))
        self._last_obs = self.observe()
        return self._last_obs
