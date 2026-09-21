"""Workbench sessions: the backend owns execution; the UI only requests.

Control modes (always labelled on every recorded step): hold, user, scripted_teacher,
learned:<policy>, debug. User joint/EE targets route through the validated controller.
Teleporting objects is a simulator intervention that contaminates evaluation. Graph edits
go through the versioned runtime transaction and drop queued chunks.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.action import NativeCommand
from rrp.contracts.errors import RRPError, VersionConflict, ControllerRejection
from rrp.control.ik import down_rotation
from rrp.control.teachers import PickPlaceTeacher
from rrp.sim.native import Session
from rrp.tasks.interventions import EditRejected

MODES = ("hold", "user", "scripted_teacher", "learned", "debug")


def robot_registry() -> dict:
    """Fixed selector keys only (no filesystem paths)."""
    from rrp.morphology.catalog import workbench_robots
    return workbench_robots()


def task_registry() -> dict:
    from rrp.sim.scenario import BUILDERS
    return {k: v for k, v in BUILDERS.items()}


def scene_geometry(model: mujoco.MjModel) -> dict:
    geoms = []
    for g in range(model.ngeom):
        if model.geom_rgba[g][3] == 0:
            continue
        geoms.append(dict(id=g, name=model.geom(g).name, type=int(model.geom_type[g]), body=int(model.geom_bodyid[g]),
                          size=model.geom_size[g].tolist(), pos=model.geom_pos[g].tolist(),
                          quat=model.geom_quat[g].tolist(), rgba=model.geom_rgba[g].tolist()))
    bodies = [dict(id=b, name=model.body(b).name, parent=int(model.body_parentid[b]),
                   joints=[model.joint(j).name for j in range(model.njnt) if int(model.jnt_bodyid[j]) == b])
              for b in range(model.nbody)]
    return dict(geoms=geoms, bodies=bodies)


@dataclass
class StepRecord:
    seq: int
    t: float
    source: str
    command: dict | None
    graph_version: int
    runtime_version: int
    statuses: dict
    rejected: str | None
    contaminated: bool


class WorkbenchSession:
    def __init__(self, robot_key: str, task: str, seed: int, *, render: dict | None = None):
        robots = robot_registry()
        if robot_key not in robots:
            raise RRPError(f"unknown robot {robot_key!r}", code="unknown_robot")
        builders = task_registry()
        if task not in builders:
            raise RRPError(f"unknown task {task!r}", code="unknown_task")
        self.id = uuid.uuid4().hex[:12]
        self.robot_key, self.task, self.seed = robot_key, task, int(seed)
        self.scenario = builders[task](robots[robot_key](), self.seed)
        self.sim = Session(self.scenario, seed=self.seed, render=None, auto_advance=False)
        self.lock = threading.RLock()
        self.mode = "hold"
        self.policy = None
        self.policy_name = None
        self.teacher = None
        self.running = False
        self.seq = 0
        self.timeline: list[StepRecord] = []
        self.initial_snapshot = self.sim.snapshot()
        self.contaminated = False
        self.user_targets: dict[str, list[float]] = {}
        self.render_cfg = render or {"width": 320, "height": 240, "camera": "front"}
        self._renderer = None
        self.layout: dict = {}
        self.events: list[dict] = []   # append-only transition/edit/command log
        self.listeners = []

    # ------------------------------------------------------------------ views
    def publish(self, kind: str, payload: dict, source: str = "system"):
        msg = dict(kind=kind, seq=self.seq, session_id=self.id, graph_version=self.sim.runtime.graph_version,
                   runtime_version=self.sim.runtime.runtime_version, source=source, t=float(self.sim.data.time),
                   payload=payload)
        self.events.append(dict(kind=kind, seq=self.seq, t=msg["t"], source=source))
        self.events = self.events[-5000:]
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:  # noqa: BLE001 - a broken client must not stop the backend
                pass
        return msg

    def poses(self) -> dict:
        d = self.sim.data
        return dict(xpos=d.xpos.round(5).tolist(), xquat=d.xquat.round(5).tolist(), time=float(d.time))

    def robot_summary(self) -> list[dict]:
        out = []
        for r in self.sim.robots:
            rs = r.spec
            out.append(dict(index=r.idx, name=rs.name, family=rs.family, spec_hash=rs.spec_hash, synthetic=rs.synthetic,
                            joints=[dict(address=j.address, name=j.name, type=j.type, range=j.range,
                                         mimic_of=j.mimic_of) for j in rs.joints],
                            assemblies=[dict(id=a.id, kind=a.kind, capabilities=a.capabilities, frame_site=a.frame.site,
                                             members=a.members) for a in rs.assemblies],
                            controller=dict(id=r.controller.contract.id, version=r.controller.version,
                                            groups=[dict(name=g.name, width=g.width, units=g.units, lower=g.lower,
                                                         upper=g.upper) for g in r.controller.contract.command_groups],
                                            current_targets={k: np.asarray(v).round(5).tolist() for k, v in
                                                             r.controller.current_targets(self.sim.data).items()}),
                            manipulators={ent: asm for ent, (ri, asm) in self.sim.manip_map.items() if ri == r.idx},
                            independent_controls=rs.independent_controls(),
                            generalized_coordinates=rs.generalized_coordinates(),
                            lineage=rs.lineage))
        return out

    def snapshot_view(self, privileged: bool = False) -> dict:
        with self.lock:
            obs = self.sim.observe()
            rt = self.sim.runtime
            view = dict(
                session_id=self.id, robot=self.robot_key, task=self.task, seed=self.seed, seq=self.seq,
                mode=self.mode, policy=self.policy_name, running=self.running, contaminated=self.contaminated,
                control_label=self.control_label(),
                time=float(self.sim.data.time), dt=self.sim.dt,
                graph=dict(document=rt.store.document(), version=rt.graph_version, priorities=rt.store.priorities,
                           layout=self.layout, history=[dict(request_id=h.request_id, version=h.graph_version,
                                                             provenance=h.provenance) for h in rt.store.history]),
                runtime=dict(version=rt.runtime_version,
                             events={k: dict(status=v.status, attempt=v.attempt, reason=v.reason,
                                             rejections=rt.rejections.get(k, [])[-5:])
                                     for k, v in rt.instances.items()},
                             receipts=[dict(event_id=r.event_id, attempt=r.attempt, output_name=r.output_name,
                                            type=r.type, version=r.version, valid=r.valid, value=r.value,
                                            invalid_reason=r.invalid_reason) for r in rt.receipts.all()],
                             owners=rt.resources_held(), succeeded=rt.succeeded()),
                observation=dict(objects=[d.model_dump(mode="json") for d in obs.object_descriptors],
                                 predicates=[p.model_dump(mode="json") for p in obs.predicate_estimates],
                                 channels=[dict(name=c.name, values=c.values.tolist()) for c in
                                           obs.declared_sensor_channels],
                                 joints=dict(addresses=obs.measured_node_state.joint_addresses,
                                             qpos=obs.measured_node_state.qpos.tolist())),
                robots=self.robot_summary(), poses=self.poses(),
                executor=dict(queued=len(self.sim.executor.queue), meta=self.sim.executor.meta,
                              log=self.sim.executor.log[-10:]),
                interventions=self.sim.intervention_log[-20:])
            if privileged:
                t = self.sim.truth(obs.observation_id)
                view["privileged_overlay"] = dict(label="PRIVILEGED SIMULATOR TRUTH (display only; never a policy input)",
                                                  object_poses=t.object_poses, held_by=t.held_by,
                                                  predicates=t.predicates,
                                                  event_completion_truth=t.event_completion_truth)
            return view

    def control_label(self) -> str:
        if self.mode == "learned":
            return f"LEARNED policy {self.policy_name}"
        if self.mode == "scripted_teacher":
            return "SCRIPTED TEACHER (privileged inputs) — not a learned policy"
        if self.mode == "user":
            return "USER teleoperation"
        if self.mode == "debug":
            return "DEBUG OVERRIDE — excluded from evaluation"
        return "HOLD"

    # ------------------------------------------------------------------ commands
    def set_mode(self, mode: str, policy=None, policy_name: str | None = None):
        if mode not in MODES:
            raise RRPError(f"unknown mode {mode}", code="unknown_mode")
        with self.lock:
            self.mode = mode
            self.sim.executor.invalidate(f"mode_change:{mode}", float(self.sim.data.time))
            if mode == "scripted_teacher":
                self.teacher = PickPlaceTeacher(self.sim) if self.task == "pick_place" else None
                if self.teacher is None:
                    self.mode = "hold"
                    raise RRPError(f"no scripted teacher for task {self.task}", code="no_teacher")
            if mode == "learned":
                if policy is None:
                    raise RRPError("learned mode requires a loaded policy", code="no_policy")
                self.policy, self.policy_name = policy, policy_name
            self.publish("mode_changed", dict(mode=self.mode, label=self.control_label()), source="user")

    def _command_for_step(self) -> tuple[NativeCommand | None, str]:
        r = self.sim.robots[0]
        if self.mode == "scripted_teacher" and self.teacher is not None:
            return self.teacher.act(), "scripted_teacher"
        if self.mode == "learned" and self.policy is not None:
            if not self.sim.executor.queue:
                self.policy.act(self.sim)   # submits a versioned chunk into the executor
            return None, f"learned:{self.policy_name}"
        if self.mode == "user" and self.user_targets:
            groups = copy.deepcopy(self.user_targets)
            self.user_targets = {}
            return NativeCommand(controller_version=r.controller.version, groups=groups, source="user"), "user"
        return None, self.mode

    def step(self, n: int = 1) -> list[dict]:
        out = []
        with self.lock:
            for _ in range(n):
                cmd, source = self._command_for_step()
                res = self.sim.step(cmd)
                self.seq += 1
                rt = self.sim.runtime
                rec = StepRecord(self.seq, res.time, source, res.command, rt.graph_version,
                                 rt.runtime_version, {k: v.status for k, v in rt.instances.items()}, res.rejected,
                                 self.contaminated)
                self.timeline.append(rec)
                if res.rejected:
                    self.publish("command_rejected", dict(reason=res.rejected), source=source)
                out.append(dict(seq=self.seq, t=res.time, source=source, rejected=res.rejected))
                if self.mode == "scripted_teacher" and self.teacher and self.teacher.done:
                    self.mode = "hold"
                    self.running = False
            self.publish("state_update", dict(poses=self.poses(), statuses=self.timeline[-1].statuses if
                                              self.timeline else {}, mode=self.mode,
                                              label=self.control_label(), queued=len(self.sim.executor.queue)),
                         source=out[-1]["source"] if out else "system")
        return out

    def user_joint_target(self, group: str, values: list[float]):
        with self.lock:
            r = self.sim.robots[0]
            cmd = NativeCommand(controller_version=r.controller.version, groups={group: values}, source="user")
            r.controller.validate(cmd)          # reject invalid requests before queueing
            self.mode = "user"
            self.user_targets[group] = [float(v) for v in values]

    def user_ee_target(self, pos: list[float], yaw: float = 0.0):
        with self.lock:
            r = self.sim.robots[0]
            if r.ik is None:
                raise RRPError("robot has no IK-capable manipulator", code="no_ik")
            q0 = r.controller.current_targets(self.sim.data)["arm"]
            q, err = r.ik.solve(self.sim.data.qpos.copy(), q0, np.asarray(pos, float), down_rotation(yaw))
            if err > 0.02:
                raise RRPError(f"end-effector target unreachable (IK error {err:.3f} m)", code="unreachable")
            self.user_joint_target("arm", q.tolist())

    def teleport(self, body: str, pos: list[float]):
        with self.lock:
            if not any(o.sim_body == body for o in self.scenario.objects):
                raise RRPError(f"unknown object {body}", code="unknown_object")
            self.sim.teleport_object(body, pos, source="user")
            self.contaminated = True
            self.publish("intervention", dict(kind="teleport", body=body, pos=pos, contaminates_evaluation=True),
                         source="user")

    def request_event(self, event_id: str, expected_version: int) -> dict:
        with self.lock:
            self.sim.runtime.tick(self.sim.observe())
            d = self.sim.runtime.request_event(event_id, expected_version, source="user")
            out = dict(accepted=d.accepted, event_id=event_id, reason_code=d.reason_code, detail=d.detail,
                       graph_version=d.graph_version)
            self.publish("event_request", out, source="user")
            return out

    def cancel_event(self, event_id: str):
        with self.lock:
            self.sim.runtime.cancel_event(event_id, "user_cancel")
            self.publish("event_cancelled", dict(event_id=event_id), source="user")

    def debug_force_success(self, event_id: str, confirm: bool):
        if not confirm:
            raise RRPError("debug override requires explicit confirmation", code="confirmation_required")
        with self.lock:
            inst = self.sim.runtime.instances[event_id]
            self.sim.runtime._set(inst, "succeeded", "DEBUG_OVERRIDE")
            self.contaminated = True
            self.mode = "debug" if self.mode == "hold" else self.mode
            self.publish("debug_override", dict(event_id=event_id, excluded_from_evaluation=True), source="debug")

    def apply_graph_edit(self, operations: list[dict], expected_version: int, request_id: str) -> dict:
        with self.lock:
            try:
                rec = self.sim.runtime.apply_edit(operations, expected_version, request_id, provenance="user")
            except VersionConflict as e:
                self.publish("command_rejected", dict(reason="version_conflict", **e.context), source="user")
                raise
            except EditRejected as e:
                self.publish("command_rejected", dict(reason=e.code, message=str(e)), source="user")
                raise
            if self.teacher is not None:
                self.teacher = PickPlaceTeacher(self.sim) if self.mode == "scripted_teacher" else self.teacher
            payload = dict(request_id=rec.request_id, graph_version=rec.graph_version, content_hash=rec.content_hash,
                           operations=rec.operations, provenance=rec.provenance, queue_dropped=True)
            self.publish("graph_committed", payload, source="user")
            return payload

    def reset(self, seed: int | None = None):
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
                self.scenario = task_registry()[self.task](robot_registry()[self.robot_key](), self.seed)
                self.sim = Session(self.scenario, seed=self.seed, auto_advance=False)
            else:
                self.sim.restore(self.initial_snapshot)
            self.initial_snapshot = self.sim.snapshot()
            self.timeline, self.seq, self.contaminated, self.mode = [], 0, False, "hold"
            self.teacher = None
            self.publish("session_reset", dict(seed=self.seed), source="user")

    # ------------------------------------------------------------------ replay
    def export_episode(self) -> dict:
        with self.lock:
            return dict(session_id=self.id, robot=self.robot_key, task=self.task, seed=self.seed,
                        contaminated=self.contaminated,
                        steps=[rec.__dict__ for rec in self.timeline],
                        graph_history=[dict(request_id=h.request_id, version=h.graph_version,
                                            operations=h.operations, provenance=h.provenance)
                                       for h in self.sim.runtime.store.history],
                        interventions=self.sim.intervention_log,
                        reproduce=f"rrp replay --episode <this file>  # robot={self.robot_key} task={self.task} "
                                  f"seed={self.seed}")

    def physics_replay(self) -> dict:
        """Re-simulate recorded commands from the initial snapshot on a fresh session and compare."""
        with self.lock:
            fresh = Session(task_registry()[self.task](robot_registry()[self.robot_key](), self.seed),
                            seed=self.seed, auto_advance=False)
            fresh.restore(self.initial_snapshot)
            r = fresh.robots[0]
            maxdev = 0.0
            for rec in self.timeline:
                cmd = NativeCommand(controller_version=r.controller.version, groups=rec.command,
                                    source="user") if rec.command else None
                fresh.step(cmd)
            ok = bool(np.allclose(fresh.data.qpos, self.sim.data.qpos, atol=1e-6))
            maxdev = float(np.abs(fresh.data.qpos - self.sim.data.qpos).max())
            return dict(steps=len(self.timeline), final_state_match=ok, max_abs_qpos_deviation=maxdev,
                        note="exact only when no graph edits/teleports/policy chunks occurred mid-episode"
                        if self.contaminated else "")

    def render_frame(self) -> bytes | None:
        """Low-load stream mode: server-rendered JPEG (software GL on host, GPU on peer)."""
        import io
        from PIL import Image
        with self.lock:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.sim.model, self.render_cfg["height"], self.render_cfg["width"])
            self._renderer.update_scene(self.sim.data, camera=self.render_cfg.get("camera", "front"))
            px = self._renderer.render()
        buf = io.BytesIO()
        Image.fromarray(px).save(buf, format="JPEG", quality=70)
        return buf.getvalue()
