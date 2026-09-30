"""Legged scenario builder (`waypoint_contact`) and LeggedSession.

LeggedSession reuses the native Session machinery (task runtime, detector/tracker, receipts,
public/privileged split, snapshots) and replaces the arm-specific parts:

* Command interface: the policy-facing group is `base_velocity` = [vx, vy, wz] (body frame,
  m/s, m/s, rad/s) at 10 Hz; a body-specific tracker (learned or scripted CPG) converts it to
  joint position targets at 50 Hz. Raw `legs` joint-target commands are also accepted (the
  joint_targets contract) for teachers/diagnostics that bypass the tracker.
* Public sensors: joint encoders, IMU (quat/gyro/acc), foot touch, a declared noisy
  localization sensor (base x, y, yaw; sigma 2 cm / 0.02 rad), and an overhead-camera detector
  for waypoint markers. Public predicates: distance_m(body, feature), upright(body),
  stance_fraction(body), base_speed(body).
* Privileged truth: true base pose, contacts, truth versions of the same predicates.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.core.observation import NodeState, PolicyObservation, SensorChannel, PredicateEstimate
from rrp.core.robot import CommandGroup, ControllerContract
from rrp.envs.mujoco.joint_targets import ChunkExecutor, JointTargetController
from rrp.envs.mujoco.legged_core import LeggedBinding, quat_rotate_inv, yaw_of
from rrp.envs.mujoco.legged_tracker import load_tracker
from rrp.bodies.compiler import compile_robot_spec
from rrp.bodies.generators import Module
from rrp.bodies.legged import legged_body, legged_world
from rrp.tasks.runtime import TaskRuntime
from rrp.envs.base import ActionSpace, StepResult
from rrp.envs.mujoco.session import RobotRuntime, Session, _obs_counter
from rrp.envs.mujoco.scenario import MountedRobot, ObjectDecl, Scenario, load_task
from rrp.envs.mujoco.sensors import DetectorConfig, ObjectTracker

WAYPOINT_COLORS = {"orange": (0.95, 0.5, 0.1, 0.9), "cyan": (0.1, 0.8, 0.85, 0.9)}
TRACKER_HZ = 50.0


def _waypoint(scene, name, xy, rgba):
    b = scene.worldbody.add_body(name=name, pos=[xy[0], xy[1], 0.0], mocap=True)
    b.add_geom(name=f"{name}_disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.12, 0.004, 0], pos=[0, 0, 0.004],
               rgba=list(rgba), contype=0, conaffinity=0)
    b.add_geom(name=f"{name}_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.15, 0], pos=[0, 0, 0.15],
               rgba=list(rgba), contype=0, conaffinity=0)
    b.add_site(name=f"{name}_site", pos=[0, 0, 0])
    return b


def tracker_contract(meta: dict, rs) -> ControllerContract:
    L = meta["legged"]
    r = L["command_ranges"]
    acts = [a.address for a in rs.actuators if a.name[3:] in L["policy_actuators"]]
    return ControllerContract(id="legged_tracker", version="lt-1.0", kind="legged_tracker", rate_hz=10.0,
                              command_groups=[CommandGroup(name="base_velocity", width=3, units="mixed",
                                                           semantic="base_velocity", actuators=acts,
                                                           lower=[r["vx"][0], r["vy"][0], r["wz"][0]],
                                                           upper=[r["vx"][1], r["vy"][1], r["wz"][1]],
                                                           hold="zero_velocity")],
                              state_required=["imu", "qpos", "qvel"], body_specific=True)


def build_waypoint_contact(robot, seed: int, task: dict | None = None, body_key: str | None = None,
                           contact: str | None = None, terrain: dict | None = None) -> Scenario:
    """robot: Module from rrp.morphology.legged (or a body key string). contact: foot-floor contact model
    ('v1' legacy default | 'v2'); None -> $RRP_CONTACT_MODEL or v1. Recorded in meta/scenario meta.
    terrain: rough-terrain flag for rrp.bodies.legged.legged_world (W6 robustness; default None = flat, unchanged)."""
    from rrp.bodies.contact import version_str
    if isinstance(robot, str):
        body_key, robot = robot, legged_body(robot)
    if not isinstance(robot, Module) or "legged" not in robot.meta:
        raise TypeError("waypoint_contact needs a legged Module (rrp.morphology.legged)")
    rng = np.random.default_rng(seed)
    meta = copy.deepcopy(robot.meta)
    body_key = body_key or meta["name"]
    scene = legged_world(f"waypoint_contact_{seed}", meta.get("source_options"), contact=contact, terrain=terrain)
    meta["contact_model"] = version_str(contact)
    scene.worldbody.add_camera(name="overhead", pos=[0, 0, 12.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=100)
    scene.worldbody.add_camera(name="front", pos=[-3.0, -3.0, 2.5], xyaxes=[0.707, -0.707, 0, 0.35, 0.35, 0.87],
                               fovy=60)
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(robot.spec.copy(), prefix="r0_", site=site)
    vmax = meta["legged"]["command_ranges"]["vx"][1]
    reach = float(np.clip(vmax * 7.0, 1.2, 3.0))
    da = reach * rng.uniform(0.6, 1.0)
    ba = rng.uniform(-math.pi / 3, math.pi / 3)
    pa = np.array([da * math.cos(ba), da * math.sin(ba)])
    db = reach * rng.uniform(0.5, 0.9)
    bb = ba + rng.uniform(-math.pi / 2, math.pi / 2)
    pb = pa + np.array([db * math.cos(bb), db * math.sin(bb)])
    _waypoint(scene, "waypoint_a", pa, WAYPOINT_COLORS["orange"])
    _waypoint(scene, "waypoint_b", pb, WAYPOINT_COLORS["cyan"])
    scene.memory = 8 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body"})
    objects = [ObjectDecl("waypoint_a", "orange waypoint marker", "feature", radius=0.12, task_entity="waypoint_a"),
               ObjectDecl("waypoint_b", "cyan waypoint marker", "feature", radius=0.12, task_entity="waypoint_b")]
    return Scenario("waypoint_contact", task or load_task("waypoint_contact"), scene, model, [mr], objects, seed,
                    meta=dict(body_key=body_key, waypoints=dict(a=pa.tolist(), b=pb.tolist()), reach_m=reach,
                              contact_model=meta["contact_model"],
                              **({"terrain": dict(terrain)} if terrain and float(terrain.get("amp_m", 0)) > 0 else {})))


class LeggedSession(Session):
    """env_id "mujoco/legged": the `base_velocity` space drives the embedded tracker (learned or CPG, recorded in
    spec.provenance["controllers"])."""
    ENV_ID = "mujoco/legged"
    LOC_SIGMA = 0.02
    YAW_SIGMA = 0.02

    def __init__(self, scenario: Scenario, *, tracker_kind: str = "auto", seed: int = 0, actuator_mode: str | None = None,
                 actuator_latency_ms: float | None = None, base_state_source: str = "truth_noise", estimator_cfg=None,
                 control: str = "base_velocity", **kw):
        """actuator_mode (D-126 #14): None -> $RRP_ACTUATOR_MODE, else rrp.physics.actuator.ACTUATOR_MODE_DEFAULT ("ideal": the
        bounded PD servo, byte-identical to before). "v1lat" / "v2" route every tracker tick through ActuatorModel (nominal
        parameters, a fixed per-episode latency: actuator_latency_ms, else $RRP_ACTUATOR_LATENCY_MS, else drawn from the seed).
        base_state_source (D-126 #27): "truth_noise" (default, unchanged) = speed from the declared noisy
        localization sensor; "estimator" = speed from rrp.envs.state_estimator (IMU + leg kinematics + contact),
        and NodeState.base_vel_estimate is filled. Localization still provides x, y, yaw for waypoint geometry.
        control (D-140): "base_velocity" (default, unchanged) = 10 Hz body-velocity commands through the embedded tracker;
        "legs" = one step() per 50 Hz tracker tick carrying absolute joint targets for the policy joints (group "legs"),
        no tracker (the body tracker stays loaded as `body_tracker`). Observations, sensing, the task runtime and the
        fall check keep their 10 Hz schedule (`boundary` is True on the ticks where they ran); with no command the
        declared fallback holds the default stance (also during the reset settle)."""
        if control not in ("base_velocity", "legs"):
            raise ValueError(f"control {control!r} not in ('base_velocity', 'legs')")
        self.control = control
        from rrp.bodies.actuator import resolve_mode
        from rrp.envs.mujoco.state_estimator import BASE_STATE_SOURCES
        if base_state_source not in BASE_STATE_SOURCES:
            raise ValueError(f"base_state_source {base_state_source!r} not in {BASE_STATE_SOURCES}")
        self.base_state_source = base_state_source
        self._estimator_cfg = estimator_cfg
        self.base_estimator = None
        self.tracker_kind = tracker_kind
        self.actuator_mode = resolve_mode(actuator_mode)
        self._act_latency_req = actuator_latency_ms
        self.actuator_model = None
        self.actuator_latency_ms = None
        kw.setdefault("detector", DetectorConfig(camera="overhead", pos_sigma=0.01, dropout=0.02, max_range=40.0))
        kw.setdefault("control_hz", 10.0)
        super().__init__(scenario, seed=seed, **kw)

    def _make_robot(self, i, mr) -> RobotRuntime:
        m = self.model
        jt = next(c for c in mr.robot_spec.controller_contracts if c.kind == "joint_targets")
        ctrl = JointTargetController(m, mr.robot_spec, jt, mr.prefix)
        jn = [j.name for j in mr.robot_spec.joints if j.type in ("hinge", "slide")]
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        touch = [mr.prefix + n for n in mr.meta["legged"]["touch_sensors"]]
        self.binding = LeggedBinding(m, mr.meta, mr.prefix)
        self.body_key = self.scenario.meta["body_key"]
        self.tracker = load_tracker(self.body_key, self.binding, mr.meta, self.tracker_kind)
        tc = next(c for c in mr.robot_spec.controller_contracts if c.kind == "legged_tracker")
        self.tracker_contract = tc
        self.tracker_version_str = f"{tc.id}:{tc.version}:{self.tracker.version}:{mr.robot_spec.spec_hash}"
        self.body_tracker = self.tracker
        if self.control == "legs":
            self.tracker = DirectTargets(self.binding)
            self.tracker_version_str = f"{jt.id}:{jt.version}:legs_direct:{mr.robot_spec.spec_hash}"
        if self.actuator_mode != "ideal":
            from rrp.bodies.actuator import ActuatorModel
            self.actuator_model = ActuatorModel(m, self.binding, 1, None, name=mr.meta["name"], randomize=False,
                                                latency_ms=0.0, mode=self.actuator_mode)
        return RobotRuntime(i, mr.prefix, mr.robot_spec, mr.meta, ctrl, jn, np.array([m.jnt_qposadr[j] for j in jids]),
                            np.array([m.jnt_dofadr[j] for j in jids]), {"body": mr.prefix + mr.meta["legged"]["imu"]["site"]},
                            touch, None, None, [])

    def controller_version(self, robot: int = 0) -> str:
        return self.tracker_version_str

    def _action_spaces(self) -> list[ActionSpace]:
        if self.control == "legs":
            b = self.binding
            return [ActionSpace(group="legs", kind="joint_position", width=int(b.n), robot=0, rate_hz=TRACKER_HZ,
                                low=np.asarray(b.lo, float).tolist(), high=np.asarray(b.hi, float).tolist(), units="rad")]
        return [ActionSpace.from_group(g, robot=0, rate_hz=self.control_hz) for g in self.tracker_contract.command_groups]

    @property
    def settle_ticks(self) -> int:
        """Tracker ticks run by reset() before the episode starts (0.3 s under the zero command / default-stance hold)."""
        return int(0.3 * TRACKER_HZ)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> PolicyObservation:
        seed = self.seed if seed is None else seed
        mujoco.mj_resetData(self.model, self.data)
        b = self.binding
        b.set_default(self.data, yaw=0.0)
        mujoco.mj_forward(self.model, self.data)
        if self.actuator_model is not None:
            from rrp.bodies.actuator import resolve_latency_ms
            self.actuator_latency_ms = resolve_latency_ms(seed, self._act_latency_req)
            self.actuator_model.fixed_latency_ms = self.actuator_latency_ms
            self.actuator_model.reset(0, self.data.ctrl[b.pol_act].copy())
        r = self.robots[0]
        r.controller.prev = r.controller.current_targets(self.data)
        r.controller.target = dict(r.controller.prev)
        self.tracker.reset(0.0)
        self.cmd = np.zeros(3)
        self.env_rng = np.random.default_rng([seed, 1])
        self.sampler_rng = np.random.default_rng([seed, 2])
        self.tracker_obj = ObjectTracker([o.descriptor for o in self.detectables], self.det_cfg)
        self.tracker_objects = self.tracker_obj
        self.tracker_ = None
        self.entity_slots = self._bind_entities()
        self.runtime = TaskRuntime(copy.deepcopy(self.scenario.task), clock=lambda: float(self.data.time),
                                   auto_advance=self.auto_advance, output_provider=self._output_provider,
                                   on_invalidate=self._on_invalidate)
        self.executor = ChunkExecutor()
        self.step_count = 0
        self.intervention_log = []
        self.fell = False
        self.loc = None
        self.loc_hist = []
        self.speed_est = float("nan")
        if self.base_state_source == "estimator":
            self._reset_estimator()
        # settle 0.3 s under the tracker holding a zero command
        for _ in range(int(0.3 * TRACKER_HZ)):
            self._tracker_tick(np.zeros(3))
        self._sense()
        obs = self.observe()
        self.runtime.tick(obs)
        self._last_obs = self.observe()
        self._legs_ticks = 0
        self.boundary = True
        return self._last_obs

    # the base class names its object tracker `self.tracker`; here `self.tracker` is the locomotion
    # tracker, so the detector-tracker lives in `self.tracker_obj` and the base helpers are overridden.
    def _sense(self):
        from rrp.envs.mujoco.sensors import camera_visibility
        meas = {}
        for i, o in enumerate(self.detectables):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)
            p = self.data.xpos[bid].copy()
            vis = camera_visibility(self.model, self.data, self.det_cfg.camera, o.sim_body, p, self.det_cfg.max_range)
            meas[i] = (p + self.env_rng.normal(0, self.det_cfg.pos_sigma, 3)) if (
                vis and self.env_rng.random() > self.det_cfg.dropout) else None
        self.tracker_obj.update(float(self.data.time), meas)
        # declared localization sensor (noisy base pose); speed from filtered differences
        b = self.binding
        q = self.data.qpos
        x = np.array([q[b.qa] + self.env_rng.normal(0, self.LOC_SIGMA), q[b.qa + 1] + self.env_rng.normal(0, self.LOC_SIGMA),
                      yaw_of(q[b.qa + 3:b.qa + 7]) + self.env_rng.normal(0, self.YAW_SIGMA)])
        self.loc_hist.append(x[:2].copy())
        w = max(2, int(round(1.0 / self.dt)) + 1)          # 1 s baseline: noise ~ 2.8 cm/s
        self.loc_hist = self.loc_hist[-w:]
        if len(self.loc_hist) >= w:
            self.speed_est = float(np.linalg.norm(self.loc_hist[-1] - self.loc_hist[0])) / ((w - 1) * self.dt)
        else:
            self.speed_est = float("nan")
        self.loc = x
        if self.base_estimator is not None:          # D-126 #27: same 1 s baseline, dead-reckoned position
            self.est_hist.append(self.base_estimator.p_w[:2].copy())
            self.est_hist = self.est_hist[-w:]
            self.speed_est = (float(np.linalg.norm(self.est_hist[-1] - self.est_hist[0])) / ((w - 1) * self.dt)
                              if len(self.est_hist) >= w else float("nan"))

    # ------------------------------------------------------------------ base-state estimator (D-126 #27)
    def _reset_estimator(self):
        from rrp.envs.mujoco.state_estimator import BaseStateEstimator, LegKinematics
        r = self.robots[0]
        if self.base_estimator is None:
            self.base_estimator = BaseStateEstimator(self._estimator_cfg)
            # encoders of every robot hinge/slide joint (the same arrays observe() publishes)
            self._leg_kin = LegKinematics.from_binding(self.binding, r.qadr, r.dadr)
        self.base_estimator.reset()
        self.est_hist = []
        # an instance-level _tracker_tick (rrp.envs.perturb.install_legged replaces it) bypasses the class hook: wrap it
        tt = self.__dict__.get("_tracker_tick")
        if tt is not None and not getattr(tt, "_bse_wrapped", False):
            def _tracker_tick(cmd, _inner=tt):
                _inner(cmd)
                self._estimator_tick(max(1, int(round(1.0 / (TRACKER_HZ * self.model.opt.timestep))))
                                     * self.model.opt.timestep)
            _tracker_tick._bse_wrapped = True
            self._tracker_tick = _tracker_tick

    def _estimator_tick(self, dt: float):
        """DEPLOYABLE inputs only: IMU (quat, gyro, acc), joint encoders, touch sensors."""
        r = self.robots[0]
        imu = self._imu()
        pos, vel = self._leg_kin.feet(self.data.qpos[r.qadr], self.data.qvel[r.dadr])
        self.base_estimator.update(quat=imu["quat"], gyro=imu["gyro"], acc=imu["acc"], foot_pos_b=pos,
                                   foot_vel_b=vel, touch=self._touch(), dt=dt)

    def _track(self, entity: str):
        s = self.entity_slots.get(entity)
        if s is None:
            return None, None
        tr = self.tracker_obj.tracks[s]
        return (np.array(tr.mean) if tr.mean is not None else None), (np.array(tr.var) if tr.var else None)

    # ------------------------------------------------------------------ public estimators
    def _touch(self) -> np.ndarray:
        vals = []
        for n in self.robots[0].touch:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            vals.append(float(self.data.sensordata[self.model.sensor_adr[sid]]))
        return np.array(vals)

    def _imu(self):
        b = self.binding
        pre = self.robots[0].prefix
        imu = self.robots[0].meta["legged"]["imu"]
        out = {}
        for k in ("quat", "gyro", "acc"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, pre + imu[k])
            a, d = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
            out[k] = self.data.sensordata[a:a + d].copy()
        return out

    def estimate(self, predicate: str, args: list[str]):
        if predicate == "distance_m" and len(args) == 2 and args[0] == "body":
            q, _ = self._track(args[1])
            if q is None or self.loc is None:
                return None, False, 0.0
            return float(np.linalg.norm(self.loc[:2] - q[:2])), True, 0.9
        if predicate == "upright" and args == ["body"]:
            quat = self._imu()["quat"]
            g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
            tilt = math.acos(max(-1.0, min(1.0, -g[2])))
            return bool(tilt < 0.8 * self.binding.tilt_limit), True, 0.95
        if predicate == "stance_fraction" and args == ["body"]:
            t = self._touch()
            return float(np.mean(t > 1.0)) if len(t) else None, bool(len(t)), 0.9
        if predicate == "base_speed" and args == ["body"]:
            ok = self.loc is not None and np.isfinite(self.speed_est)
            return (float(self.speed_est) if ok else None), ok, 0.8
        return None, False, 0.0

    def truth_predicate(self, pred: str, args: list[str], held=None):
        b = self.binding
        d = self.data
        if pred == "distance_m" and args[0] == "body":
            o = next((o for o in self.scenario.objects if o.task_entity == args[1]), None)
            if o is None:
                return None
            p = d.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)]
            return float(np.linalg.norm(d.qpos[b.qa:b.qa + 2] - p[:2]))
        if pred == "upright":
            return bool(b.tilt(d) < 0.8 * b.tilt_limit)
        if pred == "stance_fraction":
            fc, _ = b.contacts(d)
            return float(np.mean(fc))
        if pred == "base_speed":
            return float(np.linalg.norm(d.qvel[b.da:b.da + 2]))
        return None

    def _held_truth(self, contacts):
        return {"body": []}

    # ------------------------------------------------------------------ privileged StateView (D-144 R7)
    def _extra_state_entities(self):
        """`Session._extra_state_entities` override: one entity per foot link (`leg:<body>`), true pose."""
        from rrp.envs.base import EntityState
        b, m, d = self.binding, self.model, self.data
        out = []
        for bid in b.foot_bids:
            name = m.body(bid).name
            out.append(EntityState(id=f"leg:{name}", kind="link", name=name, pos=d.xpos[bid].copy(),
                                   quat=d.xquat[bid].copy(), vel=None, extent=None, mass=None, friction=None,
                                   material=None, parent="body", assembly="body", body=0, visible=None, attrs={}))
        return out

    def _extra_body_entity_map(self):
        """`Session._extra_body_entity_map` override: foot bodies resolve contacts to their `leg:<body>` id
        instead of the raw mj body name fallback."""
        m = self.model
        return {m.body(bid).name: f"leg:{m.body(bid).name}" for bid in self.binding.foot_bids}

    def observe(self) -> PolicyObservation:
        t = float(self.data.time)
        r = self.robots[0]
        q, v = self.data.qpos[r.qadr], self.data.qvel[r.dadr]
        addrs = [f"0:{j.address}" for j in r.spec.joints if j.type in ("hinge", "slide")]
        ns = NodeState(joint_addresses=addrs, qpos=q.copy(), qvel=v.copy(), qpos_mask=np.ones_like(q, dtype=bool),
                       timestamp=t)
        if self.base_estimator is not None:          # D-126 #27: body-frame base linear velocity (m/s), bse-1
            ns = ns.model_copy(update=dict(base_vel_estimate=self.base_estimator.v_b.copy(),
                                           units=dict(ns.units, base_vel_estimate=f"m/s body frame "
                                                                                  f"({self.base_estimator.cfg.version})")))
        imu = self._imu()
        tv = self._touch()
        chans = [SensorChannel(name="0:imu", kind="imu", values=np.concatenate([imu["quat"], imu["gyro"], imu["acc"]]),
                               mask=np.ones(10, bool), timestamp=t),
                 SensorChannel(name="0:touch", kind="touch", values=tv, mask=np.ones_like(tv, dtype=bool), timestamp=t)]
        if self.loc is not None:
            chans.append(SensorChannel(name="0:localization", kind="base_pose_xy_yaw", values=self.loc.copy(),
                                       mask=np.ones(3, bool), timestamp=t))
        chans.append(SensorChannel(name="0:base_velocity_command", kind="command_echo", values=self.cmd.copy(),
                                   mask=np.ones(3, bool), timestamp=t))
        pes = []
        for pred, args in self._task_predicates():
            val, known, conf = self.estimate(pred, args)
            pes.append(PredicateEstimate(predicate=pred, args=args, value=val, known=known, confidence=conf,
                                         estimator="rrp.sim.legged.public_estimators/v1", timestamp=t))
        images = self._render() if self.render_cfg else []
        oid = f"o{self.seed}_{self.step_count}_{next(_obs_counter)}"
        return PolicyObservation(observation_id=oid, sensor_time=t, robot_spec_hash=r.spec.spec_hash,
                                 sensor_images=images, measured_node_state=ns, declared_sensor_channels=chans,
                                 object_descriptors=self.tracker_obj.descriptors(t), predicate_estimates=pes,
                                 task_input=self.runtime.public_view() if hasattr(self, "runtime") else None)

    # ------------------------------------------------------------------ stepping
    def validate_command(self, cmd: NativeCommand) -> np.ndarray:
        if cmd.controller_version != self.tracker_version_str:
            raise StaleActionError(f"controller version {cmd.controller_version} != {self.tracker_version_str}")
        if set(cmd.groups) != {"base_velocity"}:
            raise ControllerRejection(f"legged tracker accepts only base_velocity, got {sorted(cmd.groups)}",
                                      code="unknown_group")
        g = self.tracker_contract.command_groups[0]
        v = np.asarray(cmd.groups["base_velocity"], float)
        if v.shape != (3,):
            raise ControllerRejection(f"base_velocity width {v.shape} != 3", code="wrong_width")
        if not np.isfinite(v).all():
            raise ControllerRejection("non-finite command", code="nonfinite")
        lo, hi = np.array(g.lower), np.array(g.upper)
        span = hi - lo
        if np.any(v < lo - 0.05 * span) or np.any(v > hi + 0.05 * span):
            raise ControllerRejection("command outside bounds", code="out_of_bounds")
        return np.clip(v, lo, hi)

    def _tracker_tick(self, cmd):
        b = self.binding
        tgt = self.tracker.act(self.data, cmd)
        act = self.actuator_model
        if act is not None:
            act.command(0, tgt)
        else:
            self.data.ctrl[b.pol_act] = tgt
        if len(b.held_act):
            self.data.ctrl[b.held_act] = b.q0_held
        n = max(1, int(round(1.0 / (TRACKER_HZ * self.model.opt.timestep))))
        for _ in range(n):
            if act is not None:
                self.data.ctrl[b.pol_act] = act.substep_ctrl(0, self.data)
            mujoco.mj_step(self.model, self.data)
        if self.base_estimator is not None:
            self._estimator_tick(n * self.model.opt.timestep)

    def actuator_record(self) -> dict | None:
        """Provenance of a non-ideal actuator mode (None for the ideal default, so default rows stay unchanged)."""
        if self.actuator_model is None:
            return None
        from rrp.bodies.actuator import mode_record
        return mode_record(self.actuator_mode, self.model, self.binding, self.robots[0].meta["name"], self.actuator_latency_ms)

    def validate_legs(self, cmd: NativeCommand) -> np.ndarray:
        if cmd.controller_version != self.tracker_version_str:
            raise StaleActionError(f"controller version {cmd.controller_version} != {self.tracker_version_str}")
        if set(cmd.groups) != {"legs"}:
            raise ControllerRejection(f"legs control accepts only the legs group, got {sorted(cmd.groups)}",
                                      code="unknown_group")
        v = np.asarray(cmd.groups["legs"], float)
        b = self.binding
        if v.shape != (b.n,):
            raise ControllerRejection(f"legs width {v.shape} != {b.n}", code="wrong_width")
        if not np.isfinite(v).all():
            raise ControllerRejection("non-finite command", code="nonfinite")
        return np.clip(v, b.lo, b.hi)

    def _step_legs(self, command) -> StepResult:
        rejected, source, executed = None, None, None
        if command is not None:
            try:
                self.tracker.pending = self.validate_legs(command)
                source = command.source
                executed = {"legs": self.tracker.pending.tolist()}
            except ControllerRejection as e:
                rejected = e.code
        self._tracker_tick(self.cmd)
        self._legs_ticks += 1
        if self._legs_ticks % max(1, int(round(self.dt * TRACKER_HZ))) == 0:
            obs = self._control_boundary()
        else:
            if not np.isfinite(self.data.qpos).all():
                raise FloatingPointError("simulation diverged (non-finite state)")
            self.boundary = False
            obs = self._last_obs
        return StepResult(obs, self.data.qpos.copy(), float(self.data.time), rejected, source, executed)

    def _control_boundary(self):
        """The 10 Hz part of a control step: divergence and fall checks, sensing, observation, task runtime."""
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError("simulation diverged (non-finite state)")
        b = self.binding
        _, bad = b.contacts(self.data)
        if bad or self.data.qpos[b.qa + 2] < b.min_h or b.tilt(self.data) > b.tilt_limit:
            self.fell = True
        self.step_count += 1
        self._sense()
        obs = self.observe()
        v0 = self.runtime.runtime_version
        self.runtime.tick(obs)
        if self.runtime.runtime_version != v0:
            obs = self.observe()
        self._last_obs = obs
        self.boundary = True
        return obs

    def step(self, command: NativeCommand | dict | None = None, robot: int = 0) -> StepResult:
        rejected, source, executed = None, None, None
        if isinstance(command, dict):
            command = command.get(0)
        if self.control == "legs":
            return self._step_legs(command)
        if command is None:
            row = self.executor.pop()
            if row is not None:
                command = NativeCommand(controller_version=self.tracker_version_str, groups=row,
                                        source=self.executor.meta["source"] if self.executor.meta else "learned")
        if command is not None:
            try:
                self.cmd = self.validate_command(command)
                source = command.source
                executed = {"base_velocity": self.cmd.tolist()}
            except ControllerRejection as e:
                rejected = e.code
        n = max(1, int(round(self.dt * TRACKER_HZ)))
        for _ in range(n):
            self._tracker_tick(self.cmd)
        obs = self._control_boundary()
        return StepResult(obs, self.data.qpos.copy(), float(self.data.time), rejected, source, executed)

    def base_pose_truth(self) -> np.ndarray:
        """PRIVILEGED: true base (x, y, yaw) - teachers/labels only."""
        b = self.binding
        q = self.data.qpos
        return np.array([q[b.qa], q[b.qa + 1], yaw_of(q[b.qa + 3:b.qa + 7])])

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        snap = super().snapshot()
        c = snap.components
        c["controller_state"] = [dict(joint_targets=c["controller_state"][0], tracker=self.tracker.state(),
                                      cmd=self.cmd.tolist(), fell=self.fell)]
        if self.actuator_model is not None:     # D-126 #14: pending-target queue (absent in ideal mode: snapshots unchanged)
            a = self.actuator_model
            c["controller_state"][0]["actuator"] = dict(
                latency_ms=self.actuator_latency_ms, lat=a.lat.tolist(), current=[None if x is None else x.tolist() for x in a.current],
                pending=[[[int(k), t.tolist()] for k, t in p] for p in a.pending])
        c["sensor_filters"]["localization"] = dict(loc=None if self.loc is None else self.loc.tolist(),
                                                   speed_est=self.speed_est,
                                                   hist=[h.tolist() for h in self.loc_hist])
        c["entity_tracker"] = self.tracker_obj.state()
        if self.base_estimator is not None:
            c["sensor_filters"]["base_estimator"] = dict(self.base_estimator.state(),
                                                         hist=[h.tolist() for h in self.est_hist])
        return snap

    def restore(self, snap):
        c = snap.components
        st = c["controller_state"][0]
        c2 = copy.copy(c)
        c2["controller_state"] = [st["joint_targets"]]
        from rrp.envs.mujoco.snapshot import Snapshot
        # base restore uses self.tracker.load for the entity tracker -> route explicitly
        tr = self.tracker
        self.tracker = self.tracker_obj
        try:
            obs = super().restore(Snapshot(c2))
        finally:
            self.tracker = tr
        self.tracker.load(st["tracker"])
        self.cmd = np.array(st["cmd"])
        self.fell = st["fell"]
        if self.actuator_model is not None and st.get("actuator"):
            a, sa = self.actuator_model, st["actuator"]
            self.actuator_latency_ms = sa["latency_ms"]
            a.fixed_latency_ms = sa["latency_ms"]
            a.lat[:] = sa["lat"]
            a.current = [None if x is None else np.array(x, float) for x in sa["current"]]
            a.pending = [[[int(k), np.array(t, float)] for k, t in p] for p in sa["pending"]]
        loc = c["sensor_filters"].get("localization", {})
        self.loc = None if loc.get("loc") is None else np.array(loc["loc"])
        self.speed_est = loc.get("speed_est", float("nan"))
        self.loc_hist = [np.array(h) for h in loc.get("hist", [])]
        if self.base_estimator is not None and "base_estimator" in c["sensor_filters"]:
            be = c["sensor_filters"]["base_estimator"]
            self.base_estimator.load(be)
            self.est_hist = [np.array(h) for h in be.get("hist", [])]
        self._last_obs = self.observe()
        return self._last_obs


class DirectTargets:
    """Tracker slot of the "legs" control mode: applies the commanded joint targets for one tracker tick; without a
    command the declared fallback holds the default stance (the behaviour of the former tracker-slot policy adapters
    before they were armed, so the reset settle is unchanged)."""
    source = "direct"
    version = "legs_direct"

    def __init__(self, binding):
        self.b = binding
        self.pending = None

    def reset(self, phase: float = 0.0):
        self.pending = None

    def act(self, data, cmd):
        t = self.pending if self.pending is not None else np.clip(self.b.q0, self.b.lo, self.b.hi)
        self.pending = None
        return t

    def state(self):
        return {}

    def load(self, st):
        pass


def make_legged_env(*, task: str, body: str, seed: int = 0, scene: dict | None = None, **kw) -> "LeggedSession":
    """env_id "mujoco/legged": waypoint_contact, loco_pick, foothold_steps (legged scenes) and h_steps, h_gap (humanoid
    scenes) on any legged/humanoid body key (rrp.bodies.legged.legged_body)."""
    from rrp.bodies.legged import legged_body
    scene = dict(scene or {})
    if task in ("h_steps", "h_gap"):
        from rrp.envs.mujoco import humanoid_scenes as hs
        sc = (hs.build_h_steps if task == "h_steps" else hs.build_h_gap)(body, seed, **scene)
        return LeggedSession(sc, seed=seed, **kw)
    if task in ("loco_pick", "foothold_steps"):
        from rrp.envs.mujoco import legged_scenes as ls
        if task == "loco_pick":
            return ls.LocoPickSession(ls.build_loco_pick(body, seed, **scene), seed=seed, **kw)
        return ls.FootholdSession(ls.build_foothold_steps(legged_body(body), seed, body_key=body, **scene), seed=seed, **kw)
    if task != "waypoint_contact":
        raise KeyError(f"mujoco/legged has no task {task!r}")
    return LeggedSession(build_waypoint_contact(body, seed, **scene), seed=seed, **kw)


def register():
    """Register the builder without editing existing entries (idempotent)."""
    from rrp.envs.mujoco import scenario as sc
    sc.BUILDERS.setdefault("waypoint_contact", build_waypoint_contact)
    return sc.BUILDERS


register()
