"""Additional legged scenes (D-126 roadmap #34 loco-manipulation, #22 legged foothold stepping). DEFAULT OFF.

Nothing here is imported by an existing module and nothing is registered into `rrp.envs.scenario.BUILDERS` unless
`register()` is called explicitly; the existing builders (waypoint_contact, pick_place, ...) are untouched.

* `loco_pick` (LOCO_PICK_VERSION): walk to a table, then pick a cube from it with an arm-bearing legged body.
  Body choice (D-126): `spot_arm` = MuJoCo Menagerie boston_dynamics_spot/spot_arm.xml, the only quadruped-with-arm
  model available. It is imported through the standard legged importer (`rrp.bodies.legged.menagerie_legged`) after
  `ensure_spot_arm_asset()` adds a LEGGED_ASSETS entry at runtime (legs = the 12 policy actuators; the 7 arm/gripper
  actuators are HELD by the locomotion tracker and are for the manipulation layer). NO spot tracker exists: a
  LeggedSession on spot_arm needs a tracker to be trained first (or tracker_kind="cpg" if a CPG is added).
  Torque limits: the menagerie model has joint actuatorfrcrange +-1000 N m and no manufacturer source is recorded here,
  so meta["actuator_limits_note"] labels them "unsourced (menagerie)".
* `foothold_steps` (FOOTHOLD_VERSION): a sequence of marked footholds; target k is stored RELATIVE TO FOOTHOLD k-1
  (dx, dy in the frame of foothold k-1, dyaw), foothold 0 relative to the robot's start frame. meta has both the world
  poses and the relative sequence (`foothold_world_from_relative` reconstructs one from the other), plus the foot
  assignment (role order: which foot per foothold; None = any foot, a null identity).

Sessions (`LocoPickSession`, `FootholdSession`) add public estimators (detector/encoder based) and privileged truth
versions of the new predicates (`height_above_m`, `foot_distance_m`), plus recorded failure reasons.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from rrp.bodies.compiler import compile_robot_spec
from rrp.bodies.generators import Module
from rrp.bodies.legged import legged_body, legged_world
from rrp.envs.legged import LeggedSession, tracker_contract
from rrp.envs.legged_core import quat_rotate_inv, yaw_of
from rrp.envs.scenario import MountedRobot, ObjectDecl, Scenario, load_task

LOCO_PICK_VERSION = "loco_pick_v0"
FOOTHOLD_VERSION = "foothold_steps_v0"
SPOT_ARM_KEY = "spot_arm"

# LEGGED_ASSETS entry for the menagerie Spot with arm (added at runtime by ensure_spot_arm_asset; default off).
SPOT_ARM_ASSET = dict(dir="boston_dynamics_spot", file="spot_arm.xml", kind="quadruped", family="boston_dynamics_spot",
                      license="BSD-3-Clause", key="home", feet=["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"],
                      gains=None,                   # keep the menagerie position servos (kp 500, kv 40)
                      legs=r"^(fl|fr|hl|hr)_", action_scale=0.25, gait_period=0.6,
                      command_ranges=dict(vx=[-0.5, 1.0], vy=[-0.3, 0.3], wz=[-0.8, 0.8]))
SPOT_ARM_GRASP_BODY = "arm_link_wr1"          # jaw body; the grasp frame site is added here
SPOT_ARM_ROOT = "arm_link_sh0"


def ensure_spot_arm_asset() -> None:
    """Add the spot_arm LEGGED_ASSETS entry (idempotent, explicit opt-in; ALL_LEGGED is unchanged)."""
    from rrp.bodies import legged as bl
    bl.LEGGED_ASSETS.setdefault(SPOT_ARM_KEY, dict(SPOT_ARM_ASSET))


def spot_arm_body() -> Module:
    """Spot + arm as a legged Module with an extra `arm` assembly (kind gripper, capability grasp) whose frame is a
    site on the jaw body. The locomotion meta (policy/held actuators, feet, IMU) is the standard importer's."""
    ensure_spot_arm_asset()
    m = legged_body(SPOT_ARM_KEY)
    m.spec.body(SPOT_ARM_GRASP_BODY).add_site(name="arm_grasp_site", pos=[0.16, 0, 0])
    meta = m.meta
    meta["assemblies"] = list(meta["assemblies"]) + [dict(id="arm", kind="gripper", root_body=SPOT_ARM_ROOT,
                                                          frame=dict(site="arm_grasp_site"),
                                                          capabilities=["grasp", "carry"])]
    meta["actuator_limits_note"] = ("unsourced (menagerie spot_arm.xml actuatorfrcrange +-1000 N m; no manufacturer "
                                    "limits recorded; not in rrp.physics.actuator.SOURCED)")
    meta["arm"] = dict(actuators=list(meta["legged"]["held_actuators"]), grasp_site="arm_grasp_site",
                       gripper_actuator="arm_f1x")
    return m


def resolve_arm_body(robot) -> tuple[Module, str]:
    if isinstance(robot, Module):
        return robot, robot.meta["name"]
    if robot == SPOT_ARM_KEY:
        return spot_arm_body(), SPOT_ARM_KEY
    raise ValueError(f"loco_pick needs an arm-bearing legged body ({SPOT_ARM_KEY!r} or a Module); got {robot!r}")


def _mount(scene, module: Module):
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(module.spec.copy(), prefix="r0_", site=site)


def _cameras(scene):
    scene.worldbody.add_camera(name="overhead", pos=[0, 0, 12.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=100)
    scene.worldbody.add_camera(name="front", pos=[-3.0, -3.0, 2.5], xyaxes=[0.707, -0.707, 0, 0.35, 0.35, 0.87],
                               fovy=60)


# ------------------------------------------------------------------ #34 loco-manipulation: walk to a table, then pick
def build_loco_pick(robot=SPOT_ARM_KEY, seed: int = 0, task: dict | None = None, contact: str | None = None, *,
                    dist_range=(1.5, 3.0), heading_range=(-math.pi / 3, math.pi / 3), table_height: float = 0.45,
                    table_half=(0.30, 0.45), cube_size: float = 0.025) -> Scenario:
    """Table (static box) at a seeded distance/heading from the start, facing the robot; a cube on the table top
    near the robot-facing edge. The standoff (where walk_to_table completes) is stored in meta."""
    from rrp.physics.contact import version_str
    module, body_key = resolve_arm_body(robot)
    rng = np.random.default_rng(seed)
    meta = copy.deepcopy(module.meta)
    scene = legged_world(f"loco_pick_{seed}", meta.get("source_options"), contact=contact)
    meta["contact_model"] = version_str(contact)
    _cameras(scene)
    _mount(scene, module)
    d = float(rng.uniform(*dist_range))
    h = float(rng.uniform(*heading_range))
    table_xy = np.array([d * math.cos(h), d * math.sin(h)])
    table_yaw = h                      # table's local -x face points back at the start
    tb = scene.worldbody.add_body(name="table", pos=[*table_xy, table_height / 2])
    tb.quat = [math.cos(table_yaw / 2), 0, 0, math.sin(table_yaw / 2)]
    tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[table_half[0], table_half[1], table_height / 2],
                rgba=[0.55, 0.4, 0.25, 1])
    tb.add_site(name="table_site", pos=[0, 0, table_height / 2])
    # cube near the robot-facing edge, lateral offset seeded
    local = np.array([-table_half[0] + 0.12, rng.uniform(-0.6, 0.6) * table_half[1]])
    c, s = math.cos(table_yaw), math.sin(table_yaw)
    cube_xy = table_xy + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
    cb = scene.worldbody.add_body(name="cube", pos=[*cube_xy, table_height + cube_size + 0.001])
    cb.add_freejoint(name="cube_free")
    cb.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_size] * 3, rgba=[0.85, 0.1, 0.1, 1],
                mass=0.05)
    standoff = table_xy - 0.75 * np.array([c, s])          # base target in front of the table edge
    scene.memory = 16 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body", "gripper": "arm"})
    objects = [ObjectDecl("table", "brown table", "feature", size=(table_half[0], table_half[1], table_height / 2),
                          radius=float(max(table_half)), task_entity="table"),
               ObjectDecl("cube", "red cube", "object", (cube_size,) * 3, task_entity="cube")]
    return Scenario("loco_pick", task or load_task("loco_pick"), scene, model, [mr], objects, seed,
                    meta=dict(scene_version=LOCO_PICK_VERSION, body_key=body_key, contact_model=meta["contact_model"],
                              table=dict(xy=table_xy.tolist(), yaw=table_yaw, height=table_height,
                                         half=list(table_half)),
                              cube=dict(xy=cube_xy.tolist(), z=table_height + cube_size, size=cube_size),
                              standoff=dict(xy=standoff.tolist(), yaw=table_yaw),
                              actuator_limits_note=meta.get("actuator_limits_note")))


# ------------------------------------------------------------------ #22 legged foothold stepping
def foot_order_for(meta: dict, n_steps: int, any_foot_every: int = 0) -> list[str | None]:
    """Default foot assignment: bipeds alternate left/right feet; quadrupeds alternate the two FRONT feet
    (the first two foot bodies). any_foot_every = k > 0 makes every k-th foothold 'any foot' (None)."""
    feet = meta["legged"]["foot_bodies"]
    pair = feet[:2]
    out = [pair[k % 2] for k in range(n_steps)]
    if any_foot_every > 0:
        out = [None if (k + 1) % any_foot_every == 0 else f for k, f in enumerate(out)]
    return out


def foothold_world_from_relative(rel: list, start=(0.0, 0.0, 0.0)) -> list[list[float]]:
    """Compose relative targets (dx, dy in the previous foothold's frame, dyaw) into world (x, y, yaw) poses."""
    x, y, yaw = map(float, start)
    out = []
    for dx, dy, dyaw in rel:
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw
        out.append([x, y, yaw])
    return out


def foothold_relative_from_world(world: list, start=(0.0, 0.0, 0.0)) -> list[list[float]]:
    prev = list(map(float, start))
    out = []
    for x, y, yaw in world:
        c, s = math.cos(prev[2]), math.sin(prev[2])
        ex, ey = x - prev[0], y - prev[1]
        out.append([c * ex + s * ey, -s * ex + c * ey, yaw - prev[2]])
        prev = [x, y, yaw]
    return out


def build_foothold_steps(robot, seed: int = 0, n_steps: int = 6, task: dict | None = None, contact: str | None = None,
                         *, radius: float = 0.07, stones: bool = False, stone_height: float = 0.04,
                         any_foot_every: int = 0, body_key: str | None = None) -> Scenario:
    """Footholds along a gently curving path. The first foothold is placed ahead of the assigned foot's start
    position; each next one is (dx, dy, dyaw) from the previous one, dx ~ step length (from the body's forward speed
    range x gait period), dy alternating sides. stones=True adds raised colliding stepping stones (height
    stone_height) under the markers; NOTE the tracker's contact logic counts only the geom named "floor", so stones
    are outside the trackers' training distribution (default False: flat, visual, non-colliding markers)."""
    from rrp.physics.contact import version_str
    if isinstance(robot, str):
        body_key, robot = robot, legged_body(robot)
    if not isinstance(robot, Module) or "legged" not in robot.meta:
        raise TypeError("foothold_steps needs a legged Module")
    rng = np.random.default_rng(seed)
    meta = copy.deepcopy(robot.meta)
    body_key = body_key or meta["name"]
    L = meta["legged"]
    scene = legged_world(f"foothold_steps_{seed}", meta.get("source_options"), contact=contact)
    meta["contact_model"] = version_str(contact)
    _cameras(scene)
    _mount(scene, robot)
    order = foot_order_for(meta, n_steps, any_foot_every)
    # start pose of the assigned feet in the default pose (kinematics of the module alone)
    m0 = robot.spec.copy().compile()
    d0 = mujoco.MjData(m0)
    key = next((i for i in range(m0.nkey)), None)
    if key is not None:
        mujoco.mj_resetDataKeyframe(m0, d0, key)
    for jn, v in L["default_pose"].items():
        jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            d0.qpos[m0.jnt_qposadr[jid]] = v
    mujoco.mj_kinematics(m0, d0)
    root = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, L["root_body"])
    foot_xy = {f: d0.xpos[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, f)][:2] - d0.xpos[root][:2]
               for f in L["foot_bodies"]}
    vmax = float(L["command_ranges"]["vx"][1])
    step = float(np.clip(0.5 * vmax * float(L["gait_period"]), 0.08, 0.35))
    half_w = float(abs(foot_xy[L["foot_bodies"][0]][1] - foot_xy[L["foot_bodies"][1]][1]) / 2) or 0.1
    f0 = order[0] or L["foot_bodies"][0]
    rel = []
    lat = lambda f: float(foot_xy[f][1]) if f is not None else 0.0      # lateral offset of the foot in the base frame
    for k in range(n_steps):
        dyaw = float(rng.uniform(-0.15, 0.15))
        if k == 0:
            dx, dy = float(foot_xy[f0][0] + max(step, 3 * radius) * rng.uniform(1.0, 1.2)), lat(f0)
        else:
            dx = float(step * rng.uniform(0.8, 1.2))
            if order[k] is None or order[k - 1] is None:
                dy = float(rng.uniform(-0.5, 0.5) * half_w)
            else:
                dy = lat(order[k]) - lat(order[k - 1])       # cross sides when the assigned foot changes
        rel.append([dx, dy, dyaw])
    world = foothold_world_from_relative(rel)
    for k, (x, y, yaw) in enumerate(world):
        rgba = (0.1, 0.8, 0.2, 0.9) if k % 2 == 0 else (0.9, 0.8, 0.1, 0.9)
        b = scene.worldbody.add_body(name=f"foothold_{k:02d}", pos=[x, y, 0.0], mocap=True)
        b.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        b.add_geom(name=f"foothold_{k:02d}_disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.003, 0],
                   pos=[0, 0, 0.003], rgba=list(rgba), contype=0, conaffinity=0)       # visual only, non-colliding
        b.add_geom(name=f"foothold_{k:02d}_dir", type=mujoco.mjtGeom.mjGEOM_BOX, size=[radius, 0.006, 0.002],
                   pos=[radius / 2, 0, 0.006], rgba=[0.1, 0.1, 0.1, 0.9], contype=0, conaffinity=0)   # heading tick
        b.add_site(name=f"foothold_{k:02d}_site", pos=[0, 0, 0])
        if stones:
            st = scene.worldbody.add_body(name=f"stone_{k:02d}", pos=[x, y, stone_height / 2])
            st.add_geom(name=f"stone_{k:02d}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius * 1.4, stone_height / 2, 0], rgba=[0.5, 0.5, 0.52, 1])
            b.pos = [x, y, stone_height]
    scene.memory = 8 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body"})
    objects = [ObjectDecl(f"foothold_{k:02d}", f"foothold pad #{k:02d}", "feature", radius=radius,
                          task_entity=f"foothold_{k:02d}") for k in range(n_steps)]
    t = task or foothold_task(n_steps, order, radius, meta)
    return Scenario("foothold_steps", t, scene, model, [mr], objects, seed,
                    meta=dict(scene_version=FOOTHOLD_VERSION, body_key=body_key, contact_model=meta["contact_model"],
                              footholds_world=world, footholds_relative=rel,
                              relative_convention="k=0 relative to the start frame (0,0,0); k>0 (dx, dy) in the frame "
                                                  "of foothold k-1, dyaw = yaw_k - yaw_{k-1}",
                              foot_order=order, radius_m=radius, stones=bool(stones),
                              stone_height_m=float(stone_height) if stones else 0.0))


def _ent(i):
    return {"kind": "entity", "entity": {"id": i, "version": 0}}


def _foot_entity(foot: str) -> str:
    return "foot_" + foot.replace("-", "_")


def foothold_task(n_steps: int, order: list, radius: float, meta: dict) -> dict:
    """Per-instance task graph: tasks/foothold_steps.json is the template (declares the predicate conventions);
    events step_00..step_{n-1} are generated here because n and the foot order vary. The foot is a second `actor`
    (ordinal 1, after the body at ordinal 0) bound to a foot entity (type body); a None foot (any foot) has NO foot role and the completion uses
    the body entity (min over feet): the null identity is explicit in the graph."""
    base = load_task("foothold_steps")
    ents = [e for e in base["entity_declarations"] if e["id"] == "body"]
    for f in sorted({f for f in order if f is not None}):
        ents.append({"id": _foot_entity(f), "type": "body", "descriptor": f"foot link {f}"})
    evs = []
    upright = {"predicate": "upright", "arguments": [_ent("body")], "comparison": "eq", "value": True,
               "source": "observation_estimate", "persistence_seconds": 0.0}
    for k, f in enumerate(order):
        fh = f"foothold_{k:02d}"
        ents.append({"id": fh, "type": "feature", "descriptor": f"foothold pad #{k:02d}"})
        roles = [{"role": "actor", "ordinal": 0, "binding": _ent("body")},
                 {"role": "target", "ordinal": 0, "binding": _ent(fh)}]
        who = "body"
        if f is not None:
            roles.append({"role": "actor", "ordinal": 1, "binding": _ent(_foot_entity(f))})   # the stepping foot
            who = _foot_entity(f)
        evs.append({"id": f"step_{k:02d}", "operator": "step_on", "roles": roles,
                    "requires_completed": [f"step_{k - 1:02d}"] if k else [], "requires_active": [],
                    "preconditions": [], "invariants": [upright], "desired_effects": [],
                    "completion": [{"predicate": "foot_distance_m", "arguments": [_ent(who), _ent(fh)],
                                    "comparison": "le", "value": float(radius), "source": "observation_estimate",
                                    "persistence_seconds": 0.1}],
                    "resources": [{"entity": {"id": "body", "version": 0}, "mode": "exclusive_control"}],
                    "produces": [], "timeout_seconds": 20, "recovery": {"max_attempts": 1, "on_failure": "fail"}})
    return dict(base, entity_declarations=ents, events=evs, success_events=[f"step_{n_steps - 1:02d}"])


# ------------------------------------------------------------------ sessions (public estimators + truth)
class LocoPickSession(LeggedSession):
    """loco_pick session: adds `height_above_m(object, table)` (public: tracked object z - tracked table top z;
    truth: body positions).
    Needs a spot_arm locomotion tracker (none exists yet)."""

    def estimate(self, predicate, args):
        if predicate == "height_above_m" and len(args) == 2:
            p, _ = self._track(args[0])
            q, _ = self._track(args[1])
            if p is None or q is None:
                return None, False, 0.0
            top = q[2] + self.scenario.meta["table"]["height"] / 2        # table body origin at mid-height
            return float(p[2] - self.scenario.object("cube").size[2] - top), True, 0.8
        return super().estimate(predicate, args)

    def truth_predicate(self, pred, args, held=None):
        if pred == "height_above_m":
            cz = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")][2]
            return float(cz - self.scenario.meta["cube"]["size"] - self.scenario.meta["table"]["height"])
        return super().truth_predicate(pred, args, held)


class FootholdSession(LeggedSession):
    """foothold_steps session. Public `foot_distance_m(foot|body, foothold)`: horizontal distance from the foot's
    touch-site position, computed by forward kinematics of the MEASURED joint encoders on the declared noisy
    localization (x, y, yaw) plus IMU roll/pitch, to the detector-tracked foothold; known only while that foot's touch
    sensor reads > 1 N (for `body`: min over loaded feet). Truth version uses the simulator's foot sites and contacts.
    `failure_reason()` records missed_foothold / wrong_foot / fell / timeout (privileged, for reports only)."""

    TOUCH_N = 1.0

    def reset(self, seed=None):
        self._fk = mujoco.MjData(self.model)
        self.step_log = []
        return super().reset(seed)

    def _foot_names(self):
        return self.robots[0].meta["legged"]["foot_bodies"]

    def _foot_index(self, ent: str):
        names = self._foot_names()
        if ent == "body":
            return list(range(len(names)))
        return [i for i, f in enumerate(names) if "foot_" + f.replace("-", "_") == ent]

    def _public_foot_xy(self):
        if self.loc is None:
            return None
        b, d, fk = self.binding, self.data, self._fk
        r = self.robots[0]
        fk.qpos[:] = 0.0
        fk.qpos[r.qadr] = d.qpos[r.qadr]        # joint encoders (public); the base pose is the public estimate below
        imu_q = self._imu()["quat"]
        g = quat_rotate_inv(imu_q, np.array([0, 0, -1.0]))
        roll, pitch = math.atan2(-g[1], -g[2]), math.asin(max(-1.0, min(1.0, g[0])))
        yaw = float(self.loc[2])
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        fk.qpos[b.qa:b.qa + 3] = [self.loc[0], self.loc[1], b.nominal_height()]
        fk.qpos[b.qa + 3:b.qa + 7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        mujoco.mj_kinematics(self.model, fk)
        return np.array([fk.site_xpos[s][:2] for s in b.foot_sids])

    def estimate(self, predicate, args):
        if predicate == "foot_distance_m" and len(args) == 2:
            idx = self._foot_index(args[0])
            q, _ = self._track(args[1])
            fxy = self._public_foot_xy()
            if not idx or q is None or fxy is None:
                return None, False, 0.0
            touch = self._touch()
            loaded = [i for i in idx if i < len(touch) and touch[i] > self.TOUCH_N]
            if not loaded:
                return None, False, 0.0               # unknown while the foot is in the air (unknown is not false)
            return float(min(np.linalg.norm(fxy[i] - q[:2]) for i in loaded)), True, 0.8
        return super().estimate(predicate, args)

    def _truth_foot(self):
        fc, _ = self.binding.contacts(self.data)
        return np.array([self.data.site_xpos[s][:2] for s in self.binding.foot_sids]), fc

    def truth_predicate(self, pred, args, held=None):
        if pred == "foot_distance_m":
            idx = self._foot_index(args[0])
            o = next((o for o in self.scenario.objects if o.task_entity == args[1]), None)
            if not idx or o is None:
                return None
            p = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)][:2]
            fxy, fc = self._truth_foot()
            ds = [np.linalg.norm(fxy[i] - p) for i in idx if fc[i]]
            return float(min(ds)) if ds else None
        return super().truth_predicate(pred, args, held)

    def failure_reason(self) -> str | None:
        """PRIVILEGED diagnostic (reports only): first unmet foothold -> fell | wrong_foot (another foot is on it) |
        missed_foothold (the assigned foot landed, but outside the radius) | timeout; None if all succeeded."""
        if self.fell:
            return "fell"
        meta = self.scenario.meta
        fxy, fc = self._truth_foot()
        for k, f in enumerate(meta["foot_order"]):
            st = self.runtime.instances.get(f"step_{k:02d}")
            if st is not None and st.status == "succeeded":
                continue
            p = np.array(meta["footholds_world"][k][:2])
            on = [i for i in range(len(fxy)) if fc[i] and np.linalg.norm(fxy[i] - p) <= meta["radius_m"]]
            names = self._foot_names()
            if on and f is not None and all(names[i] != f for i in on):
                return "wrong_foot"
            if st is not None and st.status == "failed":
                return "missed_foothold"
            return "timeout"
        return None


def register():
    """Register the builders without editing existing entries (idempotent). NOT called by any existing module."""
    from rrp.envs import scenario as sc
    sc.BUILDERS.setdefault("loco_pick", build_loco_pick)
    sc.BUILDERS.setdefault("foothold_steps", build_foothold_steps)
    return sc.BUILDERS
