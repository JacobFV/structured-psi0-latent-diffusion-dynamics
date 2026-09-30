"""W13 humanoid task scenes (D-138, research/tracks/humanoid.md section 2). Scenes are static world geoms added to the tracker
world (contact_v2 floor) and scaled by the body's leg length L (= nominal root height), so one task definition spans
toddler-size and adult humanoids. Every scene geom counts as GROUND for foot-contact / fall logic ("ground" list).

Implemented (h_steps and h_gap_sidestep, U2's h_walk / h_turn / h_reach / h_squat_pick / h_place below):
  h_steps  (L2): up N steps, a platform, down N steps, along +x; step height h (fraction of L, per episode), tread 0.6 L.
           Success: base past the far end (x > x_end) without falling; failure reasons: fell, trip (a non-sole geom of a
           foot touches a riser > 0.3 s), timeout. Per-world step heights are batchable (geom_pos / geom_size / aabb / rbound).
Task knowledge a student may receive (never outcomes): the staircase geometry (x0, tread, n, h) as the task map.

U2 (docs/architecture.md 14.4, control="wholebody"): L0 h_walk (walk to a marker, halt), L3 h_turn (turn in place to face a
marker), M1 h_reach (a palm to a point in the air), M2 h_squat_pick (squat, bimanual palm-squeeze pick of a box from a low
crate, stand, hold), M3 h_place (pick the box, twist the trunk about the waist, put the box down on a mark 0.6 rad round the
waist axis). The humanoid hands are hand-link "palm" bars without fingers (the bar is the forearm axis), so a grasp is a squeeze
between the two palms, and the arms are short (about 0.33 m from the shoulder: the crates are low and a squat brings the
shoulders down). `HumanoidSession` adds the public
predicates (palm distances from forward kinematics of the joint encoders, detector-tracked box heights, bearing, drift,
stand height) and their privileged truth twins, and names failures for the reports.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from rrp.envs.mujoco.legged import LeggedSession
from rrp.envs.mujoco.legged_core import quat_rotate_inv
from rrp.envs.mujoco.sensors import DetectorConfig

G = mujoco.mjtGeom
SCENE_VERSION = "humanoid_scenes_v2"      # v2: step boxes buried below the floor (v1 thin boxes tripped the robot)
STEPS_N = 3
STEPS_TREAD = 0.6            # x L
STEPS_X0 = 1.2               # x L, first riser
STEPS_PLATFORM = 1.2         # x L
BURY = 0.5                   # x L, step boxes extend this far below the floor


def steps_layout(L: float, h: float, n: int = STEPS_N):
    """[(name, center_x, half_x, top_z)] of the up-steps, the platform and the down-steps (boxes resting on z = 0)."""
    t, x = STEPS_TREAD * L, STEPS_X0 * L
    out = []
    for i in range(n):                                   # up: step i top at (i + 1) h
        out.append((f"step_up{i}", x + (i + 0.5) * t, 0.5 * t, (i + 1) * h))
    x += n * t
    out.append(("step_top", x + 0.5 * STEPS_PLATFORM * L, 0.5 * STEPS_PLATFORM * L, (n + 1) * h))
    x += STEPS_PLATFORM * L
    for i in range(n):                                   # down
        out.append((f"step_dn{i}", x + (i + 0.5) * t, 0.5 * t, (n - i) * h))
    return out, x + n * t


def steps_height_at(xq: np.ndarray, L: float, h: float, n: int = STEPS_N) -> np.ndarray:
    """Ground height at world x (analytic height scan of the staircase)."""
    z = np.zeros_like(np.asarray(xq, dtype=float))
    lay, _ = steps_layout(L, h, n)
    for _, cx, hx, top in lay:
        z = np.where(np.abs(xq - cx) <= hx, np.maximum(z, top), z)
    return z


def add_steps(spec: mujoco.MjSpec, L: float, h: float, floor_kw: dict, n: int = STEPS_N, width: float = 3.0, sign: float = 1.0) -> list:
    """The staircase along +x (sign 1) or mirrored behind the origin (sign -1: h_steps_carry, the robot turns to face it)."""
    names = []
    lay, _ = steps_layout(L, h, n)
    for name, cx, hx, top in lay:
        # boxes extend BURY x L below the floor, so a zero-height step is flush with the floor (no 0.2 mm edge to trip on)
        spec.worldbody.add_geom(name=name, type=G.mjGEOM_BOX, pos=[sign * cx, 0, 0.5 * (top - BURY * L)],
                                size=[hx, 0.5 * width * L, 0.5 * (top + BURY * L)],
                                rgba=[0.55, 0.5, 0.45, 1], **floor_kw)
        names.append(name)
    return names


def add_steps_mocap(spec: mujoco.MjSpec, L: float, h_max_frac: float, floor_kw: dict, n: int = STEPS_N, width: float = 3.0) -> list:
    """GPU variant: each step is a FIXED-size box on its own mocap body (per-world heights move the mocap bodies; mujoco_warp
    mis-simulates boxes whose size changes after compilation). Box half-height covers the buried part and the tallest top."""
    names = []
    lay, _ = steps_layout(L, h_max_frac * L, n)
    hz = 0.5 * (BURY + (n + 1) * h_max_frac) * L
    for name, cx, hx, top in lay:
        b = spec.worldbody.add_body(name=f"{name}_body", pos=[cx, 0, top - hz], mocap=True)
        b.add_geom(name=name, type=G.mjGEOM_BOX, pos=[0, 0, 0], size=[hx, 0.5 * width * L, hz], rgba=[0.55, 0.5, 0.45, 1],
                   **floor_kw)
        names.append(name)
    return names


def task_model(body: str, task: str, params: dict | None = None, contact: str = "v2", prefix: str = "r0_"):
    """(model, spec, meta) of `body` in the `task` scene (same construction as rrp.bodies.legged.standalone_model plus
    scene geoms). meta['scene'] = {task, version, params, ground: [geom names]}."""
    from rrp.bodies.legged import legged_body, legged_world
    from rrp.bodies.contact import apply_world, version_str
    module = legged_body(body)
    meta = copy.deepcopy(module.meta)
    L = float(meta["legged"]["nominal_height"])
    scene = legged_world(f"{meta['name']}_{task}", meta.get("source_options"), contact=contact)
    floor_kw = apply_world(mujoco.MjSpec(), contact, meta.get("source_options"))   # floor contact params, for scene geoms
    params = dict(params or {})
    if task == "h_steps":
        h = float(params.get("h_frac", 0.15)) * L
        params.setdefault("h_frac", 0.15)
        if params.get("mocap_h_max") is not None:
            ground = add_steps_mocap(scene, L, float(params["mocap_h_max"]), floor_kw)
        else:
            ground = add_steps(scene, L, h, floor_kw)
        params["x_end"] = steps_layout(L, h)[1]
    elif task == "h_gap_sidestep":
        walls = add_gap_walls(scene, L, floor_kw, float(params.get("width", 2.0 * L)), float(params.get("y_c", 0.0)),
                              mocap=bool(params.get("mocap")))
        ground = []
        params["walls"] = walls
    else:
        raise KeyError(f"unknown humanoid task scene {task!r}")
    meta["contact_model"] = version_str(contact)
    meta["scene"] = dict(task=task, version=SCENE_VERSION, params=params, ground=ground, L=L, walls=params.get("walls", []))
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(module.spec.copy(), prefix=prefix, site=site)
    scene.memory = 3 * 2 ** 20
    return scene.compile(), scene, meta


# ------------------------------------------------------------------ C-MuJoCo scenario (LeggedSession) for collection / evaluation
def build_h_steps(robot, seed: int, h_frac: float | None = None, contact: str | None = "v2"):
    """Scenario for LeggedSession: body at the origin facing +x, staircase ahead, goal marker at x_end + 0.3 L.
    h_frac (x L) defaults to U(0.10, 0.30) from the seed. Scenario meta carries the staircase (task map) and L."""
    import math as _m
    from rrp.bodies.compiler import compile_robot_spec
    from rrp.bodies.generators import Module
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged import WAYPOINT_COLORS, _waypoint, tracker_contract
    from rrp.envs.mujoco.scenario import MountedRobot, ObjectDecl, Scenario, load_task
    from rrp.bodies.contact import apply_world, version_str
    from rrp.bodies.legged import legged_world
    body_key = robot if isinstance(robot, str) else robot.meta["name"]
    if isinstance(robot, str):
        robot = legged_body(robot)
    assert isinstance(robot, Module)
    rng = np.random.default_rng([seed, 4242])
    meta = copy.deepcopy(robot.meta)
    L = float(meta["legged"]["nominal_height"])
    hf = float(rng.uniform(0.10, 0.30)) if h_frac is None else float(h_frac)
    scene = legged_world(f"h_steps_{seed}", meta.get("source_options"), contact=contact)
    floor_kw = apply_world(mujoco.MjSpec(), contact, meta.get("source_options"))
    ground = add_steps(scene, L, hf * L, floor_kw)
    x_end = steps_layout(L, hf * L)[1]
    goal = [x_end + 0.3 * L, 0.0]
    scene.worldbody.add_camera(name="overhead", pos=[0.5 * x_end, 0, 12.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=100)
    scene.worldbody.add_camera(name="side", pos=[0.5 * x_end, -3.5 * max(L, 0.5), 1.2 * max(L, 0.5)],
                               xyaxes=[1, 0, 0, 0, 0.3, 0.95], fovy=60)
    _waypoint(scene, "goal", goal, WAYPOINT_COLORS["cyan"])
    meta["contact_model"] = version_str(contact)
    meta["scene"] = dict(task="h_steps", version=SCENE_VERSION, params=dict(h_frac=hf, x_end=x_end), ground=ground, L=L)
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(robot.spec.copy(), prefix="r0_", site=site)
    scene.memory = 8 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body"})
    objects = [ObjectDecl("goal", "goal marker beyond the staircase", "feature", radius=0.12, task_entity="goal")]
    return Scenario("h_steps", load_task("h_steps"), scene, model, [mr], objects, seed,
                    meta=dict(body_key=body_key, contact_model=meta["contact_model"], L=L, h_frac=hf, x_end=x_end, goal=goal,
                              staircase=dict(x0=STEPS_X0 * L, tread=STEPS_TREAD * L, n=STEPS_N, h=hf * L,
                                             platform=STEPS_PLATFORM * L)))


SCAN_X = np.linspace(-0.3, 1.2, 11)          # privileged height-scan grid (body frame, units of L); WarpStepsEnv uses it too
SCAN_Y = np.array([-0.25, 0.0, 0.25])


def steps_scan_np(qpos_root: np.ndarray, L: float, h: float) -> np.ndarray:
    """numpy twin of WarpStepsEnv.extra_obs (PRIVILEGED expert input): 11 x 3 yaw-frame ground heights - (base z - L), / L; + h/L."""
    x, y, z = qpos_root[:3]
    w, qx, qy, qz = qpos_root[3:7]
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    sx, sy = np.meshgrid(SCAN_X, SCAN_Y, indexing="ij")
    px = x + L * (math.cos(yaw) * sx.ravel() - math.sin(yaw) * sy.ravel())
    hz = steps_height_at(px, L, h)
    return np.concatenate([(hz - (z - L)) / L, [h / L]]).astype(np.float32)


# ------------------------------------------------------------------ L1 h_gap_sidestep
GAP_X = 2.0            # x L, wall plane
WALL_T = 0.1           # x L, wall half-thickness (x)
WALL_HALF_Y = 2.0      # x L, each wall's half-length along y
WALL_HALF_Z = 0.75     # x L


def body_width(model: mujoco.MjModel, b) -> float:
    """Shoulder/hip width at the default pose: 2 x max over robot collision geoms of |y| + bounding radius (m)."""
    d = mujoco.MjData(model)
    b.set_default(d)
    mujoco.mj_kinematics(model, d)
    ys = [abs(d.geom_xpos[g][1]) + model.geom_rbound[g] for g in range(model.ngeom)
          if b.is_robot_body[model.geom_bodyid[g]] and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return 2.0 * float(max(ys))


def add_gap_walls(spec: mujoco.MjSpec, L: float, floor_kw: dict, width: float, y_c: float, mocap: bool = False) -> list:
    """Two wall boxes at x = GAP_X L leaving a gap of `width` centred at y_c. mocap=True puts each wall on its own mocap body
    (the GPU env moves them per world; geometry fixed)."""
    names = []
    for side, sg in (("left", 1), ("right", -1)):
        yc = y_c + sg * (0.5 * width + WALL_HALF_Y * L)
        size = [WALL_T * L, WALL_HALF_Y * L, WALL_HALF_Z * L]
        nm = f"wall_{side}"
        if mocap:
            body = spec.worldbody.add_body(name=f"{nm}_body", pos=[GAP_X * L, yc, WALL_HALF_Z * L], mocap=True)
            body.add_geom(name=nm, type=G.mjGEOM_BOX, size=size, rgba=[0.6, 0.35, 0.3, 1], **floor_kw)
        else:
            spec.worldbody.add_geom(name=nm, type=G.mjGEOM_BOX, pos=[GAP_X * L, yc, WALL_HALF_Z * L], size=size,
                                    rgba=[0.6, 0.35, 0.3, 1], **floor_kw)
        names.append(nm)
    return names


def build_h_gap(robot, seed: int, level: float = 1.0, contact: str | None = "v2"):
    """C-MuJoCo scenario for h_gap_sidestep (same parameter law as WarpGapEnv at `level`, drawn from the seed). Scenario meta
    carries the task map (wall x, gap centre and width, final heading) and the walls' geom names (privileged failure check)."""
    from rrp.bodies.compiler import compile_robot_spec
    from rrp.bodies.legged import legged_body, legged_world, standalone_model
    from rrp.envs.mujoco.legged import WAYPOINT_COLORS, _waypoint, tracker_contract
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.scenario import MountedRobot, ObjectDecl, Scenario, load_task
    from rrp.bodies.contact import apply_world, version_str
    body_key = robot if isinstance(robot, str) else robot.meta["name"]
    if isinstance(robot, str):
        robot = legged_body(robot)
    meta = copy.deepcopy(robot.meta)
    L = float(meta["legged"]["nominal_height"])
    m0, _, meta0 = standalone_model(robot, contact=contact)
    bw = body_width(m0, LeggedBinding(m0, meta0))
    rng = np.random.default_rng([seed, 4343])
    f = 1.6 - level * rng.uniform() * 0.4
    y_c = rng.uniform(-0.6, 0.6) * level * L
    psi_f = rng.uniform(-math.pi / 2, math.pi / 2) * level
    scene = legged_world(f"h_gap_{seed}", meta.get("source_options"), contact=contact)
    floor_kw = apply_world(mujoco.MjSpec(), contact, meta.get("source_options"))
    walls = add_gap_walls(scene, L, floor_kw, f * bw, y_c)
    goal = [GAP_X * L + 0.5 * L, y_c]
    scene.worldbody.add_camera(name="overhead", pos=[GAP_X * L, 0, 12.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=100)
    _waypoint(scene, "goal", goal, WAYPOINT_COLORS["cyan"])
    meta["contact_model"] = version_str(contact)
    meta["scene"] = dict(task="h_gap_sidestep", version=SCENE_VERSION, params=dict(width=f * bw, y_c=y_c, psi_f=psi_f), ground=[],
                         walls=walls, L=L)
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(robot.spec.copy(), prefix="r0_", site=site)
    scene.memory = 8 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body"})
    objects = [ObjectDecl("goal", "marker just beyond the gap", "feature", radius=0.12, task_entity="goal")]
    return Scenario("h_gap_sidestep", load_task("h_gap_sidestep"), scene, model, [mr], objects, seed,
                    meta=dict(body_key=body_key, contact_model=meta["contact_model"], L=L, level=level, gap_width=f * bw,
                              gap_ratio=f, body_width=bw, y_c=y_c, psi_f=psi_f, wall_x=GAP_X * L, walls=walls))


# ------------------------------------------------------------------ U2: L0 h_walk, L3 h_turn, M1 h_reach, M2 h_squat_pick, M3 h_place
MANIP_TASKS = ("h_walk", "h_turn", "h_reach", "h_squat_pick", "h_place")
MANIP_SCENE_VERSION = "humanoid_manip_v1"
# U3: C1 h_carry, C2 h_loco_pick (trained tasks) and the two HELD-OUT tasks (never in a training list: `TRAIN_TASKS` / `HELD_OUT_TASKS`,
# research/splits/humanoid_v1.json `held_out_tasks`). All four are built by `build_h_manip` and run in a HumanoidSession.
CARRY_TASKS = ("h_carry", "h_loco_pick")
HELD_OUT_TASKS = ("h_steps_carry", "h_gap_cart")
TRAIN_TASKS = ("h_steps", "h_gap") + MANIP_TASKS + CARRY_TASKS
CARRY_SCENE_VERSION = "humanoid_carry_v1"
LOCO_APPROACH = (0.8, 1.4)              # x L, h_loco_pick: how far the crate is placed beyond the M2 stance (the robot walks it first)
CARRY_GOAL_ANGLE = (1.6, 2.6)           # rad, h_carry: |bearing| of the goal from the start (behind / beside the robot: the crate is ahead)
CARRY_GOAL_DIST = (1.0, 1.6)            # x L
CARRY_TOL = 0.30                        # m, box-to-goal tolerance of the carry tasks
HELD_M = 0.25                           # m, "held": the nearest palm within this of the box / handle (a grasp has it at ~0.15)
HOLD_LOST_M = 0.35                      # m, hold_lost: the nearest palm farther than this from the box for HOLD_LOST_TICKS boundary ticks
HOLD_LOST_TICKS = 3
STEPS_CARRY_H = (0.08, 0.16)            # x L, h_steps_carry step height (2 up, platform, 2 down, mirrored behind the start)
STEPS_CARRY_N = 2
CART_HALF = (0.20, 0.5, 0.03)           # cart deck half sizes: x (m), y (x body width), z (m); the deck floats CART_Z above the floor
CART_Z = 0.12
CART_MASS = 3.0                         # kg; planar joints with viscous damping (below): a pram on a smooth floor, pushed by the squeeze friction
CART_DAMP = (20.0, 20.0, 4.0)           # N s/m, N s/m, N m s/rad of the x, y, yaw joints
CART_HANDLE = (0.30, 0.65)              # (x m in front of the start, z m): centre of the handle block (BOX_HALF-sized; the M2 palm targets sit 5 cm above it)
GAP_CART_F = (1.25, 1.5)                # gap width / cart width
BOX_HALF = (0.10, 0.18, 0.14)          # x L, box half sizes (x forward, y lateral, z): 0.13 x 0.24 x 0.13 m on t1 (the palms cannot
#                                        close inside a 0.13 m half-span: the shoulder roll limit)
BOX_MASS = 0.4                          # kg (the arms are 18 N m servos; a squeeze holds it with a wide margin)
BOX_FRICTION = (1.5, 0.02, 0.002)
CRATE_HALF_XY = (0.25, 0.50)            # x L, crate footprint half sizes
M2_CRATE = dict(top=0.50, front=0.37)  # x L: crate top height, and the distance from the start x to its front face (the squatting
M3_CRATE = dict(top=0.50, front=0.37)  #   shanks reach x = 0.21 m: the crate front stays clear of them, the box overhangs it by 1 cm)
BOX_X = 0.083                           # x L: box centre beyond the crate front (the arms reach x <= 0.30 m from a squat)
PLACE_ARC = 0.4                         # rad, h_place: the mark lies this far round the waist axis from the box (the arms cannot cross
#                                        the midline, so the box is carried sideways by twisting the trunk)
REACH_R = 0.05                          # m, palm-to-point tolerance of h_reach
HALT_SPEED = 0.15                       # m/s, "halted" of h_walk


def _lookat_xyaxes(pos, target) -> list:
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return [*x.tolist(), *np.cross(z, x).tolist()]


def _add_crate(scene, name: str, xy, top: float, half_xy, floor_kw: dict, rgba=(0.55, 0.42, 0.28, 1.0)):
    """A static box body resting on the floor (a body, so that the detector can track it)."""
    b = scene.worldbody.add_body(name=name, pos=[xy[0], xy[1], 0.5 * top])
    b.add_geom(name=f"{name}_geom", type=G.mjGEOM_BOX, size=[half_xy[0], half_xy[1], 0.5 * top], rgba=list(rgba), **floor_kw)
    return b


def _add_box(scene, name: str, xyz, half, rgba=(0.85, 0.15, 0.12, 1.0)):
    b = scene.worldbody.add_body(name=name, pos=list(xyz))
    b.add_freejoint(name=f"{name}_free")
    b.add_geom(name=f"{name}_geom", type=G.mjGEOM_BOX, size=list(half), rgba=list(rgba), mass=BOX_MASS,
               friction=list(BOX_FRICTION), condim=4)
    return b


def _add_ball(scene, name: str, xyz, radius: float, rgba):
    b = scene.worldbody.add_body(name=name, pos=list(xyz), mocap=True)
    b.add_geom(name=f"{name}_ball", type=G.mjGEOM_SPHERE, size=[radius, 0, 0], rgba=list(rgba), contype=0, conaffinity=0)
    return b


def _add_gap_cart(scene, robot, rng, contact, floor_kw, L: float, p: dict):
    """h_gap_cart (HELD OUT): the h_gap wall with a floating cart (planar x / y / yaw joints, viscous damping, no floor contact) whose
    rear handle block (BOX_HALF-sized, the palms squeeze it like the M2 box) stands in front of the robot; the cart must go
    through the gap (width = f x cart width) to the goal. Returns (camera, camera centre, objects, scene meta). The handle is the
    body `box` (task entity `box`) so that the grasp code of the pick teachers applies unchanged."""
    from rrp.bodies.legged import standalone_model
    from rrp.envs.mujoco.legged import WAYPOINT_COLORS, _waypoint
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.scenario import ObjectDecl
    m0, _, meta0 = standalone_model(robot, contact=contact)
    bw = body_width(m0, LeggedBinding(m0, meta0))
    cw = 2.0 * CART_HALF[1] * bw                         # cart width
    f = float(p.get("gap_ratio", rng.uniform(*GAP_CART_F)))
    y_c = float(p.get("y_c", rng.uniform(-0.3, 0.3) * L))
    bh = np.array(BOX_HALF) * L
    xh, zh = CART_HANDLE
    xc = xh + bh[0] + 0.02 + CART_HALF[0]                # deck centre
    zc = CART_Z + CART_HALF[2]
    cb = scene.worldbody.add_body(name="cart", pos=[xc, 0.0, zc])
    for nm, ax, kind, dmp in (("cart_x", [1, 0, 0], mujoco.mjtJoint.mjJNT_SLIDE, CART_DAMP[0]),
                              ("cart_y", [0, 1, 0], mujoco.mjtJoint.mjJNT_SLIDE, CART_DAMP[1]),
                              ("cart_yaw", [0, 0, 1], mujoco.mjtJoint.mjJNT_HINGE, CART_DAMP[2])):
        cb.add_joint(name=nm, type=kind, axis=ax, damping=dmp)
    cb.add_geom(name="cart_geom", type=G.mjGEOM_BOX, size=[CART_HALF[0], 0.5 * cw, CART_HALF[2]], mass=CART_MASS,
                rgba=[0.3, 0.45, 0.75, 1.0], friction=[0.4, 0.02, 0.002])
    cb.add_geom(name="cart_post", type=G.mjGEOM_BOX, pos=[-CART_HALF[0] + 0.01, 0.0, 0.5 * (zh - zc)], size=[0.01, 0.05, 0.5 * abs(zh - zc)],
                rgba=[0.3, 0.45, 0.75, 1.0], contype=0, conaffinity=0, mass=0.05)
    hb = cb.add_body(name="box", pos=[xh - xc, 0.0, zh - zc])
    hb.add_geom(name="box_geom", type=G.mjGEOM_BOX, size=list(bh), rgba=[0.85, 0.15, 0.12, 1.0], mass=0.05,
                friction=list(BOX_FRICTION), condim=4)
    walls = add_gap_walls(scene, L, floor_kw, f * cw, y_c)
    goal = [GAP_X * L + 0.9, y_c]
    _waypoint(scene, "goal", goal, WAYPOINT_COLORS["cyan"])
    objects = [ObjectDecl("cart", "blue cart", "object", (CART_HALF[0], 0.5 * cw, CART_HALF[2]), task_entity="cart"),
               ObjectDecl("box", "red cart handle", "object", tuple(bh), task_entity="box"),
               ObjectDecl("goal", "cart goal marker beyond the gap", "feature", radius=0.12, task_entity="goal")]
    smeta = dict(cart=dict(xy=[xc, 0.0], half=[CART_HALF[0], 0.5 * cw, CART_HALF[2]], width=cw), gap_width=f * cw, gap_ratio=f,
                 y_c=y_c, wall_x=GAP_X * L, walls=walls, goal=goal, goal_tol_m=0.35, halt_speed=HALT_SPEED, body_width=bw,
                 box=dict(xy=[xh, 0.0], half=bh.tolist(), z0=zh), crate=dict(xy=[xh, 0.0], top=zh - float(bh[2]), half=[0.0, 0.0]),
                 lift_m=0.05)
    return "overhead", [GAP_X * L, 0.0], objects, smeta


def build_h_manip(robot, seed: int, task: str, contact: str | None = "v2", **p):
    """Scenario for HumanoidSession: `task` in MANIP_TASKS on the humanoid `robot` (body key), the body at the origin facing +x.
    Everything is drawn from the seed (rng [seed, 4545]) unless a keyword (`dist_frac`, `angle`, `psi_f`, `target`, `side`) fixes
    it. Scenario meta carries the task map (marker / target / crate / box poses in the world frame) and L."""
    from rrp.bodies.compiler import compile_robot_spec
    from rrp.bodies.contact import apply_world, version_str
    from rrp.bodies.legged import legged_body, legged_world
    from rrp.envs.mujoco.legged import WAYPOINT_COLORS, _waypoint, tracker_contract
    from rrp.envs.mujoco.scenario import MountedRobot, ObjectDecl, Scenario, load_task
    if task not in MANIP_TASKS + CARRY_TASKS + HELD_OUT_TASKS:
        raise KeyError(f"unknown humanoid manipulation task {task!r}; has {MANIP_TASKS + CARRY_TASKS + HELD_OUT_TASKS}")
    body_key = robot if isinstance(robot, str) else robot.meta["name"]
    if isinstance(robot, str):
        robot = legged_body(robot)
    meta = copy.deepcopy(robot.meta)
    L = float(meta["legged"]["nominal_height"])
    rng = np.random.default_rng([seed, 4545])
    scene = legged_world(f"{task}_{seed}", meta.get("source_options"), contact=contact)
    floor_kw = apply_world(mujoco.MjSpec(), contact, meta.get("source_options"))
    objects, smeta, cam = [], {}, "overhead"
    cyan = WAYPOINT_COLORS["cyan"]
    if task == "h_walk":
        d = float(p.get("dist_frac", rng.uniform(1.0, 2.0))) * L
        ang = float(p.get("angle", rng.uniform(-0.5, 0.5)))
        goal = [d * math.cos(ang), d * math.sin(ang)]
        _waypoint(scene, "goal", goal, cyan)
        objects = [ObjectDecl("goal", "cyan goal marker", "feature", radius=0.12, task_entity="goal")]
        smeta = dict(goal=goal, reach_m=0.25, halt_speed=HALT_SPEED)
        centre = [0.5 * goal[0], 0.5 * goal[1]]
    elif task == "h_turn":
        psi = float(p.get("psi_f", rng.choice([-1, 1]) * rng.uniform(0.3 * math.pi, math.pi)))
        marker = [1.5 * L * math.cos(psi), 1.5 * L * math.sin(psi)]
        _waypoint(scene, "marker", marker, cyan)
        objects = [ObjectDecl("marker", "cyan heading marker", "feature", radius=0.12, task_entity="marker")]
        smeta = dict(psi_f=psi, marker=marker, bearing_tol=0.25, drift_max=0.5)
        centre = [0.0, 0.0]
    elif task == "h_reach":
        side = float(p.get("side", rng.choice([-1.0, 1.0])))
        tgt = list(p.get("target") or [rng.uniform(0.30, 0.45) * L, side * rng.uniform(0.15, 0.30) * L,
                                       rng.uniform(1.15, 1.40) * L])
        _add_ball(scene, "target", tgt, REACH_R, (0.95, 0.5, 0.1, 0.9))
        objects = [ObjectDecl("target", "orange target ball", "feature", radius=REACH_R, task_entity="target")]
        smeta = dict(target=tgt, side=side, tol_m=REACH_R)
        cam, centre = "front", [0.0, 0.0]
    elif task == "h_gap_cart":
        cam, centre, objects, smeta = _add_gap_cart(scene, robot, rng, contact, floor_kw, L, p)
    else:
        bh = np.array(BOX_HALF) * L
        cr = M3_CRATE if task == "h_place" else M2_CRATE
        top, half_c = cr["top"] * L, (CRATE_HALF_XY[0] * L, CRATE_HALF_XY[1] * L)
        appr = float(p.get("approach_frac", rng.uniform(*LOCO_APPROACH))) * L if task == "h_loco_pick" else 0.0
        cx = cr["front"] * L + half_c[0] + appr
        bxy = [(cr["front"] + BOX_X) * L + appr, 0.0]
        z0 = top + bh[2] + 0.002
        _add_crate(scene, "crate", [cx, 0.0], top, half_c, floor_kw)
        _add_box(scene, "box", [bxy[0], bxy[1], z0], bh)
        csz = (half_c[0], half_c[1], 0.5 * top)
        objects = [ObjectDecl("crate", "wooden crate", "feature", size=csz, radius=float(half_c[1]), task_entity="crate"),
                   ObjectDecl("box", "red box", "object", tuple(bh), task_entity="box")]
        smeta = dict(crate=dict(xy=[cx, 0.0], top=top, half=list(half_c)), box=dict(xy=bxy, half=bh.tolist(), z0=z0), lift_m=0.05)
        if task == "h_squat_pick":
            smeta["hold_s"] = 2.0
        elif task == "h_place":
            side = float(p.get("side", rng.choice([-1.0, 1.0])))
            w = waist_axis_xy(robot)
            r0 = np.array(bxy) - w
            psi = side * PLACE_ARC
            place = (w + np.array([math.cos(psi) * r0[0] - math.sin(psi) * r0[1], math.sin(psi) * r0[0] + math.cos(psi) * r0[1]])).tolist()
            _add_ball(scene, "place_mark", [place[0], place[1], top + 0.01], 0.02, (0.1, 0.8, 0.85, 0.9))
            objects.append(ObjectDecl("place_mark", "cyan placement mark", "feature", radius=0.02, task_entity="place_mark"))
            smeta.update(place=place, side=side, twist_rad=psi, waist_xy=w.tolist(), place_tol_m=0.04)
        elif task == "h_loco_pick":            # the M2 scene shifted `appr` ahead: the robot walks to the M2 stance first
            smeta.update(approach_m=appr, stance_x=0.0 + appr, stance_dist=cx - appr, stance_tol_m=0.12, hold_s=2.0)
        else:                                   # h_carry, h_steps_carry: pick, then carry the box to a goal behind / beside the start
            sg = 1.0 if p.get("side", rng.choice([-1.0, 1.0])) > 0 else -1.0
            if task == "h_carry":
                ang = sg * float(p.get("angle", rng.uniform(*CARRY_GOAL_ANGLE)))
                dist = float(p.get("dist_frac", rng.uniform(*CARRY_GOAL_DIST))) * L
                goal = [dist * math.cos(ang), dist * math.sin(ang)]
            else:
                hf = float(p.get("h_frac", rng.uniform(*STEPS_CARRY_H)))
                n = int(p.get("n_steps", STEPS_CARRY_N))
                ground = add_steps(scene, L, hf * L, floor_kw, n=n, sign=-1.0)
                x_end = steps_layout(L, hf * L, n)[1]
                goal = [-(x_end + 0.3 * L), 0.0]
                smeta.update(h_frac=hf, x_end=x_end, ground=ground,
                             staircase=dict(x0=-STEPS_X0 * L, tread=STEPS_TREAD * L, n=n, h=hf * L, platform=STEPS_PLATFORM * L,
                                            direction=-1.0))
            _waypoint(scene, "goal", goal, cyan)
            objects.append(ObjectDecl("goal", "cyan goal marker", "feature", radius=0.12, task_entity="goal"))
            smeta.update(goal=goal, goal_tol_m=CARRY_TOL, halt_speed=HALT_SPEED)
            cam = "overhead"
        if task == "h_loco_pick":
            cam, centre = "overhead", [0.5 * (cx + 0.0), 0.0]
        elif task in ("h_carry", "h_steps_carry"):
            centre = [0.5 * smeta["goal"][0], 0.5 * smeta["goal"][1]]
        else:
            cam, centre = "front", [0.4 * L, 0.0]
    scene.worldbody.add_camera(name="overhead", pos=[centre[0], centre[1], 12.0], xyaxes=[1, 0, 0, 0, 1, 0], fovy=100)
    if cam == "front" and task in MANIP_TASKS:       # a camera beyond the crate, looking back at the robot: the arms do not hide the box
        cpos = [2.2 * L, 0.0, 1.3 * L]
        scene.worldbody.add_camera(name="front", pos=cpos, xyaxes=_lookat_xyaxes(cpos, [0.2 * L, 0.0, 0.6 * L]), fovy=70)
    meta["contact_model"] = version_str(contact)
    meta["scene"] = dict(task=task, version=CARRY_SCENE_VERSION if task in CARRY_TASKS + HELD_OUT_TASKS else MANIP_SCENE_VERSION,
                         params=dict(smeta), ground=list(smeta.get("ground", [])), L=L, walls=list(smeta.get("walls", [])))
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(robot.spec.copy(), prefix="r0_", site=site)
    scene.memory = 8 * 2 ** 20
    model = scene.compile()
    rs = compile_robot_spec(model, meta, prefix="r0_", name=body_key)
    rs = rs.model_copy(update=dict(controller_contracts=rs.controller_contracts + [tracker_contract(meta, rs)])).with_hash()
    mr = MountedRobot("r0_", meta, rs, [0.0, 0.0, 0.0], 0.0, {"body": "body"})
    tdef = load_task(task)
    if task == "h_loco_pick":            # the walk_to threshold is in units of the body: the M2 stance distance + tolerance
        tdef["events"][0]["completion"][0]["value"] = round(smeta["stance_dist"] + smeta["stance_tol_m"], 4)
    return Scenario(task, tdef, scene, model, [mr], objects, seed,
                    meta=dict(body_key=body_key, contact_model=meta["contact_model"], L=L, detector_camera=cam,
                              scene_version=meta["scene"]["version"], **smeta))


def waist_joint(model, binding) -> tuple | None:
    """(held index, twist sign) of the waist yaw joint, or None: the held joint with a vertical axis whose child body is an ancestor
    of both feet (sign -1: the root is the chest above a planted pelvis, so the chest yaws by -q) or of both hands (sign +1:
    the root is the pelvis and the chest yaws by +q). Position (`waist_axis_xy`) is that of the default stance."""
    def anc(b):
        out = set()
        while b > 0:
            out.add(int(b))
            b = int(model.body_parentid[b])
        return out
    feet = [int(model.site_bodyid[s]) for s in binding.foot_sids]
    hands = [int(b) for b in binding.payload_bodies()]
    for i, a in enumerate(binding.held_act):
        j = int(model.actuator_trnid[a, 0])
        if abs(model.jnt_axis[j][2]) < 0.9:
            continue
        below = int(model.jnt_bodyid[j])
        for group, sign in ((feet, -1), (hands, 1)):
            if all(below in anc(g) for g in group):
                return i, sign
    return None


def waist_axis_xy(robot) -> np.ndarray:
    """xy of the waist axis of `robot` (a LeggedBody or its key) at the default stance, base at the origin."""
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedBinding
    if isinstance(robot, str):
        robot = legged_body(robot)
    m = robot.spec.copy().compile()
    b = LeggedBinding(m, robot.meta, "")
    w = waist_joint(m, b)
    if w is None:
        raise ValueError(f"body {robot.meta['name']!r} has no waist yaw joint; h_place needs one")
    d = mujoco.MjData(m)
    b.set_default(d)
    mujoco.mj_kinematics(m, d)
    return d.xanchor[int(m.actuator_trnid[b.held_act[w[0]], 0])][:2].copy()


class HumanoidSession(LeggedSession):
    """LeggedSession for the U2 tasks. Public predicates (estimators of the observation, all from declared sensors or the
    detector): `hand_distance_m(body, e)` (nearest palm to the tracked entity, palm positions by forward kinematics of the
    joint encoders on the declared noisy localization and the IMU roll / pitch), `height_above_m(object, support)` (tracked
    z of the object's underside above the support's top), `distance_m(a, b)` (horizontal, two tracked entities; `distance_m(body, e)` is
    the base one), `bearing_error_rad(body, e)` (|heading to the tracked entity - localized yaw|), `drift_m(body)` (horizontal
    distance from the first localized position) and `stand_frac(body)` (pelvis height over the lowest foot site by the same
    forward kinematics, as a fraction of its default-stance value). The truth versions read the simulator.
    `failure_reason()` is a PRIVILEGED diagnostic for the reports (fell, dropped, hold_lost, wall_collision, drift, no_grasp,
    not_upright, place_miss)."""

    STAND_OK = 0.9
    LOC_AVG = 5           # boundary ticks (0.5 s) of the localization average that public_fk places the hands with (the single-sample noise
    #                       is 2 cm per axis: too much for a 5 cm palm tolerance; the manipulation tasks are quasi-static)

    def __init__(self, scenario, **kw):
        kw.setdefault("detector", DetectorConfig(camera=scenario.meta.get("detector_camera", "overhead"), pos_sigma=0.01,
                                                 dropout=0.02, max_range=40.0))
        self._start = None
        self._lift_max = 0.0
        self._geo = None
        self._loc_recent = []
        self._true_xy = []
        self._hold_bad = 0          # U3: boundary ticks in a row with the nearest palm farther than HOLD_LOST_M from the box / handle
        self._wall_ticks = 0        # h_gap_cart: boundary ticks with a robot or cart geom in contact with a wall
        self._engaged = False       # the payload was lifted (box tasks) / the cart pushed 0.10 m (h_gap_cart): hold_lost applies from then on
        super().__init__(scenario, **kw)

    # ------------------------------------------------------------------ geometry constants (from the public body model)
    def _geom(self) -> dict:
        if self._geo is not None:
            return self._geo
        m, b = self.model, self.binding
        d = mujoco.MjData(m)
        b.set_default(d)
        mujoco.mj_kinematics(m, d)
        palms = {}
        for hb in b.payload_bodies():
            gid = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == hb and m.geom_contype[g])
            palms["left" if d.xpos[hb][1] > d.xpos[b.root_bid][1] else "right"] = int(gid)
        z_site = min(float(d.site_xpos[s][2]) for s in b.foot_sids)          # foot site height over the floor at the default stance
        self._geo = dict(palm_gid=palms, site_z=z_site, stand_h0=float(d.qpos[b.qa + 2]) - z_site)
        self._fk = mujoco.MjData(m)
        return self._geo

    def _capabilities(self) -> list[str]:
        """`arm_roles`: two hand links (`LeggedBinding.payload_bodies`). Tasks that need arms declare it in `TaskSpec.needs`, so `negotiate`
        (hence `rrp matrix`) records a body with an upper group but no hands as n/a with "body has no arm roles" instead of failing at the
        scene build; the relational packet binds its hand roles to null with reason `absent_limb` (docs/architecture.md 14.4)."""
        return super()._capabilities() + (["arm_roles"] if len(self.binding.payload_bodies()) >= 2 else [])

    def palm_ids(self) -> dict:
        """{"right": geom id, "left": geom id} of the palm bars (the contact geoms of the two hand links)."""
        return dict(self._geom()["palm_gid"])

    def reset(self, seed=None):
        self._geom()
        self._start = None
        self._lift_max = 0.0
        self._loc_recent = []
        self._true_xy = []
        self._hold_bad, self._wall_ticks, self._engaged = 0, 0, False
        return super().reset(seed)

    def _sense(self):
        super()._sense()
        if self._start is None and self.loc is not None:
            self._start = self.loc[:2].copy()
        self._loc_recent = (self._loc_recent + [self.loc.copy()])[-self.LOC_AVG:]
        w = max(2, int(round(1.0 / self.dt)) + 1)                         # the estimator's 1 s baseline (legged._sense)
        self._true_xy = (self._true_xy + [self.data.qpos[self.binding.qa:self.binding.qa + 2].copy()])[-w:]

    def _control_boundary(self):
        obs = super()._control_boundary()
        sc = self.scenario.meta
        if "box" in sc:
            self._lift_max = max(self._lift_max, float(self._body_pos("box")[2]) - sc["box"]["z0"])
        if self.scenario.name in CARRY_TASKS + HELD_OUT_TASKS:
            self._track_payload(sc)
        return obs

    def _track_payload(self, sc: dict):
        """U3 bookkeeping at every boundary tick (truth, for `failure_reason`): engaged, palm-to-payload distance run, wall contacts."""
        cart = self.scenario.name == "h_gap_cart"
        if not self._engaged:
            self._engaged = (float(self._body_pos("cart")[0]) - sc["cart"]["xy"][0] >= 0.10) if cart else self._lift_max >= sc["lift_m"]
        far = self.truth_predicate("hand_distance_m", ["body", "box"]) > HOLD_LOST_M
        self._hold_bad = self._hold_bad + 1 if (self._engaged and far) else 0
        if cart:
            m, d = self.model, self.data
            walls = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, w) for w in sc["walls"]}
            cart_b = m.body("cart").id
            hit = False
            for c in d.contact[:d.ncon]:
                a, b = int(c.geom1), int(c.geom2)
                if (a in walls) != (b in walls):
                    o = b if a in walls else a
                    hit = hit or bool(self.binding.is_robot_body[m.geom_bodyid[o]]) or int(m.body_rootid[m.geom_bodyid[o]]) == int(m.body_rootid[cart_b])
            self._wall_ticks = self._wall_ticks + 1 if hit else 0

    def _body_pos(self, entity: str) -> np.ndarray:
        o = next(o for o in self.scenario.objects if o.task_entity == entity)
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, o.sim_body)]

    def _decl(self, entity: str):
        return next((o for o in self.scenario.objects if o.task_entity == entity), None)

    # ------------------------------------------------------------------ public forward kinematics
    def public_fk(self):
        """(palms {side: xyz world}, pelvis height over the feet) from the joint encoders, the localized (x, y, yaw) averaged over the
        last `LOC_AVG` boundary ticks and the IMU roll / pitch; None before the first localization. The base height is not sensed:
        the pelvis is placed so that the lowest foot site touches the floor."""
        if self.loc is None:
            return None
        rec = np.array(self._loc_recent)
        loc = np.array([rec[:, 0].mean(), rec[:, 1].mean(), math.atan2(np.sin(rec[:, 2]).mean(), np.cos(rec[:, 2]).mean())])
        g = self._geom()
        b, d, fk = self.binding, self.data, self._fk
        r = self.robots[0]
        fk.qpos[:] = 0.0
        fk.qpos[r.qadr] = d.qpos[r.qadr]
        gv = quat_rotate_inv(self._imu()["quat"], np.array([0, 0, -1.0]))
        roll, pitch = math.atan2(-gv[1], -gv[2]), math.asin(max(-1.0, min(1.0, gv[0])))
        yaw = float(loc[2])
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        fk.qpos[b.qa:b.qa + 3] = [loc[0], loc[1], 0.0]
        fk.qpos[b.qa + 3:b.qa + 7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        mujoco.mj_kinematics(self.model, fk)
        h = -min(float(fk.site_xpos[s][2]) for s in b.foot_sids) + g["site_z"]
        lift = np.array([0.0, 0.0, h])
        return {s: fk.geom_xpos[i] + lift for s, i in g["palm_gid"].items()}, h

    def _truth_palms(self) -> dict:
        return {s: self.data.geom_xpos[i].copy() for s, i in self._geom()["palm_gid"].items()}

    # ------------------------------------------------------------------ public estimators
    def estimate(self, predicate, args):
        sc = self.scenario.meta
        if predicate == "distance_m" and len(args) == 2 and args[0] != "body":
            p, _ = self._track(args[0])
            q, _ = self._track(args[1])
            if p is None or q is None:
                return None, False, 0.0
            return float(np.linalg.norm(p[:2] - q[:2])), True, 0.8
        if predicate == "hand_distance_m" and len(args) == 2 and args[0] == "body":
            q, _ = self._track(args[1])
            fk = self.public_fk()
            if q is None or fk is None:
                return None, False, 0.0
            return float(min(np.linalg.norm(p - q) for p in fk[0].values())), True, 0.8
        if predicate == "height_above_m" and len(args) == 2:
            p, _ = self._track(args[0])
            q, _ = self._track(args[1])
            if p is None or q is None:
                return None, False, 0.0
            return float((p[2] - self._decl(args[0]).size[2]) - (q[2] + self._decl(args[1]).size[2])), True, 0.8
        if predicate == "bearing_error_rad" and len(args) == 2 and args[0] == "body":
            q, _ = self._track(args[1])
            if q is None or self.loc is None:
                return None, False, 0.0
            return _wrap(math.atan2(q[1] - self.loc[1], q[0] - self.loc[0]) - self.loc[2], absolute=True), True, 0.9
        if predicate == "drift_m" and args == ["body"]:
            if self._start is None or self.loc is None:
                return None, False, 0.0
            return float(np.linalg.norm(self.loc[:2] - self._start)), True, 0.9
        if predicate == "stand_frac" and args == ["body"]:
            fk = self.public_fk()
            return (None, False, 0.0) if fk is None else (float(fk[1] / self._geom()["stand_h0"]), True, 0.85)
        return super().estimate(predicate, args)

    def truth_predicate(self, pred, args, held=None):
        b, d = self.binding, self.data
        if pred == "distance_m" and len(args) == 2 and args[0] != "body":
            return float(np.linalg.norm(self._body_pos(args[0])[:2] - self._body_pos(args[1])[:2]))
        if pred == "hand_distance_m":
            q = self._body_pos(args[1])
            return float(min(np.linalg.norm(p - q) for p in self._truth_palms().values()))
        if pred == "height_above_m":
            return float((self._body_pos(args[0])[2] - self._decl(args[0]).size[2])
                         - (self._body_pos(args[1])[2] + self._decl(args[1]).size[2]))
        if pred == "bearing_error_rad":
            q, x, y, yaw = self._body_pos(args[1]), d.qpos[b.qa], d.qpos[b.qa + 1], float(self.base_pose_truth()[2])
            return _wrap(math.atan2(q[1] - y, q[0] - x) - yaw, absolute=True)
        if pred == "drift_m":
            return None if self._start is None else float(np.linalg.norm(d.qpos[b.qa:b.qa + 2] - self._start))
        if pred == "base_speed":                    # truth twin of the estimator: the same 1 s baseline of the true position
            h = self._true_xy
            return float(np.linalg.norm(h[-1] - h[0])) / ((len(h) - 1) * self.dt) if len(h) > 1 else 0.0
        if pred == "stand_frac":
            fz = min(float(d.site_xpos[s][2]) for s in b.foot_sids)
            return float((d.qpos[b.qa + 2] - fz) / self._geom()["stand_h0"])
        return super().truth_predicate(pred, args, held)

    # ------------------------------------------------------------------ PRIVILEGED failure naming (reports only)
    def failure_reason(self) -> str | None:
        """First applicable of: fell; dropped (the box is below every support top: it left the crate for the floor); wall_collision
        (h_gap_cart: a robot or cart geom in contact with a wall for 0.2 s); drift (h_turn: the base left its start by more than
        `drift_max`); hold_lost (U3: after the payload was lifted / the cart pushed, the nearest palm has been farther than
        HOLD_LOST_M from it for HOLD_LOST_TICKS boundary ticks while it is not on the floor); no_grasp (box tasks: the box never rose
        `lift_m` above the crate; h_gap_cart: the cart never moved 0.3 m); not_upright (h_squat_pick, h_loco_pick: lifted but the body is
        not standing); place_miss (h_place: lifted but not on the mark at the end). None when nothing applies (the judge then
        reports timeout)."""
        sc = self.scenario.meta
        name = self.scenario.name
        cart = name == "h_gap_cart"
        if self.fell:
            return "fell"
        if "box" in sc and not cart:
            supports = [sc["crate"]["top"]]
            if float(self._body_pos("box")[2]) - sc["box"]["half"][2] < min(supports) - 0.06:
                return "dropped"
        if cart and self._wall_ticks >= 2:
            return "wall_collision"
        if "drift_max" in sc and self._start is not None:
            if float(np.linalg.norm(self.data.qpos[self.binding.qa:self.binding.qa + 2] - self._start)) > sc["drift_max"]:
                return "drift"
        if self._hold_bad >= HOLD_LOST_TICKS:
            return "hold_lost"
        if cart:
            if float(self._body_pos("cart")[0]) - sc["cart"]["xy"][0] < 0.3:
                return "no_grasp"
        elif "box" in sc:
            if self._lift_max < sc["lift_m"]:
                return "no_grasp"
            if name in ("h_squat_pick", "h_loco_pick") and self.truth_predicate("stand_frac", ["body"]) < self.STAND_OK:
                return "not_upright"
            if name == "h_place":
                on = np.linalg.norm(self._body_pos("box")[:2] - np.array(sc["place"])) <= sc["place_tol_m"]
                if not on:
                    return "place_miss"
        return None

    def snapshot(self):
        snap = super().snapshot()
        snap.components["controller_state"][0]["humanoid"] = dict(
            start=None if self._start is None else self._start.tolist(), lift_max=self._lift_max,
            loc_recent=[x.tolist() for x in self._loc_recent],
            true_xy=[x.tolist() for x in self._true_xy], hold_bad=self._hold_bad, wall_ticks=self._wall_ticks,
            engaged=self._engaged)
        return snap

    def restore(self, snap):
        obs = super().restore(snap)
        st = snap.components["controller_state"][0].get("humanoid", {})
        self._start = None if st.get("start") is None else np.array(st["start"])
        self._lift_max = float(st.get("lift_max", 0.0))
        self._loc_recent = [np.array(x) for x in st.get("loc_recent", [])]
        self._true_xy = [np.array(x) for x in st.get("true_xy", [])]
        self._hold_bad, self._wall_ticks = int(st.get("hold_bad", 0)), int(st.get("wall_ticks", 0))
        self._engaged = bool(st.get("engaged", False))
        return obs


def _wrap(a: float, absolute: bool = False) -> float:
    a = math.atan2(math.sin(a), math.cos(a))
    return abs(a) if absolute else a


def make_humanoid_session(*, task: str, body: str, seed: int = 0, scene: dict | None = None, **kw):
    """env_id "mujoco/legged" for the humanoid tasks (the `build` entry of rrp.tasks.humanoid): the task's scene on `body`
    (`scene` = builder kwargs: h_frac | level | the U2 task parameters, contact) driven by a LeggedSession (h_steps, h_gap) or a
    HumanoidSession (h_walk, h_turn, h_reach, h_squat_pick, h_place, U3: h_carry, h_loco_pick, h_steps_carry, h_gap_cart; control defaults
    to "wholebody"); `kw` go to the session."""
    if task in MANIP_TASKS + CARRY_TASKS + HELD_OUT_TASKS:
        kw.setdefault("control", "wholebody")
        return HumanoidSession(build_h_manip(body, seed, task, **(scene or {})), seed=seed, **kw)
    builder = {"h_steps": build_h_steps, "h_gap": build_h_gap}[task]
    return LeggedSession(builder(body, seed, **(scene or {})), seed=seed, **kw)


def gap_obs_np(qpos_root: np.ndarray, sc_meta: dict, phase2: float) -> np.ndarray:
    """numpy twin of WarpGapEnv.extra_obs (PRIVILEGED expert input)."""
    x, y = qpos_root[0], qpos_root[1]
    w, qx, qy, qz = qpos_root[3:7]
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    L = sc_meta["L"]
    dx, dy = sc_meta["wall_x"] - x, sc_meta["y_c"] - y
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([(c * dx + s_ * dy) / L, (-s_ * dx + c * dy) / L, sc_meta["gap_width"] / L, sc_meta["body_width"] / L,
                     math.sin(sc_meta["psi_f"] - yaw), math.cos(sc_meta["psi_f"] - yaw), phase2, dx / L], np.float32)
