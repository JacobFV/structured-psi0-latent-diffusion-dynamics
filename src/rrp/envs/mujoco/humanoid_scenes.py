"""W13 humanoid task scenes (D-138, research/tracks/humanoid.md section 2). Scenes are static world geoms added to the tracker
world (contact_v2 floor) and scaled by the body's leg length L (= nominal root height), so one task definition spans
toddler-size and adult humanoids. Every scene geom counts as GROUND for foot-contact / fall logic ("ground" list).

Implemented so far (h_gap_sidestep below):
  h_steps  (L2): up N steps, a platform, down N steps, along +x; step height h (fraction of L, per episode), tread 0.6 L.
           Success: base past the far end (x > x_end) without falling; failure reasons: fell, trip (a non-sole geom of a
           foot touches a riser > 0.3 s), timeout. Per-world step heights are batchable (geom_pos / geom_size / aabb / rbound).
Task knowledge a student may receive (never outcomes): the staircase geometry (x0, tread, n, h) as the task map.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

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


def add_steps(spec: mujoco.MjSpec, L: float, h: float, floor_kw: dict, n: int = STEPS_N, width: float = 3.0) -> list:
    names = []
    lay, _ = steps_layout(L, h, n)
    for name, cx, hx, top in lay:
        # boxes extend BURY x L below the floor, so a zero-height step is flush with the floor (no 0.2 mm edge to trip on)
        spec.worldbody.add_geom(name=name, type=G.mjGEOM_BOX, pos=[cx, 0, 0.5 * (top - BURY * L)],
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


def make_humanoid_session(*, task: str, body: str, seed: int = 0, scene: dict | None = None, **kw):
    """env_id "mujoco/legged" for the humanoid tasks (the `build` entry of rrp.tasks.humanoid): the task's scene on `body`
    (`scene` = builder kwargs: h_frac | level, contact) driven by a LeggedSession; `kw` go to the session."""
    from rrp.envs.mujoco.legged import LeggedSession
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
