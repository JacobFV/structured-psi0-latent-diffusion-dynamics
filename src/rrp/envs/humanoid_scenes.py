"""W13 humanoid task scenes (D-138, research/tracks/humanoid.md section 2). Scenes are static world geoms added to the tracker
world (contact_v2 floor) and scaled by the body's leg length L (= nominal root height), so one task definition spans
toddler-size and adult humanoids. Every scene geom counts as GROUND for foot-contact / fall logic ("ground" list).

Implemented so far:
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
SCENE_VERSION = "humanoid_scenes_v1"
STEPS_N = 3
STEPS_TREAD = 0.6            # x L
STEPS_X0 = 1.2               # x L, first riser
STEPS_PLATFORM = 1.2         # x L


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
        spec.worldbody.add_geom(name=name, type=G.mjGEOM_BOX, pos=[cx, 0, 0.5 * top], size=[hx, 0.5 * width * L, 0.5 * top],
                                rgba=[0.55, 0.5, 0.45, 1], **floor_kw)
        names.append(name)
    return names


def task_model(body: str, task: str, params: dict | None = None, contact: str = "v2", prefix: str = "r0_"):
    """(model, spec, meta) of `body` in the `task` scene (same construction as rrp.bodies.legged.standalone_model plus
    scene geoms). meta['scene'] = {task, version, params, ground: [geom names]}."""
    from rrp.bodies.legged import legged_body, legged_world
    from rrp.physics.contact import apply_world, version_str
    module = legged_body(body)
    meta = copy.deepcopy(module.meta)
    L = float(meta["legged"]["nominal_height"])
    scene = legged_world(f"{meta['name']}_{task}", meta.get("source_options"), contact=contact)
    floor_kw = apply_world(mujoco.MjSpec(), contact, meta.get("source_options"))   # floor contact params, for scene geoms
    params = dict(params or {})
    if task == "h_steps":
        h = float(params.get("h_frac", 0.15)) * L
        params.setdefault("h_frac", 0.15)
        ground = add_steps(scene, L, h, floor_kw)
        params["x_end"] = steps_layout(L, h)[1]
    else:
        raise KeyError(f"unknown humanoid task scene {task!r}")
    meta["contact_model"] = version_str(contact)
    meta["scene"] = dict(task=task, version=SCENE_VERSION, params=params, ground=ground, L=L)
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
    from rrp.envs.legged import WAYPOINT_COLORS, _waypoint, tracker_contract
    from rrp.envs.scenario import MountedRobot, ObjectDecl, Scenario, load_task
    from rrp.physics.contact import apply_world, version_str
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


def steps_scan_np(qpos_root: np.ndarray, L: float, h: float) -> np.ndarray:
    """numpy twin of WarpStepsEnv.extra_obs (PRIVILEGED expert input): 11 x 3 yaw-frame ground heights - (base z - L), / L; + h/L."""
    from rrp.envs.warp_task_env import SCAN_X, SCAN_Y
    x, y, z = qpos_root[:3]
    w, qx, qy, qz = qpos_root[3:7]
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    sx, sy = np.meshgrid(SCAN_X, SCAN_Y, indexing="ij")
    px = x + L * (math.cos(yaw) * sx.ravel() - math.sin(yaw) * sy.ravel())
    hz = steps_height_at(px, L, h)
    return np.concatenate([(hz - (z - L)) / L, [h / L]]).astype(np.float32)
