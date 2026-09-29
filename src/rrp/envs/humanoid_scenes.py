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
