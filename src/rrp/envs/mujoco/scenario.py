"""Scenario construction: workspace + robots at mounts + task objects/features.

The scenario records the privileged entity->sim-body map (truth bus only) separately from
public descriptors (what a detector can report) and public manipulator bindings (the user
states which assembly plays which task role).
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from rrp.bodies.generators import (Module, workspace_spec, add_box_object, add_target_zone, add_mount,
                                       _quat_from_axis_angle)
from rrp.bodies.surgery import Assembled
from rrp.bodies.compiler import compile_robot_spec
from rrp.core.robot import RobotSpec

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks" / "graphs"     # package data: src/rrp/tasks/graphs/<task>.json

COLORS = {"red": (0.85, 0.12, 0.1, 1), "blue": (0.12, 0.25, 0.85, 1), "yellow": (0.9, 0.8, 0.1, 1),
          "purple": (0.55, 0.15, 0.7, 1), "orange": (0.95, 0.5, 0.1, 1), "cyan": (0.1, 0.8, 0.85, 1)}


def load_task(name: str) -> dict:
    return json.loads((TASKS_DIR / f"{name}.json").read_text())


@dataclass
class ObjectDecl:
    sim_body: str
    descriptor: str
    kind: str                 # object | feature
    size: tuple = (0.022, 0.022, 0.022)
    radius: float = 0.05
    task_entity: str | None = None   # PRIVILEGED mapping, truth bus only


@dataclass
class MountedRobot:
    prefix: str
    meta: dict
    robot_spec: RobotSpec
    base_pos: list
    base_yaw: float
    manipulator_bindings: dict       # task manipulator entity -> assembly id (public)


@dataclass
class Scenario:
    name: str
    task: dict
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    robots: list[MountedRobot]
    objects: list[ObjectDecl]
    seed: int
    meta: dict = field(default_factory=dict)

    def object(self, sim_body: str) -> ObjectDecl:
        return next(o for o in self.objects if o.sim_body == sim_body)


def _robot_module(r) -> tuple[mujoco.MjSpec, dict]:
    if isinstance(r, Assembled):
        return r.spec.copy(), copy.deepcopy(r.meta)
    if isinstance(r, Module):
        return r.spec.copy(), copy.deepcopy(r.meta)
    raise TypeError("robot must be a Module or Assembled")


def compile_scene(scene: mujoco.MjSpec, post_contact=None) -> mujoco.MjModel:
    """Compile an arm world with the selected grasp contact version (rrp.physics.grasp_contact; $RRP_GRASP_CONTACT,
    default grasp_v1 = unchanged legacy build). post_contact(scene): optional edit applied AFTER the grasp contact
    model (D-126 #35 object friction must not be overwritten by it); None = the historical build."""
    from rrp.bodies import grasp_contact
    grasp_contact.apply(scene)
    if post_contact is not None:
        post_contact(scene)
    return scene.compile()


# ------------------------------------------------------------------------------ D-126 #35: richer task objects
# Default OFF: build_pick_place(object_spec=None) is the historical scene. An object spec changes ONLY the task object
# ("cube" in the privileged sim name; its public descriptor stays "<color> cube" unless descriptor_shape is true, so the
# task binding is unchanged). Distractors and all placement draws are unchanged.
OBJECT_SHAPES = ("cube", "cylinder", "box_tall", "box_flat")
OBJECT_SPEC_VERSION = "object_spec_v1"
OBJECT_VARIATION_ENV = "RRP_OBJECT_VARIATION"


def object_half_extents(shape: str, half: float) -> tuple:
    """(x, y, z) half extents of the task object (cylinder: radius, radius, half height). Every horizontal extent is
    <= 1.2 x half, so a gripper that takes the cube takes every variant."""
    if shape == "cube":
        return (half, half, half)
    if shape == "cylinder":
        return (half, half, half)
    if shape == "box_tall":
        return (half, half, 1.5 * half)
    if shape == "box_flat":
        return (1.2 * half, 1.2 * half, 0.6 * half)
    raise ValueError(f"unknown object shape {shape!r} (known: {OBJECT_SHAPES})")


def sample_object_spec(variation: dict, seed: int) -> dict:
    """Per-episode object spec from ranges, with its own RNG stream ([seed, 35]; the scene RNG is untouched):
    variation = {"size": [lo, hi] half size (m), "mass": [lo, hi] (kg), "friction": [lo, hi] sliding mu,
                 "shapes": [...], "descriptor_shape": bool}. Missing keys keep the default for that property."""
    rng = np.random.default_rng([int(seed), 35])
    spec = {}
    if variation.get("shapes"):
        shapes = list(variation["shapes"])
        spec["shape"] = shapes[int(rng.integers(len(shapes)))]
    for k in ("size", "mass", "friction"):
        if variation.get(k) is not None:
            lo, hi = variation[k]
            spec[k] = float(rng.uniform(lo, hi))
    if variation.get("descriptor_shape"):
        spec["descriptor_shape"] = True
    return spec


def _check_object_spec(spec: dict) -> dict:
    bad = set(spec) - {"shape", "size", "mass", "friction", "descriptor_shape"}
    if bad:
        raise ValueError(f"unknown object_spec keys {sorted(bad)}")
    out = dict(shape=spec.get("shape", "cube"))
    object_half_extents(out["shape"], 0.022)
    for k, lo, hi in (("size", 0.01, 0.04), ("mass", 0.005, 2.0), ("friction", 0.05, 3.0)):
        if spec.get(k) is not None:
            v = float(spec[k])
            if not lo <= v <= hi:
                raise ValueError(f"object_spec {k}={v} outside [{lo}, {hi}]")
            out[k] = v
    out["descriptor_shape"] = bool(spec.get("descriptor_shape", False))
    return out


def mount_robots(scene: mujoco.MjSpec, robots: list, mounts: list[tuple]) -> list[tuple[str, dict, list, float]]:
    out = []
    for i, (r, (pos, yaw)) in enumerate(zip(robots, mounts)):
        spec, meta = _robot_module(r)
        site = add_mount(scene, f"mount{i}", pos, yaw)
        prefix = f"r{i}_"
        scene.attach(spec, prefix=prefix, site=site)
        out.append((prefix, meta, list(pos), yaw))
    return out


def _free_xy(rng, lo, hi, others, min_d):
    for _ in range(200):
        p = rng.uniform(lo, hi)
        if all(np.linalg.norm(p - o) >= min_d for o in others):
            return p
    raise RuntimeError("could not place object without overlap")


def build_pick_place(robot, seed: int, *, n_distractors: int = 0, cube_color: str = "red",
                     distractor_colors=("blue", "yellow", "purple"), task: dict | None = None,
                     cube_size: float = 0.022, base=((0.0, 0.0, 0.0), 0.0), object_spec: dict | None = None) -> Scenario:
    """object_spec (D-126 #35; default None = the historical scene): shape (OBJECT_SHAPES), size (half size, m; replaces
    cube_size for the task object only), mass (kg), friction (sliding mu of the object geom, set after the grasp contact
    model), descriptor_shape (public descriptor "<color> <shape>" instead of "<color> cube"). Recorded in meta.
    $RRP_OBJECT_VARIATION (JSON ranges, sample_object_spec) applies a per-seed spec when object_spec is None, so the
    evaluation harnesses (ladder, GRPO, robustness) can use varied objects without new arguments; unset = unchanged."""
    if object_spec is None and os.environ.get(OBJECT_VARIATION_ENV):
        object_spec = sample_object_spec(json.loads(os.environ[OBJECT_VARIATION_ENV]), seed)
    if object_spec is not None:
        return _build_pick_place_obj(robot, seed, n_distractors=n_distractors, cube_color=cube_color,
                                     distractor_colors=distractor_colors, task=task, cube_size=cube_size, base=base,
                                     object_spec=_check_object_spec(object_spec))
    rng = np.random.default_rng(seed)
    scene = workspace_spec(f"pick_place_{seed}")
    mounted = mount_robots(scene, [robot], [base])
    placed = []
    cube_xy = _free_xy(rng, [0.32, -0.18], [0.50, 0.18], placed, 0.1)
    placed.append(cube_xy)
    tgt_xy = _free_xy(rng, [0.30, -0.25], [0.52, 0.25], placed, 0.14)
    placed.append(tgt_xy)
    cube = add_box_object(scene, "cube", [*cube_xy, cube_size + 0.001], size=(cube_size,) * 3, rgba=COLORS[cube_color])
    cube.quat = _quat_from_axis_angle([0, 0, 1], rng.uniform(-math.pi / 4, math.pi / 4))
    add_target_zone(scene, "target_zone", [*tgt_xy, 0.0005], radius=0.05)
    objects = [ObjectDecl("cube", f"{cube_color} cube", "object", (cube_size,) * 3, task_entity="cube"),
               ObjectDecl("target_zone", "green target zone", "feature", radius=0.05, task_entity="target")]
    for k in range(n_distractors):
        xy = _free_xy(rng, [0.28, -0.3], [0.55, 0.3], placed, 0.09)
        placed.append(xy)
        col = distractor_colors[k % len(distractor_colors)]
        b = add_box_object(scene, f"distractor{k}", [*xy, cube_size + 0.001], size=(cube_size,) * 3, rgba=COLORS[col])
        b.quat = _quat_from_axis_angle([0, 0, 1], rng.uniform(-1, 1))
        objects.append(ObjectDecl(f"distractor{k}", f"{col} cube", "object", (cube_size,) * 3))
    model = compile_scene(scene)
    robots = []
    for prefix, meta, pos, yaw in mounted:
        rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
        manip = next(a.id for a in rs.assemblies if "grasp" in a.capabilities)
        robots.append(MountedRobot(prefix, meta, rs, pos, yaw, {"gripper": manip}))
    return Scenario("pick_place", task or _task("pick_place"), scene, model, robots, objects, seed,
                    meta=dict(cube_color=cube_color, n_distractors=n_distractors))


def _build_pick_place_obj(robot, seed, *, n_distractors, cube_color, distractor_colors, task, cube_size, base,
                          object_spec) -> Scenario:
    """build_pick_place with a task-object spec: identical RNG draws and placement as the default builder."""
    rng = np.random.default_rng(seed)
    scene = workspace_spec(f"pick_place_{seed}")
    mounted = mount_robots(scene, [robot], [base])
    placed = []
    cube_xy = _free_xy(rng, [0.32, -0.18], [0.50, 0.18], placed, 0.1)
    placed.append(cube_xy)
    tgt_xy = _free_xy(rng, [0.30, -0.25], [0.52, 0.25], placed, 0.14)
    placed.append(tgt_xy)
    shape = object_spec["shape"]
    half = object_spec.get("size", cube_size)
    ext = object_half_extents(shape, half)
    b = scene.worldbody.add_body(name="cube", pos=[*cube_xy, ext[2] + 0.001])
    b.add_freejoint(name="cube_free")
    gkw = dict(name="cube_geom", rgba=list(COLORS[cube_color]), density=500.0, friction=[1.2, 0.02, 0.002], condim=4)
    if shape == "cylinder":
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ext[0], ext[2], 0], **gkw)
    else:
        g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ext), **gkw)
    if object_spec.get("mass") is not None:
        g.mass = object_spec["mass"]
    b.quat = _quat_from_axis_angle([0, 0, 1], rng.uniform(-math.pi / 4, math.pi / 4))
    add_target_zone(scene, "target_zone", [*tgt_xy, 0.0005], radius=0.05)
    noun = shape.replace("_", " ") if object_spec.get("descriptor_shape") else "cube"
    objects = [ObjectDecl("cube", f"{cube_color} {noun}", "object", ext, task_entity="cube"),
               ObjectDecl("target_zone", "green target zone", "feature", radius=0.05, task_entity="target")]
    for k in range(n_distractors):
        xy = _free_xy(rng, [0.28, -0.3], [0.55, 0.3], placed, 0.09)
        placed.append(xy)
        col = distractor_colors[k % len(distractor_colors)]
        bd = add_box_object(scene, f"distractor{k}", [*xy, cube_size + 0.001], size=(cube_size,) * 3, rgba=COLORS[col])
        bd.quat = _quat_from_axis_angle([0, 0, 1], rng.uniform(-1, 1))
        objects.append(ObjectDecl(f"distractor{k}", f"{col} cube", "object", (cube_size,) * 3))
    fr = object_spec.get("friction")

    def post(sc):                     # after the grasp contact model, which sets object friction under grasp_v2+
        if fr is not None:
            gg = next(x for x in sc.geoms if x.name == "cube_geom")
            gg.friction = [fr, gg.friction[1], gg.friction[2]]
    model = compile_scene(scene, post_contact=post)
    robots = []
    for prefix, meta, pos, yaw in mounted:
        rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
        manip = next(a.id for a in rs.assemblies if "grasp" in a.capabilities)
        robots.append(MountedRobot(prefix, meta, rs, pos, yaw, {"gripper": manip}))
    tk = task or _task("pick_place")
    if object_spec.get("descriptor_shape"):
        tk = copy.deepcopy(tk)
        for e in tk["entity_declarations"]:
            if e["id"] == "cube":
                e["descriptor"] = f"{cube_color} {noun}"
    rec = dict(object_spec, version=OBJECT_SPEC_VERSION, half_extents=[float(x) for x in ext],
               mass_kg=float(model.body_mass[model.body("cube").id]),
               friction=float(model.geom_friction[model.geom("cube_geom").id][0]))
    return Scenario("pick_place", tk, scene, model, robots, objects, seed,
                    meta=dict(cube_color=cube_color, n_distractors=n_distractors, object_spec=rec))


def build_reach(robot, seed: int, task: dict | None = None, base=((0.0, 0.0, 0.0), 0.0)) -> Scenario:
    rng = np.random.default_rng(seed)
    scene = workspace_spec(f"reach_{seed}")
    mounted = mount_robots(scene, [robot], [base])
    goal = [rng.uniform(0.28, 0.5), rng.uniform(-0.25, 0.25), rng.uniform(0.08, 0.35)]
    b = scene.worldbody.add_body(name="goal_marker", pos=goal, mocap=True)
    b.add_geom(name="goal_marker_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
               rgba=[0.1, 0.3, 0.95, 0.6], contype=0, conaffinity=0)
    model = compile_scene(scene)
    robots = []
    for prefix, meta, pos, yaw in mounted:
        rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
        manip = next(a.id for a in rs.assemblies if a.kind in ("gripper", "hand")) if any(
            a.kind in ("gripper", "hand") for a in rs.assemblies) else rs.assemblies[0].id
        robots.append(MountedRobot(prefix, meta, rs, pos, yaw, {"gripper": manip}))
    objects = [ObjectDecl("goal_marker", "blue goal marker", "feature", radius=0.02, task_entity="goal")]
    return Scenario("reach_pose", task or _task("reach_pose"), scene, model, robots, objects, seed)


PAIRED_COLORS = ("red", "blue", "yellow", "purple", "orange", "cyan")


def build_pick_place_paired(robot, seed: int, *, patient: int = 0, n_objects: int = 3, cube_size: float = 0.022,
                            base=((0.0, 0.0, 0.0), 0.0)) -> Scenario:
    """Paired binding scenes (track `binding`, correction 2026-09-25 item 2).

    The PHYSICAL scene depends only on `seed`: n_objects same-size cubes with distinct colors, poses and a green
    target zone, all placed in the graspable region. `patient` (0..n_objects-1) selects which physical cube the task
    assigns as patient; the other cubes are clutter. So (seed, patient=i) and (seed, patient=j) have IDENTICAL initial
    scenes and differ only in the supplied task binding (entity descriptor), each with its own valid demonstration.
    The public slot order (tracker slots) is a seeded permutation over PHYSICAL objects incl. the target zone,
    identical within a pair and independent of the role, so neither the patient nor the destination has a fixed slot.
    Internally the assigned cube is named `cube` (privileged sim name; teachers/success checks use it); names are
    not public. The task's `cube` entity descriptor is set to the assigned cube's color descriptor."""
    if not 0 <= patient < n_objects:
        raise ValueError("patient index out of range")
    rng = np.random.default_rng(seed)
    scene = workspace_spec(f"pick_place_paired_{seed}")
    mounted = mount_robots(scene, [robot], [base])
    placed = []
    xys = []
    for _ in range(n_objects):
        xy = _free_xy(rng, [0.30, -0.22], [0.52, 0.22], placed, 0.09)
        placed.append(xy)
        xys.append(xy)
    tgt_xy = _free_xy(rng, [0.28, -0.28], [0.54, 0.28], placed, 0.12)
    colors = [PAIRED_COLORS[i] for i in rng.permutation(len(PAIRED_COLORS))[:n_objects]]
    yaws = rng.uniform(-math.pi / 4, math.pi / 4, n_objects)
    order = rng.permutation(n_objects + 1)            # public slot order over physical items (index n = target zone)
    names, k = [], 0
    for i in range(n_objects):
        if i == patient:
            names.append("cube")
        else:
            names.append(f"distractor{k}")
            k += 1
    for i in range(n_objects):                        # bodies added in PHYSICAL order: identical model up to names
        b = add_box_object(scene, names[i], [*xys[i], cube_size + 0.001], size=(cube_size,) * 3, rgba=COLORS[colors[i]])
        b.quat = _quat_from_axis_angle([0, 0, 1], yaws[i])
    add_target_zone(scene, "target_zone", [*tgt_xy, 0.0005], radius=0.05)
    decl = [ObjectDecl(names[i], f"{colors[i]} cube", "object", (cube_size,) * 3,
                       task_entity="cube" if i == patient else None) for i in range(n_objects)]
    decl.append(ObjectDecl("target_zone", "green target zone", "feature", radius=0.05, task_entity="target"))
    objects = [decl[i] for i in order]
    model = compile_scene(scene)
    robots = []
    for prefix, meta, pos, yaw in mounted:
        rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
        manip = next(a.id for a in rs.assemblies if "grasp" in a.capabilities)
        robots.append(MountedRobot(prefix, meta, rs, pos, yaw, {"gripper": manip}))
    task = copy.deepcopy(_task("pick_place"))
    for e in task["entity_declarations"]:
        if e["id"] == "cube":
            e["descriptor"] = f"{colors[patient]} cube"
    return Scenario("pick_place", task, scene, model, robots, objects, seed,
                    meta=dict(paired=True, pair_seed=seed, patient=patient, n_objects=n_objects, colors=colors,
                              cube_color=colors[patient], n_distractors=n_objects - 1,
                              slot_order=[int(x) for x in order],
                              patient_slot=int(list(order).index(patient)),
                              target_slot=int(list(order).index(n_objects))))


def _task(name):
    return load_task(name)


BUILDERS = {"pick_place": build_pick_place, "reach_pose": build_reach, "pick_place_paired": build_pick_place_paired}
