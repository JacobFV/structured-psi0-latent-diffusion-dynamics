"""Dual-arm scenarios: `support_insert` and `handover`.

Two manipulators face the table. They can be two separately mounted arm assemblies (each a
robot with its own controller contract) or ONE dual-arm body (e.g. menagerie ALOHA) whose
two gripper assemblies are bound to the task roles `left` / `right`. Task-role bindings are
public (the user states which assembly plays which role); the entity -> sim-body map stays
privileged (ObjectDecl.task_entity).

support_insert geometry (declared, public task geometry; see research/reports/dual_arm_tasks.md):
  peg      cylinder radius 12 mm (diameter 24 mm), length 90 mm, standing upright
  hole     square 28 x 28 mm, depth 39 mm, in a 220 x 220 x 45 mm fixture block
  clearance hole width - peg diameter = 4 mm (2 mm radial along the hole axes; more on diagonals)
Randomized per seed: fixture pose (x, y, yaw), hole offset inside the fixture, support pad
offset, peg position. The hole position therefore differs every seed and the downstream
align/insert commands need the `locate` output.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from rrp.morphology.compiler import compile_robot_spec
from rrp.morphology.generators import workspace_spec, add_target_zone, _quat_from_axis_angle
from .scenario import ObjectDecl, MountedRobot, Scenario, mount_robots, load_task, COLORS

PEG_RADIUS = 0.012
PEG_HALF_LEN = 0.045
HOLE_HALF_W = 0.014            # 28 mm square hole -> 4 mm diametral clearance
FIX_HALF = 0.11
FIX_HEIGHT = 0.045
FIX_BASE = 0.006
CLEARANCE_M = 2 * HOLE_HALF_W - 2 * PEG_RADIUS

BAR_HALF = (0.09, 0.011, 0.02)  # handover bar, long axis = local x
BAR_GRASP_D = 0.065             # grasp points at +-6.5 cm from the bar centre

# two separately mounted arms: left at +y, right at -y, both facing +x (the table)
DUAL_MOUNTS = (((0.0, 0.28, 0.0), 0.0), ((0.0, -0.28, 0.0), 0.0))
# a single dual-arm body is mounted at the table centre, rotated so its arms sit at +-y
DUAL_BODY_MOUNT = ((0.35, 0.0, 0.0), -math.pi / 2)


def _rot2(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def add_fixture_with_hole(scene: mujoco.MjSpec, pos_xy, yaw, hole_off, rgba=(0.55, 0.6, 0.7, 1),
                          density=600.0, friction=0.6):
    """Free-floating fixture block with a blind square hole. A geomless child body `hole`
    marks the hole-top centre (the feature a detector reports)."""
    body = scene.worldbody.add_body(name="fixture", pos=[pos_xy[0], pos_xy[1], 0.0005],
                                    quat=_quat_from_axis_angle([0, 0, 1], yaw))
    body.add_freejoint(name="fixture_free")
    L, H, a = FIX_HALF, FIX_HEIGHT, HOLE_HALF_W
    hx, hy = hole_off
    kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=list(rgba), density=density, friction=[friction, 0.02, 0.002],
              condim=4)
    body.add_geom(name="fixture_base", size=[L, L, FIX_BASE / 2], pos=[0, 0, FIX_BASE / 2], **kw)
    zc, hz = FIX_BASE + (H - FIX_BASE) / 2, (H - FIX_BASE) / 2
    blocks = {  # (xmin, xmax, ymin, ymax)
        "fixture_wx0": (-L, hx - a, -L, L), "fixture_wx1": (hx + a, L, -L, L),
        "fixture_wy0": (hx - a, hx + a, -L, hy - a), "fixture_wy1": (hx - a, hx + a, hy + a, L)}
    for n, (x0, x1, y0, y1) in blocks.items():
        body.add_geom(name=n, size=[(x1 - x0) / 2, (y1 - y0) / 2, hz], pos=[(x0 + x1) / 2, (y0 + y1) / 2, zc], **kw)
    hole = body.add_body(name="hole", pos=[hx, hy, H])
    hole.add_site(name="hole_site", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[0.9, 0.1, 0.1, 1])
    return body


def add_peg(scene, pos_xy, rgba=COLORS["orange"], density=700.0, friction=1.5):
    b = scene.worldbody.add_body(name="peg", pos=[pos_xy[0], pos_xy[1], PEG_HALF_LEN + 0.001])
    b.add_freejoint(name="peg_free")
    b.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PEG_RADIUS, PEG_HALF_LEN, 0],
               rgba=list(rgba), density=density, friction=[friction, 0.02, 0.002], condim=4)
    return b


def add_bar(scene, pos_xy, yaw, rgba=COLORS["cyan"], density=500.0, friction=1.5):
    b = scene.worldbody.add_body(name="bar", pos=[pos_xy[0], pos_xy[1], BAR_HALF[2] + 0.001],
                                 quat=_quat_from_axis_angle([0, 0, 1], yaw))
    b.add_freejoint(name="bar_free")
    b.add_geom(name="bar_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BAR_HALF), rgba=list(rgba),
               density=density, friction=[friction, 0.02, 0.002], condim=4)
    return b


def _gripper_assemblies(rs):
    return [a.id for a in rs.assemblies if a.kind in ("gripper", "hand") and "grasp" in a.capabilities]


def _mount(scene, robots):
    """robots: [left_arm, right_arm] (two bodies) or [dual_body] (one body, two grippers)."""
    if len(robots) == 2:
        return mount_robots(scene, robots, list(DUAL_MOUNTS))
    if len(robots) == 1:
        return mount_robots(scene, robots, [DUAL_BODY_MOUNT])
    raise ValueError("dual-arm scenario needs two single-arm robots or one dual-arm body")


def _mounted(model, mounted) -> list[MountedRobot]:
    out = []
    if len(mounted) == 2:
        for (prefix, meta, pos, yaw), role in zip(mounted, ("left", "right")):
            rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
            g = _gripper_assemblies(rs)
            if not g:
                raise ValueError(f"{meta.get('name')} has no grasping assembly")
            out.append(MountedRobot(prefix, meta, rs, pos, yaw, {role: g[0]}))
    else:
        prefix, meta, pos, yaw = mounted[0]
        rs = compile_robot_spec(model, meta, prefix=prefix, name=meta.get("name"))
        roles = {m["role_hint"]: m["assembly"] for m in meta["manipulators"]}
        out.append(MountedRobot(prefix, meta, rs, pos, yaw, {"left": roles["left"], "right": roles["right"]}))
    return out


def build_support_insert(robots: list, seed: int, *, task: dict | None = None) -> Scenario:
    rng = np.random.default_rng([seed, 17])
    scene = workspace_spec(f"support_insert_{seed}")
    mounted = _mount(scene, robots)
    fx = np.array([rng.uniform(0.40, 0.46), rng.uniform(-0.01, 0.05)])
    fyaw = float(rng.uniform(-0.3, 0.3))
    hole_off = (float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.08, -0.05)))
    support_off = (float(rng.uniform(-0.03, 0.03)), float(rng.uniform(0.075, 0.09)))
    add_fixture_with_hole(scene, fx, fyaw, hole_off)
    peg_xy = np.array([rng.uniform(0.30, 0.44), rng.uniform(-0.33, -0.22)])
    add_peg(scene, peg_xy)
    objects = [  # detector slot order; the hole is a feature on the workpiece
        ObjectDecl("hole", "hole feature", "feature", radius=HOLE_HALF_W, task_entity="hole"),
        ObjectDecl("fixture", "workpiece", "object", (FIX_HALF, FIX_HALF, FIX_HEIGHT / 2), task_entity="fixture"),
        ObjectDecl("peg", "peg", "object", (PEG_RADIUS, PEG_RADIUS, PEG_HALF_LEN), task_entity="peg"),
    ]
    model = scene.compile()
    rob = _mounted(model, mounted)
    hole_world = np.r_[fx + _rot2(fyaw, hole_off), FIX_HEIGHT]
    sup_world = np.r_[fx + _rot2(fyaw, support_off), FIX_HEIGHT]
    return Scenario("support_insert", task or load_task("support_and_insert"), scene, model, rob, objects, seed,
                    meta=dict(
                        # PUBLIC declared task geometry (CAD-like), usable by estimators and policies
                        declared_geometry=dict(peg=dict(radius=PEG_RADIUS, half_length=PEG_HALF_LEN),
                                               hole=dict(half_width=HOLE_HALF_W, depth=FIX_HEIGHT - FIX_BASE,
                                                         axis="fixture_up"),
                                               fixture=dict(half_extent=FIX_HALF, height=FIX_HEIGHT),
                                               clearance_m=CLEARANCE_M),
                        # PRIVILEGED layout (teacher/evaluator only)
                        privileged_layout=dict(fixture_xy=fx.tolist(), fixture_yaw=fyaw, hole_offset=hole_off,
                                               support_offset=support_off, hole_world=hole_world.tolist(),
                                               support_world=sup_world.tolist(), peg_xy=peg_xy.tolist())))


def build_handover(robots: list, seed: int, *, task: dict | None = None) -> Scenario:
    rng = np.random.default_rng([seed, 23])
    scene = workspace_spec(f"handover_{seed}")
    mounted = _mount(scene, robots)
    bar_xy = np.array([rng.uniform(0.34, 0.44), rng.uniform(0.12, 0.20)])
    bar_yaw = float(math.pi / 2 + rng.uniform(-0.35, 0.35))       # long axis roughly along y
    add_bar(scene, bar_xy, bar_yaw)
    tgt_xy = np.array([rng.uniform(0.36, 0.48), rng.uniform(-0.20, -0.10)])
    add_target_zone(scene, "target_zone", [*tgt_xy, 0.0005], radius=0.05)
    objects = [ObjectDecl("bar", "cyan bar", "object", BAR_HALF, task_entity="bar"),
               ObjectDecl("target_zone", "green target zone", "feature", radius=0.05, task_entity="target")]
    model = scene.compile()
    rob = _mounted(model, mounted)
    return Scenario("handover", task or load_task("handover"), scene, model, rob, objects, seed,
                    meta=dict(declared_geometry=dict(bar=dict(half_extents=list(BAR_HALF), grasp_offset=BAR_GRASP_D)),
                              privileged_layout=dict(bar_xy=bar_xy.tolist(), bar_yaw=bar_yaw, target_xy=tgt_xy.tolist())))


def assign_task(arm: str) -> dict:
    """Manipulator-assignment task graph: take(ARM, bar) -> place(ARM, bar, target), ARM in {left, right}.
    Built from the handover graph's `take` and `place` events with the actor rebound; the other manipulator is
    declared but bound to no event. Both variants share every scene quantity (paired tasks)."""
    import copy
    import json as _json
    if arm not in ("left", "right"):
        raise ValueError(arm)
    h = load_task("handover")
    ev = {e["id"]: e for e in h["events"]}
    rebind = lambda e, frm: _json.loads(_json.dumps(e).replace(f'"id": "{frm}"', f'"id": "{arm}"'))
    take = rebind(copy.deepcopy(ev["take"]), "left")
    place = rebind(copy.deepcopy(ev["place"]), "right")
    place["requires_completed"] = ["take"]
    decl = [dict(id="left", type="manipulator", descriptor="left manipulator assembly"),
            dict(id="right", type="manipulator", descriptor="right manipulator assembly"),
            dict(id="bar", type="object", descriptor="cyan bar to move"),
            dict(id="target", type="feature", descriptor="green target zone")]
    return dict(schema_version=h["schema_version"], task_id=f"assign_pick_place_{arm}", graph_version=0,
                entity_declarations=decl, events=[take, place], success_events=["place"])


def build_assign(robots: list, seed: int, *, arm: str = "left", task: dict | None = None) -> Scenario:
    """Paired manipulator-assignment scene: ONE layout per seed (bar and target zone in the shared central
    workspace, reachable by both arms); the task graph assigns the acting arm. The layout RNG does not depend on
    `arm`, so the two variants of a seed start from an identical initial scene."""
    rng = np.random.default_rng([seed, 29])
    scene = workspace_spec(f"assign_{seed}")
    mounted = _mount(scene, robots)
    bar_xy = np.array([rng.uniform(0.42, 0.48), rng.uniform(-0.04, 0.04)])
    bar_yaw = float(math.pi / 2 + rng.uniform(-0.35, 0.35))       # long axis roughly along y
    add_bar(scene, bar_xy, bar_yaw)
    tgt_xy = np.array([rng.uniform(0.30, 0.34), rng.uniform(-0.05, 0.05)])
    add_target_zone(scene, "target_zone", [*tgt_xy, 0.0005], radius=0.05)
    objects = [ObjectDecl("bar", "cyan bar", "object", BAR_HALF, task_entity="bar"),
               ObjectDecl("target_zone", "green target zone", "feature", radius=0.05, task_entity="target")]
    model = scene.compile()
    rob = _mounted(model, mounted)
    return Scenario(f"assign_{arm}", task or assign_task(arm), scene, model, rob, objects, seed,
                    meta=dict(declared_geometry=dict(bar=dict(half_extents=list(BAR_HALF), grasp_offset=BAR_GRASP_D)),
                              pair_family="assign_pick_place", assigned_role=arm,
                              privileged_layout=dict(bar_xy=bar_xy.tolist(), bar_yaw=bar_yaw, target_xy=tgt_xy.tolist())))


DUAL_BUILDERS = {"support_insert": build_support_insert, "handover": build_handover,
                 "assign_left": lambda robots, seed, **kw: build_assign(robots, seed, arm="left", **kw),
                 "assign_right": lambda robots, seed, **kw: build_assign(robots, seed, arm="right", **kw)}
