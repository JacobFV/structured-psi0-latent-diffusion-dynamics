"""Procedural arm generator v2 (armdiv, D-137): many distinct serial-arm kinematics for training-arm diversity.

A v2 arm is a pure function of its generator seed (`sample_arm_v2(seed)` -> ArmParamsV2 -> `procedural_arm_v2`):
  * DoF 5-8; joint 0 is always a vertical base yaw (the teacher's IK seeding rotates joint 0 by the target azimuth),
    joint 1 a shoulder pitch, the last joint a tool roll and the one before it a wrist pitch; the middle joints mix
    pitch (y), second pitch (x) and roll (z) axes;
  * fixed link twists about the link direction (so consecutive pitch axes need not be parallel), lateral shoulder /
    elbow / wrist offsets (UR-like), upper-arm and forearm lengths with a total-reach constraint, a pedestal height,
    joint ranges, link radius/density and servo gains;
  * the home pose is solved by IK to a fixed tool-down ready pose (the v1 axis-pattern heuristic does not generalize).
Labelled `synthetic`, lineage `procedural_arm_family/v2`. The v1 generator (`rrp.bodies.generators.procedural_arm`)
and every v1 key are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

from rrp.bodies.generators import Module, _base_spec, _quat_from_axis_angle, AXES

LINEAGE_V2 = "procedural_arm_family/v2"
GENERATOR_VERSION = "arm_gen_v2.0"
READY_FLANGE_POS = (0.30, 0.0, 0.34)      # preferred home: flange here, tool (flange +z) pointing down
READY_FLANGE_TARGETS = ((0.30, 0.0, 0.34), (0.40, 0.0, 0.30), (0.36, 0.0, 0.42), (0.48, 0.0, 0.26))  # in order


class ArmGenerationError(ValueError):
    code = "arm_generation_failed"


@dataclass
class ArmParamsV2:
    name: str
    seed: int
    axes: tuple                       # per joint "x" | "y" | "z" (local)
    lengths: tuple                    # link length along local z (joint i -> joint i+1; last = flange)
    twists: tuple                     # fixed rotation of link i about its parent's link direction (rad)
    offsets: tuple                    # lateral (x, y) offset of joint i's body in its parent frame (m)
    ranges: tuple                     # per joint (lo, hi) rad
    pedestal: float = 0.0
    radius: float = 0.035
    density: float = 1200.0
    kp: float = 400.0
    damping: float = 8.0
    effort: float = 80.0
    lineage: str = LINEAGE_V2
    rgba: tuple = (0.62, 0.66, 0.72, 1)
    home: tuple | None = None
    meta: dict = field(default_factory=dict)


def _choice(rng, items, p=None):
    return items[int(rng.choice(len(items), p=p))]


def sample_arm_v2(seed: int, name: str | None = None) -> ArmParamsV2:
    """Deterministic draw of one v2 arm from its generator seed ([seed, 137] RNG stream)."""
    rng = np.random.default_rng([int(seed), 137])
    dof = int(_choice(rng, [5, 6, 7, 8], p=[0.2, 0.35, 0.3, 0.15]))
    axes = ["z", "y"]
    n_mid = dof - 4                                     # joints between shoulder pitch and (wrist pitch, tool roll)
    mid = []
    for k in range(n_mid):
        for _ in range(20):
            a = _choice(rng, ["y", "z", "x"], p=[0.5, 0.35, 0.15])
            if mid and a == "z" and mid[-1] == "z":
                continue
            mid.append(a)
            break
    if n_mid and not any(a in ("y", "x") for a in mid[:2]):
        mid[int(rng.integers(min(2, n_mid)))] = "y"   # an elbow must bend
    axes += mid
    axes += [_choice(rng, ["y", "x"], p=[0.75, 0.25]), "z"]
    if n_mid == 0:                                      # 4 joints so far is impossible (dof >= 5); keep for safety
        axes.insert(2, "y")
    axes = axes[:dof]
    # segment lengths: base column, upper arm, forearm, wrist links
    base_col = float(rng.uniform(0.08, 0.20))
    upper = float(rng.uniform(0.22, 0.42))
    fore = float(rng.uniform(0.18, 0.40))
    tot = upper + fore
    if tot > 0.80:
        upper, fore = upper * 0.80 / tot, fore * 0.80 / tot
    elif tot < 0.45:
        upper, fore = upper * 0.45 / tot, fore * 0.45 / tot
    # elbow = the first bending joint after the shoulder
    elbow = next(i for i in range(2, dof) if axes[i] in ("y", "x"))
    wrist_pitch = dof - 2
    lengths = [0.0] * dof
    lengths[0] = base_col
    up_idx = list(range(1, elbow))                        # links from shoulder to elbow share the upper arm
    fo_idx = list(range(elbow, wrist_pitch)) or [elbow]   # links from elbow to wrist pitch share the forearm
    for idx, total in ((up_idx, upper), (fo_idx, fore)):
        w = rng.uniform(0.6, 1.4, size=len(idx))
        w = w / w.sum()
        for i, wi in zip(idx, w):
            lengths[i] += float(total * wi)
    for i in range(dof):
        if lengths[i] == 0.0:
            lengths[i] = float(rng.uniform(0.05, 0.12))
    lengths[-1] = float(rng.uniform(0.04, 0.08))
    twist_choices = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
    twists = [0.0] * dof
    for i in range(2, dof):
        if rng.random() < 0.35:
            twists[i] = float(_choice(rng, twist_choices) * (1 if rng.random() < 0.5 else -1))
    offsets = [(0.0, 0.0)] * dof
    if rng.random() < 0.3:                               # shoulder lateral offset
        offsets[1] = (0.0, float(rng.uniform(0.05, 0.14) * (1 if rng.random() < 0.5 else -1)))
    if rng.random() < 0.25:                              # elbow offset (UR-like, opposite side)
        oy = offsets[1][1]
        offsets[elbow] = (0.0, float(-np.sign(oy or 1.0) * rng.uniform(0.04, 0.12)))
    if rng.random() < 0.3:                               # offset wrist
        offsets[wrist_pitch] = tuple(float(v) for v in (rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)))
    shrink = rng.uniform(0.8, 1.0, size=dof)
    base_rng = {"z": 2.9, "y": 2.2, "x": 2.2}
    ranges = tuple((-float(base_rng[a] * s), float(base_rng[a] * s)) for a, s in zip(axes, shrink))
    return ArmParamsV2(name=name or f"pa2s{seed}", seed=int(seed), axes=tuple(axes), lengths=tuple(lengths),
                       twists=tuple(twists), offsets=tuple(tuple(o) for o in offsets), ranges=ranges,
                       pedestal=float(rng.uniform(0.0, 0.15)), radius=float(rng.uniform(0.028, 0.045)),
                       density=float(rng.uniform(900, 1600)), kp=float(rng.uniform(300, 600)),
                       damping=float(rng.uniform(5, 12)), effort=float(rng.uniform(60, 120)))


def _build_spec(p: ArmParamsV2) -> tuple[mujoco.MjSpec, list[str]]:
    s = _base_spec(p.name)
    base = s.worldbody.add_body(name="base", pos=[0, 0, 0])
    h = 0.03 + p.pedestal / 2
    base.add_geom(name="base_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, h, 0], pos=[0, 0, h],
                  density=p.density * 2, rgba=[0.3, 0.3, 0.35, 1])
    parent = base
    joints = []
    for i, (ax, L) in enumerate(zip(p.axes, p.lengths)):
        ox, oy = p.offsets[i]
        z = 0.06 + p.pedestal if i == 0 else p.lengths[i - 1]
        b = parent.add_body(name=f"link{i}", pos=[ox, oy, z], gravcomp=1.0,
                            quat=_quat_from_axis_angle([0, 0, 1], p.twists[i]))
        lo, hi = p.ranges[i]
        j = b.add_joint(name=f"j{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=AXES[ax], range=[lo, hi],
                        limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=p.damping, armature=0.02)
        r = p.radius * (1 - 0.06 * i)
        b.add_geom(name=f"link{i}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, L],
                   size=[r, 0, 0], density=p.density, rgba=list(p.rgba), contype=1, conaffinity=1)
        if abs(ox) + abs(oy) > 1e-6:                      # visible/physical offset strut in the child's frame
            q = np.array(_quat_from_axis_angle([0, 0, 1], -p.twists[i]))
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            back = R.reshape(3, 3) @ np.array([-ox, -oy, 0.0])
            b.add_geom(name=f"link{i}_strut", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[float(back[0]), float(back[1]), 0, 0, 0, 0], size=[r * 0.9, 0, 0], density=p.density,
                       rgba=list(p.rgba), contype=1, conaffinity=1)
        joints.append(j.name)
        parent = b
    s.add_exclude(bodyname1="base", bodyname2="link0")
    flange = parent.add_body(name="flange", pos=[0, 0, p.lengths[-1]], gravcomp=1.0)
    flange.add_geom(name="flange_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.008, 0],
                    density=p.density, rgba=[0.2, 0.2, 0.2, 1])
    flange.add_site(name="port_wrist", pos=[0, 0, 0.008])
    flange.add_site(name="flange_tcp", pos=[0, 0, 0.008])
    for i, jn in enumerate(joints):
        lo, hi = p.ranges[i]
        s.add_actuator(name=f"act_{jn}", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                       gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[p.kp] + [0] * 9,
                       biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -p.kp, -p.damping * 0.5] + [0] * 7,
                       ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=[lo, hi],
                       forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-p.effort, p.effort])
        s.add_sensor(name=f"jpos_{jn}", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                     objname=jn)
    return s, joints


def solve_home(p: ArmParamsV2, *, restarts: int = 24, pos_tol: float = 3e-3, rot_tol: float = 0.05) -> list[float]:
    """IK to the ready pose (flange at READY_FLANGE_POS, flange +z down, any of the 4 flange yaws 0/pi/+-pi/2), links
    above the table. Among the solutions within tolerance, the one with the largest joint-limit margin wins (deterministic)."""
    from rrp.bodies.ik import IKSolver
    spec, joints = _build_spec(p)
    m = spec.compile()
    ik = IKSolver(m, "flange_tcp", joints)
    rng = np.random.default_rng([p.seed, 138])
    body_ids = [m.body(f"link{i}").id for i in range(len(joints))]
    starts = [np.zeros(len(joints))]
    for _ in range(restarts):
        starts.append(rng.uniform(ik.lo * 0.7, ik.hi * 0.7))
    for target in map(np.array, READY_FLANGE_TARGETS):
        q = _home_search(ik, m, starts, target, body_ids, pos_tol, rot_tol)
        if q is not None:
            return [float(x) for x in q]
    raise ArmGenerationError(f"{p.name}: no tool-down home within tolerance")


def _home_search(ik, m, starts, target, body_ids, pos_tol, rot_tol):
    from rrp.bodies.ik import down_rotation, rot_error
    best, best_key = None, None
    for q0, yaw in ((q0, yaw) for q0 in starts for yaw in (0.0, math.pi, math.pi / 2, -math.pi / 2)):
        R = down_rotation(yaw)
        q, e = ik._solve(m.qpos0.copy(), q0, target, R, 300, 1e-4)
        pos, Rc = ik.fk(m.qpos0.copy(), q)
        er = float(np.linalg.norm(rot_error(Rc, R)))
        if e > pos_tol or er > rot_tol:
            continue
        zmin = float(min(ik.data.xpos[b][2] for b in body_ids[1:]))
        if zmin < 0.05:
            continue
        k = (round(ik.margin(q), 3), zmin)
        if best_key is None or k > best_key:
            best, best_key = q, k
    return best


def procedural_arm_v2(p: ArmParamsV2) -> Module:
    """Serial arm with a wrist attachment port (site 'port_wrist'); same metadata contract as the v1 generator."""
    if p.home is None:
        p.home = tuple(solve_home(p))
    s, joints = _build_spec(p)
    params = asdict(p)
    params.pop("meta", None)
    params["generator"] = GENERATOR_VERSION
    meta = dict(name=p.name, family="arm", synthetic=True, lineage=[p.lineage, f"{p.lineage}/{p.name}"],
                assemblies=[dict(id="arm", kind="arm", root_body="link0", frame=dict(site="flange_tcp"),
                                 capabilities=["push"])],
                ports=[dict(id="wrist", site="port_wrist", host_body="flange", max_payload_kg=3.0,
                            max_module_extent_m=0.25, interface="wrist_generic",
                            allowed_module_kinds=["gripper", "hand", "tool"])],
                controller=dict(kind="joint_targets", groups=[dict(name="arm", actuators=[f"act_{j}" for j in joints],
                                                                   semantic="joint_position", units="rad")]),
                params=params, home=list(p.home),
                home_tcp_azimuth=float(math.atan2(READY_FLANGE_POS[1], READY_FLANGE_POS[0])),
                reach_m=float(sum(p.lengths[1:]) + 0.1))
    return Module(s, meta)


def arm_v2_from_seed(seed: int) -> Module:
    return procedural_arm_v2(sample_arm_v2(seed))
