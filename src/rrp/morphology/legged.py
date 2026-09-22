"""Legged / humanoid breadth: procedural multi-leg generators + menagerie legged importers.

Every body returned here is a `Module` (MjSpec + metadata) with a common `meta["legged"]`
block that the locomotion stack (rrp.sim.legged, rrp.control.legged_env/tracker) consumes:

    root_body, imu_site, foot_bodies, foot_sites, touch_sensors, policy_actuators (ordered),
    held_actuators, default_pose {joint: rad}, nominal_height, min_height, kp/kd per actuator,
    action_scale, command_ranges (vx, vy, wz), gait_period.

Procedural bodies are labelled `synthetic=True` with a `procedural_legged/v1` lineage.
Menagerie bodies keep their pinned source lineage (`menagerie/<family>`,
`menagerie/<dir>@<sha12>`). Menagerie motor (torque) actuators are converted to bounded joint
PD position servos with the ORIGINAL torque limits (force range = ctrlrange * gear); this is a
declared actuator adapter recorded in `meta["actuator_adapter"]`, not a hidden change.

Deployable sensors added to every body: an IMU site at the root-body origin (identity
orientation) with framequat / gyro / accelerometer, and per-foot touch sensors. Because the
IMU frame equals the root frame, the free-joint quaternion and local angular velocity used
in training are exactly the IMU framequat/gyro readings (tested in tests/unit/test_legged.py).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .generators import Module, _base_spec, _quat_from_axis_angle
from .importers import MENAGERIE, MENAGERIE_SHA

G = mujoco.mjtGeom
LIM = mujoco.mjtLimited.mjLIMITED_TRUE


# ------------------------------------------------------------------ common sensor / actuator helpers
def add_imu(spec: mujoco.MjSpec, root_body_name: str, prefix: str = "imu") -> dict:
    body = spec.body(root_body_name)
    body.add_site(name=f"{prefix}_site", pos=[0, 0, 0], quat=[1, 0, 0, 0], size=[0.01, 0, 0])
    S, O = mujoco.mjtSensor, mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name=f"{prefix}_quat", type=S.mjSENS_FRAMEQUAT, objtype=O, objname=f"{prefix}_site")
    spec.add_sensor(name=f"{prefix}_gyro", type=S.mjSENS_GYRO, objtype=O, objname=f"{prefix}_site")
    spec.add_sensor(name=f"{prefix}_acc", type=S.mjSENS_ACCELEROMETER, objtype=O, objname=f"{prefix}_site")
    return dict(site=f"{prefix}_site", quat=f"{prefix}_quat", gyro=f"{prefix}_gyro", acc=f"{prefix}_acc")


def pd_servo(spec: mujoco.MjSpec, name: str, joint: str, kp: float, kd: float, effort: float, ctrlrange):
    """Bounded joint PD position servo: tau = clip(kp (u - q) - kd qdot, +-effort)."""
    return spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                             gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                             biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -kd] + [0] * 7,
                             ctrllimited=LIM, ctrlrange=list(ctrlrange),
                             forcelimited=LIM, forcerange=[-effort, effort])


# ------------------------------------------------------------------ procedural multi-leg generator
@dataclass
class LeggedParams:
    name: str = "hexapod6"
    n_legs: int = 6
    layout: str = "sprawl"                 # sprawl (insect: yaw-pitch-pitch) | mammal (roll-pitch-pitch)
    body_half: tuple = (0.15, 0.09, 0.03)
    coxa: float = 0.05
    femur: float = 0.10
    tibia: float = 0.14
    leg_radius: float = 0.012
    foot_radius: float = 0.016
    density: float = 500.0
    body_density: float = 450.0
    splay: float = 0.45                    # rad, front/back leg fan-out (sprawl)
    kp: float = 10.0
    kd: float = 0.3
    effort: float = 4.0
    stance: tuple = (0.0, 0.35, -1.6)      # default joint angles per leg
    friction: float = 1.0
    lineage: str = "procedural_legged/v1"
    seed_note: str = ""


def procedural_legged(p: LeggedParams) -> Module:
    """Real floating-base n-leg robot (3 joints/leg), positive inertias from densities."""
    if p.n_legs % 2 or p.n_legs < 4:
        raise ValueError("n_legs must be even and >= 4")
    s = _base_spec(p.name)
    s.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    s.option.impratio = 1
    torso = s.worldbody.add_body(name="torso", pos=[0, 0, 0.3])
    torso.add_freejoint(name="root")
    torso.add_geom(name="torso_geom", type=G.mjGEOM_BOX, size=list(p.body_half), density=p.body_density,
                   rgba=[0.35, 0.4, 0.5, 1])
    imu = add_imu(s, "torso")
    k = p.n_legs // 2
    xs = np.linspace(p.body_half[0] * 0.85, -p.body_half[0] * 0.85, k)
    joints, acts, feet, foot_sites, touch = [], [], [], [], []
    default = {}
    ranges = {"coxa": [-0.8, 0.8], "femur": [-1.2, 1.4], "tibia": [-2.6, 0.2]}
    if p.layout == "mammal":
        ranges = {"coxa": [-0.6, 0.6], "femur": [-1.0, 2.2], "tibia": [-2.6, -0.5]}
    for side, sgn in (("l", 1), ("r", -1)):
        for i, x in enumerate(xs):
            leg = f"leg_{side}{i}"
            if p.layout == "sprawl":
                frac = x / max(xs[0], 1e-6)
                yaw = sgn * (math.pi / 2 - p.splay * frac)
                hip = torso.add_body(name=f"{leg}_coxa", pos=[x, sgn * p.body_half[1], 0],
                                     quat=_quat_from_axis_angle([0, 0, 1], yaw))
                hip.add_joint(name=f"{leg}_coxa", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                              range=ranges["coxa"], limited=LIM, damping=0.05, armature=0.005)
                hip.add_geom(name=f"{leg}_coxa_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, p.coxa, 0, 0],
                             size=[p.leg_radius, 0, 0], density=p.density, rgba=[0.6, 0.6, 0.62, 1])
                fem = hip.add_body(name=f"{leg}_femur", pos=[p.coxa, 0, 0])
                fem.add_joint(name=f"{leg}_femur", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                              range=ranges["femur"], limited=LIM, damping=0.05, armature=0.005)
                fem.add_geom(name=f"{leg}_femur_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, p.femur, 0, 0],
                             size=[p.leg_radius, 0, 0], density=p.density, rgba=[0.6, 0.6, 0.62, 1])
                tib = fem.add_body(name=f"{leg}_tibia", pos=[p.femur, 0, 0])
                tib.add_joint(name=f"{leg}_tibia", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                              range=ranges["tibia"], limited=LIM, damping=0.05, armature=0.005)
                tib.add_geom(name=f"{leg}_tibia_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, p.tibia, 0, 0],
                             size=[p.leg_radius * 0.8, 0, 0], density=p.density, rgba=[0.5, 0.5, 0.55, 1])
                foot_pos = [p.tibia, 0, 0]
                stance = p.stance
            else:  # mammal: abduction (x axis), hip pitch (y), knee pitch (y); legs under the body
                hip = torso.add_body(name=f"{leg}_coxa", pos=[x, sgn * p.body_half[1], 0])
                hip.add_joint(name=f"{leg}_coxa", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                              range=ranges["coxa"], limited=LIM, damping=0.05, armature=0.01)
                hip.add_geom(name=f"{leg}_coxa_geom", type=G.mjGEOM_CAPSULE,
                             fromto=[0, 0, 0, 0, sgn * p.coxa, 0], size=[p.leg_radius * 1.3, 0, 0],
                             density=p.density, rgba=[0.6, 0.6, 0.62, 1])
                fem = hip.add_body(name=f"{leg}_femur", pos=[0, sgn * p.coxa, 0])
                fem.add_joint(name=f"{leg}_femur", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                              range=ranges["femur"], limited=LIM, damping=0.05, armature=0.01)
                fem.add_geom(name=f"{leg}_femur_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -p.femur],
                             size=[p.leg_radius, 0, 0], density=p.density, rgba=[0.6, 0.6, 0.62, 1])
                tib = fem.add_body(name=f"{leg}_tibia", pos=[0, 0, -p.femur])
                tib.add_joint(name=f"{leg}_tibia", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                              range=ranges["tibia"], limited=LIM, damping=0.05, armature=0.01)
                tib.add_geom(name=f"{leg}_tibia_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -p.tibia],
                             size=[p.leg_radius * 0.8, 0, 0], density=p.density, rgba=[0.5, 0.5, 0.55, 1])
                foot_pos = [0, 0, -p.tibia]
                stance = p.stance
            tib.add_geom(name=f"{leg}_foot", type=G.mjGEOM_SPHERE, pos=foot_pos, size=[p.foot_radius, 0, 0],
                         density=p.density, friction=[p.friction, 0.02, 0.001], condim=3,
                         rgba=[0.15, 0.15, 0.15, 1])
            tib.add_site(name=f"{leg}_foot_site", pos=foot_pos, size=[p.foot_radius * 1.25, 0, 0])
            s.add_sensor(name=f"{leg}_touch", type=mujoco.mjtSensor.mjSENS_TOUCH,
                         objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{leg}_foot_site")
            for jn, q0 in zip(("coxa", "femur", "tibia"), stance):
                j = f"{leg}_{jn}"
                joints.append(j)
                default[j] = float(q0)
                pd_servo(s, f"act_{j}", j, p.kp, p.kd, p.effort, ranges[jn])
                acts.append(f"act_{j}")
            feet.append(tib.name)
            foot_sites.append(f"{leg}_foot_site")
            touch.append(f"{leg}_touch")
    family = {4: "quadruped", 6: "hexapod"}.get(p.n_legs, "multipod")
    fam_lineage = f"{p.lineage}/{p.layout}{p.n_legs}"
    reach = p.coxa + p.femur + p.tibia
    meta = dict(name=p.name, family=family, synthetic=True, lineage=[p.lineage, fam_lineage, f"{fam_lineage}/{p.name}"],
                assemblies=[dict(id="body", kind="body", root_body="torso", frame=dict(site=imu["site"]),
                                 capabilities=["locomote"])],
                ports=[],
                controller=dict(kind="joint_targets", groups=[dict(name="legs", actuators=acts,
                                                                   semantic="joint_position", units="rad")]),
                params=dict(p.__dict__, lengths=[reach]),
                legged=dict(root_body="torso", imu=imu, foot_bodies=feet, foot_sites=foot_sites, touch_sensors=touch,
                            policy_actuators=acts, held_actuators=[], default_pose=default,
                            kp={a: p.kp for a in acts}, kd={a: p.kd for a in acts},
                            action_scale=0.3, command_ranges=dict(vx=[-0.15, 0.3], vy=[-0.1, 0.1], wz=[-0.6, 0.6]),
                            gait_period=0.5, min_height_frac=0.4, tilt_limit=0.9, kind=family,
                            gait="tripod" if p.n_legs == 6 else ("trot" if p.n_legs == 4 else "wave")))
    return Module(s, meta)


def hexapod(name="hexapod6", **kw) -> Module:
    return procedural_legged(LeggedParams(name=name, n_legs=6, **kw))


def procedural_quadruped(name="pquad4", **kw) -> Module:
    base = dict(layout="mammal", body_half=(0.2, 0.08, 0.04), coxa=0.04, femur=0.18, tibia=0.18, leg_radius=0.018,
                foot_radius=0.02, density=600.0, body_density=500.0, kp=30.0, kd=0.8, effort=18.0,
                stance=(0.0, 0.75, -1.5))
    base.update(kw)
    m = procedural_legged(LeggedParams(name=name, n_legs=4, **base))
    m.meta["legged"]["command_ranges"] = dict(vx=[-0.3, 0.6], vy=[-0.2, 0.2], wz=[-0.8, 0.8])
    m.meta["legged"]["action_scale"] = 0.25
    m.meta["legged"]["gait_period"] = 0.45
    return m


def leg_count_variant(n_legs: int, *, femur=0.10, tibia=0.14, name=None) -> Module:
    """4/6/8-leg sprawl diagnostic generator (same lineage family; topology holdouts)."""
    return procedural_legged(LeggedParams(name=name or f"sprawl{n_legs}_f{int(femur*100)}t{int(tibia*100)}",
                                          n_legs=n_legs, femur=femur, tibia=tibia,
                                          body_half=(0.05 * n_legs / 2 + 0.05, 0.09, 0.03)))


PROCEDURAL = {
    "hexapod6": lambda: hexapod(),
    "pquad4": lambda: procedural_quadruped(),
    "sprawl4": lambda: leg_count_variant(4),
    "sprawl8": lambda: leg_count_variant(8),
    "hexapod6_long": lambda: leg_count_variant(6, femur=0.12, tibia=0.17, name="hexapod6_long"),
}


# ------------------------------------------------------------------ menagerie legged importers
# gains: list of (regex over actuator/joint name, kp, kd, effort Nm). None effort -> keep asset torque limit.
LEGGED_ASSETS = {
    "go2": dict(dir="unitree_go2", file="go2.xml", kind="quadruped", family="unitree_quadruped",
                license="BSD-3-Clause", key="home", feet=["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
                gains=[(".*", 35.0, 0.5, None)], legs=r".*", action_scale=0.25, gait_period=0.5,
                command_ranges=dict(vx=[-0.5, 1.0], vy=[-0.4, 0.4], wz=[-1.0, 1.0])),
    "anymal_c": dict(dir="anybotics_anymal_c", file="anymal_c.xml", kind="quadruped", family="anybotics_anymal",
                     license="BSD-3-Clause", key=None, feet=["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"],
                     default={"LF_HFE": 0.4, "LF_KFE": -0.8, "RF_HFE": 0.4, "RF_KFE": -0.8,
                              "LH_HFE": -0.4, "LH_KFE": 0.8, "RH_HFE": -0.4, "RH_KFE": 0.8},
                     gains=[(".*", 80.0, 2.0, 80.0)], legs=r".*", action_scale=0.3, gait_period=0.6,
                     command_ranges=dict(vx=[-0.5, 0.8], vy=[-0.3, 0.3], wz=[-0.8, 0.8])),
    "a1": dict(dir="unitree_a1", file="a1.xml", kind="quadruped", family="unitree_quadruped",
               license="BSD-3-Clause", key="home", feet=["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
               gains=[(".*", 35.0, 0.6, 33.5)], legs=r".*", action_scale=0.25, gait_period=0.45,
               command_ranges=dict(vx=[-0.5, 0.8], vy=[-0.3, 0.3], wz=[-1.0, 1.0])),
    "h1": dict(dir="unitree_h1", file="h1.xml", kind="humanoid", family="unitree_humanoid_h1",
               license="BSD-3-Clause", key=None, feet=["left_ankle_link", "right_ankle_link"],
               default={"left_hip_pitch": -0.1, "left_knee": 0.3, "left_ankle": -0.2,
                        "right_hip_pitch": -0.1, "right_knee": 0.3, "right_ankle": -0.2},
               gains=[("hip", 150.0, 2.0, None), ("knee", 200.0, 4.0, None), ("ankle", 40.0, 2.0, None),
                      ("torso", 300.0, 6.0, None), ("shoulder|elbow", 100.0, 2.0, None)],
               legs=r"hip|knee|ankle", action_scale=0.25, gait_period=0.8,
               command_ranges=dict(vx=[-0.3, 0.8], vy=[-0.2, 0.2], wz=[-0.6, 0.6])),
    "g1": dict(dir="unitree_g1", file="g1.xml", kind="humanoid", family="unitree_humanoid_g1",
               license="BSD-3-Clause", key="stand", feet=["left_ankle_roll_link", "right_ankle_roll_link"],
               gains=[("hip_(pitch|yaw)", 100.0, 2.0, 88.0), ("hip_roll", 100.0, 2.0, 139.0),
                      ("knee", 150.0, 4.0, 139.0), ("ankle", 40.0, 2.0, 50.0), ("waist", 200.0, 5.0, 88.0),
                      ("shoulder|elbow|wrist", 40.0, 1.0, 25.0)],
               legs=r"hip|knee|ankle", action_scale=0.25, gait_period=0.8,
               command_ranges=dict(vx=[-0.3, 0.8], vy=[-0.2, 0.2], wz=[-0.6, 0.6])),
    "t1": dict(dir="booster_t1", file="t1.xml", kind="humanoid", family="booster_t1", license="Apache-2.0",
               key="home", feet=["left_foot_link", "right_foot_link"],
               gains=[("Hip", 150.0, 3.0, 60.0), ("Knee", 180.0, 4.0, 130.0), ("Ankle", 50.0, 2.0, 50.0),
                      ("Waist", 150.0, 4.0, 30.0), ("Head|Shoulder|Elbow", 40.0, 1.0, 18.0)],
               legs=r"Hip|Knee|Ankle", action_scale=0.25, gait_period=0.8,
               command_ranges=dict(vx=[-0.3, 0.8], vy=[-0.2, 0.2], wz=[-0.6, 0.6])),
    "op3": dict(dir="robotis_op3", file="op3.xml", kind="humanoid", family="robotis_op", license="Apache-2.0",
                key=None, feet=["l_ank_roll_link", "r_ank_roll_link"], default={},
                gains=[(".*", 21.1, 0.6, 5.0)], legs=r"hip|knee|ank", action_scale=0.3, gait_period=0.5,
                command_ranges=dict(vx=[-0.1, 0.25], vy=[-0.05, 0.05], wz=[-0.5, 0.5])),
    "talos": dict(dir="pal_talos", file="talos_position.xml", kind="humanoid", family="pal_talos",
                  license="Apache-2.0", key="walk_pose", feet=["leg_left_6_link", "leg_right_6_link"],
                  gains=None, legs=r"leg_", action_scale=0.2, gait_period=0.9,
                  command_ranges=dict(vx=[-0.2, 0.5], vy=[-0.1, 0.1], wz=[-0.4, 0.4])),
    "cassie": dict(dir="agility_cassie", file="cassie.xml", kind="biped", family="agility_cassie", license="MIT",
                   key="home", feet=["left-foot", "right-foot"],
                   gains=[("hip-roll|hip-yaw", 100.0, 3.0, None), ("hip-pitch|knee", 200.0, 5.0, None),
                          ("foot", 40.0, 1.0, None)],
                   legs=r".*", action_scale=0.25, gait_period=0.8,
                   command_ranges=dict(vx=[-0.3, 1.0], vy=[-0.2, 0.2], wz=[-0.6, 0.6])),
}


def _rule(name: str, rules):
    for pat, kp, kd, eff in rules:
        if re.search(pat, name):
            return kp, kd, eff
    return None


def _foot_site(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData, body: str, site_name: str,
               sensor_name: str):
    """Touch site covering the lowest collision geom of a foot body (in the default pose)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and
            (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not gids:
        raise ValueError(f"foot body {body} has no collision geoms")
    g = min(gids, key=lambda g: data.geom_xpos[g][2] - model.geom_rbound[g])
    # site in body frame: geom local pose, enlarged
    pos, quat = model.geom_pos[g].copy(), model.geom_quat[g].copy()
    b = spec.body(body)
    t = model.geom_type[g]
    if t == G.mjGEOM_SPHERE:
        b.add_site(name=site_name, pos=pos, quat=quat, type=G.mjGEOM_SPHERE, size=[model.geom_size[g][0] * 1.3, 0, 0])
    elif t in (G.mjGEOM_BOX,):
        sz = model.geom_size[g] * 1.15 + 0.004
        b.add_site(name=site_name, pos=pos, quat=quat, type=G.mjGEOM_BOX, size=list(sz))
    else:  # capsule / cylinder / mesh: aabb box
        aabb = model.geom_aabb[g]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        b.add_site(name=site_name, pos=pos + R.reshape(3, 3) @ aabb[:3], quat=quat, type=G.mjGEOM_BOX,
                   size=list(aabb[3:] * 1.15 + 0.004))
    spec.add_sensor(name=sensor_name, type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname=site_name)
    return g


def menagerie_legged(key: str) -> Module:
    info = LEGGED_ASSETS[key]
    path = MENAGERIE / info["dir"] / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"asset not fetched: {path}")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.modelname = key
    src_model = spec.copy().compile()
    free = [j for j in range(src_model.njnt) if src_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"{key}: expected one free joint, got {len(free)}")
    root_bid = int(src_model.jnt_bodyid[free[0]])
    root_body = src_model.body(root_bid).name
    # unnamed free joint -> name it (needed for addressing after prefixing)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE and not j.name:
            j.name = "root"
    for u, a in enumerate(spec.actuators):
        if not a.name:
            a.name = f"act{u}"
    # ---- actuator adapter: every actuator -> bounded joint PD servo (original torque limits kept)
    adapter = []
    m0 = src_model
    for u, a in enumerate(list(spec.actuators)):
        jname = a.target
        jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0 or m0.actuator_trntype[u] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"{key}: actuator {a.name} is not a joint actuator")
        is_pos = m0.actuator_biastype[u] == mujoco.mjtBias.mjBIAS_AFFINE and m0.actuator_biasprm[u, 1] < 0
        gear = float(m0.actuator_gear[u, 0])
        if is_pos:
            eff0 = float(m0.actuator_forcerange[u, 1] * gear) if m0.actuator_forcelimited[u] else None
            kp0, kd0 = float(m0.actuator_gainprm[u, 0]), float(-m0.actuator_biasprm[u, 2])
        else:
            eff0 = float(max(abs(m0.actuator_ctrlrange[u, 0]), abs(m0.actuator_ctrlrange[u, 1])) * gear)
            kp0, kd0 = None, None
        r = _rule(a.name, info["gains"]) if info["gains"] else None
        if r is None and not is_pos:
            raise ValueError(f"{key}: no PD gain rule for motor actuator {a.name}")
        kp, kd, eff = r if r else (kp0, kd0, eff0)
        eff = eff if eff is not None else (eff0 if eff0 is not None else 1e3)
        jr = m0.jnt_range[jid] if m0.jnt_limited[jid] else [-math.pi, math.pi]
        a.gear = [1, 0, 0, 0, 0, 0]
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm = [kp] + [0] * 9
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm = [0, -kp, -kd] + [0] * 7
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.ctrllimited = LIM
        a.ctrlrange = [float(jr[0]), float(jr[1])]
        a.forcelimited = LIM
        a.forcerange = [-eff, eff]
        adapter.append(dict(actuator=a.name, joint=jname, source_kind="position" if is_pos else "motor",
                            source_gear=gear, source_kp=kp0, kp=kp, kd=kd, effort=eff))
    # joint equalities whose DEPENDENT side (joint1) is an actuated joint: invert the (linear) relation
    # so the actuated joint is the driver (physics unchanged: q1 = c0 + c1 q2  <=>  q2 = -c0/c1 + q1/c1)
    actuated = {a.target for a in spec.actuators}
    inverted = []
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT and eq.name1 in actuated and eq.name2:
            c = list(eq.data[:5])
            if abs(c[1]) > 1e-9 and all(abs(x) < 1e-12 for x in c[2:5]):
                eq.name1, eq.name2 = eq.name2, eq.name1
                eq.data = [-c[0] / c[1], 1.0 / c[1], 0, 0, 0] + list(eq.data[5:])
                inverted.append([eq.name2, eq.name1])
            else:
                raise ValueError(f"{key}: nonlinear equality drives actuated joint {eq.name1}")
    imu = add_imu(spec, root_body)
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    # default pose
    if info.get("key"):
        k = next(i for i in range(model.nkey) if model.key(i).name == info["key"])
        mujoco.mj_resetDataKeyframe(model, data, k)
    else:
        mujoco.mj_resetData(model, data)
        for jn, v in (info.get("default") or {}).items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            data.qpos[model.jnt_qposadr[jid]] = v
    mujoco.mj_kinematics(model, data)
    acts = [model.actuator(u).name for u in range(model.nu)]
    act_joint = {model.actuator(u).name: model.joint(model.actuator_trnid[u, 0]).name for u in range(model.nu)}
    default = {act_joint[a]: float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                                    act_joint[a])]]) for a in acts}
    legs_re = re.compile(info["legs"])
    policy = [a for a in acts if legs_re.search(a) or legs_re.search(act_joint[a])]
    held = [a for a in acts if a not in policy]
    # foot touch sites (added after compile of default pose)
    foot_sites, touch = [], []
    for fb in info["feet"]:
        sn = f"{fb}_touch_site".replace("-", "_")
        tn = f"{fb}_touch".replace("-", "_")
        _foot_site(spec, model, data, fb, sn, tn)
        foot_sites.append(sn)
        touch.append(tn)
    # nominal height: root z such that the lowest foot geom touches z=0
    fbids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in info["feet"]]
    low = min(data.geom_xpos[g][2] - model.geom_rbound[g] for g in range(model.ngeom)
              if model.geom_bodyid[g] in fbids and (model.geom_contype[g] or model.geom_conaffinity[g]))
    # geom_rbound overestimates for boxes/meshes: refine with aabb along world z
    lows = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in fbids and (model.geom_contype[g] or model.geom_conaffinity[g]):
            R = data.geom_xmat[g].reshape(3, 3)
            c = data.geom_xpos[g] + R @ model.geom_aabb[g][:3]
            half = np.abs(R) @ model.geom_aabb[g][3:]
            lows.append(c[2] - half[2])
    low = min(lows) if lows else low
    root_z = float(data.xpos[root_bid][2])
    nominal = root_z - low
    meta = dict(name=key, family=info["kind"], synthetic=False,
                lineage=[f"menagerie/{info['family']}", f"menagerie/{info['dir']}@{MENAGERIE_SHA[:12]}"],
                asset=dict(source="https://github.com/google-deepmind/mujoco_menagerie", commit=MENAGERIE_SHA,
                           dir=info["dir"], file=info["file"], license=info["license"]),
                assemblies=[dict(id="body", kind="body", root_body=root_body, frame=dict(site=imu["site"]),
                                 capabilities=["locomote"])],
                ports=[],
                controller=dict(kind="joint_targets", groups=[
                    g for g in (dict(name="legs", actuators=policy, semantic="joint_position", units="rad"),
                                dict(name="upper", actuators=held, semantic="joint_position", units="rad"))
                    if g["actuators"]]),
                actuator_adapter=dict(kind="joint_pd_position_servo", note="motor actuators converted to PD servos "
                                      "with original torque limits; position actuators re-gained per table",
                                      actuators=adapter, inverted_joint_equalities=inverted),
                params=dict(lengths=[nominal]),
                legged=dict(root_body=root_body, imu=imu, foot_bodies=list(info["feet"]), foot_sites=foot_sites,
                            touch_sensors=touch, policy_actuators=policy, held_actuators=held, default_pose=default,
                            kp={d["actuator"]: d["kp"] for d in adapter}, kd={d["actuator"]: d["kd"] for d in adapter},
                            action_scale=info["action_scale"], command_ranges=info["command_ranges"],
                            gait_period=info["gait_period"], min_height_frac=0.55 if info["kind"] != "quadruped" else 0.4,
                            tilt_limit=0.7 if info["kind"] != "quadruped" else 0.9, kind=info["kind"],
                            nominal_height=nominal, gait="biped" if info["kind"] != "quadruped" else "trot",
                            source_timestep=float(src_model.opt.timestep)),
                source_options=dict(timestep=float(src_model.opt.timestep), integrator=int(src_model.opt.integrator),
                                    cone=int(src_model.opt.cone), impratio=float(src_model.opt.impratio),
                                    iterations=int(src_model.opt.iterations)))
    return Module(spec, meta)


def legged_body(key: str) -> Module:
    if key in PROCEDURAL:
        return PROCEDURAL[key]()
    return menagerie_legged(key)


ALL_LEGGED = list(PROCEDURAL) + list(LEGGED_ASSETS)


# ------------------------------------------------------------------ world + standalone model for training/validation
def legged_world(name: str, source_options: dict | None = None, *, size: float = 20.0) -> mujoco.MjSpec:
    s = _base_spec(name)
    if source_options:
        s.option.timestep = source_options["timestep"]
        s.option.integrator = source_options["integrator"]
        s.option.cone = source_options["cone"]
        s.option.impratio = source_options["impratio"]
        s.option.iterations = source_options["iterations"]
    else:
        s.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        s.option.impratio = 1
    s.visual.global_.offwidth = 640
    s.visual.global_.offheight = 480
    s.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.worldbody.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[size, size, 0.1], rgba=[0.45, 0.47, 0.44, 1],
                         friction=[1.0, 0.02, 0.001])
    return s


def standalone_model(module: Module, prefix: str = "r0_") -> tuple[mujoco.MjModel, mujoco.MjSpec, dict]:
    """Robot on a floor (no task objects): used by trainers and validation."""
    meta = copy.deepcopy(module.meta)
    scene = legged_world(f"{meta['name']}_world", meta.get("source_options"))
    site = scene.worldbody.add_site(name="mount0", pos=[0, 0, 0])
    scene.attach(module.spec.copy(), prefix=prefix, site=site)
    scene.memory = 3 * 2 ** 20      # per-MjData arena (contacts/constraints); default 14 MiB is wasteful x100s
    model = scene.compile()
    return model, scene, meta
