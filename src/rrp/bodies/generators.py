"""Procedural, physically valid robot modules built with MuJoCo's MjSpec API.

Every generated module is a real dynamical system (positive masses/inertias from geom
densities, bounded actuators, joint limits, contacts) plus explicit metadata: lineage,
assemblies with declared frames, attachment ports and controller hints. These are
labelled `synthetic` and never counted as commercial robot families.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


@dataclass
class Module:
    spec: mujoco.MjSpec
    meta: dict = field(default_factory=dict)


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), *(axis * s)]


def _base_spec(name: str) -> mujoco.MjSpec:
    s = mujoco.MjSpec()
    s.modelname = name
    s.compiler.degree = False
    s.option.timestep = 0.002
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10
    return s


# ------------------------------------------------------------------ arms
@dataclass
class ArmParams:
    name: str = "parm5"
    axes: tuple = ("z", "y", "y", "y", "z")      # joint axis pattern from base to wrist (last = wrist roll)
    lengths: tuple = (0.12, 0.34, 0.30, 0.10, 0.06)
    radius: float = 0.035
    density: float = 1200.0
    kp: float = 400.0
    damping: float = 8.0
    effort: float = 80.0
    lineage: str = "procedural_arm_family/v1"
    rgba: tuple = (0.7, 0.7, 0.75, 1)
    home: tuple | None = None   # non-singular ready pose; default derived from the axis pattern


def procedural_arm(p: ArmParams) -> Module:
    """Serial arm with a wrist attachment port (site 'port_wrist')."""
    s = _base_spec(p.name)
    parent = s.worldbody.add_body(name="base", pos=[0, 0, 0])
    parent.add_geom(name="base_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.03, 0],
                    pos=[0, 0, 0.03], density=p.density * 2, rgba=[0.3, 0.3, 0.35, 1])
    joints = []
    z = 0.06
    ranges = {"z": [-2.9, 2.9], "y": [-2.0, 2.0], "x": [-2.9, 2.9]}
    for i, (ax, L) in enumerate(zip(p.axes, p.lengths)):
        b = parent.add_body(name=f"link{i}", pos=[0, 0, z if i == 0 else 0], gravcomp=1.0)
        if i > 0:
            b.pos = [0, 0, p.lengths[i - 1]]
        j = b.add_joint(name=f"j{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=AXES[ax], range=ranges[ax],
                        limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=p.damping, armature=0.02)
        b.add_geom(name=f"link{i}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[0, 0, 0, 0, 0, L], size=[p.radius * (1 - 0.08 * i), 0, 0], density=p.density,
                   rgba=list(p.rgba), contype=1, conaffinity=1)
        joints.append(j.name)
        parent = b
    # the base is welded to the world, so MuJoCo's parent filter does not apply: exclude explicitly
    s.add_exclude(bodyname1="base", bodyname2="link0")
    flange = parent.add_body(name="flange", pos=[0, 0, p.lengths[-1]], gravcomp=1.0)
    flange.add_geom(name="flange_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.008, 0],
                    density=p.density, rgba=[0.2, 0.2, 0.2, 1])
    flange.add_site(name="port_wrist", pos=[0, 0, 0.008])
    flange.add_site(name="flange_tcp", pos=[0, 0, 0.008])
    for i, jn in enumerate(joints):
        s.add_actuator(name=f"act_{jn}", target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                       gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[p.kp] + [0] * 9,
                       biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -p.kp, -p.damping * 0.5] + [0] * 7,
                       ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=ranges[p.axes[i]],
                       forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-p.effort, p.effort])
        s.add_sensor(name=f"jpos_{jn}", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                     objname=jn)
    meta = dict(name=p.name, family="arm", synthetic=True, lineage=[p.lineage, f"{p.lineage}/{p.name}"],
                assemblies=[dict(id="arm", kind="arm", root_body="link0", frame=dict(site="flange_tcp"),
                                 capabilities=["push"])],
                ports=[dict(id="wrist", site="port_wrist", host_body="flange", max_payload_kg=3.0,
                            max_module_extent_m=0.25, interface="wrist_generic",
                            allowed_module_kinds=["gripper", "hand", "tool"])],
                controller=dict(kind="joint_targets", groups=[dict(name="arm", actuators=[f"act_{j}" for j in joints],
                                                                   semantic="joint_position", units="rad")]),
                params=p.__dict__.copy(), home=list(p.home) if p.home else default_home(p.axes))
    return Module(s, meta)


def default_home(axes) -> list[float]:
    """Elbow-up, tool-down ready pose: pitch joints share ~pi of bend, others zero."""
    pitch = [i for i, a in enumerate(axes) if a == "y"]
    home = [0.0] * len(axes)
    shares = [0.5, 1.1, 1.5, 0.3, 0.2][:len(pitch)]
    total = sum(shares)
    for k, i in enumerate(pitch):
        home[i] = shares[k] * (3.1 / total if len(pitch) >= 3 else 1.0)
    return home


# ------------------------------------------------------------------ grippers / hands
@dataclass
class GripperParams:
    name: str = "pg2"
    kind: str = "parallel"          # parallel | three_finger
    stroke: float = 0.045           # max opening per finger (m)
    finger_length: float = 0.055
    finger_width: float = 0.018
    palm: tuple = (0.035, 0.05, 0.02)
    force: float = 40.0
    lineage: str = "procedural_gripper/v1"
    density: float = 900.0
    friction: float = 1.5


def gripper_module(p: GripperParams | None = None) -> Module:
    p = p or GripperParams()
    if p.kind == "three_finger":
        return three_finger_module(p)
    s = _base_spec(p.name)
    palm = s.worldbody.add_body(name="palm", pos=[0, 0, 0], gravcomp=1.0)
    palm.add_geom(name="palm_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(p.palm),
                  pos=[0, 0, p.palm[2]], density=p.density, rgba=[0.25, 0.25, 0.3, 1])
    z0 = 2 * p.palm[2]
    palm.add_site(name="tcp", pos=[0, 0, z0 + p.finger_length * 0.75])
    palm.add_camera(name="wrist_cam", pos=[0.06, 0, z0 - 0.01], xyaxes=[0, -1, 0, 0.9, 0, 0.45], fovy=75)
    fingers = []
    for side, sgn in (("a", 1), ("b", -1)):
        f = palm.add_body(name=f"finger_{side}", pos=[0, sgn * 0.012, z0], gravcomp=1.0)
        j = f.add_joint(name=f"slide_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sgn, 0],
                        range=[0, p.stroke], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=5.0,
                        armature=0.01)
        f.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[p.finger_width / 2, 0.006, p.finger_length / 2], pos=[0, 0, p.finger_length / 2],
                   density=p.density, friction=[p.friction, 0.02, 0.002], condim=4, rgba=[0.1, 0.1, 0.1, 1],
                   solref=[0.01, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
        f.add_site(name=f"touch_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[p.finger_width / 2 + 0.001, 0.007, p.finger_length / 2 + 0.001],
                   pos=[0, 0, p.finger_length / 2])
        fingers.append(j.name)
    # finger b mimics finger a (dependent coordinate, not independently actuated)
    s.add_equality(name="mimic_b", type=mujoco.mjtEq.mjEQ_JOINT, name1="slide_b", name2="slide_a",
                   data=[0, 1, 0, 0, 0] + [0] * 6, solref=[0.005, 1])
    # opening command in meters (0 = closed, stroke = open)
    kp = p.force / 0.01
    s.add_actuator(name="act_grip", target="slide_a", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                   gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                   biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -20] + [0] * 7,
                   ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=[0, p.stroke],
                   forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-p.force, p.force])
    for side in ("a", "b"):
        s.add_sensor(name=f"touch_{side}", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname=f"touch_{side}")
    s.add_sensor(name="grip_width", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                 objname="slide_a")
    meta = dict(name=p.name, family="end_effector", synthetic=True, lineage=[p.lineage, f"{p.lineage}/{p.name}"],
                module_kind="gripper", root_body="palm",
                assemblies=[dict(id="gripper", kind="gripper", root_body="palm", frame=dict(site="tcp"),
                                 capabilities=["grasp", "push", "insert", "support"])],
                controller=dict(groups=[dict(name="gripper", actuators=["act_grip"], semantic="gripper",
                                             units="m", open_value=p.stroke, closed_value=0.0)]),
                touch_sensors=["touch_a", "touch_b"], width_sensor="grip_width", stroke=p.stroke,
                cameras=["wrist_cam"], params=p.__dict__.copy())
    return Module(s, meta)


def three_finger_module(p: GripperParams) -> Module:
    """Three hinged fingers, one actuated, two coupled by joint equalities (mimic)."""
    s = _base_spec(p.name)
    palm = s.worldbody.add_body(name="palm", pos=[0, 0, 0], gravcomp=1.0)
    palm.add_geom(name="palm_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.06, p.palm[2], 0],
                  pos=[0, 0, p.palm[2]], density=p.density, rgba=[0.3, 0.25, 0.25, 1])
    z0 = 2 * p.palm[2]
    palm.add_site(name="tcp", pos=[0, 0, z0 + p.finger_length * 0.85])
    palm.add_camera(name="wrist_cam", pos=[0.07, 0, z0 - 0.01], xyaxes=[0, -1, 0, 0.9, 0, 0.45], fovy=75)
    names = []
    for k in range(3):
        ang = 2 * math.pi * k / 3
        r = 0.052
        pos = [r * math.cos(ang), r * math.sin(ang), z0]
        # hinge axis tangent to the palm circle so positive angle closes toward center
        axis = [-math.sin(ang), math.cos(ang), 0]
        f = palm.add_body(name=f"finger_{k}", pos=pos, gravcomp=1.0)
        j = f.add_joint(name=f"hinge_{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[-a for a in axis],
                        range=[-0.6, 0.9], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.5, armature=0.005)
        f.add_geom(name=f"fpad_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, p.finger_length],
                   size=[0.009, 0, 0], density=p.density, friction=[p.friction, 0.02, 0.002], condim=4,
                   rgba=[0.12, 0.1, 0.1, 1])
        f.add_site(name=f"touch_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, p.finger_length],
                   size=[0.0105, 0, 0])
        names.append(j.name)
    for k in (1, 2):
        s.add_equality(name=f"mimic_{k}", type=mujoco.mjtEq.mjEQ_JOINT, name1=names[k], name2=names[0],
                       data=[0, 1, 0, 0, 0] + [0] * 6, solref=[0.005, 1])
    kp = p.force / 0.05
    s.add_actuator(name="act_grip", target=names[0], trntype=mujoco.mjtTrn.mjTRN_JOINT,
                   gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                   biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, -2] + [0] * 7,
                   ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=[-0.6, 0.9],
                   forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-p.force * 0.2, p.force * 0.2])
    for k in range(3):
        s.add_sensor(name=f"touch_{k}", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname=f"touch_{k}")
    s.add_sensor(name="grip_width", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                 objname=names[0])
    meta = dict(name=p.name, family="end_effector", synthetic=True, lineage=[p.lineage, f"{p.lineage}/{p.name}"],
                module_kind="gripper", root_body="palm",
                assemblies=[dict(id="gripper", kind="hand", root_body="palm", frame=dict(site="tcp"),
                                 capabilities=["grasp", "push", "support"])],
                controller=dict(groups=[dict(name="gripper", actuators=["act_grip"], semantic="gripper",
                                             units="rad", open_value=-0.6, closed_value=0.9)]),
                touch_sensors=["touch_0", "touch_1", "touch_2"], width_sensor="grip_width",
                cameras=["wrist_cam"], params=dict(p.__dict__, finger_base_radius=0.052))
    return Module(s, meta)


# ------------------------------------------------------------------ world
def workspace_spec(name: str = "scene", *, cameras: bool = True) -> mujoco.MjSpec:
    s = _base_spec(name)
    s.visual.global_.offwidth = 640
    s.visual.global_.offheight = 480
    s.worldbody.add_light(pos=[0, -0.5, 2.5], dir=[0, 0.2, -1], diffuse=[0.8, 0.8, 0.8])
    s.worldbody.add_light(pos=[1.5, 1.0, 2.0], dir=[-0.5, -0.4, -1], diffuse=[0.4, 0.4, 0.4])
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                         rgba=[0.45, 0.45, 0.42, 1])
    s.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.8, 0.02], pos=[0.35, 0, -0.02],
                         rgba=[0.62, 0.52, 0.4, 1], friction=[1.0, 0.01, 0.001])
    if cameras:
        s.worldbody.add_camera(name="front", pos=[1.25, 0.0, 0.65], xyaxes=[0, 1, 0, -0.45, 0, 0.9], fovy=55)
        s.worldbody.add_camera(name="top", pos=[0.45, 0.0, 1.3], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    return s


def add_box_object(s: mujoco.MjSpec, name: str, pos, size=(0.022, 0.022, 0.022), rgba=(0.85, 0.15, 0.1, 1),
                   density=500.0, friction=1.2):
    b = s.worldbody.add_body(name=name, pos=list(pos))
    b.add_freejoint(name=f"{name}_free")
    b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), rgba=list(rgba),
               density=density, friction=[friction, 0.02, 0.002], condim=4)
    return b


def add_target_zone(s: mujoco.MjSpec, name: str, pos, radius=0.05, rgba=(0.1, 0.7, 0.2, 0.35)):
    """Visual target region (no collision); its pose is declared task geometry."""
    b = s.worldbody.add_body(name=name, pos=list(pos), mocap=True)
    b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, 0.001, 0], rgba=list(rgba),
               contype=0, conaffinity=0)
    b.add_site(name=f"{name}_site", pos=[0, 0, 0])
    return b


def add_mount(s: mujoco.MjSpec, name: str, pos, yaw=0.0):
    return s.worldbody.add_site(name=name, pos=list(pos), quat=_quat_from_axis_angle([0, 0, 1], yaw))
