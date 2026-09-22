"""MuJoCo Menagerie ALOHA 2 (bimanual ViperX-300 derivative) as ONE dual-arm body.

Imported as a single Module with two arm assemblies and two gripper assemblies, each with its
own controller groups (left_arm/left_gripper/right_arm/right_gripper). Declared surgery (all
recorded in meta['surgery']; no physics parameters of the asset are changed):
  * a TCP site per gripper at the fingertip midpoint, z = approach axis, y = closing axis;
  * a box touch site + touch sensor on every finger link (the asset has no touch sensors);
  * a joint-position width sensor on each actuated finger joint;
  * keyframes removed from the module (the scene sets the declared home pose explicitly);
  * the finger joint equality leader swapped to the actuated finger (symmetric coupling, same physics).
Lineage: menagerie/aloha (trossen family; derived from the ViperX-300 6DOF model).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .generators import Module
from .importers import MENAGERIE, MENAGERIE_SHA

ALOHA_DIR = MENAGERIE / "aloha"
ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
HOME = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]     # the asset's 'neutral_pose' keyframe (arm part)
TCP_INSET = 0.012                            # TCP this far inside the fingertip ends (pad overlap)


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q.tolist()


def aloha_body() -> Module:
    path = ALOHA_DIR / "aloha.xml"
    if not path.exists():
        raise FileNotFoundError(f"asset not fetched: {path}")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.modelname = "aloha"
    for k in list(spec.keys):
        spec.delete(k)
    surgery = []
    # the asset couples the fingers with q_left = q_right but declares the ACTUATED left finger as the
    # dependent joint; swap the (symmetric, identical-physics) equality so the actuator drives the leader
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT and eq.name1.endswith("/left_finger"):
            eq.name1, eq.name2 = eq.name2, eq.name1
            surgery.append(dict(op="swap_equality_leader", joints=[eq.name1, eq.name2],
                                reason="actuated joint must be the driver (polycoef 0 1: identical physics)"))
    m = spec.copy().compile()
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        for jn, v in zip(ARM_JOINTS, HOME):
            d.qpos[m.jnt_qposadr[m.joint(f"{side}/{jn}").id]] = v
    mujoco.mj_kinematics(m, d)
    manipulators, assemblies, groups, touch_all = [], [], [], []
    for side in ("left", "right"):
        gl = m.body(f"{side}/gripper_link").id
        xg, Rg = d.xpos[gl].copy(), d.xmat[gl].reshape(3, 3).copy()
        pl = d.geom_xpos[m.geom(f"{side}/left_g1").id].copy()
        pr = d.geom_xpos[m.geom(f"{side}/right_g1").id].copy()
        z = Rg[:, 0]                                  # approach axis of the ViperX gripper link
        y = pr - pl
        y = y - (y @ z) * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        mid = (pl + pr) / 2 - z * TCP_INSET
        R_local = Rg.T @ np.stack([x, y, z], axis=1)
        body = next(b for b in spec.bodies if b.name == f"{side}/gripper_link")
        body.add_site(name=f"{side}/tcp", pos=(Rg.T @ (mid - xg)).tolist(), quat=_mat2quat(R_local),
                      size=[0.005, 0, 0], group=5)
        surgery.append(dict(op="add_site", name=f"{side}/tcp", body=f"{side}/gripper_link",
                            rule="fingertip midpoint, z=approach, y=closing", inset_m=TCP_INSET))
        touch = []
        for fl in ("left_finger_link", "right_finger_link"):
            fb = m.body(f"{side}/{fl}").id
            gid = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                       and m.geom_contype[g] != 0)
            c, h = m.geom_aabb[gid][:3], m.geom_aabb[gid][3:]
            Rgeom = np.zeros(9)
            mujoco.mju_quat2Mat(Rgeom, m.geom_quat[gid])
            pos = m.geom_pos[gid] + Rgeom.reshape(3, 3) @ c
            sb = next(b for b in spec.bodies if b.name == f"{side}/{fl}")
            sname = f"{side}/{fl}_touch"
            sb.add_site(name=sname, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos.tolist(),
                        quat=m.geom_quat[gid].tolist(), size=(h * 1.05 + 0.001).tolist(), group=5)
            spec.add_sensor(name=sname, type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                            objname=sname)
            touch.append(sname)
            surgery.append(dict(op="add_touch_sensor", site=sname, body=f"{side}/{fl}", shape="finger mesh AABB x1.05"))
        wname = f"{side}/grip_width"
        spec.add_sensor(name=wname, type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"{side}/left_finger")
        surgery.append(dict(op="add_width_sensor", sensor=wname, joint=f"{side}/left_finger"))
        touch_all += touch
        assemblies += [dict(id=f"{side}_arm", kind="arm", root_body=f"{side}/base_link",
                            frame=dict(site=f"{side}/tcp"), capabilities=["push"]),
                       dict(id=f"{side}_gripper", kind="gripper", root_body=f"{side}/gripper_base",
                            frame=dict(site=f"{side}/tcp"), capabilities=["grasp", "push", "insert", "support"],
                            parent=f"{side}_arm")]
        groups += [dict(name=f"{side}_arm", actuators=[f"{side}/{j}" for j in ARM_JOINTS], semantic="joint_position",
                        units="rad"),
                   dict(name=f"{side}_gripper", actuators=[f"{side}/gripper"], semantic="gripper", units="m",
                        open_value=0.037, closed_value=0.002)]
        base = d.xpos[m.body(f"{side}/base_link").id]
        manipulators.append(dict(assembly=f"{side}_gripper", role_hint=side, arm_group=f"{side}_arm",
                                 grip_group=f"{side}_gripper", touch_sensors=touch, width_sensor=wname,
                                 home=list(HOME), gripper_kind="aloha", reach_m=0.75,
                                 base_local=[float(v) for v in base]))
    meta = dict(name="aloha", family="dual_arm", synthetic=False,
                lineage=["menagerie/trossen", f"menagerie/aloha@{MENAGERIE_SHA[:12]}"],
                asset=dict(source="https://github.com/google-deepmind/mujoco_menagerie", commit=MENAGERIE_SHA,
                           dir="aloha", file="aloha.xml", license="BSD-3-Clause"),
                assemblies=assemblies, ports=[],
                controller=dict(kind="joint_targets", groups=groups),
                manipulators=manipulators, touch_sensors=[], width_sensor=None, home=None,
                surgery=surgery, reach_m=0.75, params=dict(lengths=[0.75]))
    return Module(spec, meta)
