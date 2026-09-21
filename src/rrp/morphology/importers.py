"""Importers for pinned third-party assets (MuJoCo Menagerie @ source lock).

An imported arm becomes a Module with declared metadata: arm assembly, attachment port at
the asset's own attachment_site (or an added flange site), controller groups from its
position actuators, home pose from its 'home' keyframe, and lineage
`menagerie/<dir>@<sha>` so near-duplicate models (Panda/FR3...) share a family key.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

from .generators import Module

REPO = Path(__file__).resolve().parents[3]
MENAGERIE = REPO / ".cache" / "assets" / "mujoco_menagerie"
MENAGERIE_SHA = "c96a32d28fb5da84da38c1da4d749e7a13212855"

# family keys group near-duplicate lineages (docs: lineage cautions in source audit)
ARMS = {
    "panda": dict(dir="franka_emika_panda", file="panda_nohand.xml", family="franka", license="Apache-2.0"),
    "fr3": dict(dir="franka_fr3", file="fr3.xml", family="franka", license="Apache-2.0"),
    "ur5e": dict(dir="universal_robots_ur5e", file="ur5e.xml", family="universal_robots", license="BSD-3-Clause"),
    "lite6": dict(dir="ufactory_lite6", file="lite6.xml", family="ufactory", license="BSD-3-Clause"),
    "xarm7": dict(dir="ufactory_xarm7", file="xarm7_nohand.xml", family="ufactory", license="BSD-3-Clause"),
    "sawyer": dict(dir="rethink_robotics_sawyer", file="sawyer.xml", family="rethink", license="Apache-2.0"),
    "z1": dict(dir="unitree_z1", file="z1.xml", family="unitree_arm", license="BSD-3-Clause",
               add_site=dict(body="link06", pos=[0.05, 0, 0], zaxis=[1, 0, 0])),
}


def _quat_z_to(axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, axis)
    s, c = np.linalg.norm(v), float(z @ axis)
    if s < 1e-9:
        return [1, 0, 0, 0] if c > 0 else [0, 1, 0, 0]
    ang = math.atan2(s, c)
    v = v / s
    return [math.cos(ang / 2), *(v * math.sin(ang / 2))]


def menagerie_arm(key: str) -> Module:
    info = ARMS[key]
    path = MENAGERIE / info["dir"] / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"asset not fetched: {path} (run scripts/fetch_menagerie.sh)")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.modelname = key
    if "add_site" in info:
        a = info["add_site"]
        body = next(b for b in spec.bodies if b.name == a["body"])
        body.add_site(name="attachment_site", pos=a["pos"], quat=_quat_z_to(a["zaxis"]))
    m = spec.copy().compile()
    joints = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]
    acts = [m.actuator(u).name or f"act{u}" for u in range(m.nu)]
    # name unnamed actuators so they can be namespaced and addressed
    for u, a in enumerate(spec.actuators):
        if not a.name:
            a.name = f"act{u}"
    acts = [a.name for a in spec.actuators]
    site = "attachment_site"
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    host_body = m.body(m.site_bodyid[sid]).name
    root_body = next(m.body(b).name for b in range(1, m.nbody) if m.body_jntnum[b] > 0)
    home = None
    if m.nkey:
        k = next((i for i in range(m.nkey) if m.key(i).name == "home"), 0)
        home = [float(m.key_qpos[k][m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]])
                for j in joints]
    # home TCP azimuth (for IK seeding independent of the asset's zero-yaw convention)
    d = mujoco.MjData(m)
    if home:
        for j, v in zip(joints, home):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_kinematics(m, d)
    p = d.site_xpos[sid]
    reach = float(np.linalg.norm(p[:2])) + 0.25
    meta = dict(name=key, family="arm", synthetic=False,
                lineage=[f"menagerie/{info['family']}", f"menagerie/{info['dir']}@{MENAGERIE_SHA[:12]}"],
                asset=dict(source="https://github.com/google-deepmind/mujoco_menagerie", commit=MENAGERIE_SHA,
                           dir=info["dir"], file=info["file"], license=info["license"]),
                assemblies=[dict(id="arm", kind="arm", root_body=root_body, frame=dict(site=site),
                                 capabilities=["push"])],
                ports=[dict(id="wrist", site=site, host_body=host_body, max_payload_kg=2.0,
                            max_module_extent_m=0.3, interface="flange_iso9409",
                            allowed_module_kinds=["gripper", "hand", "tool"])],
                controller=dict(kind="joint_targets", groups=[dict(name="arm", actuators=acts, semantic="joint_position",
                                                                   units="rad")]),
                home=home, home_tcp_azimuth=float(math.atan2(p[1], p[0])), reach_m=reach,
                params=dict(lengths=[reach]))
    return Module(spec, meta)
