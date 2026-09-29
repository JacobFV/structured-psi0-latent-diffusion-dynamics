"""Importers for pinned third-party assets (MuJoCo Menagerie @ source lock).

An imported arm becomes a Module with declared metadata: arm assembly, attachment port at
the asset's own attachment_site (or an added flange site), controller groups from its
position actuators, home pose from its 'home' keyframe, and lineage
`menagerie/<dir>@<sha>` so near-duplicate models (Panda/FR3...) share a family key.
"""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from rrp.bodies.generators import Module

from rrp.contracts.paths import rrp_home  # noqa: E402

REPO = rrp_home()                  # checkout (unchanged); $RRP_HOME / cwd for an installed rrp
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

# armdiv (D-137): more menagerie arms at the same pinned SHA. Adapter fields: `strip` = bodies (with their subtrees)
# removed together with every actuator/equality/exclude/keyframe that references them (the asset's own gripper, so a
# declared module can be attached); `add_site.pos = "auto"` = flange on the approach axis at the far end of the host
# body's geometry. Kept in a separate dict so the historical ARMS keys (and workbench keys) are unchanged; the
# workbench exposes them only through the armdiv registry (rrp.bodies.armdiv).
ARMS_V2 = {
    "gen3": dict(dir="kinova_gen3", file="gen3.xml", family="kinova", license="BSD-3-Clause",
                 exclude=[("base_link", "shoulder_link")],      # welded root: MuJoCo's parent filter does not apply
                 add_site=dict(body="bracelet_link", pos="auto", zaxis=[0, 0, -1])),
    "iiwa14": dict(dir="kuka_iiwa_14", file="iiwa14.xml", family="kuka", license="BSD-3-Clause"),
    "rizon4": dict(dir="flexiv_rizon4", file="flexiv_rizon4.xml", family="flexiv", license="Apache-2.0",
                   add_site=dict(body="link7", pos="auto", zaxis=[0, 0, 1])),
    "ur10e": dict(dir="universal_robots_ur10e", file="ur10e.xml", family="universal_robots", license="BSD-3-Clause"),
    "vx300s": dict(dir="trossen_vx300s", file="vx300s.xml", family="trossen", license="BSD-3-Clause",
                   strip=["gripper_prop_link", "left_finger_link", "right_finger_link"],
                   add_site=dict(body="gripper_link", pos="auto", zaxis=[1, 0, 0])),
    "wx250s": dict(dir="trossen_wx250s", file="wx250s.xml", family="trossen", license="BSD-3-Clause",
                   strip=["wx250s/left_finger_link", "wx250s/right_finger_link"],
                   add_site=dict(body="wx250s/gripper_link", pos="auto", zaxis=[1, 0, 0])),
    "piper": dict(dir="agilex_piper", file="piper.xml", family="agilex", license="MIT",
                  strip=["link7", "link8"], add_site=dict(body="link6", pos="auto", zaxis=[0, 0, 1])),
    "arxl5": dict(dir="arx_l5", file="arx_l5.xml", family="arx", license="BSD-3-Clause",
                  strip=["link7", "link8"], add_site=dict(body="link6", pos="auto", zaxis=[1, 0, 0])),
    "yam": dict(dir="i2rt_yam", file="yam.xml", family="i2rt", license="MIT",
                strip=["link_left_finger", "link_right_finger"], add_site=dict(body="link_6", pos="auto", zaxis=[0, 0, 1])),
}


def _arm_info(key: str) -> dict:
    return ARMS[key] if key in ARMS else ARMS_V2[key]


def _strip_bodies(spec: mujoco.MjSpec, names: list[str]) -> list[str]:
    """Delete the named bodies (and subtrees) plus actuators/equalities/excludes/keyframes that reference them.
    Returns the names of the removed joints."""
    m = spec.copy().compile()
    gone_b, gone_j = set(), set()
    for n in names:
        root = m.body(n).id
        for b in range(m.nbody):
            a = b
            while a > 0 and a != root:
                a = m.body_parentid[a]
            if a == root:
                gone_b.add(m.body(b).name)
                gone_j.update(m.joint(j).name for j in range(m.njnt) if m.jnt_bodyid[j] == b)
    for a in list(spec.actuators):
        if a.target in gone_j:
            spec.delete(a)
    for e in list(spec.equalities):
        if e.name1 in gone_j or e.name2 in gone_j or e.name1 in gone_b or e.name2 in gone_b:
            spec.delete(e)
    for x in list(spec.excludes):
        if x.bodyname1 in gone_b or x.bodyname2 in gone_b:
            spec.delete(x)
    for k in list(spec.keys):
        spec.delete(k)
    for s_ in list(spec.sensors):
        if s_.objname in gone_j or s_.objname in gone_b:
            spec.delete(s_)
    for li in list(spec.lights):
        if getattr(li, "targetbody", "") in gone_b:
            spec.delete(li)
    for n in names:
        spec.delete(spec.body(n))
    return sorted(gone_j)


def _limit_continuous(spec: mujoco.MjSpec, bound: float = 2 * math.pi) -> None:
    """Continuous (unlimited) hinges get a declared range of +-bound (and matching actuator ctrlrange): the IK clips to
    joint ranges, and an unlimited joint compiles to range (0, 0)."""
    m = spec.copy().compile()
    free = {m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == 3 and not m.jnt_limited[j]}
    for j in spec.joints:
        if j.name in free:
            j.range = [-bound, bound]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    for a in spec.actuators:
        if a.target in free:
            a.ctrlrange = [-bound, bound]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE


def _auto_flange(spec: mujoco.MjSpec, body: str, zaxis) -> list[float]:
    """Point on the approach axis at the far end of the host body's own geoms (axis-aligned boxes, local frame)."""
    m = spec.copy().compile()
    bid = m.body(body).id
    ax = np.asarray(zaxis, float) / np.linalg.norm(zaxis)
    far = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        R = m.geom_quat[g]
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, R)
        Rm = Rm.reshape(3, 3)
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    p = m.geom_pos[g] + Rm @ (c + h * np.array([sx, sy, sz]))
                    far = max(far, float(p @ ax))
    return [float(v) for v in ax * far]


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
    info = _arm_info(key)
    path = MENAGERIE / info["dir"] / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"asset not fetched: {path} (run scripts/fetch_menagerie.sh)")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.modelname = key
    home_all = None
    if info.get("strip"):
        m0 = spec.copy().compile()                    # home keyframe read BEFORE the keyframes are removed
        k0 = next((i for i in range(m0.nkey) if m0.key(i).name == "home"), 0) if m0.nkey else None
        if k0 is not None:
            home_all = {m0.joint(j).name: float(m0.key_qpos[k0][m0.jnt_qposadr[j]]) for j in range(m0.njnt)
                        if m0.jnt_type[j] in (2, 3)}
        _strip_bodies(spec, info["strip"])
    if key in ARMS_V2:
        _limit_continuous(spec)
    for b1, b2 in info.get("exclude", []):
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
    if "add_site" in info:
        a = info["add_site"]
        body = next(b for b in spec.bodies if b.name == a["body"])
        pos = _auto_flange(spec, a["body"], a["zaxis"]) if a["pos"] == "auto" else a["pos"]
        body.add_site(name="attachment_site", pos=pos, quat=_quat_z_to(a["zaxis"]))
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
    if home_all is not None:
        home = [home_all[j] for j in joints]
    elif m.nkey:
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
