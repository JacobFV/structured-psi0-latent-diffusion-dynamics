"""Compile a MuJoCo model (+ declared module metadata) into a typed RobotSpec.

Structural addresses come from the kinematic tree (child order by body id), so identity
does not depend on names. Mimic joints are detected from joint-equality constraints and
never counted as independent controls.
"""
from __future__ import annotations

import mujoco
import numpy as np

from rrp.contracts.robot import (RobotSpec, LinkSpec, JointSpec, ActuatorSpec, SensorSpec, AssemblySpec,
                                 FrameDef, AttachmentPort, ControllerContract, CommandGroup, TypedEdge)

JT = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
      int(mujoco.mjtJoint.mjJNT_BALL): "ball", int(mujoco.mjtJoint.mjJNT_FREE): "free"}
QW = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}
VW = {"hinge": 1, "slide": 1, "ball": 3, "free": 6}
SENS = {int(k): v for k, v in {mujoco.mjtSensor.mjSENS_JOINTPOS: ("joint_pos", "rad"), mujoco.mjtSensor.mjSENS_JOINTVEL: ("joint_vel", "rad/s"),
        mujoco.mjtSensor.mjSENS_TOUCH: ("touch", "N"), mujoco.mjtSensor.mjSENS_FORCE: ("force", "N"),
        mujoco.mjtSensor.mjSENS_TORQUE: ("torque", "Nm"), mujoco.mjtSensor.mjSENS_ACCELEROMETER: ("imu", "m/s2"),
        mujoco.mjtSensor.mjSENS_GYRO: ("imu", "rad/s"), mujoco.mjtSensor.mjSENS_FRAMEQUAT: ("imu", "quat")}.items()}


def robot_body_ids(model: mujoco.MjModel, prefix: str) -> list[int]:
    """Bodies belonging to the robot: named with prefix, plus unnamed descendants."""
    ids = []
    for b in range(1, model.nbody):
        name = model.body(b).name
        if name.startswith(prefix):
            ids.append(b)
    return ids


def structural_addresses(model: mujoco.MjModel, body_ids: list[int]) -> dict[int, str]:
    bset = set(body_ids)
    roots = [b for b in body_ids if model.body_parentid[b] not in bset]
    addr = {}
    for ri, r in enumerate(sorted(roots)):
        addr[r] = f"r{ri}"
        stack = [r]
        while stack:
            p = stack.pop()
            kids = sorted(b for b in body_ids if model.body_parentid[b] == p)
            for k, c in enumerate(kids):
                addr[c] = f"{addr[p]}/{k}"
                stack.append(c)
    return addr


def compile_robot_spec(model: mujoco.MjModel, meta: dict, prefix: str = "", *, name: str | None = None,
                       controller_rate_hz: float = 20.0, asset_source: dict | None = None) -> RobotSpec:
    bids = robot_body_ids(model, prefix)
    if not bids:
        raise ValueError(f"no bodies with prefix {prefix!r}")
    baddr = structural_addresses(model, bids)
    links, joints, acts, sensors, edges = [], [], [], [], []
    body_name_to_addr = {model.body(b).name: baddr[b] for b in bids}
    for b in bids:
        a = baddr[b]
        pj = None
        jids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        ext = np.zeros(3)
        vol = 0.0
        for g in geoms:
            ext = np.maximum(ext, np.abs(model.geom_pos[g]) + model.geom_rbound[g])
            vol += 4.0 / 3.0 * np.pi * model.geom_rbound[g] ** 3
        for k, j in enumerate(jids):
            jt = JT[int(model.jnt_type[j])]
            jaddr = f"{a}:j{k}"
            pj = jaddr if pj is None else pj
            rng = [float(x) for x in model.jnt_range[j]] if model.jnt_limited[j] else None
            joints.append(JointSpec(address=jaddr, name=model.joint(j).name, type=jt,
                                    parent_link=baddr.get(model.body_parentid[b]), child_link=a,
                                    axis=[float(x) for x in model.jnt_axis[j]], range=rng,
                                    damping=float(model.dof_damping[model.jnt_dofadr[j]]),
                                    armature=float(model.dof_armature[model.jnt_dofadr[j]]),
                                    qpos_width=QW[jt], qvel_width=VW[jt]))
        links.append(LinkSpec(address=a, name=model.body(b).name, parent_joint=pj,
                              mass=max(float(model.body_mass[b]), 1e-6),
                              inertia_diag=[max(float(x), 1e-9) for x in model.body_inertia[b]],
                              com=[float(x) for x in model.body_ipos[b]],
                              pos_in_parent=[float(x) for x in model.body_pos[b]],
                              quat_in_parent_wxyz=[float(x) for x in model.body_quat[b]],
                              geom_features=[float(len(geoms)), *[float(x) for x in ext], float(vol)]))
        par = model.body_parentid[b]
        if par in baddr:
            edges.append(TypedEdge(src=baddr[par], dst=a, type="parent_of"))
    jname_to_addr = {j.name: j.address for j in joints}
    # mimic constraints from joint equalities: eq obj1 = dependent, obj2 = driver
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] >= 0:
            dep, drv = model.joint(model.eq_obj1id[e]).name, model.joint(model.eq_obj2id[e]).name
            if dep in jname_to_addr and drv in jname_to_addr:
                for js in joints:
                    if js.name == dep:
                        js.mimic_of = jname_to_addr[drv]
                edges.append(TypedEdge(src=jname_to_addr[drv], dst=jname_to_addr[dep], type="mimics"))
    act_name_to_addr = {}
    for u in range(model.nu):
        nm = model.actuator(u).name
        if not nm.startswith(prefix):
            continue
        if model.actuator_trntype[u] != mujoco.mjtTrn.mjTRN_JOINT:
            jn = None
        else:
            jn = jname_to_addr.get(model.joint(model.actuator_trnid[u, 0]).name)
        is_pos = model.actuator_biastype[u] == mujoco.mjtBias.mjBIAS_AFFINE and model.actuator_biasprm[u, 1] < 0
        is_vel = model.actuator_biastype[u] == mujoco.mjtBias.mjBIAS_AFFINE and model.actuator_biasprm[u, 1] == 0 \
            and model.actuator_biasprm[u, 2] < 0
        kind = "position" if is_pos else ("velocity" if is_vel else "motor")
        addr = f"act:{jn or nm}"
        act_name_to_addr[nm] = addr
        acts.append(ActuatorSpec(address=addr, name=nm, kind=kind, joint=jn, gear=float(model.actuator_gear[u, 0]),
                                 ctrl_range=[float(x) for x in model.actuator_ctrlrange[u]] if model.actuator_ctrllimited[u] else None,
                                 force_range=[float(x) for x in model.actuator_forcerange[u]] if model.actuator_forcelimited[u] else None,
                                 kp=float(model.actuator_gainprm[u, 0]) if is_pos else None))
        if jn:
            edges.append(TypedEdge(src=addr, dst=jn, type="actuates"))
    for s in range(model.nsensor):
        nm = model.sensor(s).name
        if not nm.startswith(prefix):
            continue
        kind, units = SENS.get(int(model.sensor_type[s]), ("force", "raw"))
        mount = None
        if model.sensor_objtype[s] == mujoco.mjtObj.mjOBJ_SITE:
            mount = baddr.get(model.site_bodyid[model.sensor_objid[s]])
        elif model.sensor_objtype[s] == mujoco.mjtObj.mjOBJ_JOINT:
            mount = baddr.get(model.jnt_bodyid[model.sensor_objid[s]])
        sensors.append(SensorSpec(address=f"sen:{kind}:{nm[len(prefix):]}", name=nm, kind=kind, mount_link=mount,
                                  width=int(model.sensor_dim[s]), units=units, rate_hz=1.0 / model.opt.timestep))
    for c in range(model.ncam):
        nm = model.camera(c).name
        if nm.startswith(prefix):
            sensors.append(SensorSpec(address=f"sen:camera:{nm[len(prefix):]}", name=nm, kind="camera",
                                      mount_link=baddr.get(model.cam_bodyid[c]), width=3, units="rgb8", rate_hz=20.0))
    assemblies = []
    for asm in meta.get("assemblies", []):
        rb = prefix + asm["root_body"]
        root_addr = body_name_to_addr.get(rb)
        if root_addr is None:
            raise ValueError(f"assembly {asm['id']} root body {rb} missing")
        members = [ad for n, ad in body_name_to_addr.items() if ad == root_addr or ad.startswith(root_addr + "/")]
        site = prefix + asm["frame"]["site"]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise ValueError(f"assembly frame site {site} missing")
        assemblies.append(AssemblySpec(id=asm["id"], kind=asm["kind"], members=members,
                                       frame=FrameDef(link=baddr[model.site_bodyid[sid]],
                                                      pos=[float(x) for x in model.site_pos[sid]],
                                                      quat_wxyz=[float(x) for x in model.site_quat[sid]], site=site),
                                       capabilities=asm["capabilities"], parent_assembly=asm.get("parent")))
    ports = []
    for p in meta.get("ports", []):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + p["site"])
        if sid < 0:
            continue
        ports.append(AttachmentPort(id=p["id"], host_link=baddr[model.site_bodyid[sid]],
                                    pos=[float(x) for x in model.site_pos[sid]],
                                    quat_wxyz=[float(x) for x in model.site_quat[sid]],
                                    max_payload_kg=p["max_payload_kg"], max_module_extent_m=p["max_module_extent_m"],
                                    interface=p["interface"], allowed_module_kinds=p["allowed_module_kinds"],
                                    occupied_by=p.get("occupied_by")))
    groups = []
    for g in meta.get("controller", {}).get("groups", []):
        owned = [act_name_to_addr[prefix + a] for a in g["actuators"] if prefix + a in act_name_to_addr]
        if not owned:
            continue
        lo, hi = [], []
        for a in g["actuators"]:
            u = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + a)
            lo.append(float(model.actuator_ctrlrange[u, 0]))
            hi.append(float(model.actuator_ctrlrange[u, 1]))
        groups.append(CommandGroup(name=g["name"], width=len(owned), units=g["units"], semantic=g["semantic"],
                                   actuators=owned, lower=lo, upper=hi,
                                   hold="hold_last" if g["semantic"] == "gripper" else "hold_measured"))
    contracts = []
    if groups:
        contracts.append(ControllerContract(id=f"{meta.get('controller', {}).get('kind', 'joint_targets')}",
                                            version="jt-1.0", kind=meta.get("controller", {}).get("kind", "joint_targets"),
                                            rate_hz=controller_rate_hz, command_groups=groups,
                                            state_required=["qpos", "qvel"], body_specific=False))
    caps = sorted({c for a in assemblies for c in a.capabilities})
    spec = RobotSpec(name=name or meta.get("name", prefix.rstrip("_")), family=meta.get("family", "arm"),
                     asset_source=asset_source or {"kind": "procedural", "generator": "rrp.morphology.generators"},
                     lineage=list(meta.get("lineage", [])), floating_base=any(j.type == "free" for j in joints),
                     links=links, joints=joints, actuators=acts, sensors=sensors, assemblies=assemblies,
                     attachment_ports=ports, controller_contracts=contracts, typed_edges=edges,
                     capability_tags=caps, synthetic=bool(meta.get("synthetic", False)))
    return spec.with_hash()
