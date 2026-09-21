"""Module attachment at declared ports with physical validation and lineage.

Procedure: validate port/module compatibility -> MjSpec.attach (namespaces all
bodies/joints/actuators/sensors/equalities/cameras with a prefix and rebinds references)
-> merge metadata -> compile -> validate masses/inertias/unique names/finite dynamics/
initial penetration/holding stability -> fresh spec hash + recorded lineage.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from rrp.contracts.robot import RobotSpec, LinkSpec
from .compiler import compile_robot_spec
from .generators import Module


class AttachmentError(ValueError):
    code = "attachment_invalid"


@dataclass
class Assembled:
    spec: mujoco.MjSpec
    meta: dict
    model: mujoco.MjModel
    robot_spec: RobotSpec
    validation: dict

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.robot_spec.joints]

    @property
    def rigid_links(self) -> list[LinkSpec]:
        return self.robot_spec.links


def module_mass(m: Module) -> float:
    model = m.spec.copy().compile()
    return float(model.body_mass.sum())


def module_extent(m: Module) -> float:
    model = m.spec.copy().compile()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    pts = [d.geom_xpos[g] for g in range(model.ngeom)]
    r = [np.linalg.norm(p) + model.geom_rbound[g] for g, p in enumerate(pts)]
    return float(max(r)) if r else 0.0


def attach(body: Module | Assembled, module: Module, port_id: str, *, prefix: str | None = None,
           validate_steps: int = 400) -> Assembled:
    host_spec = body.spec.copy()
    meta = copy.deepcopy(body.meta)
    ports = {p["id"]: p for p in meta.get("ports", [])}
    if port_id not in ports:
        raise AttachmentError(f"port {port_id!r} not declared on {meta.get('name')}")
    port = ports[port_id]
    if port.get("occupied_by"):
        raise AttachmentError(f"port {port_id} already occupied by {port['occupied_by']}")
    kind = module.meta.get("module_kind", "gripper")
    if kind not in port["allowed_module_kinds"]:
        raise AttachmentError(f"module kind {kind} not allowed at port {port_id}")
    mass = module_mass(module)
    if mass > port["max_payload_kg"]:
        raise AttachmentError(f"module mass {mass:.3f} kg exceeds port payload {port['max_payload_kg']} kg")
    ext = module_extent(module)
    if ext > port["max_module_extent_m"]:
        raise AttachmentError(f"module extent {ext:.3f} m exceeds port limit {port['max_module_extent_m']} m")
    site = next((s for s in host_spec.sites if s.name == port["site"]), None)
    if site is None:
        raise AttachmentError(f"port site {port['site']} missing in host spec")
    pre = prefix or f"{port_id}_"
    host_spec.attach(module.spec.copy(), prefix=pre, site=site)
    # merged metadata with namespaced module references
    mm = copy.deepcopy(module.meta)
    for a in mm.get("assemblies", []):
        a["root_body"] = pre + a["root_body"]
        a["frame"]["site"] = pre + a["frame"]["site"]
        a["parent"] = meta.get("assemblies", [{}])[0].get("id")
    meta["assemblies"] = meta.get("assemblies", []) + mm.get("assemblies", [])
    groups = meta.setdefault("controller", {}).setdefault("groups", [])
    for g in mm.get("controller", {}).get("groups", []):
        g = dict(g)
        g["actuators"] = [pre + a for a in g["actuators"]]
        groups.append(g)
    for k in ("touch_sensors", "cameras"):
        meta[k] = meta.get(k, []) + [pre + x for x in mm.get(k, [])]
    if mm.get("width_sensor"):
        meta["width_sensor"] = pre + mm["width_sensor"]
    for k in ("stroke",):
        if k in mm:
            meta[k] = mm[k]
    meta["gripper_params"] = mm.get("params")
    ports[port_id]["occupied_by"] = mm.get("name")
    meta["ports"] = list(ports.values())
    meta["lineage"] = list(meta.get("lineage", [])) + list(mm.get("lineage", []))
    meta["composition"] = meta.get("composition", []) + [dict(port=port_id, module=mm.get("name"), prefix=pre)]
    meta["name"] = f"{meta.get('name')}+{mm.get('name')}"
    try:
        model = host_spec.copy().compile()
    except ValueError as e:
        raise AttachmentError(f"compile failed after attach: {e}") from e
    validation = validate_physics(model, steps=validate_steps)
    if not validation["ok"]:
        raise AttachmentError(f"physics validation failed: {validation['failures']}")
    rs = compile_robot_spec(model, meta, prefix="", name=meta["name"])
    return Assembled(host_spec, meta, model, rs, validation)


def validate_physics(model: mujoco.MjModel, steps: int = 400) -> dict:
    """Finite dynamics, positive inertia, unique names, bounded motion under hold commands,
    and no deep initial penetration between non-adjacent bodies."""
    failures = []
    names = [model.joint(j).name for j in range(model.njnt)] + [model.actuator(u).name for u in range(model.nu)]
    if len(names) != len(set(names)):
        failures.append("duplicate_names")
    for b in range(1, model.nbody):
        if model.body_mass[b] <= 0 and model.body_dofnum[b] > 0:
            failures.append(f"nonpositive_mass:{model.body(b).name}")
        if np.any(model.body_inertia[b] < 0):
            failures.append(f"negative_inertia:{model.body(b).name}")
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    # hold commands: position actuators hold current joint positions
    for u in range(model.nu):
        if model.actuator_trntype[u] == mujoco.mjtTrn.mjTRN_JOINT:
            j = model.actuator_trnid[u, 0]
            d.ctrl[u] = np.clip(d.qpos[model.jnt_qposadr[j]], *model.actuator_ctrlrange[u]) \
                if model.actuator_ctrllimited[u] else d.qpos[model.jnt_qposadr[j]]
    # every contact MuJoCo generates is physically active (its own filters already applied,
    # including the case where a parent is welded to the world and NOT filtered)
    max_pen = 0.0
    for c in range(d.ncon):
        max_pen = max(max_pen, -d.contact[c].dist)
    if max_pen > 0.01:
        failures.append(f"initial_penetration:{max_pen:.4f}")
    q0 = d.qpos.copy()
    maxv = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failures.append("nonfinite_state")
            break
        maxv = max(maxv, float(np.abs(d.qvel).max()) if model.nv else 0.0)
    drift = float(np.abs(d.qpos - q0).max()) if model.nq else 0.0
    if maxv > 50:
        failures.append(f"unstable_velocity:{maxv:.1f}")
    return dict(ok=not failures, failures=failures, max_initial_penetration=max_pen, hold_max_qvel=maxv,
                hold_drift=drift, total_mass=float(model.body_mass.sum()))
