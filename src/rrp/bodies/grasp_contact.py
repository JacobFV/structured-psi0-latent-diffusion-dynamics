"""Versioned finger/object ("grasp") contact models for the arm worlds (W7, D-108).

`grasp_v1` (legacy, still the default): the physics of every arm result up to 2026-09-27.
    Finger pads (parallel pg2: boxes; three-finger tf3: capsules) friction [1.5, 0.02, 0.002], condim 4, priority 0;
    objects friction [1.2, 0.02, 0.002], condim 4; the pad-object contact takes the element-wise MAX (1.5).
    Default soft contact solref [0.02, 1] (tf3, objects) / [0.01, 1] (pg2 pads), solimp [0.9, 0.95, 0.001] /
    [0.95, 0.99, 0.001]. MuJoCo's soft constraint scales its stiffness with the constraint's effective mass, and the
    cube weighs 42 g, so a 40-145 N squeeze sinks the fingers 7-24 mm into it (median, measured by the teacher
    quality harness); a gripper then holds by interpenetration (geometric caging) rather than friction, which is why
    W6 found grasps insensitive to friction x0.05 (D-108).
    Gripper force: pg2 40 N per finger; tf3 8 N m per hinge = ~145 N at the 55 mm fingertip.

`grasp_v2`:
    * Pads get priority 1, so the PAD's friction/condim/solref/solimp define every pad-object contact (no max-mixing).
    * Pad friction [1.0, 0.004, 0.0001] (rubber-like pad on a hard object: mu ~ 0.8-1.2); condim 4, i.e. torsional
      friction with a 4 mm patch radius (a pad resists twisting the object about the grip axis), no rolling term.
    * Objects: friction [0.9, 0.004, 0.0001], condim 4 (object-table and object-object contacts).
    * Stiff, nearly incompressible pad and object contact: solref [0.004, 1.0] (time constant = 2 physics steps,
      critically damped) and solimp [0.99, 0.999, 0.0005, 0.5, 2] (full impedance after 0.5 mm). The soft-constraint
      stiffness grows ~ 1/timeconst^2 and ~ d/(1-d), so penetration under the same squeeze drops by orders of
      magnitude; measured by the rig (rrp.evaluation.grasp_rig) and tests/unit/test_grasp_contact.py.
    * Realistic grip force (sourced, see the archived armexpert note, "grasp contact v2" (D-126)): pg2 stays at 40 N per
      finger; tf3 hinge torque 2.2 N m = 40 N at the 55 mm fingertip (was 145 N).
    * margin/gap 0 (no action at a distance).
The world options (elliptic cone, impratio 10, dt 2 ms, implicitfast) are already set by every arm module and are not
changed. The version is written into the model as a text element `grasp_contact_version` (grasp_v2 only; v1 models
are byte-identical to the legacy build) and read by rrp.contracts.provenance.physics_provenance.
"""
from __future__ import annotations

import os
import re

import mujoco

ENV = "RRP_GRASP_CONTACT"
TEXT = "grasp_contact_version"
PAD_RE = re.compile(r"(^|_)(pad_[ab]|fpad_\d+)$")

GRASP_MODELS = {
    "v1": dict(version="grasp_v1"),
    "v2": dict(version="grasp_v2",
               pad=dict(friction=[1.0, 0.004, 0.0001], condim=4, priority=1, solref=[0.004, 1.0],
                        solimp=[0.99, 0.999, 0.0005, 0.5, 2.0], margin=0.0, gap=0.0),
               obj=dict(friction=[0.9, 0.004, 0.0001], condim=4, solref=[0.004, 1.0],
                        solimp=[0.99, 0.999, 0.0005, 0.5, 2.0], margin=0.0, gap=0.0),
               grip_force=dict(parallel_N=40.0, three_finger_Nm=2.2)),
}
# grasp_v2.1 (D-118): grasp_v2 + the same stiff contact (solref/solimp, margin/gap 0) on EVERY other robot geom that can
# touch a task object (palm, wrist, links). Diagnosis: in v5dart DART episodes the object penetration came from the PALM
# (default soft contact, solref 0.02) pressed onto the cube during the noisy descent (up to 68 mm), not from the pads.
# Pads keep their friction priority; other robot geoms keep their own friction/condim/priority.
GRASP_MODELS["v2.1"] = dict(GRASP_MODELS["v2"], version="grasp_v2.1",
                            robot=dict(solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.0005, 0.5, 2.0], margin=0.0, gap=0.0))
ROBOT_BODY_RE = re.compile(r"^r\d+_")
# research candidates for D-118 (not for data until verified): world support geoms (table/floor) stiff too; stiffer robot
_STIFF = dict(solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.0005, 0.5, 2.0], margin=0.0, gap=0.0)
GRASP_MODELS["v2.1w"] = dict(GRASP_MODELS["v2.1"], version="grasp_v2.1w_candidate", world=dict(_STIFF))
GRASP_MODELS["v2.1h"] = dict(GRASP_MODELS["v2.1"], version="grasp_v2.1h_candidate", world=dict(_STIFF),
                             robot=dict(_STIFF, solimp=[0.999, 0.9999, 0.0005, 0.5, 2.0]),
                             obj=dict(GRASP_MODELS["v2"]["obj"], solimp=[0.999, 0.9999, 0.0005, 0.5, 2.0]))


def resolve(v: str | None = None) -> str:
    """'v1' | 'v2' (accepts 'grasp_v1'/'grasp_v2'); None -> $RRP_GRASP_CONTACT or 'v1'."""
    c = v if v is not None else os.environ.get(ENV, "v1")
    c = str(c).lower().replace("grasp_", "").replace("v2_1", "v2.1").replace("_candidate", "")
    if c not in GRASP_MODELS:
        raise ValueError(f"unknown grasp contact model {v!r}; known: {sorted(GRASP_MODELS)}")
    return c


def version_str(v: str | None = None) -> str:
    return GRASP_MODELS[resolve(v)]["version"]


def _free_bodies(spec: mujoco.MjSpec) -> set[str]:
    out = set()
    for b in spec.bodies:
        if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints):
            out.add(b.name)
    return out


def apply(spec: mujoco.MjSpec, version: str | None = None) -> dict:
    """Apply a grasp contact version to an arm-world spec BEFORE compile. v1: no change at all (legacy bytes).
    Returns a summary of what was changed (for scenario meta)."""
    c = resolve(version)
    if c == "v1":
        return dict(version="grasp_v1", changed=0)
    cm = GRASP_MODELS[c]
    free = _free_bodies(spec)
    n_pad = n_obj = n_act = n_rob = 0
    for g in spec.geoms:
        name = g.name or ""
        body = g.parent.name if g.parent is not None else ""
        if PAD_RE.search(name):
            p = cm["pad"]
            n_pad += 1
        elif body in free:
            p = cm["obj"]
            n_obj += 1
        elif "robot" in cm and ROBOT_BODY_RE.match(body or "") and (g.contype or g.conaffinity):
            p = cm["robot"]
            n_rob += 1
        elif "world" in cm and body == "world" and (g.contype or g.conaffinity):
            p = cm["world"]
        else:
            continue
        if "friction" in p:
            g.friction = list(p["friction"])
        if "condim" in p:
            g.condim = p["condim"]
        g.solref = list(p["solref"])
        g.solimp = list(p["solimp"])
        g.margin = p["margin"]
        g.gap = p["gap"]
        if "priority" in p:
            g.priority = p["priority"]
    for a in spec.actuators:
        if not (a.name or "").endswith("act_grip"):
            continue
        tgt = a.target or ""
        if "hinge" in tgt:          # three-finger: torque at the actuated hinge
            f = cm["grip_force"]["three_finger_Nm"]
        else:                       # parallel: force on the slide
            f = cm["grip_force"]["parallel_N"]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-f, f]
        n_act += 1
    if not any((t.name or "") == TEXT for t in spec.texts):
        spec.add_text(name=TEXT, data=cm["version"])
    return dict(version=cm["version"], pads=n_pad, objects=n_obj, robot_geoms=n_rob, grip_actuators=n_act)


def model_grasp_version(model: mujoco.MjModel) -> str | None:
    """grasp contact version embedded in a compiled model; None for legacy (grasp_v1) models."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, TEXT)
    if tid < 0:
        return None
    adr, n = model.text_adr[tid], model.text_size[tid]
    return bytes(model.text_data[adr:adr + n - 1]).decode()
