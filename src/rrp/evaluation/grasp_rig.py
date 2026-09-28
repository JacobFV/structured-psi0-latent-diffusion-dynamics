"""Grasp contact rig (python -m rrp.evaluation.grasp_rig v2 pg2 [friction_scale]) (W7, D-108): the real gripper modules (pg2 / tf3) on a vertical carriage, a cube between the fingers.
1. close (full command -> the actuator's force limit, the worst case a policy can command), 2. lift 10 cm with a
min-jerk profile (peak acc ~2.3 m/s^2), 3. hold, then ramp the cube mass x1.08 every 0.25 s until it slips (>5 mm
relative to the palm). Reports penetration (max -dist pad<->cube) during the held lift, pad normal forces, and the slip
onset vs the Coulomb prediction m* = mu * sum(N) / g. Usage: grasp_contact_rig.py [v1|v2] [pg2|tf3] [friction_scale]"""
from __future__ import annotations

import json
import math
import sys

import mujoco
import numpy as np

from rrp.bodies.fixtures import gripper_module, three_finger_module
from rrp.physics import grasp_contact as GC


def build(version: str, kind: str, half: float = 0.022, fscale: float = 1.0, yaw: float = 0.0):
    s = mujoco.MjSpec()
    s.option.timestep = 0.002
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10
    w = s.worldbody
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.01], pos=[0, 0, -0.01],
               friction=[1.0, 0.01, 0.001])
    car = w.add_body(name="carriage", pos=[0, 0, 0])
    car.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.1, 0.3],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=50.0)
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=2.0, contype=0, conaffinity=0)
    mod = gripper_module() if kind == "pg2" else three_finger_module()
    gs = mod.spec
    # tool frame: module z (palm -> fingertips) points DOWN; tcp site at z0 + 0.75/0.85 finger length
    L = 0.055
    tcp_off = 0.04 + L * (0.75 if kind == "pg2" else 0.85)
    fr = car.add_frame(pos=[0, 0, half + 0.004 + tcp_off], quat=[0, 1, 0, 0])
    s.attach(gs, prefix="r0_", frame=fr)
    s.add_actuator(name="lift_act", target="lift", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                   gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[20000] + [0] * 9,
                   biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -20000, -400] + [0] * 7)
    b = w.add_body(name="cube", pos=[0, 0, half + 0.0005], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    b.add_freejoint(name="cube_free")
    b.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3, density=500.0,
               friction=[1.2, 0.02, 0.002], condim=4)
    info = GC.apply(s, version)
    m = s.compile()
    if fscale != 1.0:
        m.geom_friction[:] *= fscale
    return m, info


def run(version="v2", kind="pg2", fscale=1.0, t_close=0.8, verbose=False, yaw=0.0):
    m, info = build(version, kind, fscale=fscale, yaw=yaw)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "r0_act_grip")
    lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
    cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r0_palm")
    cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    pads = [g for g in range(m.ngeom) if GC.PAD_RE.search(m.geom(g).name or "")]
    closed = m.actuator_ctrlrange[grip][0] if kind == "pg2" else m.actuator_ctrlrange[grip][1]
    opened = m.actuator_ctrlrange[grip][1] if kind == "pg2" else -0.1
    d.ctrl[grip] = opened
    mu = float(max(m.geom_friction[pads[0]][0], m.geom_friction[cg][0]) if m.geom_priority[pads[0]] == m.geom_priority[cg]
               else (m.geom_friction[pads[0]][0] if m.geom_priority[pads[0]] > m.geom_priority[cg] else m.geom_friction[cg][0]))

    def contacts():
        pen, N, touching = 0.0, 0.0, set()
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if cg in (c.geom1, c.geom2) and (c.geom1 in pads or c.geom2 in pads):
                pen = max(pen, -float(c.dist))
                mujoco.mj_contactForce(m, d, i, f)
                N += abs(f[0])
                touching.add(c.geom1 if c.geom2 == cg else c.geom2)
        return pen, N, len(touching)

    def rel():
        return d.xpos[cube] - d.xpos[palm]

    # 1. settle, close
    T = 0.0
    for _ in range(int(0.2 / dt)):
        mujoco.mj_step(m, d)
    for k in range(int(t_close / dt)):
        a = min(1.0, k * dt / 0.4)
        d.ctrl[grip] = opened + (closed - opened) * a
        mujoco.mj_step(m, d)
    pen_close, N_close, nt = contacts()
    rel0 = rel().copy()
    # 2. lift 10 cm, min-jerk over 0.5 s, then hold 0.5 s
    pens, Ns = [], []
    Tl = 0.5
    for k in range(int((Tl + 0.5) / dt)):
        u = min(1.0, k * dt / Tl)
        d.ctrl[lift] = 0.10 * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
        mujoco.mj_step(m, d)
        p, n_, _ = contacts()
        pens.append(p)
        Ns.append(n_)
    lift_slip = float(np.linalg.norm(rel() - rel0))
    held = d.xpos[cube][2] > 0.08
    pen_hold, N_hold, nt_hold = contacts()
    # 3. mass ramp until slip
    m0 = float(m.body_mass[cube])
    I0 = m.body_inertia[cube].copy()
    base = rel().copy()
    slip_mass, N_at_slip = None, None
    N_prev = N_hold
    scale = 1.0
    for step in range(200):
        scale *= 1.08
        m.body_mass[cube] = m0 * scale
        m.body_inertia[cube] = I0 * scale
        Nw = []
        for _ in range(int(0.25 / dt)):
            mujoco.mj_step(m, d)
            Nw.append(contacts()[1])
        if np.linalg.norm(rel() - base) > 0.005 or d.xpos[cube][2] < 0.02:
            slip_mass = m0 * scale
            N_at_slip = float(np.median(Nw))
            break
        N_prev = float(np.median(Nw))
    g = 9.81
    pred = None if N_at_slip is None else mu * N_prev / g
    out = dict(version=info["version"], gripper=kind, friction_scale=fscale, cube_yaw_deg=round(math.degrees(yaw), 1), mu_pad_obj=round(mu, 4), cube_mass_kg=round(m0, 4),
               grip_force_limit=float(np.abs(m.actuator_forcerange[grip]).max()),
               n_fingers_touching=nt, pen_after_close_mm=round(1000 * pen_close, 2),
               pen_max_during_lift_mm=round(1000 * max(pens), 2), pen_hold_mm=round(1000 * pen_hold, 2),
               normal_force_sum_hold_N=round(N_hold, 2), lift_held=bool(held), lift_slip_mm=round(1000 * lift_slip, 2),
               slip_mass_kg=None if slip_mass is None else round(slip_mass, 4),
               coulomb_pred_mass_kg=None if pred is None else round(pred, 4),
               slip_ratio=None if (pred is None or not pred) else round(slip_mass / pred, 3),
               slip_load_over_cube_weight=None if slip_mass is None else round(slip_mass / m0, 1))
    return out


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "v2"
    k = sys.argv[2] if len(sys.argv) > 2 else "pg2"
    fs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    print(json.dumps(run(v, k, fs)))


def palm_press(version="v2.1", kind="tf3", depth=0.04, T=0.6):
    """D-118: the carriage drives the (open) gripper DOWN onto the cube top, `depth` below first palm contact, like a
    noisy descent; returns the max palm/robot<->cube and cube<->table penetration (the lift actuator is stiff: 20 kN/m)."""
    m, info = build(version, kind)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "r0_act_grip")
    cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    rob = {b for b in range(m.nbody) if (m.body(b).name or "").startswith("r0_")}
    d.ctrl[grip] = m.actuator_ctrlrange[grip][1] if kind == "pg2" else -0.1
    # the tcp starts at the grasp height; the palm is ~(finger reach) above the cube top: lower until well past contact
    L = 0.055 * (0.75 if kind == "pg2" else 0.85)
    target = -(L - 0.004 + depth)
    pr = pt = 0.0
    for _ in range(int(0.4 / dt)):          # open the fingers first (the module starts closed, overlapping the cube)
        mujoco.mj_step(m, d)
    for k in range(int((T + 0.5) / dt)):
        u = min(1.0, k * dt / T)
        d.ctrl[lift] = target * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            gs = (c.geom1, c.geom2)
            if cg in gs:
                other = gs[1] if gs[0] == cg else gs[0]
                if m.geom_bodyid[other] in rob:
                    pr = max(pr, -float(c.dist))
                elif other == table:
                    pt = max(pt, -float(c.dist))
    return dict(version=info["version"], gripper=kind, press_depth_m=depth, robot_cube_pen_mm=round(1000 * pr, 2),
                cube_table_pen_mm=round(1000 * pt, 2), robot_geoms_stiffened=info.get("robot_geoms", 0))
