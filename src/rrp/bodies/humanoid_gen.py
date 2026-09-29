"""Procedural humanoid family `phum` (W13, D-138): primitive-geom floating-base humanoids from a seeded parameter vector.

Every body is a `Module` with the same `meta["legged"]` block as rrp.bodies.legged.menagerie_legged, so the tracker stack
(LeggedBinding, LeggedEnv, WarpTrackerEnv, tracker_validation) consumes it unchanged. Lineage `procedural_humanoid/v1`,
`synthetic=True`, torque limits `limits_source="procedural_scaling"` (declared scaling law below, not a manufacturer value).

Parameter space (research/tracks/humanoid.md section 1; research/splits/humanoid_v1.json):
  height H 0.35-1.8 m (log-uniform); mass M = 22 (H/1.2)^2.3 x U(0.75, 1.25) kg; thigh/shin ratio 0.8-1.25;
  torso/leg ratio 0.6-1.1; hip width 0.18-0.32 x H; foot length 0.12-0.2 x H; foot shape {box, capsule pair};
  leg DoF {5: no hip yaw, 6, 7: + toe}; arms {0, 3, 4, 7} DoF per arm; waist {0, 1, 3}; link-mass / CoM jitter +-15%.
SEALED region (never trained): thigh/shin > 1.15 AND leg DoF 7 AND H > 1.5, plus generator seeds >= 9,000,000.
Torque scaling law (per joint, N m): effort = f x M g L_leg with f ~ U: hip pitch 0.25-0.45, hip roll/yaw 0.7 x hip pitch,
knee 0.35-0.65, ankle 0.10-0.22, toe 0.05-0.1; arms 0.05-0.1 x M g L_arm; waist 0.3 x hip pitch. Ranges bracket the sourced
t1 / g1 / h1 values (knee 0.32-0.67, hip 0.24-0.44, ankle 0.09-0.22 of M g L_leg). PD gains: the declared auto rule of
rrp.bodies.legged (kp = clip(effort, 10, 300), kd = 0.025 kp).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from rrp.bodies.generators import Module, _base_spec

GEN_VERSION = "phum_v1"
LINEAGE = "procedural_humanoid/v1"
SEALED_SEED_MIN = 9_000_000
TRAIN_SEED_MAX = 1_000_000
G = mujoco.mjtGeom
HINGE = mujoco.mjtJoint.mjJNT_HINGE
LIM = mujoco.mjtLimited.mjLIMITED_TRUE


@dataclass
class PhumParams:
    seed: int
    height: float
    mass: float
    thigh_shin: float
    torso_leg: float
    hip_width_frac: float
    foot_len_frac: float
    foot_shape: str
    leg_dof: int
    arm_dof: int
    waist_dof: int
    mass_jitter: tuple
    f_hip: float
    f_knee: float
    f_ankle: float
    f_toe: float
    f_arm: float

    @property
    def topology(self) -> str:
        """Bodies with the same topology differ only in batchable model fields (sizes, masses, ranges, gains)."""
        return f"l{self.leg_dof}a{self.arm_dof}w{self.waist_dof}{self.foot_shape[0]}"


def in_sealed_region(p: PhumParams) -> bool:
    return p.thigh_shin > 1.15 and p.leg_dof == 7 and p.height > 1.5


def sample_params(seed: int) -> PhumParams:
    """Deterministic in `seed`. Training seeds (< TRAIN_SEED_MAX) never land in the sealed region: the draw is repeated with
    a derived stream until it leaves it (the rejection is part of the generator, so a seed always maps to one body)."""
    k = 0
    while True:
        r = np.random.default_rng([int(seed), 1729, k])
        H = float(math.exp(r.uniform(math.log(0.35), math.log(1.8))))
        p = PhumParams(
            seed=int(seed), height=H, mass=float(22.0 * (H / 1.2) ** 2.3 * r.uniform(0.75, 1.25)),
            thigh_shin=float(r.uniform(0.8, 1.25)), torso_leg=float(r.uniform(0.6, 1.1)),
            hip_width_frac=float(r.uniform(0.18, 0.32)), foot_len_frac=float(r.uniform(0.12, 0.2)),
            foot_shape=str(r.choice(["box", "capsule"])), leg_dof=int(r.choice([5, 6, 7])),
            arm_dof=int(r.choice([0, 3, 4, 7])), waist_dof=int(r.choice([0, 1, 3])),
            mass_jitter=tuple(float(x) for x in r.uniform(0.85, 1.15, 8)),
            f_hip=float(r.uniform(0.25, 0.45)), f_knee=float(r.uniform(0.35, 0.65)), f_ankle=float(r.uniform(0.10, 0.22)),
            f_toe=float(r.uniform(0.05, 0.1)), f_arm=float(r.uniform(0.05, 0.1)))
        if seed >= SEALED_SEED_MIN or not in_sealed_region(p):
            return p
        k += 1


def _auto(eff):
    from rrp.bodies.legged import _auto_gains
    return _auto_gains(eff)


def build(p: PhumParams, name: str | None = None) -> Module:
    from rrp.bodies.legged import add_imu, pd_servo
    name = name or f"phum_{p.seed}"
    H, M, g = p.height, p.mass, 9.81
    leg = 0.47 * H                                   # hip-to-sole length
    foot_h = 0.02 * H
    shin = (leg - foot_h) / (1 + p.thigh_shin)
    thigh = shin * p.thigh_shin
    torso = p.torso_leg * leg * 0.62
    hipw = p.hip_width_frac * H
    footL = p.foot_len_frac * H
    r_limb = 0.035 * H
    armL = 0.38 * H
    J = p.mass_jitter
    mfrac = dict(pelvis=0.15 * J[0], torso=0.30 * J[1], thigh=0.10 * J[2], shin=0.05 * J[3], foot=0.015 * J[4],
                 uarm=0.025 * J[5], farm=0.015 * J[6], head=0.06 * J[7])
    MgL, MgA = M * g * leg, M * g * armL
    s = _base_spec(name)
    z0 = leg + 0.02
    pelvis = s.worldbody.add_body(name="pelvis", pos=[0, 0, z0])
    pelvis.add_freejoint(name="root")
    pelvis.add_geom(name="pelvis_geom", type=G.mjGEOM_BOX, size=[0.35 * hipw, 0.5 * hipw + r_limb, 0.05 * H],
                    mass=M * mfrac["pelvis"], rgba=[0.3, 0.35, 0.45, 1])
    imu = add_imu(s, "pelvis")
    acts, feet, default, held, eff_tab = [], [], {}, [], {}

    def joint(body, jn, axis, rng, eff, q0, policy=True):
        body.add_joint(name=jn, type=HINGE, axis=axis, range=list(rng), limited=LIM, damping=0.02 * eff / 10 + 0.05,
                       armature=0.01 * (H / 1.2) ** 2)
        kp, kd = _auto(eff)
        pd_servo(s, jn, jn, kp, kd, eff, rng)
        default[jn] = float(q0)
        eff_tab[jn] = float(eff)
        (acts if policy else held).append(jn)

    # upper body: waist chain -> torso (+ head, arms)
    parent, wz = pelvis, 0.05 * H
    for i, (ax, rng) in enumerate([([0, 0, 1], [-0.8, 0.8]), ([1, 0, 0], [-0.3, 0.3]), ([0, 1, 0], [-0.3, 0.6])][:p.waist_dof]):
        wb = parent.add_body(name=f"waist{i}", pos=[0, 0, wz if i == 0 else 0])
        wb.add_geom(name=f"waist{i}_geom", type=G.mjGEOM_SPHERE, size=[0.6 * r_limb, 0, 0], mass=0.01 * M, contype=0, conaffinity=0)
        joint(wb, f"waist_{['yaw', 'roll', 'pitch'][i]}", ax, rng, 0.3 * p.f_hip * MgL, 0.0, policy=False)
        parent, wz = wb, 0.0
    tor = parent.add_body(name="torso", pos=[0, 0, wz])
    tor.add_geom(name="torso_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0.5 * r_limb, 0, 0, torso], size=[0.45 * hipw, 0, 0],
                 mass=M * mfrac["torso"], rgba=[0.35, 0.4, 0.5, 1])
    tor.add_geom(name="head_geom", type=G.mjGEOM_SPHERE, pos=[0, 0, torso + 0.07 * H], size=[0.06 * H, 0, 0],
                 mass=M * mfrac["head"], rgba=[0.35, 0.4, 0.5, 1])
    if p.arm_dof:
        axes = {3: [("shoulder_pitch", [0, 1, 0], [-2.5, 1.0], 0.0), ("shoulder_roll", [1, 0, 0], [-0.3, 1.5], 0.15),
                    ("elbow", [0, 1, 0], [-2.2, 0.0], -0.3)]}
        axes[4] = axes[3][:2] + [("shoulder_yaw", [0, 0, 1], [-1.3, 1.3], 0.0)] + axes[3][2:]
        axes[7] = axes[4] + [("wrist_roll", [0, 0, 1], [-1.5, 1.5], 0.0), ("wrist_pitch", [0, 1, 0], [-1.0, 1.0], 0.0),
                             ("wrist_yaw", [1, 0, 0], [-0.8, 0.8], 0.0)]
        for side, sg in (("left", 1), ("right", -1)):
            b = tor.add_body(name=f"{side}_shoulder", pos=[0, sg * (0.45 * hipw + 1.2 * r_limb), torso * 0.92])
            chain = axes[p.arm_dof]
            for i, (jn, ax, rng, q0) in enumerate(chain):
                rng_ = rng if sg > 0 or ax != [1, 0, 0] else [-rng[1], -rng[0]]
                q0_ = q0 if sg > 0 or ax != [1, 0, 0] else -q0
                if jn == "elbow":
                    b = b.add_body(name=f"{side}_forearm", pos=[0, 0, -0.5 * armL])
                    b.add_geom(name=f"{side}_forearm_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -0.45 * armL],
                               size=[0.7 * r_limb, 0, 0], mass=M * mfrac["farm"], rgba=[0.55, 0.55, 0.6, 1])
                elif i == 0:
                    b.add_geom(name=f"{side}_uarm_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -0.5 * armL],
                               size=[0.8 * r_limb, 0, 0], mass=M * mfrac["uarm"], rgba=[0.55, 0.55, 0.6, 1])
                joint(b, f"{side}_{jn}", ax, rng_, p.f_arm * MgA, q0_, policy=False)
    # legs
    kb = 0.35                                          # default knee bend (rad)
    for side, sg in (("left", 1), ("right", -1)):
        b = pelvis.add_body(name=f"{side}_hip", pos=[0, sg * 0.5 * hipw, -0.03 * H])
        b.add_geom(name=f"{side}_hip_geom", type=G.mjGEOM_SPHERE, size=[r_limb, 0, 0], mass=0.01 * M, contype=0, conaffinity=0)
        fh = p.f_hip * MgL
        chain = ([("hip_yaw", [0, 0, 1], [-0.6, 0.6], 0.0, 0.7 * fh)] if p.leg_dof >= 6 else []) + [
            ("hip_roll", [1, 0, 0], [-0.35, 0.5] if sg > 0 else [-0.5, 0.35], 0.0, 0.7 * fh),
            ("hip_pitch", [0, 1, 0], [-2.0, 0.8], -0.5 * kb, fh)]
        for jn, ax, rng, q0, eff in chain:
            joint(b, f"{side}_{jn}", ax, rng, eff, q0)
            if jn != "hip_pitch":
                b = b.add_body(name=f"{side}_{jn}_link", pos=[0, 0, 0])
                b.add_geom(name=f"{side}_{jn}_geom", type=G.mjGEOM_SPHERE, size=[0.8 * r_limb, 0, 0], mass=0.005 * M,
                           contype=0, conaffinity=0)
        b.add_geom(name=f"{side}_thigh_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -thigh], size=[r_limb, 0, 0],
                   mass=M * mfrac["thigh"], rgba=[0.6, 0.6, 0.65, 1])
        kn = b.add_body(name=f"{side}_shin", pos=[0, 0, -thigh])
        joint(kn, f"{side}_knee", [0, 1, 0], [-0.05, 2.4], p.f_knee * MgL, kb)
        kn.add_geom(name=f"{side}_shin_geom", type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -shin], size=[0.85 * r_limb, 0, 0],
                    mass=M * mfrac["shin"], rgba=[0.55, 0.55, 0.6, 1])
        an = kn.add_body(name=f"{side}_ankle", pos=[0, 0, -shin])
        an.add_geom(name=f"{side}_ankle_geom", type=G.mjGEOM_SPHERE, size=[0.6 * r_limb, 0, 0], mass=0.005 * M,
                    contype=0, conaffinity=0)
        joint(an, f"{side}_ankle_pitch", [0, 1, 0], [-0.9, 0.6], p.f_ankle * MgL, -0.5 * kb)
        ft = an.add_body(name=f"{side}_foot", pos=[0, 0, 0])
        joint(ft, f"{side}_ankle_roll", [1, 0, 0], [-0.35, 0.35], 0.7 * p.f_ankle * MgL, 0.0)
        heel, toe_len = 0.3 * footL, (0.7 * footL if p.leg_dof < 7 else 0.45 * footL)

        def sole(body, x0, x1, nm):
            cx, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
            if p.foot_shape == "box":
                body.add_geom(name=nm, type=G.mjGEOM_BOX, pos=[cx, 0, -0.5 * foot_h], size=[hx, 0.2 * footL, 0.5 * foot_h],
                              mass=M * mfrac["foot"] * (x1 - x0) / footL, rgba=[0.15, 0.15, 0.15, 1])
            else:
                for dy in (-0.12 * footL, 0.12 * footL):
                    body.add_geom(name=f"{nm}{'L' if dy > 0 else 'R'}", type=G.mjGEOM_CAPSULE,
                                  fromto=[x0 + 0.5 * foot_h, dy, -0.5 * foot_h, x1 - 0.5 * foot_h, dy, -0.5 * foot_h],
                                  size=[0.5 * foot_h, 0, 0], mass=0.5 * M * mfrac["foot"] * (x1 - x0) / footL,
                                  rgba=[0.15, 0.15, 0.15, 1])
        sole(ft, -heel, toe_len, f"{side}_sole")
        if p.leg_dof == 7:
            tb = ft.add_body(name=f"{side}_toe", pos=[toe_len, 0, 0])
            joint(tb, f"{side}_toe", [0, 1, 0], [-0.6, 0.6], p.f_toe * MgL, 0.0)
            sole(tb, 0.0, 0.25 * footL, f"{side}_toe_sole")
        ft.add_site(name=f"{side}_foot_touch_site", pos=[0.5 * (toe_len - heel), 0, -0.5 * foot_h],
                    size=[0.5 * (toe_len + heel) * 1.15 + 0.004, 0.2 * footL * 1.15 + 0.004, 0.5 * foot_h * 1.15 + 0.004],
                    type=G.mjGEOM_BOX)
        s.add_sensor(name=f"{side}_foot_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                     objname=f"{side}_foot_touch_site")
        feet.append(f"{side}_foot")
    # nominal height from the compiled default pose
    model = s.copy().compile()
    d = mujoco.MjData(model)
    for jn, q in default.items():
        d.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = q
    mujoco.mj_kinematics(model, d)
    fb = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in feet}
    fb |= {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sd}_toe") for sd in ("left", "right")} - {-1}
    lows = []
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] in fb and model.geom_contype[gi]:
            R = d.geom_xmat[gi].reshape(3, 3)
            c = d.geom_xpos[gi] + R @ model.geom_aabb[gi][:3]
            lows.append(c[2] - (np.abs(R) @ model.geom_aabb[gi][3:])[2])
    nominal = float(d.xpos[1][2] - min(lows))
    vmax = float(min(1.0, 0.8 * math.sqrt(H / 1.2)))
    period = float(0.8 * math.sqrt(H / 1.2))
    pitch = {sd: [f"{sd}_hip_pitch", f"{sd}_knee", f"{sd}_ankle_pitch"] for sd in ("left", "right")}
    sealed = p.seed >= SEALED_SEED_MIN or in_sealed_region(p)
    meta = dict(name=name, family="humanoid", synthetic=True, lineage=[LINEAGE, f"{LINEAGE}/{p.topology}", f"{LINEAGE}/{name}"],
                assemblies=[dict(id="body", kind="body", root_body="pelvis", frame=dict(site=imu["site"]),
                                 capabilities=["locomote"] + (["manipulate"] if p.arm_dof else []))],
                ports=[],
                controller=dict(kind="joint_targets", groups=[g_ for g_ in (
                    dict(name="legs", actuators=acts, semantic="joint_position", units="rad"),
                    dict(name="upper", actuators=held, semantic="joint_position", units="rad")) if g_["actuators"]]),
                params=dict(asdict(p), lengths=[nominal], topology=p.topology, gen_version=GEN_VERSION),
                actuator_limits="procedural_scaling", limits_source="procedural_scaling", sealed=bool(sealed),
                legged=dict(root_body="pelvis", imu=imu, foot_bodies=feet, foot_sites=[f"{sd}_foot_touch_site" for sd in ("left", "right")],
                            touch_sensors=[f"{sd}_foot_touch" for sd in ("left", "right")], policy_actuators=acts,
                            held_actuators=held, default_pose=default,
                            kp={a: _auto(eff_tab[a])[0] for a in acts + held}, kd={a: _auto(eff_tab[a])[1] for a in acts + held},
                            action_scale=0.25, command_ranges=dict(vx=[-0.3 * vmax, vmax], vy=[-0.25 * vmax, 0.25 * vmax],
                                                                   wz=[-0.6, 0.6]),
                            gait_period=period, min_height_frac=0.55, tilt_limit=0.7, kind="humanoid", nominal_height=nominal,
                            gait="biped", swing_height=float(0.08 * H / 1.2), pitch_actuators=pitch))
    return Module(s, meta)


def sealed_region_seeds(n: int, start: int = SEALED_SEED_MIN) -> list[int]:
    """The first n sealed generator seeds whose body lies in the sealed region (S5 targets; ~1% of sealed seeds)."""
    out, s = [], start
    while len(out) < n:
        if in_sealed_region(sample_params(s)):
            out.append(s)
        s += 1
    return out


def phum_body(seed: int) -> Module:
    return build(sample_params(int(seed)))
