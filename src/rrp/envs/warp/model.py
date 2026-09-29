"""Legged model building for the batched GPU envs (MuJoCo Warp): the body on its floor with EXPLICIT, recorded model
adaptations, and the mujoco_warp import guard. Kept from the D-126 #15 / W13 bake-off prototypes (rrp.envs.mjx_legged,
rrp.envs.warp_legged), whose MJX / bake-off stepping classes were deleted in D-140 S4; `build_model(..., adapt=[...])`
applies only the named adaptations and every result records them, so a changed physics is never silent.
"""
from __future__ import annotations

import mujoco

TRACKER_HZ = 50.0


INSTALL_HINT = ("rrp.envs.warp needs mujoco_warp + warp-lang (not rrp dependencies); peer: "
                "PYTHONPATH=src:$HOME/work/ext/pylibs/mjwarp")


def _wp():
    try:
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as e:
        raise ImportError(f"{INSTALL_HINT} ({e})") from e
    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    return wp, mjw



def _pyramidal(m):
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL


def _no_robot_self_collision(m):
    # robot geoms collide with the floor only (floor keeps contype/conaffinity 1): robot contype 2, conaffinity 1
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for g in range(m.ngeom):
        if g != floor and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
    m.geom_contype[floor], m.geom_conaffinity[floor] = 1, 2


def _leg_cross_collision(m, ground=None):
    """W13: robot collides with the ground AND left-leg geoms collide with right-leg geoms (legs cannot pass through each
    other, the failure mode of `no_self_collision` policies under pushes); every other robot-robot pair is off.
    Bits: ground c=1 a=14; other robot c=2 a=1; left leg c=4 a=9; right leg c=8 a=5. Leg sides come from the leg actuators'
    joint bodies (rrp.envs.morph_obs.slot_of) and all their descendant bodies."""
    from rrp.envs.mujoco.morph_obs import slot_of
    ground = set(ground if ground is not None else [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")]) - {-1}
    side_root = {}
    for a in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        so = slot_of(nm.split("_", 1)[1] if nm.startswith("r0_") else nm)
        if so is None or m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        b = int(m.jnt_bodyid[m.actuator_trnid[a, 0]])
        depth = lambda x: 0 if x == 0 else 1 + depth(int(m.body_parentid[x]))
        if so[0] not in side_root or depth(b) < depth(side_root[so[0]]):
            side_root[so[0]] = b
    side_of = {}
    for b in range(m.nbody):
        p = b
        while p > 0:
            for sd, r in side_root.items():
                if p == r:
                    side_of[b] = sd
            if b in side_of:
                break
            p = int(m.body_parentid[p])
    for g in range(m.ngeom):
        if g in ground:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 14
        elif m.geom_contype[g] or m.geom_conaffinity[g]:
            sd = side_of.get(int(m.geom_bodyid[g]))
            m.geom_contype[g], m.geom_conaffinity[g] = {"left": (4, 9), "right": (8, 5)}.get(sd, (2, 1))
            if sd is not None:
                m.geom_margin[g] = 0.0      # mujoco_warp: no margin on MULTICCD pairs (apollo soles had 0.5 mm)


def _condim3(m):
    m.geom_condim[m.geom_condim > 3] = 3


def _newton_small(m):
    m.opt.iterations = min(int(m.opt.iterations), 4)
    m.opt.ls_iterations = min(int(m.opt.ls_iterations), 8)


def _solver10(m):
    m.opt.iterations = min(int(m.opt.iterations), 10)
    m.opt.ls_iterations = min(int(m.opt.ls_iterations), 20)


ADAPTATIONS = {
    "pyramidal_cone": ("contact_v2 elliptic cone -> pyramidal", _pyramidal),
    "no_self_collision": ("robot geoms collide with the floor only (no robot-robot contacts)", _no_robot_self_collision),
    "leg_cross_collision": ("robot collides with the ground, and left-leg geoms with right-leg geoms; other self-contacts off",
                            _leg_cross_collision),
    "condim3": ("condim 6 (torsional + rolling friction) -> 3", _condim3),
    "few_solver_iters": ("solver iterations <= 4, line-search <= 8 (MJX-typical)", _newton_small),
    "solver_iters_10": ("solver iterations <= 10, line-search <= 20 (W13 P1a)", _solver10),
}


def build_model(body: str, contact: str = "v2", adapt=()):
    """(model, meta, binding, adaptations) for `body` on the flat floor of `contact`. `adapt`: names from ADAPTATIONS."""
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    m, _, meta = standalone_model(legged_body(body), contact=contact)
    done = []
    for a in adapt:
        if a not in ADAPTATIONS:
            raise KeyError(f"unknown model adaptation {a!r}; known {sorted(ADAPTATIONS)}")
        ADAPTATIONS[a][1](m)
        done.append(dict(name=a, description=ADAPTATIONS[a][0]))
    return m, meta, LeggedBinding(m, meta), done


def substeps_of(m) -> int:
    return max(1, int(round(1.0 / (TRACKER_HZ * m.opt.timestep))))


def default_data(m, b) -> mujoco.MjData:
    d = mujoco.MjData(m)
    b.set_default(d, yaw=0.0)
    mujoco.mj_forward(m, d)
    return d
