"""MJX legged stepping: FEASIBILITY PROTOTYPE (D-126, roadmap #15 "MJX for throughput"). Not used by any training,
collection or evaluation path; nothing imports it.

It steps the legged tracker physics (a body from rrp.bodies.legged on the contact_v1/v2 floor, standalone_model, the
bounded PD servos addressed through LeggedBinding) with mujoco.mjx, batched with jax.vmap under jit, and compares it with
the C MuJoCo reference on the SAME compiled model.

jax / mujoco-mjx are NOT dependencies of rrp. They live in an isolated venv (peer: ~/work/ext/venvs/mjx, built with
    uv venv --python /usr/bin/python3.12 ~/work/ext/venvs/mjx
    uv pip install --python ~/work/ext/venvs/mjx/bin/python "mujoco==3.14.0" "mujoco-mjx==3.14.0" "jax[cuda13]" numpy pydantic
). This module imports without them; every function that needs them raises an ImportError saying so.

Model adaptations: MJX (JAX implementation) does not support every MuJoCo feature. `build_model(..., adapt=[...])`
applies ONLY the adaptations explicitly named (ADAPTATIONS), and every result records them (`adaptations`) together with
their effect on the C reference (`adaptation_effect_max_dqpos`), so a changed physics is never silent. The default is no
adaptation.

CLI (JSON on stdout):
    python -m rrp.envs.mjx_legged --body go2 [--contact v2] [--device cpu|gpu] [--x64] [--ticks 50]
        [--n-envs 1,64,512] [--tp-ticks 20] [--adapt a,b]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import mujoco
import numpy as np

PROTOTYPE_VERSION = "mjx_legged_proto_v0"
SOURCE_LABEL = "prototype:mjx_feasibility (not a training/eval path)"
TRACKER_HZ = 50.0
INSTALL_HINT = ("rrp.envs.mjx_legged needs jax and mujoco-mjx (same version as mujoco). They are not rrp dependencies: "
                "use the isolated venv ~/work/ext/venvs/mjx (see the module docstring).")


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
    except ImportError as e:            # explicit, never silent: the prototype cannot run without them
        raise ImportError(f"{INSTALL_HINT} ({e})") from e
    return jax, jnp, mjx


# explicit model adaptations (name -> (description, function(model))); applied only when requested
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
    from rrp.envs.morph_obs import slot_of
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
    from rrp.envs.legged_core import LeggedBinding
    m, _, meta = standalone_model(legged_body(body), contact=contact)
    done = []
    for a in adapt:
        if a not in ADAPTATIONS:
            raise KeyError(f"unknown MJX adaptation {a!r}; known {sorted(ADAPTATIONS)}")
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


def open_loop_targets(b, t: int, amp: float = 0.15) -> np.ndarray:
    """Deterministic open-loop joint targets (default pose + a clock-phased oscillation; no policy). Used by parity and
    throughput so both sides see identical commands."""
    ph = 2 * math.pi * t / (TRACKER_HZ * b.period)
    s = np.sin(ph + np.arange(b.n) * 0.7)
    return np.clip(b.q0 + amp * s, b.lo, b.hi)


class MjxLegged:
    """n_envs copies of one legged model stepped by mjx at the 50 Hz tracker rate (PD targets held over the substeps)."""

    def __init__(self, body: str, n_envs: int, contact: str = "v2", adapt=(), device=None):
        jax, jnp, mjx = _jax()
        self.jax, self.jnp, self.mjx = jax, jnp, mjx
        self.m, self.meta, self.b, self.adaptations = build_model(body, contact, adapt)
        self.body, self.n = body, int(n_envs)
        self.device = device
        self.mx = mjx.put_model(self.m, device=device) if device is not None else mjx.put_model(self.m)
        self.substeps = substeps_of(self.m)
        b = self.b
        pol = jnp.asarray(b.pol_act)
        held = jnp.asarray(b.held_act) if len(b.held_act) else None
        q0h = jnp.asarray(b.q0_held) if len(b.held_act) else None
        mx, ns = self.mx, self.substeps

        def one(d, tgt):
            ctrl = d.ctrl.at[pol].set(tgt)
            if held is not None:
                ctrl = ctrl.at[held].set(q0h)
            d = d.replace(ctrl=ctrl)
            return jax.lax.fori_loop(0, ns, lambda _, dd: mjx.step(mx, dd), d)

        self._step = jax.jit(jax.vmap(one))
        self.data = None

    def reset(self, key=None, noise: float = 0.0):
        """Default pose (LeggedBinding.set_default on a host MjData) broadcast to n_envs; optional U(-noise, noise) joint
        noise per env from a jax PRNG key."""
        jax, jnp, mjx = self.jax, self.jnp, self.mjx
        d0 = default_data(self.m, self.b)
        dx = mjx.put_data(self.m, d0, device=self.device) if self.device is not None else mjx.put_data(self.m, d0)
        dx = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.n,) + x.shape) if hasattr(x, "shape") else x, dx)
        if noise and key is not None:
            qadr = jnp.asarray(self.b.pol_qadr)
            dq = jax.random.uniform(key, (self.n, len(self.b.pol_qadr)), minval=-noise, maxval=noise)
            dx = dx.replace(qpos=dx.qpos.at[:, qadr].add(dq))
        self.data = dx
        return dx

    def step(self, targets):
        """targets: (n_envs, n_policy_actuators) joint position targets. Returns (qpos, qvel) after one 50 Hz tick."""
        self.data = self._step(self.data, self.jnp.asarray(targets, dtype=self.data.qpos.dtype))
        return self.data.qpos, self.data.qvel


def c_rollout(m, b, ticks: int, targets_fn=None) -> np.ndarray:
    """C MuJoCo reference: qpos per tick, (ticks + 1, nq)."""
    targets_fn = targets_fn or (lambda t: open_loop_targets(b, t))
    d = default_data(m, b)
    ns = substeps_of(m)
    out = [d.qpos.copy()]
    for t in range(ticks):
        d.ctrl[b.pol_act] = targets_fn(t)
        if len(b.held_act):
            d.ctrl[b.held_act] = b.q0_held
        for _ in range(ns):
            mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.array(out)


def parity_rollout(body: str, ticks: int = 25, targets_fn=None, contact: str = "v2", adapt=(), device=None) -> dict:
    """Same model, same open-loop targets: C MuJoCo vs MJX (1 env). Reports max |dqpos| over all ticks and the base
    position/height difference at the end, plus the effect of any adaptation on the C reference."""
    env = MjxLegged(body, 1, contact=contact, adapt=adapt, device=device)
    b, m = env.b, env.m
    user_fn = targets_fn
    targets_fn = targets_fn or (lambda t: open_loop_targets(b, t))
    ref = c_rollout(m, b, ticks, targets_fn)
    env.reset()
    got = [np.asarray(env.data.qpos[0])]
    t0 = time.time()
    for t in range(ticks):
        q, _ = env.step(np.asarray(targets_fn(t))[None])
        got.append(np.asarray(q[0]))
    wall = time.time() - t0
    got = np.array(got)
    err = np.abs(got - ref)
    res = dict(body=body, contact=contact, ticks=ticks, substeps=env.substeps, dtype=str(got.dtype),
               max_dqpos=float(err.max()), max_dqpos_joints=float(err[:, 7:].max()) if err.shape[1] > 7 else None,
               final_base_dxyz=(got[-1, :3] - ref[-1, :3]).tolist(), c_final_height=float(ref[-1, 2]),
               mjx_final_height=float(got[-1, 2]), mjx_wall_s_incl_compile=wall, adaptations=env.adaptations)
    if adapt:
        m0, _, b0, _ = build_model(body, contact, ())
        ref0 = c_rollout(m0, b0, ticks, user_fn)
        res["adaptation_effect_max_dqpos"] = float(np.abs(ref0 - ref).max())
    return res


def throughput(body: str, n_envs: int, ticks: int = 20, contact: str = "v2", adapt=(), device=None) -> dict:
    """Env-ticks/s of the jitted vmapped step (compile excluded: first tick timed separately)."""
    env = MjxLegged(body, n_envs, contact=contact, adapt=adapt, device=device)
    env.reset()
    tg = np.stack([open_loop_targets(env.b, 0)] * n_envs)
    t0 = time.time()
    q, _ = env.step(tg)
    q.block_until_ready()
    compile_s = time.time() - t0
    t0 = time.time()
    for t in range(ticks):
        q, _ = env.step(np.stack([open_loop_targets(env.b, t + 1)] * n_envs))
    q.block_until_ready()
    dt = time.time() - t0
    fin = bool(np.isfinite(np.asarray(q)).all())
    return dict(n_envs=n_envs, ticks=ticks, compile_s=compile_s, wall_s=dt, env_ticks_per_s=n_envs * ticks / dt,
                sim_s_per_wall_s=n_envs * ticks / TRACKER_HZ / dt, finite=fin)


def c_throughput(body: str, ticks: int = 50, contact: str = "v2", adapt=()) -> dict:
    m, _, b, _ = build_model(body, contact, adapt)
    t0 = time.time()
    c_rollout(m, b, ticks)
    dt = time.time() - t0
    return dict(n_envs=1, ticks=ticks, env_ticks_per_s=ticks / dt, note="C MuJoCo, one thread")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--body", default="go2")
    ap.add_argument("--contact", default="v2")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--ticks", type=int, default=25)
    ap.add_argument("--n-envs", default="1,64")
    ap.add_argument("--tp-ticks", type=int, default=20)
    ap.add_argument("--adapt", default="", help="comma list of ADAPTATIONS; tried in order only if the plain model fails")
    ap.add_argument("--try-plain", action="store_true", help="first try the unadapted model and record its error")
    a = ap.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if a.device == "gpu" else "cpu")
    jax, _, _ = _jax()
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    adapt = [x for x in a.adapt.split(",") if x]
    out = dict(version=PROTOTYPE_VERSION, source=SOURCE_LABEL, jax=jax.__version__, mujoco=mujoco.__version__,
               devices=[str(d) for d in jax.devices()], x64=a.x64, requested_adaptations=adapt)
    if a.try_plain and adapt:
        try:
            out["plain_parity"] = parity_rollout(a.body, 3, contact=a.contact)
        except Exception as e:  # noqa: BLE001 - recorded, the point of the prototype
            out["plain_error"] = f"{type(e).__name__}: {str(e)[:400]}"
    try:
        out["parity"] = parity_rollout(a.body, a.ticks, contact=a.contact, adapt=adapt)
        out["throughput"] = [throughput(a.body, int(n), a.tp_ticks, contact=a.contact, adapt=adapt)
                             for n in a.n_envs.split(",") if n]
        out["c_throughput"] = c_throughput(a.body, 50, contact=a.contact, adapt=adapt)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
