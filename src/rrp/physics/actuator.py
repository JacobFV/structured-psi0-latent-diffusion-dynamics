"""Actuator realism for legged trackers ("actuator_v2", W1 task 3).

The legacy actuator (v1) is an ideal bounded joint PD servo: tau = clip(kp (u - q) - kd qdot, +-effort), and the new target
applies within the 20 ms tick (contact_v2 randomises only 0-8 ms of intra-tick latency). actuator_v2 adds:

* Reflected rotor inertia (armature): per joint, max(model armature, ARMATURE_PER_NM x effort). The rule of thumb
  (about 4e-4 kg m^2 per N m of peak torque) matches go2's menagerie value (0.01-0.02 at 24-45 N m) and h1's knee (0.1 at 300 N m is above it,
  so the model value is kept). Training randomises x U(0.8, 1.2).
* Joint viscous damping and Coulomb friction (MuJoCo dof_damping / dof_frictionloss): the model values plus U(0, 1%) of the
  joint's peak torque (N m s/rad and N m), resampled per worker.
* Torque-speed limit: the available torque derates linearly from peak at |qdot| = knee x vmax to 0 at vmax (a DC-motor
  back-EMF envelope). vmax per body family is an ASSUMPTION from public spec sheets (not in the MJCF models): humanoids 20 rad/s,
  go2 30, anymal_c 12, procedural 8; knee 0.5. It is enforced exactly for the affine PD servo, per physics substep, by clamping
  the effective target.
* Actuation latency 0-30 ms (0-15 substeps at dt 0.002), per episode. It is applied across control ticks through a pending-target queue.

Validation uses the nominal values (x1 armature, +0.5% damping/friction) and a FIXED latency (`latency_ms`), so robustness can be
reported per latency.
"""
from __future__ import annotations

import numpy as np

ARMATURE_PER_NM = 4e-4
VMAX = {"humanoid": 20.0, "biped": 20.0, "quadruped:go2": 30.0, "quadruped:anymal_c": 12.0, "quadruped": 20.0,
        "hexapod": 8.0, "multipod": 8.0}
KNEE = 0.5
RAND = dict(armature=(0.8, 1.2), damping_frac=(0.0, 0.01), friction_frac=(0.0, 0.01), latency_ms=(0.0, 30.0))


def vmax_for(kind: str, name: str) -> float:
    return VMAX.get(f"{kind}:{name}", VMAX.get(kind, 20.0))


class ActuatorModel:
    """Applies actuator_v2 to a compiled model (joint params) and wraps target application (latency + torque-speed)."""

    def __init__(self, model, binding, n_envs: int, rng: np.random.Generator | None, *, name: str = "",
                 randomize: bool = True, latency_ms: float | None = None):
        b = binding
        self.m, self.b, self.n = model, b, n_envs
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.randomize = randomize
        self.fixed_latency_ms = latency_ms
        j = model.actuator_trnid[b.pol_act, 0]
        self.dof = model.jnt_dofadr[j]
        eff = b.effort
        a0 = model.dof_armature[self.dof].copy()
        d0 = model.dof_damping[self.dof].copy()
        f0 = model.dof_frictionloss[self.dof].copy()
        if randomize:
            arm_s = self.rng.uniform(*RAND["armature"])
            dfr, ffr = self.rng.uniform(*RAND["damping_frac"]), self.rng.uniform(*RAND["friction_frac"])
        else:
            arm_s, dfr, ffr = 1.0, 0.005, 0.005
        model.dof_armature[self.dof] = np.maximum(a0, ARMATURE_PER_NM * eff) * arm_s
        model.dof_damping[self.dof] = d0 + dfr * eff
        model.dof_frictionloss[self.dof] = f0 + ffr * eff
        self.params = dict(armature_scale=float(arm_s), damping_frac=float(dfr), friction_frac=float(ffr),
                           armature=model.dof_armature[self.dof].round(4).tolist())
        self.kp = model.actuator_gainprm[b.pol_act, 0].copy()
        self.kd = -model.actuator_biasprm[b.pol_act, 2].copy()
        self.eff = eff.copy()
        self.vmax = vmax_for(b.kind, name)
        self.dt = float(model.opt.timestep)
        self.lat = np.zeros(n_envs, int)
        self.pending = [[] for _ in range(n_envs)]     # [(remaining substeps, target)]
        self.current = [None] * n_envs

    def reset(self, i: int, target0: np.ndarray):
        ms = self.fixed_latency_ms if self.fixed_latency_ms is not None else self.rng.uniform(*RAND["latency_ms"])
        self.lat[i] = int(round(ms / 1000.0 / self.dt))
        self.pending[i] = []
        self.current[i] = np.array(target0, float)

    def command(self, i: int, target: np.ndarray):
        self.pending[i].append([int(self.lat[i]), np.array(target, float)])

    def substep_ctrl(self, i: int, d) -> np.ndarray:
        """Target to write to d.ctrl for the next physics substep (latency and torque-speed clamp)."""
        p = self.pending[i]
        while p and p[0][0] <= 0:
            self.current[i] = p.pop(0)[1]
        for e in p:
            e[0] -= 1
        u = self.current[i]
        q = d.qpos[self.b.pol_qadr]
        qd = d.qvel[self.b.pol_dadr]
        lim = self.eff * np.clip((self.vmax - np.abs(qd)) / ((1 - KNEE) * self.vmax), 0.0, 1.0)
        tau = self.kp * (u - q) - self.kd * qd
        tau_c = np.clip(tau, -lim, lim)
        return np.clip(q + (tau_c + self.kd * qd) / self.kp, self.b.lo, self.b.hi)
