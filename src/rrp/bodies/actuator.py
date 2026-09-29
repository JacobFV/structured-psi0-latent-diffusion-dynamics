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

# SOURCED per-joint limits (W1 follow-up, 2026-09-27): (joint-name regex, peak torque N m, max joint speed rad/s), read from
# the manufacturers' published URDFs (<limit effort= velocity=>). Sources and sha256 (first 16 hex) of the files read:
#   t1       BoosterRobotics/booster_gym resources/T1/T1_serial.urdf (027a5333ce4ed0a1); torques also equal the
#            menagerie booster_t1/t1.xml actuatorfrcrange
#   h1       unitreerobotics/unitree_ros robots/h1_description/urdf/h1.urdf (ebd495cba7887406)
#   g1       unitreerobotics/unitree_mujoco unitree_robots/g1/g1_29dof.xml (423e28bd718b19f7; motor ctrlrange) and
#            unitreerobotics/unitree_rl_gym resources/robots/g1_description/g1_29dof.urdf (f055210bb33df351) AGREE: hip 88,
#            knee 139, ankle 50 N m; speeds hip 32, knee 20, ankle 37. (Conflicting older files: unitree_ros g1_29dof.urdf
#            ankle 35 / 30 rad/s; g1_12dof.urdf and the menagerie g1.xml hip_roll 139. The two current Unitree files win.)
#   go2      unitreerobotics/unitree_ros robots/go2_description/urdf/go2_description.urdf (7d19fe48e2e689ee)
#   anymal_c ANYbotics/anymal_c_simple_description urdf/anymal.urdf (3902c3957ac82176)
# The URDF velocity is used as the zero-torque speed of the linear torque-speed envelope (a modelling choice, labelled).
# Procedural bodies have no source: they keep the ESTIMATED VMAX below.
SOURCED = {
    "t1": [("Hip_Pitch", 45.0, 12.5), ("Hip_Roll|Hip_Yaw", 30.0, 10.9), ("Knee", 60.0, 11.7), ("Ankle_Pitch", 20.0, 18.8),
           ("Ankle_Roll", 15.0, 12.4)],
    "h1": [("hip", 200.0, 23.0), ("knee", 300.0, 14.0), ("ankle", 40.0, 9.0)],
    "g1": [("hip", 88.0, 32.0), ("knee", 139.0, 20.0), ("ankle", 50.0, 37.0)],
    "go2": [("hip_joint|thigh", 23.7, 30.1), ("calf", 45.43, 15.70)],
    "anymal_c": [("HAA|HFE|KFE", 80.0, 7.5)],
}

# Body-model default (W1, D-107): the legged body adapter uses these SOURCED peak torques as the PD servo force limits.
# "legacy_gains_v0" = the pre-2026-09-27 hand-written gains table (t1 2-3x too strong, g1 hip_roll 139).
ACTUATOR_LIMITS_DEFAULT = "sourced_v1"
ACTUATOR_LIMITS = ("legacy_gains_v0", "sourced_v1")
# bodies whose effective limits DIFFER between the two versions (all others have identical values, so trackers stay compatible)
LIMITS_CHANGED = {"t1", "g1"}


def resolve_limits(limits: str | None = None) -> str:
    import os
    v = limits if limits is not None else os.environ.get("RRP_ACTUATOR_LIMITS", ACTUATOR_LIMITS_DEFAULT)
    if v not in ACTUATOR_LIMITS:
        raise ValueError(f"unknown actuator limits {v!r}; known {ACTUATOR_LIMITS}")
    return v


def sourced_effort(body: str, joint: str) -> float | None:
    import re
    for pat, e, _v in SOURCED.get(body, []):
        if re.search(pat, joint):
            return e
    return None


def model_actuator_limits(model) -> str | None:
    """Actuator-limits version embedded by the legged body builder (text element `actuator_limits`), or None."""
    import mujoco
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "actuator_limits")
    if tid < 0:
        tid = next((t for t in range(model.ntext)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, t) or "").endswith("actuator_limits")), -1)
    if tid < 0:
        return None
    adr, n = model.text_adr[tid], model.text_size[tid]
    return bytes(model.text_data[adr:adr + n - 1]).decode()


# ------------------------------------------------------------------ actuator MODE switch (D-126 #14)
# One place decides which actuator model every legged consumer uses (tracker training, tracker validation, the legged pipeline's
# collect / DAgger / eval / edit subprocesses, legged_latent_eval). The DEFAULT stays the ideal PD servo ("ideal" = the legacy
# "v1"), so nothing changes unless a caller or $RRP_ACTUATOR_MODE asks. Flipping the project default later is this one constant.
#   ideal  bounded joint PD servo, targets applied within the tick (legacy "v1"; contact_v2 training adds 0-8 ms intra-tick latency)
#   v1lat  ideal joints + SOURCED torque/speed envelope + 0-30 ms actuation latency (ActuatorModel mode "v1lat")
#   v2     armature + joint damping/friction + torque-speed envelope + 0-30 ms latency (ActuatorModel mode "v2")
ACTUATOR_MODE_VERSION = "actuator_mode_v1"
ACTUATOR_MODES = ("ideal", "v1lat", "v2")
ACTUATOR_MODE_DEFAULT = "ideal"
_MODE_ALIASES = {"v1": "ideal", "ideal_pd": "ideal", "ideal": "ideal", "v1lat": "v1lat", "v2": "v2"}


def resolve_mode(mode: str | None = None) -> str:
    """Canonical actuator mode: explicit value, else $RRP_ACTUATOR_MODE, else ACTUATOR_MODE_DEFAULT ("v1" is an alias of "ideal")."""
    import os
    v = mode if mode is not None else (os.environ.get("RRP_ACTUATOR_MODE") or ACTUATOR_MODE_DEFAULT)
    if v not in _MODE_ALIASES:
        raise ValueError(f"unknown actuator mode {v!r}; known {ACTUATOR_MODES} (alias v1 = ideal)")
    return _MODE_ALIASES[v]


def legacy_mode_name(mode: str | None) -> str:
    """The name the tracker trainer / validator historically record ("v1" for ideal), so default records stay byte-identical."""
    m = resolve_mode(mode)
    return "v1" if m == "ideal" else m


def resolve_latency_ms(seed: int | None = None, latency_ms: float | None = None) -> float:
    """Deployment/eval latency for a non-ideal mode: explicit value, else $RRP_ACTUATOR_LATENCY_MS, else drawn per episode from
    U(RAND latency) with rng([seed, 9101]) (reproducible from the episode seed)."""
    import os
    if latency_ms is not None:
        return float(latency_ms)
    env = os.environ.get("RRP_ACTUATOR_LATENCY_MS")
    if env not in (None, ""):
        return float(env)
    lo, hi = RAND["latency_ms"]
    return float(np.random.default_rng([int(seed or 0), 9101]).uniform(lo, hi))


def speed_sources(name: str, joint_names) -> dict:
    """Per-joint provenance of the max joint speed used by the torque-speed envelope: 'sourced_urdf' (manufacturer URDF) or
    'estimate' (VMAX family default). D-103/D-107: any 'estimate' must be flagged wherever a result depends on it."""
    import re
    out = {}
    for n in joint_names:
        hit = any(re.search(pat, n) for pat, _e, _v in SOURCED.get(name, []))
        out[n] = "sourced_urdf" if hit else "estimate"
    return out


def mode_record(mode: str, model=None, binding=None, name: str = "", latency_ms: float | None = None) -> dict:
    """Provenance block for a non-ideal actuator mode (eval rows, collect metadata, tracker meta)."""
    rec = dict(actuator_mode=resolve_mode(mode), version=ACTUATOR_MODE_VERSION)
    if latency_ms is not None:
        rec["latency_ms"] = float(latency_ms)
    if model is not None and binding is not None:
        j = model.actuator_trnid[binding.pol_act, 0]
        jn = [model.joint(int(x)).name[len(binding.prefix):] for x in j]
        src = speed_sources(name, jn)
        est = sorted(k for k, v in src.items() if v == "estimate")
        rec.update(speed_source="estimate" if est else "sourced_urdf", speed_estimated_joints=est)
    return rec


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
                 randomize: bool = True, latency_ms: float | None = None, mode: str = "v2"):
        """mode "v2": all effects. mode "v1lat": ideal joints (no armature/damping/friction change) plus SOURCED torque and
        speed limits plus 0-30 ms latency (the lead's "actuator_v1 with sourced limits and latency")."""
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
        if mode == "v1lat":
            arm_s, dfr, ffr = None, 0.0, 0.0
        elif randomize:
            arm_s = self.rng.uniform(*RAND["armature"])
            dfr, ffr = self.rng.uniform(*RAND["damping_frac"]), self.rng.uniform(*RAND["friction_frac"])
        else:
            arm_s, dfr, ffr = 1.0, 0.005, 0.005
        if arm_s is not None:
            model.dof_armature[self.dof] = np.maximum(a0, ARMATURE_PER_NM * eff) * arm_s
        model.dof_damping[self.dof] = d0 + dfr * eff
        model.dof_frictionloss[self.dof] = f0 + ffr * eff
        self.params = dict(mode=mode, armature_scale=None if arm_s is None else float(arm_s), damping_frac=float(dfr), friction_frac=float(ffr),
                           armature=model.dof_armature[self.dof].round(4).tolist())
        self.kp = model.actuator_gainprm[b.pol_act, 0].copy()
        self.kd = -model.actuator_biasprm[b.pol_act, 2].copy()
        self.eff = eff.copy()
        self.vmax = np.full(b.n, vmax_for(b.kind, name))
        self.limit_source = "estimate"
        if name in SOURCED:
            import re
            jn = [model.joint(int(x)).name[len(b.prefix):] for x in j]
            for k, n in enumerate(jn):
                hit = next(((e, v) for pat, e, v in SOURCED[name] if re.search(pat, n)), None)
                if hit is None:
                    raise KeyError(f"{name}: no sourced limit for joint {n}")
                self.eff[k] = min(self.eff[k], hit[0])
                self.vmax[k] = hit[1]
            self.limit_source = "sourced_urdf"
        self.params.update(limit_source=self.limit_source, effort=self.eff.round(2).tolist(), vmax=self.vmax.round(2).tolist())
        # D-126 #14: joints whose max speed is still an ESTIMATE (kept out of `params`, which existing rows embed)
        self.speed_estimated = [] if self.limit_source == "sourced_urdf" else [
            model.joint(int(x)).name[len(b.prefix):] for x in j]
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
