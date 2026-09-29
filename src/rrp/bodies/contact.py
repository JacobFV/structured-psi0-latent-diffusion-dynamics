"""Versioned foot-floor contact models for the legged stack.

`contact_v1` (legacy, still the default): the physics every tracker up to 2026-09-26 was trained in.
    Procedural bodies and the humanoids (t1/h1/g1): PYRAMIDAL friction cone, impratio 1, no noslip.
    Menagerie quadrupeds (go2/anymal_c) inherit the asset options (ELLIPTIC, impratio 100).
    Floor: friction [1, 0.02, 0.001], default solref [0.02, 1] / solimp [0.9, 0.95, 0.001, 0.5, 2],
    priority 0, so each foot geom's own friction/condim/solref is mixed with the floor's.
    Randomisation: a per-worker scale on the sliding coefficient only (0.6-1.25).

`contact_v2`: realistic stance (feet planted, no soft tangential creep).
    * ELLIPTIC cone with impratio 10 for every body. A larger impratio makes friction "harder" relative to
      normal contact; with an elliptic cone that removes the pyramidal cone's creep (see
      tests/unit/test_contact_model.py: creep at 0.9*mu on a slope drops about 10x or more).
    * The floor gets priority 2 (above every asset geom), so the floor's friction, condim, solref and solimp
      define EVERY robot-floor contact. The parameters are then the same across bodies and are not mixed
      with per-asset foot values. This overrides go2's menagerie foot values, a declared change.
    * condim 3 (sliding friction only). All humanoid feet are multi-point flat contacts (t1: box; h1: 3
      capsules; g1: 4 spheres per foot), which already resist yaw through several contact points. The
      quadruped and hexapod feet are small spheres, where torsional/rolling resistance is physically
      negligible. The torsional/rolling coefficients are still stored (scaled with mu) for condim 4/6 use.
    * Semi-soft sole: solref [0.008, 1.0] (time constant 8 ms = 4 physics steps at dt 0.002, critically
      damped); solimp [0.9, 0.99, 0.003, 0.5, 2] (impedance ramps over 3 mm, which gives a slightly compliant
      sole). Measured penetration stays at 0-2 mm under body weight and landing (unit test).
    * Physics timestep min(source, 0.002); 10 substeps per 50 Hz tracker tick.
    * Stiction: MuJoCo has ONE Coulomb coefficient (no static/kinetic split). Stiction is approximated by
      the elliptic cone plus a high impratio: below mu*N the tangential constraint behaves like a
      stiff, critically damped spring, so there is no macroscopic creep (slope test: 0.1 mm/s at 0.9*mu,
      against 8-13 mm/s for v1). `noslip_iterations` 5 removes even that residual creep (to 1e-8 m/s) at
      about 20% extra step cost; the residual is negligible for walking, so v2 keeps noslip at 0
      (artifacts/runs/contact_v2/slope_table.json). No velocity-dependent kinetic-friction proxy is added: MuJoCo has no clean
      hook for it, and we do not hack it.
    Training randomisation (per worker, resampled every `resample_steps` control ticks, since the model
    is shared by a worker's envs): mu ~ U(0.4, 1.25) with torsional/rolling scaled to match; solref
    timeconst ~ U(0.006, 0.012) s and damping ratio ~ U(0.8, 1.2) (restitution proxy); root mass x U(0.9, 1.1)
    and root CoM shift U(+-2 cm) per axis (sampled once per worker); actuator/PD latency: per episode the new
    joint targets take effect after U{0..4} physics substeps (0-8 ms).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

DEFAULT_ENV = "RRP_CONTACT_MODEL"

CONTACT_MODELS = {
    "v1": dict(version="contact_v1"),
    "v2": dict(version="contact_v2", cone="elliptic", impratio=10.0, noslip_iterations=0, max_timestep=0.002,
               floor=dict(mu=0.9, torsional_per_mu=0.02, rolling_per_mu=0.001, condim=3, priority=2,
                          solref=[0.008, 1.0], solimp=[0.9, 0.99, 0.003, 0.5, 2.0]),
               randomization=dict(mu=[0.4, 1.25], timeconst=[0.006, 0.012], dampratio=[0.8, 1.2],
                                  root_mass_scale=[0.9, 1.1], root_com_shift_m=0.02, latency_substeps=[0, 4],
                                  resample_steps=500)),
}


def resolve(contact: str | None = None) -> str:
    """'v1' | 'v2' (accepts 'contact_v1'/'contact_v2'); None -> $RRP_CONTACT_MODEL or 'v1'."""
    c = contact if contact is not None else os.environ.get(DEFAULT_ENV, "v1")
    c = str(c).lower().replace("contact_", "")
    if c not in CONTACT_MODELS:
        raise ValueError(f"unknown contact model {contact!r}; known: {sorted(CONTACT_MODELS)}")
    return c


def version_str(contact: str | None) -> str:
    return CONTACT_MODELS[resolve(contact)]["version"]


def model_contact_version(model: mujoco.MjModel) -> str | None:
    """The contact version a compiled legged world was built with ("contact_v1" | "contact_v2"), read from the
    model's `contact_version` text element (set by legged_world); None for models not built by legged_world."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "contact_version")
    if tid < 0:
        return None
    adr, n = model.text_adr[tid], model.text_size[tid]
    return bytes(model.text_data[adr:adr + n - 1]).decode()


def floor_friction(mu: float, c: str = "v2") -> list:
    f = CONTACT_MODELS[resolve(c)]["floor"]
    return [float(mu), float(mu * f["torsional_per_mu"]), float(mu * f["rolling_per_mu"])]


def apply_world(spec: mujoco.MjSpec, contact: str | None, source_options: dict | None) -> dict:
    """Set physics options for the world spec; returns kwargs for the floor geom."""
    c = resolve(contact)
    spec.add_text(name="contact_version", data=CONTACT_MODELS[c]["version"])   # queryable from the compiled model
    if c == "v1":           # legacy, byte-identical to the pre-v2 legged_world
        if source_options:
            spec.option.timestep = source_options["timestep"]
            spec.option.integrator = source_options["integrator"]
            spec.option.cone = source_options["cone"]
            spec.option.impratio = source_options["impratio"]
            spec.option.iterations = source_options["iterations"]
        else:
            spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            spec.option.impratio = 1
        return dict(friction=[1.0, 0.02, 0.001])
    cm = CONTACT_MODELS[c]
    if source_options:
        spec.option.integrator = source_options["integrator"]
        spec.option.iterations = source_options["iterations"]
        spec.option.timestep = min(float(source_options["timestep"]), cm["max_timestep"])
    else:
        spec.option.timestep = min(float(spec.option.timestep), cm["max_timestep"])
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = cm["impratio"]
    spec.option.noslip_iterations = cm["noslip_iterations"]
    f = cm["floor"]
    return dict(friction=floor_friction(f["mu"], c), condim=f["condim"], priority=f["priority"],
                solref=list(f["solref"]), solimp=list(f["solimp"]))


class ContactRandomizer:
    """Per-worker domain randomisation for contact_v2 (model-level: the model is shared by a worker's envs)."""

    def __init__(self, model: mujoco.MjModel, root_bid: int, rng: np.random.Generator, contact: str = "v2"):
        self.m, self.rng, self.c = model, rng, resolve(contact)
        self.R = CONTACT_MODELS[self.c]["randomization"]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.root = root_bid
        R = self.R
        s = float(rng.uniform(*R["root_mass_scale"]))
        shift = rng.uniform(-R["root_com_shift_m"], R["root_com_shift_m"], 3)
        model.body_mass[root_bid] *= s
        model.body_inertia[root_bid] *= s
        model.body_ipos[root_bid] += shift
        self.static = dict(root_mass_scale=s, root_com_shift=shift.tolist())
        self.resample()

    def resample(self) -> dict:
        R, m = self.R, self.m
        self.mu = float(self.rng.uniform(*R["mu"]))
        tc = float(self.rng.uniform(*R["timeconst"]))
        dr = float(self.rng.uniform(*R["dampratio"]))
        m.geom_friction[self.floor] = floor_friction(self.mu, self.c)
        m.geom_solref[self.floor] = [tc, dr]
        self.current = dict(mu=self.mu, timeconst=tc, dampratio=dr, **self.static)
        return self.current

    def latency(self) -> int:
        lo, hi = self.R["latency_substeps"]
        return int(self.rng.integers(lo, hi + 1))

    @property
    def nominal_mu(self) -> float:
        return float(CONTACT_MODELS[self.c]["floor"]["mu"])
