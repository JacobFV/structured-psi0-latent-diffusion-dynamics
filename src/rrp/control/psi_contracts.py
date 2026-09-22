"""psi0 action/state contracts, exactly as audited (research/methods/source-audit.md section b).

Two INCOMPATIBLE contracts:
  * psi0_original (AMO prior command): 36-dim action, 32-real-dim state zero-padded to 36.
  * psi0_sonic (released multi-task.psi-dream / postpre.sonic1.0): 80-dim action
    (64 SONIC latent token + 14 Dex3 hand joints + 2 neck), 45-dim state.

Validation rules:
  * widths must match exactly (no silent padding/truncation);
  * group layout is fixed (index ranges below); units marked UNVERIFIED stay flagged;
  * a contract may only be applied to a body listed in its `validated_bodies` (spec hash or body key)
    -> applying the G1-derived SONIC or AMO contract to any other body raises ContractBodyError.
    There is NO fallback / automatic retargeting (no silent G1 -> other body).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ContractWidthError(ValueError):
    code = "psi_contract_wrong_width"


class ContractBodyError(ValueError):
    code = "psi_contract_body_not_validated"


@dataclass(frozen=True)
class Group:
    name: str
    start: int
    dim: int
    units: str
    verified_units: bool = True
    note: str = ""

    @property
    def stop(self):
        return self.start + self.dim


@dataclass
class PsiContract:
    id: str
    action_groups: tuple
    state_groups: tuple
    action_dim: int
    state_dim: int
    state_real_dim: int
    rate_hz: float
    chunk: int
    source_body: str                         # body the checkpoints were trained on
    validated_bodies: set = field(default_factory=set)   # body keys / spec hashes validated in OUR sim
    normalization: str = "per-dim min/max bounds -> [-1, 1] (bounds from the checkpoint run config)"

    def __post_init__(self):
        for groups, dim in ((self.action_groups, self.action_dim), (self.state_groups, self.state_real_dim)):
            pos = 0
            for g in groups:
                if g.start != pos:
                    raise AssertionError(f"{self.id}: group {g.name} starts at {g.start}, expected {pos}")
                pos = g.stop
            if pos != dim:
                raise AssertionError(f"{self.id}: groups cover {pos} dims, declared {dim}")

    def split_action(self, a) -> dict[str, np.ndarray]:
        a = np.asarray(a, float)
        if a.ndim not in (1, 2) or a.shape[-1] != self.action_dim:
            raise ContractWidthError(f"{self.id}: action width {a.shape[-1] if a.ndim else 0} != {self.action_dim}")
        if not np.isfinite(a).all():
            raise ContractWidthError(f"{self.id}: non-finite action")
        return {g.name: a[..., g.start:g.stop] for g in self.action_groups}

    def pack_state(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        missing = [g.name for g in self.state_groups if g.name not in groups]
        extra = [k for k in groups if k not in {g.name for g in self.state_groups}]
        if missing or extra:
            raise ContractWidthError(f"{self.id}: state groups missing={missing} extra={extra}")
        out = np.zeros(self.state_dim)
        for g in self.state_groups:
            v = np.asarray(groups[g.name], float)
            if v.shape != (g.dim,):
                raise ContractWidthError(f"{self.id}: state group {g.name} width {v.shape} != ({g.dim},)")
            out[g.start:g.stop] = v
        return out     # zero padding beyond state_real_dim is part of the audited contract

    def check_body(self, body: str, spec_hash: str | None = None):
        if body not in self.validated_bodies and (spec_hash is None or spec_hash not in self.validated_bodies):
            raise ContractBodyError(
                f"{self.id} is validated only for {sorted(self.validated_bodies) or '[]'} (source body "
                f"{self.source_body}); refusing to apply it to {body!r}. Validate the body+controller first.")

    def apply(self, body: str, action, spec_hash: str | None = None) -> dict[str, np.ndarray]:
        """The only entry point that turns a psi0 action into body commands: body check, then width check."""
        self.check_body(body, spec_hash)
        return self.split_action(action)

    def describe(self) -> dict:
        f = lambda gs: [dict(name=g.name, range=[g.start, g.stop], dim=g.dim, units=g.units,
                             verified_units=g.verified_units, note=g.note) for g in gs]
        return dict(id=self.id, action_dim=self.action_dim, state_dim=self.state_dim,
                    state_real_dim=self.state_real_dim, rate_hz=self.rate_hz, chunk=self.chunk,
                    source_body=self.source_body, validated_bodies=sorted(self.validated_bodies),
                    action_groups=f(self.action_groups), state_groups=f(self.state_groups),
                    normalization=self.normalization)


def psi0_original() -> PsiContract:
    """AMO prior command (raw_to_lerobot.py:215-258; psi-inference_rtc.py:291-315)."""
    return PsiContract(
        id="psi0_original_36",
        action_groups=(Group("q_hand", 0, 14, "rad", False, "left 7, right 7; H1 Inspire 6-DOF hands padded"),
                       Group("q_arm", 14, 14, "rad", False, "IK joint targets sol_q[15:29], left 7, right 7"),
                       Group("torso_rpy", 28, 3, "rad", False),
                       Group("base_height", 31, 1, "m", True, "default 0.75"),
                       Group("v_x", 32, 1, "UNVERIFIED", False, "client snaps to {0, 0.6}"),
                       Group("v_y", 33, 1, "UNVERIFIED", False, "client snaps to {0, +-0.5}"),
                       Group("v_yaw", 34, 1, "UNVERIFIED", False),
                       Group("p_yaw", 35, 1, "rad", False, "target yaw; torso_dyaw excluded")),
        state_groups=(Group("hands", 0, 14, "rad", False), Group("arms", 14, 14, "rad", False),
                      Group("torso_rpy", 28, 3, "rad", False), Group("torso_height", 31, 1, "m")),
        action_dim=36, state_dim=36, state_real_dim=32, rate_hz=30.0, chunk=30, source_body="unitree_h1_inspire")


def psi0_sonic() -> PsiContract:
    """SONIC 80-dim action (transform_psi0_sonic.py:268-283) + 45-dim state (posttrain preflight)."""
    return PsiContract(
        id="psi0_sonic_80",
        action_groups=(Group("sonic_token", 0, 64, "unitless", True,
                             "FSQ grid 1/16; client clip [-0.625, 0.625] vs dataset min -0.875 UNRESOLVED"),
                       Group("hands", 64, 14, "rad", False,
                             "Dex3: L thumb0/1/2, middle0/1, index0/1, then R"),
                       Group("neck", 78, 2, "UNVERIFIED", False, "yaw, pitch; post-train variant zero-pads")),
        state_groups=(Group("legs", 0, 12, "rad", False, "L hip p/r/y, knee, ankle p/r; then R"),
                      Group("waist", 12, 3, "rad", False, "yaw, roll, pitch"),
                      Group("arms", 15, 14, "rad", False, "L shoulder p/r/y, elbow, wrist r/p/y; then R"),
                      Group("hands", 29, 14, "rad", False),
                      Group("neck", 43, 2, "UNVERIFIED", False, "ranges look suspicious")),
        action_dim=80, state_dim=45, state_real_dim=45, rate_hz=30.0, chunk=30, source_body="unitree_g1_dex3",
        normalization="bounds -> [-1, 1] with clipping; near-constant dims pass through")


# Bodies validated in OUR simulator for each contract. Empty until a psi0 checkpoint + its
# whole-body controller (AMO / SONIC WBC) is reproduced on the matching body; no other body may
# be added without its own validation receipt.
VALIDATED: dict[str, set] = {"psi0_original_36": set(), "psi0_sonic_80": set()}


def contracts() -> dict[str, PsiContract]:
    out = {}
    for c in (psi0_original(), psi0_sonic()):
        c.validated_bodies = set(VALIDATED[c.id])
        out[c.id] = c
    return out
