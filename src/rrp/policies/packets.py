"""Packet builders (W4-deferred dedup, resolved in W5): ONE builder of a single-robot LatentActionChunk.

- `arm_packet` is the former `latent_semantic_edits.build_packet` (unchanged).
- `ladder.make_packet(s, f, z, knot_times, lsv, rcv, validity, *, source, policy_version)` is now
  `arm_packet(f, s, s.observe(), z, ..., sampling=None)`. The one behaviour difference of the old copy: it stored
  `np.asarray(z, float32)` (a non-contiguous z stayed a strided view) where build_packet stores
  `np.ascontiguousarray(z, float32)`; values, dtype and shape are identical, only the memory layout differs
  (tests/unit/test_packets.py compares the two layout-insensitively, the W5 R1/R2 parity runs cover the rollouts).
- `dual_packet` (masked assemblies, dual handles) stays a separate builder: different packet semantics.

The tensor-level EDIT REGISTRY (`EDITS`, D-146 P3) is the one definition of a causal packet edit for every policy that
hands a packet z[..., K, M, D] (knots, assemblies, latent dims) to a system 0: `LatentStackPolicy.packet_hook(i, chunk)`
(arm / dual / pointer / legged packets: `chunk_hook`) and the Ψ₀ policy (`psi0.OursModel`, a bare tensor: `apply_edit`
/ `tensor_hook`). An edit is a pure function of z (numpy or torch, never in place); the assembly axis is -2.
"""
from __future__ import annotations

import time

import numpy as np


def _copy(z):
    return z.clone() if hasattr(z, "clone") else np.array(z, copy=True)


def mean_packet(z, *, mean):
    """Every packet replaced by the reference mean packet (`mean` broadcasts to z: [K, M, D] or [D]): the null edit that
    keeps the packet's statistics and removes everything it said about THIS state."""
    out = _copy(z)
    out[...] = mean
    return out


def zero_slot(z, *, slot):
    """Assembly `slot` (index on axis -2) zeroed."""
    out = _copy(z)
    out[..., slot, :] = 0
    return out


def swap_assembly(z, *, a=0, b=1):
    """Assemblies `a` and `b` exchange their latents (handles unchanged): the left arm is driven by the right one's z."""
    out = _copy(z)
    out[..., a, :] = z[..., b, :]
    out[..., b, :] = z[..., a, :]
    return out


EDITS = {"mean_packet": mean_packet, "zero_slot": zero_slot, "swap_assembly": swap_assembly}


def apply_edit(name: str, z, **kw):
    if name not in EDITS:
        raise KeyError(f"unknown packet edit {name!r}; registered: {sorted(EDITS)}")
    return EDITS[name](z, **kw)


def tensor_hook(name: str, **kw):
    """`(i, z) -> z` hook for a bare tensor packet (`Psi0Policy.packet_hook`)."""
    return lambda i, z: apply_edit(name, z, **kw)


def chunk_hook(name: str, **kw):
    """`(i, chunk) -> chunk` hook for `LatentStackPolicy.packet_hook`: the edited packet is labelled source="debug" and
    `sampling.intervention` names the edit (as every edited packet does, harness.eval.latent_causal)."""
    def hook(i, p):
        return p.model_copy(update={"z": np.ascontiguousarray(apply_edit(name, p.z, **kw), np.float32), "source": "debug",
                                    "sampling": {**(p.sampling or {}), "intervention": name}})
    return hook


def arm_packet(f, s, o, z, *, lsv, rcv, knot_times, source, name, sampling=None, validity=0.8):
    from rrp.core.latent_action import AssemblyHandle, EntityHandle, LatentActionChunk
    M = z.shape[1]
    gasms = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")][:M]
    now = float(s.data.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=np.ascontiguousarray(z, np.float32),
        knot_times=list(knot_times),
        assemblies=[AssemblyHandle(handle=f"asm:{f.spec.spec_hash}:{a.frame.link}", robot_index=0) for a in gasms],
        assembly_mask=[True] * M, entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
        observation_id=o.observation_id, graph_version=s.runtime.graph_version,
        runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec.spec_hash, generated_at=time.time(),
        valid_from=now, valid_until=now + validity, source=source, policy_version=name, sampling=dict(sampling or {}))


def dual_packet(f, s, o, z, *, lsv, rcv, knot_times, source, name, sampling, validity=0.8):
    from rrp.core.latent_action import EntityHandle, LatentActionChunk
    from rrp.policies.features.multi import assembly_handles
    hs, mask = assembly_handles(f, z.shape[1])
    z = np.ascontiguousarray(z, np.float32)
    z[:, ~np.array(mask)] = 0.0
    now = float(s.data.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=z, knot_times=list(knot_times), assemblies=hs,
        assembly_mask=mask, entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
        observation_id=o.observation_id, graph_version=s.runtime.graph_version,
        runtime_version=s.runtime.runtime_version, robot_spec_hash=f.spec_hash, generated_at=time.time(),
        valid_from=now, valid_until=now + validity, source=source, policy_version=name, sampling=sampling)
