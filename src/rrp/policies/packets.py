"""Packet builders (W4-deferred dedup, resolved in W5): ONE builder of a single-robot LatentActionChunk.

- `arm_packet` is the former `latent_semantic_edits.build_packet` (unchanged).
- `ladder.make_packet(s, f, z, knot_times, lsv, rcv, validity, *, source, policy_version)` is now
  `arm_packet(f, s, s.observe(), z, ..., sampling=None)`. The one behaviour difference of the old copy: it stored
  `np.asarray(z, float32)` (a non-contiguous z stayed a strided view) where build_packet stores
  `np.ascontiguousarray(z, float32)`; values, dtype and shape are identical, only the memory layout differs
  (tests/unit/test_packets.py compares the two layout-insensitively, the W5 R1/R2 parity runs cover the rollouts).
- `dual_packet` (masked assemblies, dual handles) stays a separate builder: different packet semantics.
"""
from __future__ import annotations

import time

import numpy as np


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
