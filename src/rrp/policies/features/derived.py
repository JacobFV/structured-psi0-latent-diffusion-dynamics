"""Public quantities derived from a PolicyInput (W4: moved unchanged from rrp.learning.packed, which re-exports them).

System 0 reads `local_sensors` at deployment (rrp.control.latent_realizer), so it lives in the features layer.
"""
from __future__ import annotations

import numpy as np

from rrp.policies.features.featurizer import HASH_DIM

OPERATORS = ["none", "grasp", "place", "reach", "maintain_support", "estimate_frame", "align_axis", "insert",
             "give", "receive", "walk_to", "halt", "maintain_hold", "release"]   # appended only (indices are stable)


def local_sensors(pi) -> np.ndarray:
    """Declared local sensors for system 0: touch summary (log max, count>0.2N, log mean) + gripper width."""
    from rrp.policies.features.featurizer import text_hash
    th = text_hash("touch")
    out = np.zeros(4, np.float32)
    it, ik = pi.tokens["interact"], pi.token_kind["interact"]
    for j in range(len(ik)):
        if ik[j] != 1:
            continue
        if np.allclose(it[j, 16:], th):
            out[:3] = it[j, 13:16]
        else:
            out[3] = it[j, 13]
    return out


def active_operator(pi) -> int:
    """Public subtask label: operator of the first ACTIVE event (runtime status is public)."""
    from rrp.policies.features.featurizer import text_hash
    tt, tk = pi.tokens["task"], pi.token_kind["task"]
    for j in range(len(tk)):
        if tk[j] == 0 and tt[j, HASH_DIM + 2] > 0.5:
            for k, op in enumerate(OPERATORS):
                if np.allclose(tt[j, :HASH_DIM], text_hash(op), atol=1e-5):
                    return k
    return 0
