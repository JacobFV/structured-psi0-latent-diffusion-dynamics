"""Training-batch data mix (D-144, docs/relations.md 5.3): each batch = the scheduler's shares of composed relgen
samples + the main (task) data, fraction-exact per batch and deterministic under the seed. `allocate` is the exact
integer split; `mixed_batches` (unit R10) streams batches from the main iterator and the relgen shards."""
from __future__ import annotations

import numpy as np


def allocate(n: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of n rows over shares (sum <= 1; the remainder goes to key "main")."""
    s = {k: max(0.0, float(v)) for k, v in shares.items()}
    tot = sum(s.values())
    if tot > 1 + 1e-9:
        raise ValueError(f"shares sum to {tot} > 1")
    s["main"] = s.get("main", 0.0) + max(0.0, 1.0 - tot)
    raw = {k: n * v for k, v in s.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


def mixed_batches(main, scheduler, shards, batch_size: int, rng):
    """Yield training batches mixing `main` with scheduler-chosen relgen rows (unit R10)."""
    raise NotImplementedError("harness.data.mix.mixed_batches is implemented by fanout unit R10")
