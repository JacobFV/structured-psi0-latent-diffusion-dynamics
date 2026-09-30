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


def mask_missing_labels(samples, label_names):
    """Every sample gets every name in `label_names`: a label it already carries passes through; one it does not
    carry becomes an all-invalid placeholder (same shape as another label already on the sample, or scalar if it
    carries none) -- so a probe's loss over a mixed batch simply ignores rows that do not carry it, rather than
    erroring on a missing key (docs/relations.md 5.3: "labels a sample does not carry are masked")."""
    from rrp.harness.data.relgen import Label
    ref_shape = None
    for s in samples:
        for lab in s.get("labels", {}).values():
            ref_shape = lab.valid.shape
            break
        if ref_shape is not None:
            break
    shape = ref_shape or (1,)
    out = []
    for s in samples:
        labels = dict(s.get("labels", {}))
        for name in label_names:
            if name not in labels:
                labels[name] = Label(value=np.zeros(shape + (1,)), valid=np.zeros(shape, dtype=bool),
                                     prov="none", version="")
        out.append({**s, "labels": labels})
    return out


def mixed_batches(main, scheduler, shards, batch_size: int, rng, *, label_names=None):
    """Yield training batches mixing `main` with scheduler-chosen relgen rows (unit R10).

    `main`: an iterable of main (task) batches, each an iterable of samples (dicts with a "labels" key; plain task
    rows carry none). `scheduler`: a `relgen.curriculum.Scheduler` (`.sample(step, n)` drives the split; its shares
    ARE `allocate`'s split, so the count per key is exact). `shards`: `{factor_name: [Sample, ...]}` (e.g.
    `pipelines.relations.load_shard_rows` per factor) -- the relgen row pools `sample` draws from, keyed by the
    active set's first (sorted) factor name (composed multi-factor active sets are unit R11's `compose`; until then
    each factor's own pool stands in for its active set). `rng`: a `numpy.random.Generator` -- same seed => the same
    draw sequence (row indices), so two runs over the same shards are byte-identical.

    Each yielded batch is `{"step", "main": [...], "relgen": [...], "counts": {key: n}}`; `sum(counts.values()) ==
    batch_size` exactly (the same integer split `allocate` computes from the scheduler's current shares) and every
    row in "main" + "relgen" is masked (`mask_missing_labels`) over the union of labels present anywhere in the
    batch (plus `label_names`, when given), so downstream code can stack them uniformly."""
    step = 0
    for main_batch in main:
        main_rows = list(main_batch)
        active = scheduler.sample(step, batch_size)
        counts = {"main": batch_size}
        rel_rows = []
        for factor_set, count in active:
            if count <= 0:
                continue
            key = sorted(factor_set)[0]
            counts[key] = count
            counts["main"] -= count
            pool = shards.get(key) or []
            if not pool:
                continue
            idx = rng.integers(0, len(pool), size=count)
            rel_rows.extend(pool[int(i)] for i in idx)
        n_main = max(0, counts["main"])
        batch_main = [main_rows[i % len(main_rows)] for i in range(n_main)] if main_rows else []
        names = set(label_names or [])
        for s in batch_main + rel_rows:
            names |= set(s.get("labels", {}))
        # mask the combined batch in one call so a main row missing a label borrows its shape from a relgen row
        # that carries it, not only from another main row (mask_missing_labels' reference shape is batch-wide)
        masked = mask_missing_labels(batch_main + rel_rows, names)
        yield {"step": step, "main": masked[:len(batch_main)], "relgen": masked[len(batch_main):], "counts": counts}
        step += 1
