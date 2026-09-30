"""Generic, factor-agnostic sample transforms (D-144 unit R9; design: docs/relations.md 5.2, 5.4, 5.5; catalog:
research/relations_catalog.md J "epistemic").

Every transform here is a PURE function of `(sample, rng, params) -> list[sample]` (the `TransformDef.fn` contract
of `rrp.harness.data.relgen`): it never mutates its `sample` argument, it draws randomness only from the `rng` it is
given (so a fixed `numpy.random.Generator` seed reproduces byte-identical output), and it appends one record to
`sample["provenance"]["transforms"]` describing what it did. They know nothing about any particular factor; a factor
names one of them in `FactorDef.gen` and supplies its knobs via `FactorSpec.params` / `FactorDef.params`.

Sample shapes each transform reads (a `Sample` may carry only the keys the transforms it uses need):
- `reveal` / `surprise` read `sample["inputs"]["candidates"]` (a sequence of hashable candidate ids), optionally
  `sample["inputs"]["prior"]` (`{candidate: weight}`, default uniform) and `sample["inputs"]["evidence"]`
  (`[{"t": int, "excludes": [candidate, ...]}, ...]`, each event ruling candidates OUT as of its time `t`).
- `cf_swap` reads `sample["inputs"]["tokens"]` (`{set_name: {"fields": {field_name: array[T, ...]}, ...}}`),
  `sample["inputs"]["edges"]` (`{edge_name: {"sets": (q_set, k_set), "data": array[Q, K, ...]}}`) and
  `sample["labels"]` (`Label`s whose arity-2 values are indexed the same way as an edge over the same token set).
- `noise` / `occlude` read `sample["inputs"]["tokens"][set_name]`.
- `subsample` reads `sample["inputs"]["knots"]` (`{"fields": {name: array[T, ...]}}`) and, if present,
  `sample["inputs"]["evidence"]` (remapped into the subsampled time axis).

None of this is a new registry: `rrp.harness.data.relgen.{LabelDef, Label, ScenePart, TransformDef, ...}` is the
foundation (F4); this module only fills in the six `TransformDef.fn`s and registers them in `TRANSFORMS`.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np

from rrp.harness.data.relgen import Sample, TransformDef, register_transform

__all__ = ["reveal", "surprise", "cf_swap", "noise", "occlude", "subsample", "recovery_steps", "kl_divergence"]

_EPS = 1e-12


# ------------------------------------------------------------------ shared: candidate-set Bayes posterior
def _posterior(candidates: Sequence[Any], prior: dict | None, survivors: set) -> np.ndarray:
    """q(e) ~ prior(e) * 1[e in survivors], renormalized; uniform prior by default; uniform over ALL candidates if
    `survivors` is empty (a contradictory evidence set collapses to "no information" rather than dividing by zero)."""
    n = len(candidates)
    w = np.array([1.0 if prior is None else float(prior.get(c, 0.0)) for c in candidates], dtype=np.float64)
    if prior is None:
        w[:] = 1.0
    mask = np.array([c in survivors for c in candidates], dtype=np.float64)
    q = w * mask
    tot = q.sum()
    if tot <= _EPS:
        q = w.copy()
        tot = q.sum()
    if tot <= _EPS:
        q = np.full(n, 1.0 / max(n, 1))
        tot = q.sum()
    return q / tot


def _cum_survivors(candidates: Sequence[Any], evidence: Sequence[dict], t: int) -> set:
    """Candidates not yet excluded by any evidence event with `event["t"] <= t`."""
    survivors = set(candidates)
    for ev in evidence:
        if int(ev.get("t", 0)) <= t:
            survivors -= set(ev.get("excludes", ()))
    return survivors


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    """KL(q || p) for two discrete distributions over the same support (clamped away from 0 for stability)."""
    q = np.clip(np.asarray(q, dtype=np.float64), _EPS, None)
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, None)
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))


def _append_provenance(sample: Sample, record: dict) -> Sample:
    out = copy.deepcopy(sample)
    prov = out.setdefault("provenance", {})
    prov.setdefault("transforms", []).append(record)
    return out


# ------------------------------------------------------------------ reveal
def reveal(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Progressive information (docs/relations.md 5.4): target `q_t(e) ~ prior(e) * 1[e consistent with evidence <=
    t]` at every time in `params["schedule"]` (default: the sorted, de-duplicated evidence times). Deterministic
    given `sample`; `rng` is accepted only to satisfy the transform contract."""
    del rng
    inputs = sample.get("inputs", {})
    candidates = list(inputs.get("candidates", ()))
    prior = inputs.get("prior")
    evidence = sorted(inputs.get("evidence", ()), key=lambda e: int(e.get("t", 0)))
    schedule = list(params.get("schedule") or sorted({int(e.get("t", 0)) for e in evidence}))
    field_name = params.get("field", "reveal")

    q_by_t = np.stack([_posterior(candidates, prior, _cum_survivors(candidates, evidence, t)) for t in schedule]) \
        if schedule else np.zeros((0, len(candidates)))

    out = _append_provenance(sample, {
        "transform": "reveal", "version": TRANSFORMS_VERSION, "schedule": list(schedule),
        "candidates": list(candidates), "field": field_name,
    })
    out.setdefault("labels", {})[field_name] = {
        "value": q_by_t, "valid": np.ones(len(schedule), dtype=bool),
        "prov": "synthetic:reveal", "version": TRANSFORMS_VERSION, "candidates": list(candidates), "t": list(schedule),
    }
    return [out]


# ------------------------------------------------------------------ surprise
def surprise(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Contradiction after collapse (docs/relations.md 5.4): once the posterior collapses onto one candidate (its
    mass >= `params.get("collapse", 0.999)`), at every later schedule step (after an `params["after"]`-step grace
    period) draw from `rng` and, with probability `params["rate"]`, trigger belief revision: the collapsed candidate
    is now known wrong, so the survivor set reopens to every OTHER candidate (`params["flip_survivors"]` overrides
    with an explicit replacement set, e.g. when only some alternatives remain consistent). The FIRST trigger is
    recorded as the switch; later schedule steps keep applying any remaining evidence on top of the reopened
    survivors. Pure and seed-deterministic given `rng`'s state."""
    inputs = sample.get("inputs", {})
    candidates = list(inputs.get("candidates", ()))
    prior = inputs.get("prior")
    evidence = sorted(list(inputs.get("evidence", ())), key=lambda e: int(e.get("t", 0)))
    schedule = list(params.get("schedule") or sorted({int(e.get("t", 0)) for e in evidence}))
    rate = float(params.get("rate", 0.0))
    after = int(params.get("after", 0))
    collapse_thr = float(params.get("collapse", 0.999))
    field_name = params.get("field", "surprise")

    collapsed_id = None
    collapse_t = None
    switch_t = None
    q_stream = []
    survivors = set(candidates)
    pending = list(evidence)
    for t in schedule:
        while pending and int(pending[0].get("t", 0)) <= t:
            survivors -= set(pending.pop(0).get("excludes", ()))
        q = _posterior(candidates, prior, survivors)
        if collapsed_id is None and q.max(initial=0.0) >= collapse_thr:
            collapsed_id = candidates[int(np.argmax(q))]
            collapse_t = t
        if (collapsed_id is not None and switch_t is None and collapse_t is not None
                and t >= collapse_t + after and rng.random() < rate):
            switch_t = t
            # belief revision, not further narrowing: the collapsed candidate is now known wrong, so the survivor
            # set reopens to everyone else (`params["flip_survivors"]` overrides with an explicit replacement set).
            flip_to = params.get("flip_survivors")
            survivors = set(flip_to) if flip_to is not None else (set(candidates) - {collapsed_id})
            q = _posterior(candidates, prior, survivors)
        q_stream.append(q)
    q_by_t = np.stack(q_stream) if q_stream else np.zeros((0, len(candidates)))

    out = _append_provenance(sample, {
        "transform": "surprise", "version": TRANSFORMS_VERSION, "schedule": list(schedule),
        "collapse_at": collapse_t, "switch_at": switch_t, "field": field_name,
    })
    out.setdefault("labels", {})[field_name] = {
        "value": q_by_t, "valid": np.ones(len(schedule), dtype=bool),
        "prov": "synthetic:surprise", "version": TRANSFORMS_VERSION, "candidates": list(candidates),
        "t": list(schedule), "switch_at": switch_t,
    }
    return [out]


def recovery_steps(target: np.ndarray, estimate: np.ndarray, switch_index: int, eps: float = 0.05) -> int | None:
    """Steps from `switch_index` (inclusive) until `KL(target_t || estimate_t) < eps` first holds, or `None` if it
    never does within the two streams. `target`/`estimate`: `[T, C]` distributions over the same candidate order (as
    produced by `reveal`/`surprise`'s `labels[...]["value"]` and a model's own posterior estimate, respectively)."""
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if target.shape != estimate.shape:
        raise ValueError(f"target/estimate shape mismatch: {target.shape} vs {estimate.shape}")
    for i in range(switch_index, target.shape[0]):
        if kl_divergence(target[i], estimate[i]) < eps:
            return i - switch_index
    return None


# ------------------------------------------------------------------ cf_swap
def _swap_rows(arr, i: int, j: int):
    arr = np.array(arr, copy=True)
    arr[[i, j], ...] = arr[[j, i], ...]
    return arr


def _swap_pair_axes(arr, i: int, j: int, axis: int):
    arr = np.array(arr, copy=True)
    idx = [slice(None)] * arr.ndim
    idx[axis] = [i, j]
    swap_idx = list(idx)
    swap_idx[axis] = [j, i]
    arr[tuple(idx)] = arr[tuple(swap_idx)]
    return arr


def cf_swap(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Counterfactual swap (docs/relations.md 5.2; today's `binding_cf`): swap two token slots of one token set
    consistently everywhere they are referenced -- ALL of that token set's fields (so identity, position, every
    attribute move together, not just `params["field"]`), every edge whose query or key axis is that token set (rows
    / columns `i`, `j` swapped), and every arity-2 label over that set. `params["field"]` names the field whose two
    highest-valid slots the swap picks by default (`params["pair"]` overrides with an explicit `(i, j)`, for exact,
    reproducible tests); `params["set"]` names the token set (default: the first one carrying `params["field"]`)."""
    inputs = sample.get("inputs", {})
    tokens = inputs.get("tokens", {})
    field_name = params.get("field")
    if field_name is None:
        raise ValueError("cf_swap needs params['field']")
    set_name = params.get("set") or next((s for s, t in tokens.items() if field_name in t.get("fields", {})), None)
    if set_name is None or set_name not in tokens:
        raise ValueError(f"cf_swap: no token set carries field {field_name!r}")
    fields = tokens[set_name]["fields"]
    n = len(next(iter(fields.values())))

    pair = params.get("pair")
    if pair is not None:
        i, j = int(pair[0]), int(pair[1])
    else:
        valid = fields.get(f"{field_name}.valid")
        slots = [k for k in range(n) if valid is None or bool(valid[k])]
        if len(slots) < 2:
            raise ValueError("cf_swap: fewer than two valid slots to swap")
        i, j = (int(x) for x in rng.choice(slots, size=2, replace=False))

    out = copy.deepcopy(sample)
    out_tokens = out["inputs"]["tokens"]
    for f_name, arr in out_tokens[set_name]["fields"].items():
        out_tokens[set_name]["fields"][f_name] = _swap_rows(arr, i, j)
    if "mask" in out_tokens[set_name]:
        out_tokens[set_name]["mask"] = _swap_rows(out_tokens[set_name]["mask"], i, j)
    if "kind" in out_tokens[set_name]:
        out_tokens[set_name]["kind"] = _swap_rows(out_tokens[set_name]["kind"], i, j)

    for e_name, edge in out.get("inputs", {}).get("edges", {}).items():
        q_set, k_set = edge.get("sets", (None, None))
        data = edge["data"]
        if q_set == set_name:
            data = _swap_pair_axes(data, i, j, axis=0)
        if k_set == set_name:
            data = _swap_pair_axes(data, i, j, axis=1)
        edge["data"] = data

    for l_name, label in out.get("labels", {}).items():
        if not isinstance(label, dict) or label.get("set") != set_name:
            continue
        value, valid = label.get("value"), label.get("valid")
        arity = label.get("arity", 1)
        if value is not None:
            label["value"] = _swap_rows(value, i, j) if arity == 1 else _swap_pair_axes(
                _swap_pair_axes(value, i, j, axis=0), i, j, axis=1)
        if valid is not None:
            label["valid"] = _swap_rows(valid, i, j) if arity == 1 else _swap_pair_axes(
                _swap_pair_axes(valid, i, j, axis=0), i, j, axis=1)

    out = _append_provenance(out, {"transform": "cf_swap", "version": TRANSFORMS_VERSION, "field": field_name,
                                    "set": set_name, "swapped": [i, j]})
    return [out]


# ------------------------------------------------------------------ noise
def noise(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Estimator-style noise on a public field (docs/relations.md 5.2): adds `Normal(0, sigma)` to
    `sample["inputs"]["tokens"][set]["fields"][field]` and writes `.var = sigma**2` (broadcast, masked by
    `<field>.valid` when present so invalid / null slots stay untouched), for confidence scaling downstream."""
    field_name, sigma = params.get("field"), float(params.get("sigma", 0.0))
    if field_name is None:
        raise ValueError("noise needs params['field']")
    set_name = params.get("set") or next(
        (s for s, t in sample.get("inputs", {}).get("tokens", {}).items() if field_name in t.get("fields", {})), None)
    if set_name is None:
        raise ValueError(f"noise: no token set carries field {field_name!r}")

    out = copy.deepcopy(sample)
    fields = out["inputs"]["tokens"][set_name]["fields"]
    arr = np.array(fields[field_name], dtype=np.float64, copy=True)
    valid = fields.get(f"{field_name}.valid")
    eps = rng.normal(0.0, sigma, size=arr.shape)
    var = np.full(arr.shape, sigma ** 2)
    if valid is not None:
        m = np.asarray(valid, dtype=bool)
        bshape = (slice(None),) + (None,) * (arr.ndim - m.ndim)
        eps = eps * m[bshape]
        var = var * m[bshape]
    fields[field_name] = arr + eps
    fields[f"{field_name}.var"] = var

    out = _append_provenance(out, {"transform": "noise", "version": TRANSFORMS_VERSION, "field": field_name,
                                    "set": set_name, "sigma": sigma})
    return [out]


# ------------------------------------------------------------------ occlude
def occlude(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Drop tokens / mark invisible (docs/relations.md 5.2): each currently-valid slot of `params["set"]` (default:
    every token set) becomes a null identity independently with probability `params["p"]`, by clearing its `mask`
    (kept as a masked slot per the token model, section 2, not deleted)."""
    p = float(params.get("p", 0.0))
    sets = [params["set"]] if params.get("set") else list(sample.get("inputs", {}).get("tokens", {}))

    out = copy.deepcopy(sample)
    occluded: dict[str, list[int]] = {}
    for set_name in sets:
        tset = out["inputs"]["tokens"].get(set_name)
        if tset is None or "mask" not in tset:
            continue
        mask = np.array(tset["mask"], dtype=bool, copy=True)
        n = mask.shape[0]
        draws = rng.random(n)
        drop = mask & (draws < p)
        mask[drop] = False
        tset["mask"] = mask
        occluded[set_name] = [int(i) for i in np.nonzero(drop)[0]]

    out = _append_provenance(out, {"transform": "occlude", "version": TRANSFORMS_VERSION, "p": p,
                                    "occluded": occluded})
    return [out]


# ------------------------------------------------------------------ subsample
def subsample(sample: Sample, rng: np.random.Generator, params: dict) -> list[Sample]:
    """Frame / knot subsampling for temporal factors (docs/relations.md 5.2): keeps every `params["stride"]`-th
    index (default 1: identity) of `sample["inputs"]["knots"]["fields"][*]` along its time axis (0), and remaps
    `sample["inputs"]["evidence"]` times into the kept index positions (an event whose time falls strictly between
    two kept indices is attached to the next kept index, so evidence is never silently dropped). Deterministic; `rng`
    is accepted only to satisfy the transform contract."""
    del rng
    stride = int(params.get("stride", 1))
    if stride < 1:
        raise ValueError("subsample: stride must be >= 1")

    out = copy.deepcopy(sample)
    knots = out.get("inputs", {}).get("knots")
    kept: list[int] = []
    if knots is not None and knots.get("fields"):
        n = len(next(iter(knots["fields"].values())))
        kept = list(range(0, n, stride))
        for name, arr in knots["fields"].items():
            knots["fields"][name] = np.asarray(arr)[kept]

    if kept:
        remap = {}
        for new_t, old_t in enumerate(kept):
            lo = old_t if new_t == 0 else kept[new_t - 1] + 1
            for t in range(lo, old_t + 1):
                remap[t] = new_t
        evidence = out.get("inputs", {}).get("evidence")
        if evidence:
            max_new = len(kept) - 1
            for ev in evidence:
                t = int(ev.get("t", 0))
                ev["t"] = remap.get(t, max_new if t > kept[-1] else 0)

    out = _append_provenance(out, {"transform": "subsample", "version": TRANSFORMS_VERSION, "stride": stride,
                                    "kept": kept})
    return [out]


# ------------------------------------------------------------------ registration
TRANSFORMS_VERSION = "1"

for _name, _fn in (("reveal", reveal), ("surprise", surprise), ("cf_swap", cf_swap), ("noise", noise),
                    ("occlude", occlude), ("subsample", subsample)):
    register_transform(TransformDef(name=_name, version=TRANSFORMS_VERSION, fn=_fn))
