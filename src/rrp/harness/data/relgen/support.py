"""Support / force-flow data generation (D-144 unit R17; design: docs/relations.md sections 5, 6, 10 row R17;
catalog: research/relations_catalog.md D "support / stacking", "force transfer").

  `support_pairs`    LABELS entry: "a supports b" iff a and b are in contact and the contact normal (world,
                      pointing from its `.a` to its `.b`, `rrp.envs.base.ContactState.normal`) is within
                      `SUPPORT_ANGLE_DEG` of gravity-up, evaluated at the contact -- which is "at a's top" exactly
                      when that normal points from a up into b (the lower / upper assignment below); a contact
                      whose position falls below the lower entity's own origin along "up" is rejected as a stray /
                      degenerate contact, not a real "resting on top of" contact, whenever poses are available for
                      that entity. Arity-2 label over one token set ([T, T, 1] 0/1; valid where both slots hold an
                      entity id and i != j).
  `support_closure`  LABELS entry: the transitive closure of `support_pairs`'s direct graph -- upstream (ancestors)
                      / downstream (descendants) along the support chain. This is the ground-truth target
                      `ix.force_flow` (its `flow` operator recomputes the identical closure over the estimated /
                      given edge graph at forward time, docs/relations.md 3.2) trains and is evaluated against.
  `stack`            PARTS entry: `n` objects stacked directly on top of one another, `activates={"support",
                      "force_flow"}`; `vary(draft, rng, factor)` returns copies that differ ONLY in which object
                      occupies which position in the stack (docs 5.2's decoupling pairs) -- every object's own
                      physical properties (size, mass, color) travel with its id, only `order` (bottom -> top)
                      changes.

Both label functions are pure `(StateView, TokenIndex) -> Label` functions (docs 5.2): no backend-specific code,
they run in every env whose `caps` cover `LabelDef.needs`. `stack` is likewise backend-agnostic: it mutates a
`SceneDraft` (declarative), never touches a simulator -- an env's own builder (outside this unit) turns the
resolved draft into real bodies.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from rrp.envs.base import StateView
from rrp.harness.data.relgen import (Label, LabelDef, SceneDraft, ScenePart, TokenIndex, register_label,
                                     register_part)

__all__ = ["SUPPORT_ANGLE_DEG", "support_matrix", "support_closure", "support_pairs_fn", "support_closure_fn"]

VERSION = "1"
SUPPORT_ANGLE_DEG = 30.0


# ------------------------------------------------------------------ label `support_pairs` / `support_closure`
def _up(gravity) -> np.ndarray:
    """World "up" = -gravity, normalized (falls back to +z for a ~zero-gravity fixture)."""
    g = np.asarray(gravity, dtype=np.float64)
    n = float(np.linalg.norm(g))
    return np.array([0.0, 0.0, 1.0]) if n < 1e-9 else -g / n


def support_matrix(view: StateView, angle_deg: float = SUPPORT_ANGLE_DEG) -> dict[str, dict[str, bool]]:
    """Direct support graph over entity ids: `out[a][b]` is True iff some contact between `a` and `b` has a world
    normal within `angle_deg` of gravity-up (`a` below, `b` above) or within `angle_deg` of gravity-DOWN (`b` below,
    `a` above -- the same physical relation read from the other contact endpoint order). Several contact points
    between one pair (a real box-on-box rest typically reports more than one) just OR into the same `out[a][b]`.
    When entity poses are known (`view.entities()`), a contact whose position falls below the lower entity's own
    origin along "up" is rejected: a genuine "resting on top" contact is always at or above the supporting body's
    own origin, so this catches a stray side contact whose normal happens to pass the angle test by coincidence.
    Entities the caller never returns from `entities()` (e.g. an unmapped/world body) skip that corroboration --
    the angle test alone still decides them."""
    up = _up(view.gravity)
    cos_thr = float(np.cos(np.deg2rad(angle_deg)))
    poses = {e.id: np.asarray(e.pos, dtype=np.float64) for e in view.entities()}
    out: dict[str, dict[str, bool]] = {}
    for c in view.contacts():
        n = np.asarray(c.normal, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        cosang = float(np.dot(n, up) / nn)
        if cosang >= cos_thr:
            lower, upper = c.a, c.b
        elif cosang <= -cos_thr:
            lower, upper = c.b, c.a
        else:
            continue
        lo = poses.get(lower)
        if lo is not None and float(np.dot(np.asarray(c.pos, dtype=np.float64) - lo, up)) < -1e-6:
            continue
        out.setdefault(lower, {})[upper] = True
    return out


def support_closure(direct: dict[str, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    """Transitive closure of a direct support graph (`support_matrix`'s output, or any `{a: {b: True, ...}, ...}`):
    `out[a][b]` is True iff `b` is downstream of `a` (equivalently, `a` is upstream of `b`) along one or more
    support edges. Pure, no simulator; ids with an empty reachable set are simply absent from `out`."""
    nodes = set(direct) | {b for row in direct.values() for b in row}
    reach: dict[str, set] = {a: set(direct.get(a, {})) for a in nodes}
    changed = True
    while changed:
        changed = False
        for a in nodes:
            grown = set()
            for b in reach[a]:
                grown |= reach.get(b, set())
            if not grown <= reach[a]:
                reach[a] |= grown
                changed = True
    return {a: {b: True for b in bs} for a, bs in reach.items() if bs}


def _slots(idx: TokenIndex) -> list:
    if not idx.sets:
        raise ValueError("support labels need at least one token set in TokenIndex.sets")
    name = "ctx" if "ctx" in idx.sets else next(iter(idx.sets))
    return idx.sets[name]


def _to_label(graph: dict[str, dict[str, bool]], ids: list) -> Label:
    T = len(ids)
    value = np.zeros((T, T, 1), dtype=np.float64)
    valid = np.zeros((T, T), dtype=bool)
    for i, ai in enumerate(ids):
        if ai is None:
            continue
        row = graph.get(ai, {})
        for j, bj in enumerate(ids):
            if bj is None or i == j:
                continue
            valid[i, j] = True
            if row.get(bj):
                value[i, j, 0] = 1.0
    return Label(value=value, valid=valid, prov="gt", version=VERSION)


def support_pairs_fn(view: StateView, idx: TokenIndex) -> Label:
    return _to_label(support_matrix(view), _slots(idx))


def support_closure_fn(view: StateView, idx: TokenIndex) -> Label:
    return _to_label(support_closure(support_matrix(view)), _slots(idx))


register_label(LabelDef(name="support_pairs", version=VERSION, arity=2, needs=frozenset({"contacts", "poses"}),
                        fn=support_pairs_fn, prov="gt"))
register_label(LabelDef(name="support_closure", version=VERSION, arity=2, needs=frozenset({"contacts", "poses"}),
                        fn=support_closure_fn, prov="gt"))


# ------------------------------------------------------------------ part `stack`
def _stack_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    """`draft.kwargs["stack_n"]` objects (default 3), stacked bottom -> top. Every object's physical properties are
    drawn once, keyed by its own id, and stored in `draft.kwargs["stack"]["objects"]`; `draft.kwargs["stack"]
    ["order"]` is the sole record of which object occupies which position (`vary` below only ever touches it)."""
    n = int(draft.kwargs.get("stack_n", 3))
    size = float(draft.kwargs.get("stack_size", 0.03))
    tag = f"stack{sum(1 for p in draft.parts if p == 'stack')}"
    ids = [f"{tag}_{i}" for i in range(n)]
    objects = {oid: {"size": [size, size, size], "mass": float(rng.uniform(0.05, 0.2)),
                     "color": [float(x) for x in rng.uniform(0.2, 0.9, size=3)]} for oid in ids}
    draft.entities = list(draft.entities) + [dict(objects[oid], id=oid, kind="object") for oid in ids]
    draft.kwargs = dict(draft.kwargs, stack={"order": ids, "objects": objects})
    draft.active = draft.active | {"support", "force_flow"}
    draft.parts = draft.parts + ("stack",)
    draft.provenance = dict(draft.provenance, stack={"version": VERSION, "n": n, "order": list(ids)})


def _stack_vary(draft: SceneDraft, rng: np.random.Generator, factor: str) -> list[SceneDraft]:
    """Decoupling pairs for `stack` (docs 5.2: "k copies differing ONLY in the named factor's value"): `stack` has
    exactly one varyable dynamic -- which object occupies which position -- so every `factor` name reaches the same
    permutation of `draft.kwargs["stack"]["order"]`; nothing else in the draft (object identities, their sizes /
    masses / colors, entities, events, every other kwarg) changes. Up to 3 distinct permutations, deterministic
    given `rng`; fewer than 2 objects has nothing to vary (`[draft]`)."""
    del factor
    spec = draft.kwargs.get("stack")
    base = list(spec["order"]) if spec else []
    if len(base) < 2:
        return [draft]
    seen = {tuple(base)}
    variants: list[SceneDraft] = []
    budget = 20
    while len(variants) < min(3, len(base) - 1) and budget > 0:
        budget -= 1
        perm = list(base)
        rng.shuffle(perm)
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        new_kwargs = copy.deepcopy(draft.kwargs)
        new_kwargs["stack"]["order"] = perm
        new_prov = copy.deepcopy(draft.provenance)
        if "stack" in new_prov:
            new_prov["stack"]["order"] = list(perm)
        variants.append(replace(draft, kwargs=new_kwargs, provenance=new_prov))
    return variants


register_part(ScenePart(name="stack", version=VERSION, activates=frozenset({"support", "force_flow"}),
                        build=_stack_build, vary=_stack_vary, envs=("mujoco/arm", "mujoco/dual")))
