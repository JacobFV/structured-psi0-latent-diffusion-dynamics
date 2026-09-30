"""Contact / grasp / handover label family (D-144 unit R16; design: docs/relations.md sections 5-6, 10; catalog:
`research/relations_catalog.md` D "physical interaction"). Registers the three `LabelDef`s of `catalog.py`'s
`ix.contact` / `ix.held_by` / `ix.handover` (all `hidden` / `bilinear` / `aug`, source `probe`, label from here) and
the scene part `grasp_target` that makes a graspable object reachable by one or two manipulators.

Every label is a pure function `(StateView, TokenIndex) -> Label` (`rrp.harness.data.relgen`), computed once against
`StateView.contacts()` / `StateView.entities()`; it never touches a factor, a net or an attention site (that is what
`ix.*` in `catalog.py` and `FactorSite` do with the label at training time). `held_pairs` reproduces today's
`Session._held_truth` rule (an object touched by >= 2 distinct hand bodies of one manipulator assembly) at the
`StateView` level: `StateView.contacts()` already resolves every hand body of an assembly to that ONE assembly entity
id (`Session._body_entity_map`), so >= 2 distinct hand bodies shows up as >= 2 separate contact RECORDS between the
same (object, assembly) entity pair -- one mjData contact per contacting geom pair, not per assembly.
"""
from __future__ import annotations

from dataclasses import replace as _replace

import numpy as np

from rrp.envs.base import StateView
from rrp.harness.data.relgen import Label, LabelDef, SceneDraft, ScenePart, TokenIndex, register_label, register_part

__all__ = ["contact_pairs", "held_pairs", "handover_pairs", "grasp_target_build", "grasp_target_vary"]

VERSION = "1"
_MIN_HELD_CONTACTS = 2          # >= 2 finger bodies of one hand assembly (docs/relations.md 6, today's `held` label)


def _ctx_ids(idx: TokenIndex) -> list:
    """The `ctx` token set's per-slot entity ids (every family that carries contact-relevant entities -- arm, dual,
    legged -- names its scene/assembly token set `ctx`, docs/relations.md 2)."""
    if "ctx" not in idx.sets:
        raise KeyError(f"contact labels need a 'ctx' token index; got {sorted(idx.sets)}")
    return idx.sets["ctx"]


def _pair_label(ids: list, pair_true, prov: str) -> Label:
    """`[T,T,1]` float-0/1 `value`; `valid[i,j]` true only when BOTH slots resolve to a known entity (an unmapped /
    null-identity slot's row and column are masked, never scored). `pair_true(a, b)` is evaluated only for valid
    `(i, j)`; every predicate this module passes is symmetric (`pair_true(a, b) == pair_true(b, a)`), matching the
    `Algebra(direction="symmetric")` of all three `ix.*` factors."""
    T = len(ids)
    value = np.zeros((T, T, 1), dtype=np.float32)
    valid = np.zeros((T, T), dtype=bool)
    for i in range(T):
        if ids[i] is None:
            continue
        for j in range(T):
            if ids[j] is None:
                continue
            valid[i, j] = True
            if pair_true(ids[i], ids[j]):
                value[i, j, 0] = 1.0
    return Label(value=value, valid=valid, prov=prov, version=VERSION)


def _contact_id_pairs(sv: StateView) -> set:
    """Unordered `(a, b)` entity-id pairs with >= 1 contact record, excluding self-pairs (`a == b`, seen when both
    sides of one contact resolve to the same manipulator-assembly entity, e.g. two fingertips of one gripper)."""
    return {frozenset((c.a, c.b)) for c in sv.contacts() if c.a != c.b}


def _contact_counts(sv: StateView) -> dict:
    """Unordered `(a, b)` entity-id pair -> number of contact records between them."""
    counts: dict = {}
    for c in sv.contacts():
        if c.a == c.b:
            continue
        k = frozenset((c.a, c.b))
        counts[k] = counts.get(k, 0) + 1
    return counts


def _entity_kinds(sv: StateView) -> dict:
    return {e.id: e.kind for e in sv.entities()}


# ------------------------------------------------------------------ contact_pairs (`ix.contact`)
def contact_pairs(sv: StateView, idx: TokenIndex) -> Label:
    """Any two entities with >= 1 contact record between them (docs/relations.md 6: `ix.contact`)."""
    pairs = _contact_id_pairs(sv)
    return _pair_label(_ctx_ids(idx), lambda a, b: frozenset((a, b)) in pairs, prov="gt")


# ------------------------------------------------------------------ held_pairs (`ix.held_by`)
def held_pairs(sv: StateView, idx: TokenIndex) -> Label:
    """An object entity held by a manipulator-assembly entity: >= `_MIN_HELD_CONTACTS` contact records between them
    (see module docstring; today's `Session._held_truth`). Manipulator<->manipulator and object<->object pairs are
    never "held"."""
    counts = _contact_counts(sv)
    kinds = _entity_kinds(sv)

    def held(a, b):
        a_asm, b_asm = kinds.get(a) == "assembly", kinds.get(b) == "assembly"
        if a_asm == b_asm:                          # exactly one side must be the manipulator assembly
            return False
        return counts.get(frozenset((a, b)), 0) >= _MIN_HELD_CONTACTS

    return _pair_label(_ctx_ids(idx), held, prov="gt")


# ------------------------------------------------------------------ handover_pairs (`ix.handover`)
def handover_pairs(sv: StateView, idx: TokenIndex) -> Label:
    """Two manipulator-assembly entities simultaneously in contact with the same object entity: a handover in
    progress (docs/relations.md 6: `ix.handover`, "dual only"; `research/relations_catalog.md` D "handover")."""
    kinds = _entity_kinds(sv)
    holders_of: dict = {}                            # object entity id -> set of assembly entity ids touching it
    for c in sv.contacts():
        for obj, other in ((c.a, c.b), (c.b, c.a)):
            if kinds.get(other) == "assembly" and kinds.get(obj) != "assembly":
                holders_of.setdefault(obj, set()).add(other)
    handoff_pairs = {frozenset((x, y)) for holders in holders_of.values() for x in holders for y in holders
                     if x != y}
    return _pair_label(_ctx_ids(idx), lambda a, b: frozenset((a, b)) in handoff_pairs, prov="gt")


for _name, _fn in (("contact_pairs", contact_pairs), ("held_pairs", held_pairs), ("handover_pairs", handover_pairs)):
    register_label(LabelDef(name=_name, version=VERSION, arity=2, needs=frozenset({"poses", "contacts"}), fn=_fn,
                            prov="gt"))


# ------------------------------------------------------------------ part `grasp_target`
def grasp_target_build(draft: SceneDraft, rng: np.random.Generator) -> None:
    """Adds one graspable object within reach of the scene's manipulator(s) (docs/relations.md 5.2, 10;
    `research/relations_catalog.md` D "grasp" / "handover"). Declarative only, at the `SceneDraft` level this fanout
    unit owns: wiring `compose` (R11) into a live env's scenario builder is that unit's job, not this one's.
    `rng`-driven placement only (no other global randomness), so a fixed generator reproduces a fixed scene."""
    obj_id = f"grasp_obj_{sum(1 for e in draft.entities if e.get('role') == 'graspable')}"
    pos = rng.uniform([-0.05, -0.05, 0.0], [0.05, 0.05, 0.02]).tolist()
    draft.entities.append({"id": obj_id, "kind": "object", "pos": pos, "role": "graspable"})
    draft.events.append({"event": "grasp", "entity": obj_id})


def grasp_target_vary(draft: SceneDraft, rng: np.random.Generator, factor: str) -> list:
    """Decoupling pairs for `factor` (docs 5.2: "k copies differing ONLY in the named factor's value"). `factor ==
    "reach"`: the part's own object moved between inside (near) and outside (far) a manipulator's reach, everything
    else in the draft held fixed -- the pair that lets an `ix.contact` / `ix.held_by` label toggle without any other
    scene change confounding it. Requires `grasp_target_build` to have run on `draft` already."""
    if factor != "reach":
        raise ValueError(f"grasp_target: no vary axis {factor!r} (only 'reach')")
    if not draft.entities or draft.entities[-1].get("role") != "graspable":
        raise ValueError("grasp_target.vary: call build() first (no graspable object on this draft)")
    del rng
    near = _replace(draft, entities=[*draft.entities[:-1], {**draft.entities[-1], "pos": [0.02, 0.0, 0.01]}])
    far = _replace(draft, entities=[*draft.entities[:-1], {**draft.entities[-1], "pos": [0.6, 0.6, 0.6]}])
    return [near, far]


register_part(ScenePart(name="grasp_target", version="1", activates=frozenset({"contact"}),
                        envs=("mujoco/arm", "mujoco/dual"), build=grasp_target_build, vary=grasp_target_vary))
